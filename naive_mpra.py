"""naive_mpra.py [--six] [--device cuda:0] TABLE... — mRNABench's naive-4 / naive-6 rows on the MPRA tables.

naive_ridge.py's features and readout (3-7-mer counts + GC ratio, 21,825 features; --six adds CDS length and the
splice-site count; raw features, grids A and B, LOO alpha, intercept unpenalized), in primal form: the dual n x n
kernel of naive_ridge.py does not fit 318,468 rows, so cp.Fitter sums the 21,825-wide Gram in float64 on the card
over row chunks taken from host memory. Same seeds and splits (cp.seed_cuts). Writes results/naive/<table>[.six].json
in naive_ridge.py's format.
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from paths import CODE, WORK

spec = importlib.util.spec_from_file_location("nr", CODE / "naive_ridge.py")
nr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nr)
cp = nr.cp
OUT = __import__("pathlib").Path(os.environ.get("NAIVE_OUT", str(nr.OUT)))  # a check writes elsewhere


def main():
    six = "--six" in sys.argv
    dev = torch.device(sys.argv[sys.argv.index("--device") + 1]) if "--device" in sys.argv else torch.device("cuda:0")
    tables = [t for i, t in enumerate(sys.argv[1:], 1) if not t.startswith("--") and sys.argv[i - 1] != "--device"]
    frac = os.environ.get("EMBED_MEM_FRACTION")
    if frac and dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(frac), dev)
    for table in tables:
        started = time.monotonic()
        key = cp.KEYS[table]
        df = pd.read_parquet(cp.DATA / f"{key}.parquet")
        order = np.concatenate([df.index[df["split"] == s].to_numpy() for s in ("train", "val", "test")])
        seqs = [(a + b + c).upper() for a, b, c in zip(*(df[col].fillna("").to_numpy()[order] for col in ("UTR5_sequence", "CDS_sequence", "UTR3_sequence")))]
        X = torch.from_numpy(nr.kmer_features(seqs))
        if six:
            lc = df["CDS_sequence"].fillna("").str.len().to_numpy()[order].astype(np.float32)
            sites = sum(df[c].apply(lambda v: 0 if v is None else len(v)).to_numpy()[order].astype(np.float32)
                        for c in ("UTR5_splice_coord", "CDS_splice_coord", "UTR3_splice_coord"))
            X = torch.cat([X, torch.from_numpy(np.stack([lc, sites], 1))], 1)
        print(f"{table}: {X.shape[0]} rows x {X.shape[1]} features ({(time.monotonic() - started) / 60:.1f} min)", flush=True)
        cuts = cp.seed_cuts(key, df, order)
        tcols = [c for c in df.columns if c.startswith("target")]
        ys = {c: torch.tensor(df[c].to_numpy(float)[order], dtype=torch.float64, device=dev) for c in tcols}
        groups = {}
        for c, y in ys.items():
            have = torch.isfinite(y)
            groups.setdefault(have.cpu().numpy().tobytes(), (have, []))[1].append(c)
        res = {c: {g: {"val": [], "test": [], "alpha": []} for g in ("A", "B")} for c in tcols}
        for cut in cuts:
            for have, cs in groups.values():
                tr, va, te = (torch.from_numpy(cut == s).to(dev) & have for s in ("train", "val", "test"))
                fit = cp.Fitter(X, tr, [ys[c][tr] for c in cs], ("raw",))
                for ti, c in enumerate(cs):
                    for g in ("A", "B"):
                        pv, alpha, _ = fit.predict(va, "raw", ti, g)
                        pt, _, _ = fit.predict(te, "raw", ti, g)
                        res[c][g]["val"].append(cp.pearson(pv, ys[c][va]))
                        res[c][g]["test"].append(cp.pearson(pt, ys[c][te]))
                        res[c][g]["alpha"].append(alpha)
                del fit
        for c in tcols:
            print(f"naive-{6 if six else 4} {table} {c}: A test {np.mean(res[c]['A']['test']):.4f}  B test {np.mean(res[c]['B']['test']):.4f}", flush=True)
        (OUT / f"{table}{'.six' if six else ''}.json").write_text(json.dumps({"table": table, "features": int(X.shape[1]), "results": res}))
        print(f"wrote {table} ({(time.monotonic() - started) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
