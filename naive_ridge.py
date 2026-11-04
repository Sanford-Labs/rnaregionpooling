"""naive_ridge.py [--six] TABLE... — mRNABench's naive-4 baseline (with --six: naive-6, + CDS length and the count of
splice sites across the three regions, as naive_baseline.py's six-track variant; TE tables read the te_refetch copies,
which carry splice coordinates) (3-7-mer counts over ACGT + GC ratio, 21,825 features,
naive_baseline.py) on the non-MPRA regression tables, ridge in dual form (n x n kernel), float64, CPU.
Same ten seeds, cutter and LOO-alpha rule as fit.py; grid A (mRNABench) and grid B. Writes results/naive/<table>.json.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from paths import CODE, WORK

spec = importlib.util.spec_from_file_location("cp", CODE / "cpu_probe.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)
torch.set_num_threads(8)
OUT = WORK / "results/naive"
CODE = np.full(256, -1, np.int64)
for i, ch in enumerate("ACGT"):
    CODE[ord(ch)] = i


def kmer_features(seqs):
    blocks = []
    for k in range(3, 8):
        m = np.zeros((len(seqs), 4 ** k), np.float32)
        for r, s in enumerate(seqs):
            c = CODE[np.frombuffer(s.encode(), np.uint8)]
            if len(c) < k:
                continue
            win = np.lib.stride_tricks.sliding_window_view(c, k)
            ok = (win >= 0).all(1)
            idx = (win[ok] * (4 ** np.arange(k - 1, -1, -1))).sum(1)
            m[r] = np.bincount(idx, minlength=4 ** k)
        blocks.append(m)
    gc = np.array([[(s.count("G") + s.count("C")) / max(sum(s.count(x) for x in "ACGT"), 1)] for s in seqs], np.float32)
    return np.concatenate(blocks + [gc], 1)


def dual_fit(K, ytr, alphas):
    """Ridge with an unpenalized intercept from a centred train kernel; LOO over `alphas` (cpu_probe's rule)."""
    n = K.shape[0]
    s, U = torch.linalg.eigh(K)
    s = s.clamp_min(0)
    yc = ytr - ytr.mean()
    b = U.T @ yc
    U2 = U * U
    errs = []
    for a in alphas:
        f = s / (s + a)
        fitted = U @ (b * f)
        lev = 1.0 / n + U2 @ f
        r = (yc - fitted) / (1 - lev).clamp_min(1e-12)
        errs.append(float((r * r).mean()))
    return s, U, b, errs




def main():
    OUT.mkdir(parents=True, exist_ok=True)
    six = "--six" in sys.argv
    for table in [t for t in sys.argv[1:] if t != "--six"]:
        started = time.monotonic()
        key = cp.KEYS[table]
        df = pd.read_parquet(cp.DATA / f"{key}.parquet")
        order = np.concatenate([df.index[df["split"] == s].to_numpy() for s in ("train", "val", "test")])
        seqs = [(a + b + c).upper() for a, b, c in zip(*(df[col].fillna("").to_numpy()[order] for col in ("UTR5_sequence", "CDS_sequence", "UTR3_sequence")))]
        X = torch.from_numpy(kmer_features(seqs)).double()
        if six:
            lc = df["CDS_sequence"].fillna("").str.len().to_numpy()[order].astype(float)
            sites = sum(df[c].apply(lambda v: 0 if v is None else len(v)).to_numpy()[order].astype(float)
                        for c in ("UTR5_splice_coord", "CDS_splice_coord", "UTR3_splice_coord"))
            X = torch.cat([X, torch.from_numpy(np.stack([lc, sites], 1))], 1)
        cuts = cp.seed_cuts(key, df, order)
        tcols = [c for c in df.columns if c.startswith("target")]
        res = {}
        for c in tcols:
            y = torch.tensor(df[c].to_numpy(float)[order], dtype=torch.float64)
            per = {"A": {"val": [], "test": [], "alpha": []}, "B": {"val": [], "test": [], "alpha": []}}
            for cut in cuts:
                have = torch.isfinite(y)
                tr, va, te = (torch.from_numpy(cut == s) & have for s in ("train", "val", "test"))
                mu = X[tr].mean(0)
                Xc = X - mu
                Ktr = Xc[tr] @ Xc[tr].T
                s, U, b, errs = dual_fit(Ktr, y[tr], cp.GRID_B)
                errs = np.array(errs)
                for g, idx in (("A", cp.A_IN_B), ("B", list(range(len(cp.GRID_B))))):
                    k = idx[int(np.argmin(errs[idx]))]
                    a = cp.GRID_B[k]
                    coef = U @ (b / (s + a))                        # dual coefficients on centred train rows
                    for split, rows in (("val", va), ("test", te)):
                        pred = (Xc[rows] @ Xc[tr].T) @ coef + y[tr].mean()
                        per[g][split].append(cp.pearson(pred, y[rows]))
                    per[g]["alpha"].append(a)
            res[c] = per
            print(f"naive-{6 if six else 4} {table} {c}: A test {np.mean(per['A']['test']):.4f}  B test {np.mean(per['B']['test']):.4f}  "
                  f"({(time.monotonic() - started) / 60:.1f} min)", flush=True)
        (OUT / f"{table}{'.six' if six else ''}.json").write_text(json.dumps({"table": table, "features": int(X.shape[1]), "results": res}))


if __name__ == "__main__":
    main()
