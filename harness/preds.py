"""preds.py [--device cuda:0] FEATS_DIR... — the test-row predictions behind fit.json for two readouts: mbmean with
mRNABench's probe (raw features, grid A) and regions with ours (raw features, grid B). Splits, labels, the Evo 1 bf16
mbmean and cp.Fitter as fit_ext.py, so the per-seed test Pearson r matches fit.json's `mbmean|rawA` / `regions|rawB`.

Skips a FEATS_DIR whose output exists. Writes results/preds/<model dir>/<table>.npz (through a temporary file): `order` (parquet rows, fit.py's order), `test` (10 x n, bool),
`y|<target>` (n), `mbmean|<target>` and `regions|<target>` (10 x n, float32, NaN off the seed's test rows), and prints
the largest |r - fit.json| per file.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import CODE, WORK

spec = importlib.util.spec_from_file_location("fe", CODE / "harness/fit_ext.py")
fe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fe)
cp = fe.cp
OUT = WORK / "results/preds"
READ = (("mbmean", "A"), ("regions", "B"))


def main():
    args = sys.argv[1:]
    device = "cuda:0"
    if "--device" in args:
        i = args.index("--device")
        device = args[i + 1]
        del args[i:i + 2]
    dev = torch.device(device)
    for fd in map(Path, args):
        started = time.monotonic()
        dest = OUT / fd.parent.parent.name
        if (dest / f"{fd.parent.name}.npz").exists():
            continue
        meta = json.loads((fd / "meta.json").read_text())
        table = meta["table"]
        key = cp.KEYS[table]
        df = pd.read_parquet(cp.DATA / f"{key}.parquet")
        order = np.array(meta["order"])
        cuts = cp.seed_cuts(key, df, order, meta["model"])
        names = {c: fe.LABELS.get(table, {}).get(c, "batch.y") for c in meta["targets"]}
        labels = {names[c]: torch.tensor(v, dtype=torch.float64, device=dev) for c, v in meta["targets"].items()}
        X = {nm: torch.from_numpy(np.load(fd / f"{nm}.npy")) for nm in ("mbmean", "bases", "regions")}
        if meta["model"].startswith("evo-1") and meta.get("mbmean") != "bf16":
            X["mbmean"] = X["bases"].bfloat16().float()
        n = X["bases"].shape[0]
        feats = {"mbmean": X["mbmean"].contiguous().to(dev), "regions": X["regions"].reshape(n, -1).contiguous().to(dev)}
        groups: dict = {}
        for k, y in labels.items():
            have = torch.isfinite(y)
            groups.setdefault(have.cpu().numpy().tobytes(), (have, []))[1].append(k)
        out = {"order": order, "test": np.stack([c == "test" for c in cuts])}
        out |= {f"y|{t}": y.cpu().numpy() for t, y in labels.items()}
        for fset, _ in READ:
            out |= {f"{fset}|{t}": np.full((len(cuts), n), np.nan, np.float32) for t in labels}
        for si, cut in enumerate(cuts):
            for have, tnames in groups.values():
                tr = torch.from_numpy(cut == "train").to(dev) & have
                te = torch.from_numpy(cut == "test").to(dev) & have
                rows = te.nonzero().squeeze(1).cpu().numpy()
                for fset, grid in READ:
                    fit = cp.Fitter(feats[fset], tr, [labels[t][tr] for t in tnames], ("raw",))
                    for ti, t in enumerate(tnames):
                        pt, _, _ = fit.predict(te, "raw", ti, grid)
                        out[f"{fset}|{t}"][si, rows] = pt.cpu().numpy()
                    del fit
        fit_json = json.loads((fd.parent / "fit.json").read_text())["results"]
        worst = 0.0
        for t in labels:
            y = out[f"y|{t}"]
            for fset, grid in READ:
                ref = fit_json[t][f"{fset}|raw{grid}"]["test"]
                for si in range(len(cuts)):
                    p = out[f"{fset}|{t}"][si]
                    ok = ~np.isnan(p)
                    worst = max(worst, abs(np.corrcoef(p[ok].astype(float), y[ok])[0, 1] - ref[si]))
        dest.mkdir(parents=True, exist_ok=True)
        tmp = dest / f"{table}.{os.getpid()}.tmp.npz"
        np.savez(tmp, **out)
        os.replace(tmp, dest / f"{table}.npz")
        print(f"{fd.parent.parent.name} {table}: {n} rows, max |test r - fit.json| {worst:.1e}  ({(time.monotonic() - started) / 60:.1f} min)",
              flush=True)


if __name__ == "__main__":
    main()
