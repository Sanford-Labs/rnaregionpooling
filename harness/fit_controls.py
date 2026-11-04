"""fit_controls.py [--cpu] FEATS_DIR... — the feature sets missing from FEATS_DIR/../fit.json on raw features (mRNABench's
RidgeCV estimator, grids A and B), fitted and merged into it: regions and split3 (and their PCA-d, grid B), rbal, the single
blocks r5 / rc / r3, F6 alone and appended to bases / split3 / regions (regression tables but MPRA, and mrl-varying), the
boundary shifts and the random boundaries. Splits, labels, F6 and Fitter as fit_ext.py; keys already in fit.json are left as
they are. --cpu fits on the CPU (no card, no fit slot).
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


def main():
    if "--cpu" in sys.argv:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        slot, card = fe.take_slot()  # noqa: F841  (held until exit)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(card)
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    for fd in map(Path, [a for a in sys.argv[1:] if not a.startswith("--")]):
        started = time.monotonic()
        fit_p = fd.parent / "fit.json"
        out = json.loads(fit_p.read_text())
        res = out["results"]
        have = set(next(iter(res.values())))
        meta = json.loads((fd / "meta.json").read_text())
        table = meta["table"]
        key = cp.KEYS[table]
        df = pd.read_parquet(cp.DATA / f"{key}.parquet")
        order = np.array(meta["order"])
        cuts = cp.seed_cuts(key, df, order, meta["model"])
        names = {c: fe.LABELS.get(table, {}).get(c, "batch.y") for c in meta["targets"]}
        labels = {names[c]: torch.tensor(v, dtype=torch.float64, device=dev) for c, v in meta["targets"].items()}
        X = {nm: torch.from_numpy(np.load(fd / f"{nm}.npy")) for nm in ("bases", "regions", "split3")}
        n, d = X["bases"].shape
        R, S3 = X["regions"].reshape(n, -1), X["split3"].reshape(n, -1)
        static = []  # (fset, x, variants, top)
        miss = lambda fset: f"{fset}|rawB" not in have  # noqa: E731
        for fset, x in (("regions", R), ("split3", S3)):
            pca = f"pca_{fset}|rawB" not in have
            if miss(fset) or pca:
                static.append((fset, x, ("raw",) if miss(fset) else (), d if pca else None))
        Rb = X["regions"]
        if miss("rbal"):
            empty = Rb.abs().sum(-1) == 0
            static.append(("rbal", ((Rb.double() * (~empty).unsqueeze(-1)).sum(1) / (~empty).sum(1, keepdim=True).clamp_min(1)), ("raw",), None))
        static += [(k, Rb[:, i], ("raw",), None) for i, k in enumerate(("r5", "rc", "r3")) if miss(k)]
        if (fd / "shift.npy").exists():
            sh = np.load(fd / "shift.npy", mmap_mode="r")
            rd = np.load(fd / "random.npy", mmap_mode="r")
            static += [(f"shift{s:+d}", torch.from_numpy(np.array(sh[:, i])).reshape(n, -1), ("raw",), None)
                       for i, s in enumerate(meta["shifts"]) if miss(f"shift{s:+d}")]
            static += [(f"random{j}", torch.from_numpy(np.array(rd[:, j])).reshape(n, -1), ("raw",), None)
                       for j in range(meta["draws"]) if miss(f"random{j}")]
        use_f6 = ((table not in cp.MPRA) or table == "mrl-varying") and miss("regions+F6")
        if use_f6:
            l5, lc, l3 = (np.array(meta[k], float) for k in ("l5", "lc", "l3"))
            u5, cds, u3 = (df[c].fillna("").to_numpy()[order] for c in ("UTR5_sequence", "CDS_sequence", "UTR3_sequence"))
            gc = [np.array([(s.upper().count("G") + s.upper().count("C")) / len(s) if len(s) else np.nan for s in v]) for v in (u5, cds, u3)]
            f6_raw = np.stack([np.log1p(l5), np.log1p(lc), np.log1p(l3), *gc], 1)
        if not static and not use_f6:
            print(f"{meta['model']} {table}: nothing missing", flush=True)
            continue
        groups: dict = {}
        for k, y in labels.items():
            ok = torch.isfinite(y)
            groups.setdefault(ok.cpu().numpy().tobytes(), (ok, []))[1].append(k)
        new = {k: {} for k in labels}
        for si, cut in enumerate(cuts):
            for ok, tnames in groups.values():
                m = {s: torch.from_numpy(cut == s).to(dev) & ok for s in ("train", "val", "test")}
                tr, va, te = m["train"], m["val"], m["test"]
                ys = [labels[t][tr] for t in tnames]
                plan = list(static)
                if use_f6:
                    f6 = torch.from_numpy(f6_raw.copy())
                    f6 = torch.where(torch.isnan(f6), torch.nanmean(f6[tr.cpu()], 0).expand_as(f6), f6)
                    plan += [("F6", f6, ("raw",), None), ("bases+F6", torch.cat([X["bases"].double(), f6], 1), ("raw",), None),
                             ("split3+F6", torch.cat([S3.double(), f6], 1), ("raw",), None),
                             ("regions+F6", torch.cat([R.double(), f6], 1), ("raw",), None)]
                for fset, x, variants, top in plan:
                    fit = cp.Fitter(x.contiguous().to(dev), tr, ys, variants, top=top)
                    for ti, t in enumerate(tnames):
                        for variant in fit.fits:
                            for grid in ("A", "B") if variant == "raw" else ("B",):
                                pv, alpha, _ = fit.predict(va, variant, ti, grid)
                                pt, _, _ = fit.predict(te, variant, ti, grid)
                                k = f"pca_{fset}|rawB" if variant == "pca" else f"{fset}|{variant}{grid}"
                                r = new[t].setdefault(k, {"val": [None] * 10, "test": [None] * 10, "alpha": [None] * 10})
                                r["val"][si], r["test"][si], r["alpha"][si] = cp.pearson(pv, labels[t][va]), cp.pearson(pt, labels[t][te]), alpha
                    del fit
        for t in res:
            res[t].update(new[t])
        out["controls_seconds"] = time.monotonic() - started
        tmp = fit_p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out))
        os.replace(tmp, fit_p)
        added = sorted(new[next(iter(new))])
        print(f"{meta['model']} {table}: added {len(added)} sets ({', '.join(added[:4])}{' ...' if len(added) > 4 else ''})  "
              f"({(time.monotonic() - started) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
