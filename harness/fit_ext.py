"""fit_ext.py FEATS_DIR... — cpu_probe.py's ridge (float64, on the card of the fit slot it takes) (grid A, grid B, standardized grid B) on embed_mm.py features.

Feature sets: mbmean, bases, regions, split3, rbal, r5/rc/r3, shift<d>, random<j>, and F6 (three log1p lengths,
three GC) alone and appended to bases / split3 / regions. Writes FEATS_DIR/../fit.json in cpu_probe.py's format.
"""
import importlib.util
import json
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

spec = importlib.util.spec_from_file_location("cp", CODE / "cpu_probe.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)
torch.set_num_threads(8)

LABELS = {"mrl-egfp": {"target_mrl_egfp_unmod": "batch.unmod", "target_mrl_egfp_pseudo": "batch.pseudo",
                       "target_mrl_egfp_m1pseudo": "batch.m1pseudo"},
          "mrl-hl": {"target_ribosome_load": "batch.mrl", "target_in_cell_half_life": "batch.hl"}}


SLOTS = 2  # at most this many fits on the CPU at once, across every worker


def take_slot():
    """Block until one of SLOTS lock files is free; the lock lives as long as this process."""
    import fcntl
    import time as _t
    d = WORK / "results/.fit-slots"
    d.mkdir(parents=True, exist_ok=True)
    while True:
        for i in range(SLOTS):
            fh = open(d / f"slot{i}", "w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fh, i
            except BlockingIOError:
                fh.close()
        _t.sleep(30)


def main():
    import os
    slot, card = take_slot()  # noqa: F841  (held until exit)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(card)  # slot i runs on card i
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    for fd in map(Path, sys.argv[1:]):
        started = time.monotonic()
        meta = json.loads((fd / "meta.json").read_text())
        table = meta["table"]
        key = cp.KEYS[table]
        df = pd.read_parquet(cp.DATA / f"{key}.parquet")
        order = np.array(meta["order"])
        full = np.concatenate([df.index[df["split"] == s].to_numpy() for s in ("train", "val", "test")])
        assert len(order) == len(full) and (order == full).all(), "rows are not fit.py's order"
        cuts = cp.seed_cuts(key, df, order, meta["model"])
        names = {c: LABELS.get(table, {}).get(c, "batch.y") for c in meta["targets"]}
        labels = {names[c]: torch.tensor(v, dtype=torch.float64, device=dev) for c, v in meta["targets"].items()}
        l5, lc, l3 = (np.array(meta[k], float) for k in ("l5", "lc", "l3"))
        u5, cds, u3 = (df[c].fillna("").to_numpy()[order] for c in ("UTR5_sequence", "CDS_sequence", "UTR3_sequence"))
        gc = [np.array([(s.upper().count("G") + s.upper().count("C")) / len(s) if len(s) else np.nan for s in v]) for v in (u5, cds, u3)]
        f6_raw = np.stack([np.log1p(l5), np.log1p(lc), np.log1p(l3), *gc], 1)

        f = fd
        # float32 in host memory as written; each feature set goes to the card only for its own fit (cp.Fitter sums in float64)
        X = {nm: torch.from_numpy(np.load(f / f"{nm}.npy")) for nm in ("mbmean", "bases", "regions", "split3")}
        if meta["model"].startswith("evo-1") and meta.get("mbmean") != "bf16":  # written before embed_evo1.py rounded mbmean
            X["mbmean"] = X["bases"].bfloat16().float()
        n, d = X["bases"].shape
        R = X["regions"]
        empty = R.abs().sum(-1) == 0
        rbal = (R.double() * (~empty).unsqueeze(-1)).sum(1) / (~empty).sum(1, keepdim=True).clamp_min(1)
        mpra = table in cp.MPRA
        lean = meta["model"].startswith("evo2") or X["bases"].shape[1] >= 2048  # wide models: main pools only; no shifts, F6, PCA
        extra = {}
        if (f / "shift.npy").exists() and not lean:
            sh = np.load(f / "shift.npy", mmap_mode="r")
            rd = np.load(f / "random.npy", mmap_mode="r")
            extra |= {f"shift{s:+d}": torch.from_numpy(np.array(sh[:, i])).reshape(n, -1) for i, s in enumerate(meta["shifts"])}
            extra |= {f"random{j}": torch.from_numpy(np.array(rd[:, j])).reshape(n, -1) for j in range(meta["draws"])}
        groups: dict = {}
        for k, y in labels.items():
            have = torch.isfinite(y)
            groups.setdefault(have.cpu().numpy().tobytes(), (have, []))[1].append(k)
        res = {k: {} for k in labels}
        use_f6 = (not mpra) or table == "mrl-varying"
        for si, cut in enumerate(cuts):
            for have, tnames in groups.values():
                m = {s: torch.from_numpy(cut == s).to(dev) & have for s in ("train", "val", "test")}
                tr, va, te = m["train"], m["val"], m["test"]
                ys = [labels[t][tr] for t in tnames]
                plan = [("mbmean", X["mbmean"], ("raw", "std")), ("bases", X["bases"], ("raw", "std")),
                        ("regions", R.reshape(n, -1), ("raw", "std")), ("split3", X["split3"].reshape(n, -1), ("raw", "std")),
                        ("rbal", rbal, ("raw", "std")),
                        ("r5", R[:, 0], ("std",)), ("rc", R[:, 1], ("std",)), ("r3", R[:, 2], ("std",))]
                if lean:
                    plan = [("mbmean", X["mbmean"], ("raw", "std")), ("bases", X["bases"], ("raw", "std")),
                            ("regions", R.reshape(n, -1), ("std",)), ("split3", X["split3"].reshape(n, -1), ("std",)),
                            ("rbal", rbal, ("std",)), ("r5", R[:, 0], ("std",)), ("rc", R[:, 1], ("std",)), ("r3", R[:, 2], ("std",))]
                plan += [(k, v, ("raw", "std")) for k, v in extra.items()]
                if use_f6 and not lean:
                    f6 = torch.from_numpy(f6_raw.copy())
                    f6 = torch.where(torch.isnan(f6), torch.nanmean(f6[tr.cpu()], 0).expand_as(f6), f6)
                    plan += [("F6", f6, ("std",)), ("bases+F6", torch.cat([X["bases"].double(), f6], 1), ("std",)),
                             ("split3+F6", torch.cat([X["split3"].reshape(n, -1).double(), f6], 1), ("std",)),
                             ("regions+F6", torch.cat([R.reshape(n, -1).double(), f6], 1), ("std",))]
                for fset, x, variants in plan:
                    fit = cp.Fitter(x.contiguous().to(dev), tr, ys, variants, top=d if fset in ("regions", "split3") and not lean else None)
                    for ti, t in enumerate(tnames):
                        for variant, grids in (("raw", ("A", "B")), ("std", ("B",)), ("pca", ("B",))):
                            if variant not in fit.fits:
                                continue
                            for grid in grids:
                                pv, alpha, _ = fit.predict(va, variant, ti, grid)
                                pt, _, _ = fit.predict(te, variant, ti, grid)
                                key = f"pca_{fset}|rawB" if variant == "pca" else f"{fset}|{variant}{grid}"
                                r = res[t].setdefault(key, {"val": [None] * 10, "test": [None] * 10, "alpha": [None] * 10})
                                r["val"][si], r["test"][si], r["alpha"][si] = cp.pearson(pv, labels[t][va]), cp.pearson(pt, labels[t][te]), alpha
                    del fit
        out = {"model": meta["model"], "table": table, "n": n, "d": d, "results": res, "seconds": time.monotonic() - started}
        (fd.parent / "fit.json").write_text(json.dumps(out))
        line = "  ".join(f"{k} {np.mean(res[t][k + '|stdB']['test']):.4f}" for t in res for k in ("mbmean", "bases", "split3", "regions"))
        print(f"{meta['model']} {table}: {line}  ({(time.monotonic() - started) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
