"""decomp_ext.py [--complete] [--raw] [--cpu] FEATS_DIR... — on embed_mm.py-layout features: per seed and target, the regions probe of
fit_ext.py's protocol S (cp.Fitter, columns standardized on the train rows, grid B, LOO alpha), and on the test rows
each block's contribution c_k = ((x - mu) / sd)_k . w_k (k = 5'UTR, CDS, 3'UTR): var_share_k = var(c_k) / var(sum c),
cov_share = the off-diagonal covariances / var(sum c), as cpu_probe.py's decomposition. Writes FEATS_DIR/../decomp.json;
--complete fits and decomposes on the rows whose three regions are all non-empty only (decomp.complete.json). --raw uses
the probe on raw features instead (mRNABench's RidgeCV, grid B; c_k = (x - mu)_k . w_k; decomp[.complete].rawB.json).
--cpu fits on the CPU (no card, no fit slot).
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
    raw = "--raw" in sys.argv
    variant = "raw" if raw else "std"
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    complete = "--complete" in sys.argv
    for fd in map(Path, [a for a in sys.argv[1:] if not a.startswith("--")]):
        started = time.monotonic()
        meta = json.loads((fd / "meta.json").read_text())
        table = meta["table"]
        key = cp.KEYS[table]
        df = pd.read_parquet(cp.DATA / f"{key}.parquet")
        order = np.array(meta["order"])
        cuts = cp.seed_cuts(key, df, order, meta["model"])
        names = {c: fe.LABELS.get(table, {}).get(c, "batch.y") for c in meta["targets"]}
        labels = {names[c]: torch.tensor(v, dtype=torch.float64, device=dev) for c, v in meta["targets"].items()}
        R = torch.from_numpy(np.load(fd / "regions.npy"))
        n, _, d = R.shape
        x = R.reshape(n, -1).contiguous().to(dev)
        full = torch.from_numpy(np.all([np.array(meta[k]) > 0 for k in ("l5", "lc", "l3")], 0)).to(dev)
        groups: dict = {}
        for k, y in labels.items():
            have = torch.isfinite(y) & (full if complete else True)
            groups.setdefault(have.cpu().numpy().tobytes(), (have, []))[1].append(k)
        res = {k: {"var_share": [None] * 10, "cov_share": [None] * 10, "test": [None] * 10, "alpha": [None] * 10} for k in labels}
        for si, cut in enumerate(cuts):
            for have, tnames in groups.values():
                tr, te = (torch.from_numpy(cut == s).to(dev) & have for s in ("train", "test"))
                fit = cp.Fitter(x, tr, [labels[t][tr] for t in tnames], (variant,))
                xc = fit._rows(te.nonzero().squeeze(1)) - fit.mu
                if not raw:
                    xc = xc / fit.sd
                for ti, t in enumerate(tnames):
                    pt, alpha, w = fit.predict(te, variant, ti, "B")
                    c = torch.stack([xc[:, k * d:(k + 1) * d] @ w[k * d:(k + 1) * d] for k in range(3)], 1)
                    vt = float(c.sum(1).var())
                    cov = torch.cov(c.T)
                    r = res[t]
                    r["var_share"][si] = [float(cov[k, k]) / vt for k in range(3)]
                    r["cov_share"][si] = float(cov.sum() - cov.diagonal().sum()) / vt
                    r["test"][si], r["alpha"][si] = cp.pearson(pt, labels[t][te]), alpha
                del fit, xc
        out = {"model": meta["model"], "table": table, "n": n, "d": d, "complete_rows_only": complete, "rows": int(full.sum()) if complete else n,
               "probe": f"{variant}B", "results": res, "seconds": time.monotonic() - started}
        (fd.parent / f"decomp{'.complete' if complete else ''}{'.rawB' if raw else ''}.json").write_text(json.dumps(out))
        line = "  ".join(f"{t}: 5' {np.mean([s[0] for s in r['var_share']]):.2f} CDS {np.mean([s[1] for s in r['var_share']]):.2f} "
                         f"3' {np.mean([s[2] for s in r['var_share']]):.2f} cov {np.mean(r['cov_share']):+.2f} (r {np.mean(r['test']):.4f})"
                         for t, r in res.items())
        print(f"{meta['model']} {table}{' (complete rows)' if complete else ''}: {line}  ({(time.monotonic() - started) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
