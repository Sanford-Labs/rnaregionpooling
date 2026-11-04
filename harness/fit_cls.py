"""fit_cls.py [--device cuda:0] FEATS_DIR... — mRNABench's classification columns on embed_mm.py features.

Logistic readout = `newton` below (Newton's method on sklearn LogisticRegression's objective at C = 1, intercept
unpenalized; matches sklearn max_iter=5000 to 1e-4 micro AUPRC), scored by `metrics.score_logistic` (`ap` = micro,
what mRNABench's multilabel and binary columns report; `ap_macro` = mean of per-label AP, its eCLIP columns).
Variants: A = raw features, C = 1 (mRNABench); B = raw, C in CGRID by val; T = standardized on train, C in TGRID
by val, the logistic side of protocol S; S = T's C = 1 member. C enters as features scaled by sqrt(C), and each C
starts from the previous C's solution. Wide models (Evo 2, or bases width >= 2048) run the main pools only, A and T.
Splits as fit.py (ten seeds, 0.7/0.15/0.15, the catalog's cutter).
Writes FEATS_DIR/../fit_cls.json.
"""
import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import CODE, WORK
from metrics import score_logistic  # noqa: E402

spec = importlib.util.spec_from_file_location("cp", CODE / "cpu_probe.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)

METRIC = {"vep-traitgym-complex": "ap", "vep-traitgym-mendelian": "ap", "eclip-binding-hepg2": "ap_macro",
          "eclip-binding-k562": "ap_macro", "rna-loc-fazal": "ap", "rna-lifecycle-ietswaart": "ap", "prot-loc": "ap",
          "go-bp": "ap", "go-cc": "ap", "go-mf": "ap"}
CGRID = (0.01, 0.1, 1.0, 10.0)
TGRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
SOLVER = "newton-centred-f64fallback-jitter-h64"  # fit_queue.py refits classification outputs without this tag


def proba(x, w, b):
    return torch.sigmoid(x.to(w.dtype) @ w + b)


def chol64(h):
    """Cholesky of a batch of float64 Hessians; a factor that fails gets a diagonal jitter, raised until it holds.
    Newton's fixed point is where the gradient vanishes, so the jitter moves the step, not the optimum."""
    c, info = torch.linalg.cholesky_ex(h)
    scale = torch.diagonal(h, dim1=-2, dim2=-1).abs().mean(-1)
    for eps in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
        bad = info > 0
        if not bool(bad.any()):
            return c
        eye = torch.eye(h.shape[-1], dtype=h.dtype, device=h.device)
        cj, ij = torch.linalg.cholesky_ex(h[bad] + (eps * scale[bad])[:, None, None] * eye)
        c[bad], info[bad] = cj, ij
    if bool((info > 0).any()):
        raise FloatingPointError("a logistic Hessian stayed indefinite under jitter")
    return c


def newton(x, y, steps=25, tol=1e-7, refine=2, budget=4 << 30, init=None):
    """Newton's method on sklearn's LogisticRegression objective at C = 1, intercept unpenalized. Probabilities and gradients in
    float64; Hessians and their Cholesky factors in float32 for every label in one batch, each solve refined
    `refine` times against the float64 product of the same Hessian, and solved again in float64 where that leaves a
    residual; a host sync or two a step. Features are centred first (the intercept is unpenalized, so this moves only
    the intercept). With fewer rows than features the fit runs in the span of the centred rows (w = Q beta, Q from a
    QR of x^T, |w| = |beta|), where the optimum lies."""
    y = y.unsqueeze(1) if y.dim() == 1 else y
    dev = x.device
    x = x_in = x.double()
    mu = x.mean(0, keepdim=True)
    x = x - mu
    basis = None
    if x.shape[0] < x.shape[1]:
        basis, _ = torch.linalg.qr(x.T)
        x = x @ basis
    n, dim = x.shape
    labels = y.shape[1]
    xb = torch.cat([x, torch.ones(n, 1, dtype=torch.float64, device=dev)], 1)
    x32 = xb.float()
    yb = y.double()
    pen = torch.ones(dim + 1, 1, dtype=torch.float64, device=dev)
    pen[dim] = 0.0
    eye = torch.diag(pen[:, 0]).float()
    live = y.amin(0) != y.amax(0)
    theta = torch.zeros(dim + 1, labels, dtype=torch.float64, device=dev)
    if init is not None:  # (w, bias) in the caller's features
        theta[:dim] = basis.T @ init[0] if basis is not None else init[0]
        theta[dim] = init[1] + (mu @ init[0])[0]
    per = max(1, budget // (12 * (dim + 1) ** 2))
    for _ in range(steps):
        p = torch.sigmoid(xb @ theta)
        g = (xb.T @ (p - yb) + pen * theta) * live
        if float(g.abs().max()) < tol:
            break
        wgt = (p * (1 - p)).clamp_min(1e-10).float()
        for j0 in range(0, labels, per):
            j1 = min(labels, j0 + per)
            h = torch.stack([(x32 * wgt[:, j:j + 1]).T @ x32 for j in range(j0, j1)]) + eye
            c, _ = torch.linalg.cholesky_ex(h)
            h64 = h.double()
            rhs = g[:, j0:j1].T.unsqueeze(-1)
            step = torch.cholesky_solve(rhs.float(), c).double()
            for _ in range(refine):
                step = step + torch.cholesky_solve((rhs - h64 @ step).float(), c).double()
            off = (rhs - h64 @ step).norm(dim=1) > 1e-4 * rhs.norm(dim=1)
            if bool(off.any() | ~torch.isfinite(step).all()):
                # float32 accumulation can leave a near-flat direction (a saturated label's intercept) indefinite:
                # these Hessians again, accumulated in float64
                w64 = (p[:, j0:j1] * (1 - p[:, j0:j1])).clamp_min(1e-10)
                h64 = torch.stack([(xb * w64[:, j:j + 1]).T @ xb for j in range(j1 - j0)]) + torch.diag(pen[:, 0])
                step = torch.cholesky_solve(rhs, chol64(h64))
            theta[:, j0:j1] -= step[..., 0].T
            del h, h64, c
    if not bool(torch.isfinite(theta).all()):
        if init is not None:
            return newton(x_in, y, steps, tol, refine, budget)
        raise FloatingPointError("a logistic Newton step left the finite range")
    w, bias = theta[:dim], theta[dim]
    if basis is not None:
        w = basis @ w
    bias = bias - (mu @ w)[0]
    return torch.where(live, w, torch.zeros_like(w)), torch.where(live, bias, torch.zeros_like(bias)), live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feats", nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--table", default="", help="score these features against another table's labels (development)")
    a = ap.parse_args()
    dev = torch.device(a.device)
    frac = os.environ.get("EMBED_MEM_FRACTION")  # this process's share of the card's memory
    if frac and dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(frac), dev)
    torch.set_num_threads(8)
    for fd in map(Path, a.feats):
        started = time.monotonic()
        meta = json.loads((fd / "meta.json").read_text())
        table = a.table or meta["table"]
        key = cp.KEYS.get(table, table)
        df = pd.read_parquet(cp.DATA / f"{key}.parquet")
        order = np.array(meta["order"])
        full = np.concatenate([df.index[df["split"] == s].to_numpy() for s in ("train", "val", "test")])
        assert len(order) == len(full) and (order == full).all(), "rows are not fit.py's order"
        cuts = cp.seed_cuts(key, df, order, meta["model"])
        tcols = sorted(c for c in df.columns if c.startswith("target"))
        Y = torch.tensor(np.stack([df[c].to_numpy(float)[order] for c in tcols], 1), dtype=torch.float64, device=dev)
        if a.table:  # development: binarize a regression target at its median
            Y = (Y > Y.nanmedian(0).values).double()
        metric = METRIC.get(table, "ap")
        l5, lc, l3 = (np.array(meta[k], float) for k in ("l5", "lc", "l3"))
        u5, cds, u3 = (df[c].fillna("").to_numpy()[order] for c in ("UTR5_sequence", "CDS_sequence", "UTR3_sequence"))
        gc = [np.array([(s.upper().count("G") + s.upper().count("C")) / len(s) if len(s) else np.nan for s in v]) for v in (u5, cds, u3)]
        f6_raw = np.stack([np.log1p(l5), np.log1p(lc), np.log1p(l3), *gc], 1)
        X = {nm: torch.from_numpy(np.load(fd / f"{nm}.npy")).double().to(dev) for nm in ("mbmean", "bases", "regions", "split3")}
        n, d = X["bases"].shape
        R = X["regions"]
        empty = R.abs().sum(-1) == 0
        rbal = (R * (~empty).unsqueeze(-1)).sum(1) / (~empty).sum(1, keepdim=True).clamp_min(1)
        res: dict = {}
        for si, cut in enumerate(cuts):
            m = {s: torch.from_numpy(cut == s).to(dev) for s in ("train", "val", "test")}
            tr, va, te = m["train"], m["val"], m["test"]
            f6 = torch.from_numpy(f6_raw.copy()).to(dev)
            f6 = torch.where(torch.isnan(f6), torch.nanmean(f6[tr], 0).expand_as(f6), f6)
            plan = [("mbmean", X["mbmean"], "ABT"), ("bases", X["bases"], "ABT"), ("regions", R.reshape(n, -1), "ABT"),
                    ("split3", X["split3"].reshape(n, -1), "ABT"), ("rbal", rbal, "AT"),
                    ("r5", R[:, 0], "T"), ("rc", R[:, 1], "T"), ("r3", R[:, 2], "T"),
                    ("F6", f6, "T"), ("bases+F6", torch.cat([X["bases"], f6], 1), "T"),
                    ("split3+F6", torch.cat([X["split3"].reshape(n, -1), f6], 1), "T"),
                    ("regions+F6", torch.cat([R.reshape(n, -1), f6], 1), "T")]
            if meta["model"].startswith("evo2") or d >= 2048:
                plan = [("mbmean", X["mbmean"], "AT"), ("bases", X["bases"], "AT"), ("regions", R.reshape(n, -1), "T"),
                        ("split3", X["split3"].reshape(n, -1), "T"), ("rbal", rbal, "T"),
                        ("r5", R[:, 0], "T"), ("rc", R[:, 1], "T"), ("r3", R[:, 2], "T")]
            for fset, x, variants in plan:
                mu = x[tr].mean(0)
                sd = x[tr].std(0).clamp_min(1e-12)
                for v in variants:
                    base, grid = {"A": (x, (1.0,)), "B": (x, CGRID), "T": ((x - mu) / sd, TGRID)}[v]
                    best, prev, at = None, None, {}
                    for c in grid:
                        xx = base * c ** 0.5
                        init = None if prev is None else (prev[0] * (prev[2] / c) ** 0.5, prev[1])
                        w, b, _ = newton(xx[tr], Y[tr], init=init)
                        prev = (w, b, c)
                        sv = score_logistic(Y[va], proba(xx[va], w, b))
                        st = score_logistic(Y[te], proba(xx[te], w, b))
                        at[c] = (sv, st, c)
                        if best is None or sv[metric] > best[0][metric]:
                            best = at[c]
                    for name, pick in [(v, best)] + ([("S", at[1.0])] if v == "T" else []):
                        r = res.setdefault(f"{fset}|{name}", {"val": [None] * 10, "test": [None] * 10, "C": [None] * 10,
                                                              "test_ap": [None] * 10, "test_ap_macro": [None] * 10})
                        r["val"][si], r["test"][si], r["C"][si] = pick[0][metric], pick[1][metric], pick[2]
                        r["test_ap"][si], r["test_ap_macro"][si] = pick[1]["ap"], pick[1]["ap_macro"]
            print(f"{meta['model']} {table} seed {si + 1}/10 {(time.monotonic() - started) / 60:.1f} min", flush=True)
        out = {"model": meta["model"], "table": table, "metric": metric, "labels": tcols, "n": n, "d": d, "solver": SOLVER,
               "results": res, "seconds": time.monotonic() - started}
        (fd.parent / ("fit_cls.json" if not a.table else "fit_cls.dev.json")).write_text(json.dumps(out))
        line = "  ".join(f"{k} {np.mean(res[k + '|T']['test']):.4f}" for k in ("mbmean", "bases", "split3", "regions"))
        print(f"{meta['model']} {table} [{metric}]: {line}  ({(time.monotonic() - started) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
