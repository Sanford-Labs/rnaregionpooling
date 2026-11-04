"""cpu_probe.py — what the fitting scripts share: the dataset keys, mRNABench's ten splits of a table (`seed_cuts`) and
the ridge probe (`Fitter`).

The ridge has an unpenalized intercept and chooses its penalty by exact leave-one-out error on the training rows,
from one eigendecomposition per feature set shared by the table's targets: grid A is mRNABench's five alphas, grid B
runs from 1e-6 to 1e6, and both are read off the same decomposition. The `std` variant standardizes the columns on
each split's training rows.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from paths import CODE, DATA, WORK

torch.set_num_threads(8)
DATA = DATA
SEEDS = [2541, 413, 411, 412, 2547, 321, 421, 311, 2516, 2515]
RATIOS = (0.7, 0.15, 0.15)
GRID_A = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
GRID_B = tuple(10.0 ** k for k in range(-6, 7))
A_IN_B = [GRID_B.index(a) for a in GRID_A]
KEYS = {"mrl-egfp": "mrl-sample-egfp", "mrl-mcherry": "mrl-sample-mcherry", "mrl-varying": "mrl-sample-varying",
        "mrl-designed": "mrl-sample-designed", "rnahl-human": "rnahl-human", "rnahl-mouse": "rnahl-mouse",
        "sugimoto": "mrl-sugimoto", "mrl-hl": "mrl-hl-lbkwk", "te-human": "translation-efficiency-human",
        "te-mouse": "translation-efficiency-mouse"}
MPRA = {"mrl-egfp", "mrl-mcherry", "mrl-varying", "mrl-designed"}

spec = importlib.util.spec_from_file_location("fetch", CODE / "fetch_mrnabench.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


# mRNABench's rnahl-mouse runs (overall_df; the paper's HL column) cut seeds 1-5 along the mouse homology map and
# seeds 6-10 along the human one: by_modelname.py hardcodes species="human", 2 of the 13,723 mouse gene names are in
# that map, and split_homologous sends every other row on its own. Every other table takes its own species for all ten.
SPECIES_BY_SEED = {"rnahl-mouse": ("mouse",) * 5 + ("human",) * 5}
# their Evo 2 rows of rnahl-mouse take the human map on all ten seeds (seeds 1-5 match it to 3e-4, the mouse map to 0.035)
SPECIES_BY_SEED_MODEL = {("rnahl-mouse", "evo2-7b-base"): ("human",) * 10}


def seed_cuts(key, df, order, model=""):
    """mRNABench's ten train / val / test columns of one table for one model, in SEEDS order, aligned to `order`."""
    ds = fetch.CATALOG[key]
    genes = df["gene"].tolist() if ds.homologous else None
    species = SPECIES_BY_SEED_MODEL.get((key, model.replace("_", "-")), SPECIES_BY_SEED.get(key, (ds.species,) * len(SEEDS)))
    maps = {s: fetch.homology_map(s, DATA / ".cache") for s in set(species)} if ds.homologous else {}
    return [fetch.three_way(len(order), fetch.cutter(genes, maps.get(sp)), RATIOS, seed)[order]
            for sp, seed in zip(species, SEEDS)]


def decompose(gram_c, n):
    s, v = torch.linalg.eigh(gram_c)
    return s.clamp_min(0), v


def solve(xc_tr, s, v, ys, x_mean_term, grid_idx_sets):
    """LOO over GRID_B for each target; returns per target {grid: (w_in_eigbasis_scale, alpha)}."""
    n = xc_tr.shape[0]
    g = xc_tr @ v
    g2 = g * g
    out = []
    for y in ys:
        yc = y - y.mean()
        b = g.T @ yc
        errs = []
        for a in GRID_B:
            scale = 1.0 / (s + a)
            r = (yc - g @ (b * scale)) / (1.0 - (1.0 / n + g2 @ scale)).clamp_min(1e-12)
            errs.append(float((r * r).mean()))
        errs = np.array(errs)
        picks = {}
        for name, idx in grid_idx_sets.items():
            k = idx[int(np.argmin(errs[idx]))]
            picks[name] = (v @ (b * (1.0 / (s + GRID_B[k]))), GRID_B[k], float(y.mean()))
        out.append(picks)
    return out


def pearson(p, y):
    p, y = p - p.mean(), y - y.mean()
    return float((p @ y) / (p.norm() * y.norm()).clamp_min(1e-30))


class Fitter:
    """One feature matrix, one seed: raw (grid A, B), standardized (grid B) and top-PC (grid B) fits for every target.

    FitterDense's arithmetic, taken over row chunks: the Gram matrix, X'y and the LOO residuals are summed in float64
    chunk by chunk, so `x` may stay float32 (the features are float32 on disk) and memory is `x` plus one chunk
    rather than several float64 copies of it. The training rows are `tr`'s in ascending order, as `x[tr]` is.
    With fewer training rows than columns and no standardized variant, the fits come off the n x n Gram matrix of the
    centered rows instead (sklearn's RidgeCV gcv_mode="eigen"): the same hat matrix, LOO residuals and weights.
    """

    CHUNK_BYTES = 1 << 30
    DUAL = True

    def __init__(self, x, tr, ys_tr, variants, top=None):
        self.x = x
        self.dev = dev = tr.device  # x may stay in host memory: rows go to tr's device a chunk at a time
        d = x.shape[1]
        idx = tr.nonzero().squeeze(1)
        n = idx.numel()
        self.step = max(1, self.CHUNK_BYTES // (8 * d))
        chunks = [idx[i:i + self.step] for i in range(0, n, self.step)]
        mu = torch.zeros(d, dtype=torch.float64, device=dev)
        for c in chunks:
            mu += self._rows(c).sum(0)
        self.mu = mu / n
        ymeans = [float(y.mean()) for y in ys_tr]
        yc = torch.stack([y - y.mean() for y in ys_tr], 1)
        if self.DUAL and n < d and "std" not in variants:
            self._dual(chunks, yc, ymeans, n, variants, top)
            return
        gram = torch.zeros(d, d, dtype=torch.float64, device=dev)
        xty = torch.zeros(d, yc.shape[1], dtype=torch.float64, device=dev)
        at = 0
        for c in chunks:
            xc = self._rows(c) - self.mu
            gram += xc.T @ xc
            xty += xc.T @ yc[at:at + len(c)]
            at += len(c)
        self.fits = {}
        if "raw" in variants or top is not None:
            s, v = decompose(gram, n)
            if "raw" in variants:
                self.fits["raw"] = (None, self._solve(chunks, None, s, v, xty, yc, ymeans, n,
                                                      {"A": A_IN_B, "B": list(range(len(GRID_B)))}))
            if top is not None:
                keep = slice(v.shape[1] - top, v.shape[1])
                self.fits["pca"] = (v[:, keep], self._solve(chunks, None, s[keep], v[:, keep], xty, yc, ymeans, n,
                                                            {"B": list(range(len(GRID_B)))}, reduced=True))
        if "std" in variants:
            sd = (torch.diagonal(gram) / n).sqrt()
            sd = torch.where(sd > 1e-12, sd, torch.ones_like(sd))
            self.sd = sd
            gs = gram / sd[:, None] / sd[None, :]
            s, v = decompose(gs, n)
            self.fits["std"] = (None, self._solve(chunks, sd, s, v, xty / sd[:, None], yc, ymeans, n,
                                                  {"B": list(range(len(GRID_B)))}))
        del gram

    def _dual(self, chunks, yc, ymeans, n, variants, top):
        """The raw and top-PC fits off K = XcXc' = U diag(s) U': hat diagonal 1/n + (U*U) s/(s+a), weights Xc'U (U'y)/(s+a)."""
        xc = torch.cat([self._rows(c) - self.mu for c in chunks])
        s, u = decompose(xc @ xc.T, n)
        uty = u.T @ yc
        sets = {}
        if "raw" in variants:
            sets["raw"] = (slice(None), {"A": A_IN_B, "B": list(range(len(GRID_B)))})
        if top is not None:
            sets["pca"] = (slice(max(0, n - top), n), {"B": list(range(len(GRID_B)))})
        self.fits = {}
        for variant, (keep, grid_idx_sets) in sets.items():
            sk, uk, utyk = s[keep], u[:, keep], uty[keep]
            ratio = torch.stack([sk / (sk + a) for a in GRID_B], 1)
            den = (1.0 - (1.0 / n + (uk * uk) @ ratio)).clamp_min(1e-12)
            err = torch.stack([(((yc[:, t:t + 1] - uk @ (utyk[:, t:t + 1] * ratio)) / den) ** 2).sum(0)
                               for t in range(yc.shape[1])])
            err = (err / n).cpu().numpy()
            out = []
            for t in range(yc.shape[1]):
                picks = {}
                for name, idx in grid_idx_sets.items():
                    k = idx[int(np.argmin(err[t, idx]))]
                    picks[name] = (xc.T @ (uk @ (utyk[:, t] / (sk + GRID_B[k]))), GRID_B[k], ymeans[t])
                out.append(picks)
            self.fits[variant] = (None, out)

    def _solve(self, chunks, sd, s, v, xty, yc, ymeans, n, grid_idx_sets, reduced=False):
        """`solve` over row chunks: b = v'X'y, then each chunk's LOO residuals for every alpha at once."""
        b = v.T @ xty
        scale = torch.stack([1.0 / (s + a) for a in GRID_B], 1)
        err = torch.zeros(yc.shape[1], len(GRID_B), dtype=torch.float64, device=yc.device)
        at = 0
        for c in chunks:
            xc = self._rows(c) - self.mu
            if sd is not None:
                xc = xc / sd
            g = xc @ v
            den = (1.0 - (1.0 / n + (g * g) @ scale)).clamp_min(1e-12)
            for t in range(yc.shape[1]):
                r = (yc[at:at + len(c), t:t + 1] - g @ (b[:, t:t + 1] * scale)) / den
                err[t] += (r * r).sum(0)
            at += len(c)
        err = (err / n).cpu().numpy()
        out = []
        for t in range(yc.shape[1]):
            picks = {}
            for name, idx in grid_idx_sets.items():
                k = idx[int(np.argmin(err[t, idx]))]
                w = b[:, t] * scale[:, k]
                picks[name] = (w if reduced else v @ w, GRID_B[k], ymeans[t])
            out.append(picks)
        return out

    def _rows(self, c):
        return self.x[c.to(self.x.device)].to(self.dev, non_blocking=True).double()

    def predict(self, rows, variant, t, grid):
        basis, per = self.fits[variant]
        w, alpha, ymean = per[t][grid]
        idx = rows.nonzero().squeeze(1)
        out = [torch.zeros(0, dtype=torch.float64, device=self.dev)]
        for i in range(0, idx.numel(), self.step):
            xc = self._rows(idx[i:i + self.step]) - self.mu
            if variant == "std":
                xc = xc / self.sd
            if basis is not None:
                xc = xc @ basis
            out.append(xc @ w + ymean)
        return torch.cat(out), alpha, w


class FitterDense:
    """One feature matrix, one seed: raw (grid A, B) and standardized (grid B) fits for every target."""

    def __init__(self, x, tr, ys_tr, variants, top=None):
        self.x, self.tr = x, tr
        xt = x[tr]
        self.mu = xt.mean(0)
        xc = xt - self.mu
        n = xc.shape[0]
        self.fits = {}
        gram = xc.T @ xc
        if "raw" in variants or top is not None:
            s, v = decompose(gram, n)
            if "raw" in variants:
                self.fits["raw"] = (None, solve(xc, s, v, ys_tr, None, {"A": A_IN_B, "B": list(range(len(GRID_B)))}))
            if top is not None:
                keep = slice(v.shape[1] - top, v.shape[1])
                self.fits["pca"] = (v[:, keep], solve(xc @ v[:, keep], s[keep], torch.eye(top, dtype=x.dtype, device=x.device),
                                                     ys_tr, None, {"B": list(range(len(GRID_B)))}))
        if "std" in variants:
            sd = (torch.diagonal(gram) / n).sqrt()
            sd = torch.where(sd > 1e-12, sd, torch.ones_like(sd))
            self.sd = sd
            gs = gram / sd[:, None] / sd[None, :]
            s, v = decompose(gs, n)
            self.fits["std"] = (None, solve(xc / sd, s, v, ys_tr, None, {"B": list(range(len(GRID_B)))}))
        del gram

    def predict(self, rows, variant, t, grid):
        basis, per = self.fits[variant]
        w, alpha, ymean = per[t][grid]
        xc = self.x[rows] - self.mu
        if variant == "std":
            xc = xc / self.sd
        if basis is not None:
            xc = xc @ basis
        return xc @ w + ymean, alpha, w
