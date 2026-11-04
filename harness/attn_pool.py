"""attn_pool.py FEATS_DIR [--device cuda:0] [--variants none,regemb,perregion] — learned attention pooling on
frozen per-base vectors (embed_mm2.py --tokens), ten fit.py seeds, early stop on val; ridge(regions) as reference.

  none       one learned query; softmax over the row's bases; no annotation
  regemb     the same, keys plus a learned 5'UTR / CDS / 3'UTR embedding
  perregion  one query per region, softmax inside each region, the three pooled vectors concatenated
The pooled vector goes through one linear layer to the (train-standardized) target; MSE, AdamW.

  --seeds 0-4   only these splits, for running the ten on several cards
  --shares      for the single-query variants, the softmax weight each test row puts on its 5'UTR, CDS and 3'UTR,
                beside the nucleotide shares, in attn_shares.<tag>.json; the fitted parameters in attn_states/
"""
import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import CODE, WORK

spec = importlib.util.spec_from_file_location("cp", CODE / "cpu_probe.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)


class AttnPool(nn.Module):
    def __init__(self, H, variant, dk=64):
        super().__init__()
        self.variant = variant
        self.norm = nn.LayerNorm(H)
        self.key = nn.Linear(H, dk)
        self.q = nn.Parameter(torch.randn(3 if variant == "perregion" else 1, dk) / math.sqrt(dk))
        self.reg = nn.Embedding(3, dk) if variant == "regemb" else None
        self.head = nn.Linear(H * (3 if variant == "perregion" else 1), 1)
        self.dk = dk

    def weights(self, x, g):
        """The softmax weight of each base, for the single-query variants."""
        k = self.key(self.norm(x))
        if self.reg is not None:
            k = k + self.reg(g)
        return torch.softmax(k @ self.q[0] / math.sqrt(self.dk), 0)

    def forward(self, xs, regs, spans):
        out = []
        for x, g, sp in zip(xs, regs, spans):
            k = self.key(self.norm(x))
            if self.reg is not None:
                k = k + self.reg(g)
            if self.variant == "perregion":  # the regions are consecutive spans: slices, no mask (each mask is a host sync)
                parts = []
                for j, (a, b) in enumerate(sp):
                    if b <= a:
                        parts.append(torch.zeros(x.shape[1], device=x.device))
                        continue
                    w = torch.softmax(k[a:b] @ self.q[j] / math.sqrt(self.dk), 0)
                    parts.append(w @ x[a:b])
                out.append(torch.cat(parts))
            else:
                w = torch.softmax(k @ self.q[0] / math.sqrt(self.dk), 0)
                out.append(w @ x)
        return self.head(torch.stack(out)).squeeze(1)


def pearson(p, y):
    p, y = p - p.mean(), y - y.mean()
    return float((p @ y) / (p.norm() * y.norm()).clamp_min(1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feats")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--variants", default="none,regemb,perregion")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--shares", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.seeds.split("-"))
    fd = Path(a.feats)
    dev = torch.device(a.device)
    meta = json.loads((fd / "meta.json").read_text())
    table = meta["table"]
    key = cp.KEYS[table]
    df = pd.read_parquet(cp.DATA / f"{key}.parquet")
    order = np.array(meta["order"])
    cuts = cp.seed_cuts(key, df, order, meta["model"])
    tcol = sorted(meta["targets"])[0]
    y_all = torch.tensor(meta["targets"][tcol], dtype=torch.float32)
    H = np.load(fd / "bases.npy", mmap_mode="r").shape[1]
    idx = np.load(fd / "tokens_index.npy")
    flat = torch.from_numpy(np.memmap(fd / "tokens.bf16", dtype=np.int16, mode="r").reshape(-1, H)).view(torch.bfloat16)
    l5, lc = np.array(meta["l5"]), np.array(meta["lc"])

    def row(r):
        s, n = idx[r]
        x = flat[s:s + n].to(dev, non_blocking=True).float()
        g = torch.full((n,), 2, dtype=torch.long, device=dev)
        g[:l5[r]] = 0
        g[l5[r]:l5[r] + lc[r]] = 1
        e5, ec = int(l5[r]), int(l5[r] + lc[r])
        return x, g, ((0, e5), (e5, ec), (ec, int(n)))

    fj = fd.parent / "fit.json"
    if not fj.exists():  # a token cache has no fit of its own: the reference is the model's fit on the same table
        fj = fd.parents[2] / meta["model"] / table / "fit.json"
    fit = json.loads(fj.read_text())["results"] if fj.exists() else None
    ref = fit[sorted(fit)[0]]["regions|stdB"]["test"] if fit else None
    res = {}
    started = time.monotonic()
    for variant in a.variants.split(","):
        res[variant] = {"test": [], "val": [], "epochs": []}
        for si, cut in enumerate(cuts):
            if not lo <= si <= hi:
                continue
            torch.manual_seed(si)
            have = torch.isfinite(y_all).numpy()
            tr, va, te = (np.nonzero((cut == s) & have)[0] for s in ("train", "val", "test"))
            mu, sd = float(y_all[tr].mean()), float(y_all[tr].std())
            y = ((y_all - mu) / sd).to(dev)
            model = AttnPool(H, variant).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

            def predict(rows):
                model.eval()
                outs = []
                with torch.no_grad():
                    for i in range(0, len(rows), 64):
                        xs, gs, ss = zip(*(row(r) for r in rows[i:i + 64]))
                        outs.append(model(xs, gs, ss))
                model.train()
                return torch.cat(outs)

            best, best_state, bad, ep = -1e9, None, 0, 0
            for ep in range(a.epochs):
                perm = np.random.default_rng(1000 * si + ep).permutation(tr)
                for i in range(0, len(perm), a.batch):
                    rows = perm[i:i + a.batch]
                    xs, gs, ss = zip(*(row(r) for r in rows))
                    loss = ((model(xs, gs, ss) - y[torch.as_tensor(rows, device=dev)]) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                v = pearson(predict(va), y[torch.as_tensor(va, device=dev)])
                if v > best:
                    best, bad = v, 0
                    best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= a.patience:
                        break
            model.load_state_dict(best_state)
            t = pearson(predict(te), y[torch.as_tensor(te, device=dev)])
            res[variant]["val"].append(best)
            res[variant]["test"].append(t)
            res[variant]["epochs"].append(ep + 1)
            if a.shares and variant != "perregion":
                (fd.parent / "attn_states").mkdir(exist_ok=True)
                torch.save(best_state, fd.parent / "attn_states" / f"{variant}-seed{si}.pt")
                model.eval()
                sh, nt = [], []
                with torch.no_grad():
                    for r in te:
                        x, g, ((_, e5), (_, ec), (_, n)) = row(r)
                        w = model.weights(x, g)
                        sh.append([float(w[:e5].sum()), float(w[e5:ec].sum()), float(w[ec:].sum())])
                        nt.append([e5 / n, (ec - e5) / n, (n - ec) / n])
                res[variant].setdefault("seeds", []).append(si)
                res[variant].setdefault("shares", []).append(sh)
                res[variant].setdefault("nt", []).append(nt)
                res[variant].setdefault("rows", []).append([int(r) for r in te])
                s, q = np.mean(sh, 0), np.mean(nt, 0)
                print(f"{variant} seed {si + 1}: attention 5'UTR {s[0]:.3f} CDS {s[1]:.3f} 3'UTR {s[2]:.3f}; "
                      f"nucleotides {q[0]:.3f} {q[1]:.3f} {q[2]:.3f}", flush=True)
                (fd.parent / f"attn_shares.{a.tag or variant}.json").write_text(json.dumps({"table": table, "results": res}))
            print(f"{variant} seed {si + 1}/10: val {best:.4f} test {t:.4f} (epochs {ep + 1}, {(time.monotonic() - started) / 60:.1f} min)", flush=True)
    out = {"table": table, "feats": str(fd), "results": res, "ridge_regions_stdB": ref}
    (fd.parent / (f"attn_pool.{a.tag}.json" if a.tag else "attn_pool.json")).write_text(json.dumps(out))
    for v, r in res.items():
        print(f"{v:10} test {np.mean(r['test']):.4f} ± {np.std(r['test']):.4f}")
    if ref:
        print(f"ridge regions (std B) test {np.mean(ref):.4f}")


if __name__ == "__main__":
    main()
