"""embed_evo2.py MODEL TABLE OUT [--device cuda:0] [--isolate] [--parquet P] [--limit N] — Evo 2 through the evo2
package, as mRNABench's wrapper: chunks of 8,192 nt, one token per base, embeddings read at the middle block's
pre-norm and at the final `norm`.
  mbmean      mRNABench's readout: mean over tokens at both layers, concatenated (2H)
  mbmean_mid  the middle layer's mean alone
  bases / regions / split3 / shift / random   embed_mm.py's pools on the MIDDLE layer (H); the last layer of an
              autoregressive model is shaped for the next token, which is why mRNABench reads the middle one too.
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import CODE, WORK

spec = importlib.util.spec_from_file_location("mm", CODE / "harness/embed_mm.py")
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)
MIDDLE = {"evo2_7b_base": "blocks.16.pre_norm", "evo2_7b": "blocks.16.pre_norm", "evo2_1b_base": "blocks.12.pre_norm"}
CHUNK = 8192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(MIDDLE))
    ap.add_argument("table")
    ap.add_argument("out")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--isolate", action="store_true")
    ap.add_argument("--parquet", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from evo2 import Evo2
    model = Evo2(a.model)
    layers = [MIDDLE[a.model], "norm"]
    if a.parquet:
        df = pd.read_parquet(a.parquet)
        order = np.arange(len(df))
    else:
        df = pd.read_parquet(mm.DATA / f"{mm.KEYS[a.table]}.parquet")
        order = np.concatenate([df.index[df["split"] == s].to_numpy() for s in ("train", "val", "test")])
    if a.limit:
        order = order[:a.limit]
    u5, cds, u3 = ([s.upper().replace("U", "T") for s in df[c].fillna("").to_numpy()[order]]
                   for c in ("UTR5_sequence", "CDS_sequence", "UTR3_sequence"))
    l5 = np.array([len(x) for x in u5]); lc = np.array([len(x) for x in cds]); l3 = np.array([len(x) for x in u3])
    L = l5 + lc + l3
    n = len(order)
    extra = (not a.parquet) and a.table in mm.REGRESSION6 and not a.isolate
    rng = np.random.default_rng(0)
    donors = []
    for _ in range(mm.DRAWS):
        p = rng.permutation(n)
        donors.append(np.where(p == np.arange(n), np.roll(p, 1), p))
    arr = None
    feats = Path(a.out) / "features"
    feats.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    done = 0
    with torch.inference_mode():
        for r in np.argsort(-L, kind="stable"):
            segs = [u5[r], cds[r], u3[r]] if a.isolate else [u5[r] + cds[r] + u3[r]]
            mids, lasts = [], []
            for s in segs:
                for st in range(0, len(s), CHUNK):
                    ids = torch.tensor(model.tokenizer.tokenize(s[st:st + CHUNK]), dtype=torch.int).unsqueeze(0).to(a.device)
                    _, emb = model(ids, return_embeddings=True, layer_names=layers)
                    mids.append(emb[layers[0]][0].float())
                    lasts.append(emb[layers[1]][0].float())
                    done += ids.shape[1]
            e = torch.cat(mids)
            last = torch.cat(lasts)
            assert e.shape[0] == L[r], (int(r), tuple(e.shape), int(L[r]))
            if arr is None:
                H = e.shape[1]
                shapes = {"mbmean": (n, 2 * H), "mbmean_mid": (n, H), "bases": (n, H), "regions": (n, 3, H), "split3": (n, 3, H)}
                if extra:
                    shapes |= {"shift": (n, len(mm.SHIFTS), 3 * H), "random": (n, mm.DRAWS, 3 * H)}
                arr = {nm: np.lib.format.open_memmap(feats / f"{nm}.npy", mode="w+", dtype=np.float32, shape=shp) for nm, shp in shapes.items()}
            arr["mbmean"][r] = torch.cat([e.mean(0), last.mean(0)]).cpu().numpy()
            arr["mbmean_mid"][r] = e.mean(0).cpu().numpy()
            arr["bases"][r] = e.mean(0).cpu().numpy()
            arr["regions"][r] = mm.spans_pool(e, (int(l5[r]), int(l5[r] + lc[r]))).cpu().numpy()
            arr["split3"][r] = mm.spans_pool(e, (int(round(L[r] / 3)), int(round(2 * L[r] / 3)))).cpu().numpy()
            if extra:
                arr["shift"][r] = torch.stack([mm.spans_pool(e, (int(l5[r] + d), int(l5[r] + lc[r] + d))).reshape(-1) for d in mm.SHIFTS]).cpu().numpy()
                rand = []
                for p in donors:
                    dn = p[r]
                    f5, fc = l5[dn] / max(L[dn], 1), lc[dn] / max(L[dn], 1)
                    rand.append(mm.spans_pool(e, (int(round(f5 * L[r])), int(round((f5 + fc) * L[r])))).reshape(-1))
                arr["random"][r] = torch.stack(rand).cpu().numpy()
    for v in arr.values():
        v.flush()
    meta = {"model": a.model, "table": a.table, "parquet": a.parquet, "isolate": a.isolate, "layers": layers, "chunk": CHUNK,
            "rows": n, "order": order.tolist(), "sets": df["split"].to_numpy()[order].tolist() if "split" in df else None,
            "targets": {c: df[c].to_numpy(float)[order].tolist() for c in df.columns if c.startswith("target")},
            "shifts": mm.SHIFTS, "draws": mm.DRAWS, "l5": l5.tolist(), "lc": lc.tolist(), "l3": l3.tolist(),
            "seconds": time.monotonic() - started}
    (feats / "meta.json").write_text(json.dumps(meta))
    print(f"wrote {feats}: {n} rows, {done:,} tokens, {(time.monotonic() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
