"""embed_evo1.py MODEL TABLE OUT [--device cuda:0] [--isolate] [--parquet P] [--limit N] — Evo 1 through the evo-model
package, as mRNABench's evo1.py: `Evo(model_version)`, unembed replaced by the identity (the output is the final norm),
chunks of 8,192 nt, one token per base (byte level, no special tokens), mean over every position.
  mbmean      mRNABench's readout: mean over all positions of the last layer, rounded to bfloat16 (their wrapper
              averages the model's bfloat16 output and keeps that dtype until dataset_embedder's .float())
  bases / regions / split3 / shift / random   embed_mm.py's pools on the same last-layer output
The weights are the 1.1_fix revision's safetensors (evo-model's load_checkpoint), read from the local Hugging Face cache\n(snapshot_download asked for the safetensors and json only).
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

spec = importlib.util.spec_from_file_location("mm", CODE / "harness/embed_mm.py")
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)
VERSIONS = ("evo-1-8k-base", "evo-1-131k-base")
CHUNK = 8192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=VERSIONS)
    ap.add_argument("table")
    ap.add_argument("out")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--isolate", action="store_true")
    ap.add_argument("--parquet", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--revision", default="", help="Hugging Face revision of the weights (evo-model's own choice when empty: 1.1_fix)")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import huggingface_hub
    from torch import nn
    full = huggingface_hub.snapshot_download

    def safetensors_only(repo_id, **kw):  # evo-model reads only the safetensors; the cache holds no .bin / .pt
        if a.revision:
            kw["revision"] = a.revision
        return full(repo_id, allow_patterns=["*.safetensors", "*.json"], **kw)

    huggingface_hub.snapshot_download = safetensors_only
    from evo import Evo
    evo_model = Evo(a.model)
    net = evo_model.model.to(a.device).eval()
    tokenize = evo_model.tokenizer.tokenize

    class IdentityEmbedding(nn.Module):  # mRNABench's evo1.py: return the embedding, not the logits
        def unembed(self, u):
            return u

    net.unembed = IdentityEmbedding()
    frac = os.environ.get("EMBED_MEM_FRACTION")
    if frac and torch.device(a.device).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(frac), torch.device(a.device))
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
            outs = []
            for s in segs:
                for st in range(0, len(s), CHUNK):
                    ids = torch.tensor(tokenize(s[st:st + CHUNK]), dtype=torch.int).unsqueeze(0).to(a.device)
                    emb, _ = net(ids)
                    outs.append(emb[0].float())
                    done += ids.shape[1]
            e = torch.cat(outs)
            assert e.shape[0] == L[r], (int(r), tuple(e.shape), int(L[r]))
            if arr is None:
                H = e.shape[1]
                shapes = {"mbmean": (n, H), "bases": (n, H), "regions": (n, 3, H), "split3": (n, 3, H)}
                if extra:
                    shapes |= {"shift": (n, len(mm.SHIFTS), 3 * H), "random": (n, mm.DRAWS, 3 * H)}
                arr = {nm: np.lib.format.open_memmap(feats / f"{nm}.npy", mode="w+", dtype=np.float32, shape=shp) for nm, shp in shapes.items()}
            arr["bases"][r] = e.mean(0).cpu().numpy()
            arr["mbmean"][r] = e.mean(0).bfloat16().float().cpu().numpy()
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
    meta = {"model": a.model, "table": a.table, "parquet": a.parquet, "isolate": a.isolate, "layers": ["norm"], "revision": a.revision or "1.1_fix", "chunk": CHUNK, "mbmean": "bf16",
            "rows": n, "order": order.tolist(), "sets": df["split"].to_numpy()[order].tolist() if "split" in df else None,
            "targets": {c: df[c].to_numpy(float)[order].tolist() for c in df.columns if c.startswith("target")},
            "shifts": mm.SHIFTS, "draws": mm.DRAWS, "l5": l5.tolist(), "lc": lc.tolist(), "l3": l3.tolist(),
            "seconds": time.monotonic() - started}
    (feats / "meta.json").write_text(json.dumps(meta))
    print(f"wrote {feats}: {n} rows, {done:,} tokens, {(time.monotonic() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
