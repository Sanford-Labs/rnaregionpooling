"""embed_hf.py MODEL TABLE OUT [options] — embed_mm.py's pools for transformers checkpoints outside multimolecule.

  nt-2.5b-multi         InstaDeepAI/nucleotide-transformer-2.5b-multi-species: 6-mer tokens (single bases where a
                        6-mer does not fit). As mRNABench: the whole sequence tokenized once, the ids cut into chunks
                        of model_max_length (only the first chunk carries <cls>), last hidden state, mean over tokens.
                        A base takes the vector of the token that covers it.
  hyenadna-medium-450k  LongSafari/hyenadna-medium-450k-seqlen-hf: one token per base; chunks of the model's context
                        (450k), bf16, the model's first output, mean over tokens.
Both read DNA (T). Pools, --isolate, --parquet, row grouping and outputs are embed_mm.py's.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import importlib.util
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import CODE, WORK
spec = importlib.util.spec_from_file_location("mm", CODE / "harness/embed_mm.py")
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)

MODELS = {"nt-2.5b-multi": ("InstaDeepAI/nucleotide-transformer-2.5b-multi-species", "nt"),
          "hyenadna-medium-450k": ("LongSafari/hyenadna-medium-450k-seqlen-hf", "hyena")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("table")
    ap.add_argument("out")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget", type=int, default=32768, help="padded tokens per forward")
    ap.add_argument("--isolate", action="store_true")
    ap.add_argument("--parquet", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
    hf, kind = MODELS[a.model]
    dev = torch.device(a.device)
    # EMBED_MEM_FRACTION caps this process's share of the card's memory (a card shared with other work)
    frac = os.environ.get("EMBED_MEM_FRACTION")
    if frac and dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(frac), dev)
    tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    if kind == "nt":
        model = AutoModelForMaskedLM.from_pretrained(hf, trust_remote_code=True, torch_dtype=torch.bfloat16)
        chunk_tokens = tok.model_max_length
    else:
        model = AutoModel.from_pretrained(hf, trust_remote_code=True, torch_dtype=torch.bfloat16)
        chunk_tokens = 450_000
    model = model.to(dev).eval()
    if a.parquet:
        df = pd.read_parquet(a.parquet)
        order = np.arange(len(df))
    else:
        df = pd.read_parquet(mm.DATA / f"{mm.KEYS[a.table]}.parquet")
        order = np.concatenate([df.index[df["split"] == s].to_numpy() for s in ("train", "val", "test")])
    if a.limit:
        order = order[:a.limit]
    u5, cds, u3 = ([s.upper().replace("U", "T") for s in df[c].fillna("").to_numpy()[order]] for c in mm.SEQ_COLS) \
        if hasattr(mm, "SEQ_COLS") else ([s.upper().replace("U", "T") for s in df[c].fillna("").to_numpy()[order]]
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
    H = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.d_model
    shapes = {"mbmean": (n, H), "bases": (n, H), "regions": (n, 3, H), "split3": (n, 3, H)}
    if extra:
        shapes |= {"shift": (n, len(mm.SHIFTS), 3 * H), "random": (n, mm.DRAWS, 3 * H)}
    feats = Path(a.out) / "features"
    feats.mkdir(parents=True, exist_ok=True)
    arr = {nm: np.lib.format.open_memmap(feats / f"{nm}.npy", mode="w+", dtype=np.float32, shape=shp) for nm, shp in shapes.items()}
    specials = set(tok.all_special_ids)

    def tokenize(s):
        """ids and the number of bases each id covers (0 for special tokens)."""
        ids = tok(s, add_special_tokens=True)["input_ids"]
        toks = tok.convert_ids_to_tokens(ids)
        span = [0 if i in specials else len(t) for i, t in zip(ids, toks)]
        return ids, span

    started = time.monotonic()
    done = 0
    order_rows = np.argsort(-L, kind="stable")
    with torch.inference_mode():
        for r in order_rows:
            segs = [u5[r], cds[r], u3[r]] if a.isolate else [u5[r] + cds[r] + u3[r]]
            all_vecs, nts = [], []
            for s in segs:
                if not s:
                    continue
                ids, span = tokenize(s)
                assert sum(span) == len(s), (a.model, int(r), sum(span), len(s))
                outs = []
                for st in range(0, len(ids), chunk_tokens):
                    piece = torch.tensor([ids[st:st + chunk_tokens]], device=dev)
                    with torch.autocast(dev.type, dtype=torch.bfloat16):
                        if kind == "nt":
                            h = model(piece, attention_mask=torch.ones_like(piece), output_hidden_states=True).hidden_states[-1]
                        else:
                            h = model(piece)[0]
                    outs.append(h[0].float())
                    done += piece.shape[1]
                v = torch.cat(outs)
                all_vecs.append(v)
                nts.append(torch.repeat_interleave(v, torch.tensor(span, device=dev), dim=0))
            e = torch.cat(nts)
            assert e.shape[0] == L[r]
            arr["mbmean"][r] = torch.cat(all_vecs).mean(0).cpu().numpy()
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
    meta = {"model": a.model, "hf": hf, "table": a.table, "parquet": a.parquet, "isolate": a.isolate, "dtype": "bf16",
            "chunk_tokens": chunk_tokens, "hidden": H, "rows": n, "order": order.tolist(),
            "sets": df["split"].to_numpy()[order].tolist() if "split" in df else None,
            "targets": {c: df[c].to_numpy(float)[order].tolist() for c in df.columns if c.startswith("target")},
            "shifts": mm.SHIFTS, "draws": mm.DRAWS, "l5": l5.tolist(), "lc": lc.tolist(), "l3": l3.tolist(),
            "seconds": time.monotonic() - started}
    (feats / "meta.json").write_text(json.dumps(meta))
    print(f"wrote {feats}: {n} rows, {done:,} tokens, {(time.monotonic() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
