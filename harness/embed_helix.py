"""embed_helix.py TABLE OUT [options] — one table through helical-ai/Helix-mRNA, as mRNABench's wrapper feeds it.

mRNABench (helix_mrna.py, six-track path) writes an 'E' before every codon of the CDS, upper-cases, T -> U, and
hands the string to helical's HelixmRNA: a character tokenizer that appends one [SEP], truncation at 12,288 tokens,
attention_mask = 1 - special_tokens_mask, last hidden state, mean over every position (the E's and the [SEP] too).
The model (helix_ref/, helical 3.0.3's modelling code; its torch SSD fallback summed the inter-chunk decay over
the target chunk, fixed to the source chunk and checked against a step-by-step scan) is causal (Mamba2 + causal
attention), so rows are batched right-padded: a position never sees the padding after it. float32, SDPA.
Pools as embed_mm.py:
  mbmean   mRNABench's mean: the first 12,287 characters + [SEP] of the row (its truncation)
  bases / regions / split3 / shift / random   over the row's nucleotide positions (E's and [SEP] dropped); a row
           longer than 12,287 characters is read in consecutive windows of 12,287 characters, each with its [SEP]
"""
import argparse
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
from paths import CODE, WORK, MODEL_DIR

sys.path.insert(0, CODE / "harness")
spec = importlib.util.spec_from_file_location("mm", CODE / "harness/embed_mm.py")
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)
WEIGHTS = str(MODEL_DIR / "helix-mrna-hf")  # the gated helical-ai/helix-mRNA weights, downloaded here
MAXLEN = 12288
CHARS = {ch: i + 7 for i, ch in enumerate(("A", "C", "G", "U", "N", "E"))} | {"T": 10}
SEP = PAD = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("out")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget", type=int, default=65536, help="padded tokens per forward")
    ap.add_argument("--group", type=int, default=2_000_000, help="tokens per row group")
    ap.add_argument("--parquet", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--bf16", action="store_true", help="comparison only: run the model in bfloat16")
    ap.add_argument("--no-e", action="store_true", help="the plain sequence, no 'E' before each codon (same weights)")
    ap.add_argument("--ref", default="fixed", choices=("fixed", "orig", "scanbug"), help="comparison only: orig = helical 3.0.3 as shipped; scanbug = layers chained, shipped scan")
    a = ap.parse_args()
    import importlib
    mh = importlib.import_module("helix_ref.modeling_helix_mrna" + {"fixed": "", "orig": "_orig", "scanbug": "_scanbug"}[a.ref])

    mixer_forward = mh.Mamba2Mixer.forward

    def unmasked(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        # mRNABench feeds one row at a time, and the mixer applies the mask only to batches; rows here are
        # right-padded, which the causal scan never reads back
        return mixer_forward(self, hidden_states, cache_params, cache_position, None)

    mh.Mamba2Mixer.forward = unmasked
    if a.ref == "orig":  # the shipped model drops every block's output but the last (an MLP): skip the scans
        mh.Mamba2Mixer.forward = lambda self, hidden_states, *args, **kwargs: torch.zeros_like(hidden_states)
    HelixmRNAPretrainedModel = mh.HelixmRNAPretrainedModel

    dev = torch.device(a.device)
    # EMBED_MEM_FRACTION caps this process's share of the card's memory (a card shared with other work)
    frac = os.environ.get("EMBED_MEM_FRACTION")
    if frac and dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(frac), dev)
    model = HelixmRNAPretrainedModel.from_pretrained(a.weights, attn_implementation="sdpa", torch_dtype=torch.bfloat16 if a.bf16 else torch.float32)
    model = model.to(dev).eval()
    H = model.config.hidden_size
    if a.parquet:
        df = pd.read_parquet(a.parquet)
        order = np.arange(len(df))
    else:
        df = pd.read_parquet(mm.DATA / f"{mm.KEYS[a.table]}.parquet")
        order = np.concatenate([df.index[df["split"] == s].to_numpy() for s in ("train", "val", "test")])
    if a.limit:
        order = order[:a.limit]
    u5, cds, u3 = (df[c].fillna("").to_numpy()[order] for c in ("UTR5_sequence", "CDS_sequence", "UTR3_sequence"))
    clean = lambda s: s.upper().replace("T", "U")  # noqa: E731
    u5, cds, u3 = ([clean(x) for x in v] for v in (u5, cds, u3))
    l5 = np.array([len(x) for x in u5]); lc = np.array([len(x) for x in cds]); l3 = np.array([len(x) for x in u3])
    L = l5 + lc + l3
    n = len(order)
    extra = (not a.parquet) and a.table in mm.REGRESSION6
    rng = np.random.default_rng(0)
    donors = []
    for _ in range(mm.DRAWS):
        p = rng.permutation(n)
        donors.append(np.where(p == np.arange(n), np.roll(p, 1), p))

    def text(r):  # the string mRNABench builds: an E before each codon of the CDS
        if a.no_e:
            return u5[r] + cds[r] + u3[r]
        return u5[r] + "".join("E" + cds[r][i:i + 3] for i in range(0, lc[r], 3)) + u3[r]

    shapes = {"mbmean": (n, H), "bases": (n, H), "regions": (n, 3, H), "split3": (n, 3, H)}
    if extra:
        shapes |= {"shift": (n, len(mm.SHIFTS), 3 * H), "random": (n, mm.DRAWS, 3 * H)}
    feats = Path(a.out) / "features"
    feats.mkdir(parents=True, exist_ok=True)
    arr = {nm: np.lib.format.open_memmap(feats / f"{nm}.npy", mode="w+", dtype=np.float32, shape=shp) for nm, shp in shapes.items()}
    width = MAXLEN - 1
    T = np.array([len(text(r)) for r in range(n)])
    by_len = np.argsort(-T, kind="stable")
    started = time.monotonic()
    tokens_done = rows_done = 0
    with torch.inference_mode():
        g0 = 0
        while g0 < n:
            g1, acc = g0, 0
            while g1 < n and (acc == 0 or acc + T[by_len[g1]] <= a.group):
                acc += T[by_len[g1]] + 1
                g1 += 1
            rows = by_len[g0:g1]
            g0 = g1
            pieces = []  # (row, window index, string)
            for r in rows:
                s = text(r)
                for wi, st in enumerate(range(0, max(len(s), 1), width)):
                    pieces.append((r, wi, s[st:st + width]))
            outs = {}
            order_p = sorted(range(len(pieces)), key=lambda j: -len(pieces[j][2]))
            j = 0
            while j < len(order_p):
                longest = len(pieces[order_p[j]][2]) + 1
                take = max(1, a.budget // longest)
                batch = order_p[j:j + take]
                j += len(batch)
                ids = torch.full((len(batch), longest), PAD, dtype=torch.long)
                mask = torch.zeros((len(batch), longest), dtype=torch.long)
                for bi, b in enumerate(batch):
                    s = pieces[b][2]
                    ids[bi, :len(s)] = torch.tensor([CHARS.get(ch, 6) for ch in s])
                    ids[bi, len(s)] = SEP
                    mask[bi, :len(s)] = 1  # [SEP] and padding are special tokens: mask 0
                h = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False)[0].float()
                for bi, b in enumerate(batch):
                    m = len(pieces[b][2]) + 1
                    outs[b] = h[bi, :m]
                    tokens_done += m
            index = {}
            for b, (r, wi, _) in enumerate(pieces):
                index.setdefault(r, []).append((wi, b))
            got = {nm: [] for nm in arr}
            for r in rows:
                parts = [outs[b] for _, b in sorted(index[r])]
                got["mbmean"].append(parts[0].mean(0))
                s = text(r)
                keep = torch.tensor([ch != "E" for ch in s], device=dev)
                e = torch.cat([p[:-1] for p in parts])[keep]
                assert e.shape[0] == L[r], (int(r), tuple(e.shape), int(L[r]))
                got["bases"].append(e.mean(0))
                got["regions"].append(mm.spans_pool(e, (int(l5[r]), int(l5[r] + lc[r]))))
                got["split3"].append(mm.spans_pool(e, (int(round(L[r] / 3)), int(round(2 * L[r] / 3)))))
                if extra:
                    got["shift"].append(torch.stack([mm.spans_pool(e, (int(l5[r] + d), int(l5[r] + lc[r] + d))).reshape(-1)
                                                     for d in mm.SHIFTS]))
                    rand = []
                    for p in donors:
                        dn = p[r]
                        f5, fc = l5[dn] / max(L[dn], 1), lc[dn] / max(L[dn], 1)
                        rand.append(mm.spans_pool(e, (int(round(f5 * L[r])), int(round((f5 + fc) * L[r])))).reshape(-1))
                    got["random"].append(torch.stack(rand))
            at = np.asarray(rows)
            for nm, v in got.items():
                arr[nm][at] = torch.stack(v).cpu().numpy()
            rows_done += len(rows)
            del outs
            spent = time.monotonic() - started
            print(f"  {rows_done}/{n} rows, {tokens_done / spent:,.0f} tok/s, {spent / 60:.1f} min", flush=True)
    for v in arr.values():
        v.flush()
    meta = {"model": "helix-mrna", "hf": "helical-ai/Helix-mRNA", "table": a.table, "parquet": a.parquet, "isolate": False,
            "dtype": "fp32", "scan": a.ref, "chunk": width, "mb": "truncate_12288", "tokens": "single" if a.no_e else "single+E", "hidden": H, "rows": n,
            "order": order.tolist(), "sets": df["split"].to_numpy()[order].tolist() if "split" in df else None,
            "targets": {c: df[c].to_numpy(float)[order].tolist() for c in df.columns if c.startswith("target")},
            "shifts": list(mm.SHIFTS), "draws": mm.DRAWS, "l5": l5.tolist(), "lc": lc.tolist(), "l3": l3.tolist(),
            "seconds": time.monotonic() - started}
    (feats / "meta.json").write_text(json.dumps(meta))
    print(f"wrote {feats}: {n} rows, {tokens_done:,} tokens, {(time.monotonic() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
