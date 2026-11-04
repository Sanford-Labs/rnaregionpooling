"""embed_mm2.py MODEL TABLE OUT [options] — embed_mm.py plus DNA models (T kept) and BPE tokens (offsets).

embed_mm.py MODEL TABLE OUT [options] — one table through a frozen multimolecule model, pooled per nucleotide.

TABLE is an mRNABench table (rows in fit.py's order: parquet split train, val, test) or, with --parquet PATH,
any parquet with UTR5_sequence / CDS_sequence / UTR3_sequence (rows in file order; TABLE names the output).
Chunking follows mRNABench's wrapper for the model: non-overlapping chunks of `chunk` nucleotides, each with its
own special tokens. Last-layer pools:
  mbmean   mRNABench's own mean (which special tokens it keeps is per model, `mb`)
  bases    mean over nucleotide positions
  regions  (3, H): 5'UTR, CDS, 3'UTR means (zeros for an empty region)
  split3   (3, H): equal thirds by nucleotide index
  shift    (8, H*3): regions with both boundaries moved by SHIFTS, clipped to the row      } the six regression
  random   (5, H*3): regions at another row's region proportions (5 fixed draws)         } tables only
--isolate encodes the 5'UTR, CDS and 3'UTR as three separate inputs (each chunked on its own) and pools the
three encodings: `regions` there is the iso-regions readout. Rows are processed in groups of about --group
tokens, pooled on the device, written, and dropped.
Token -> nucleotide: `single` (one token per base), `kmer6` (overlapping 6-mers; a base takes the mean of the
6-mers covering it).
"""
import argparse
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
from paths import DATA

DATA = DATA
KEYS = {"mrl-egfp": "mrl-sample-egfp", "mrl-mcherry": "mrl-sample-mcherry", "mrl-varying": "mrl-sample-varying",
        "mrl-designed": "mrl-sample-designed", "rnahl-human": "rnahl-human", "rnahl-mouse": "rnahl-mouse",
        "sugimoto": "mrl-sugimoto", "mrl-hl": "mrl-hl-lbkwk", "te-human": "translation-efficiency-human",
        "te-mouse": "translation-efficiency-mouse"}
REGRESSION6 = {"rnahl-human", "rnahl-mouse", "sugimoto", "mrl-hl", "te-human", "te-mouse"}
KEYS |= {t: t for t in ("vep-traitgym-complex", "vep-traitgym-mendelian", "eclip-binding-hepg2", "eclip-binding-k562",
                       "rna-loc-fazal", "rna-lifecycle-ietswaart", "prot-loc", "go-bp", "go-cc", "go-mf")}
SHIFTS = (-300, -100, -30, -10, 10, 30, 100, 300)
DRAWS = 5
# id, nucleotides per chunk, mRNABench's special-token rule, token map
MODELS = {
    "ernierna": ("multimolecule/ernierna", 1022, "keep_all", "single"),
    "rnaernie": ("multimolecule/rnaernie", 510, "keep_all", "single"),
    "rnamsm": ("multimolecule/rnamsm", 1022, "keep_all", "single"),
    "rnabert": ("multimolecule/rnabert", 438, "keep_all", "single"),
    "utrlm-te_el": ("multimolecule/utrlm-te_el", 1024, "keep_all", "single"),
    "rnafm": ("multimolecule/rnafm", 1022, "strip_inner", "single"),
    "splicebert": ("multimolecule/splicebert", 1024, "keep_all", "single"),
    "rinalmo-giga": ("multimolecule/rinalmo-giga", 8190, "keep_all", "single"),
    "rinalmo-mega": ("multimolecule/rinalmo-mega", 8190, "keep_all", "single"),
    "rinalmo-micro": ("multimolecule/rinalmo-micro", 8190, "keep_all", "single"),
    "utrbert-6mer": ("multimolecule/utrbert-6mer", 510, "cls_only", "kmer6"),
    "dnabert2": ("multimolecule/dnabert2", 8192, "keep_all", "bpe"),
    "dnaberts": ("multimolecule/dnaberts", 8192, "keep_all", "bpe"),
}
DNA = {"dnabert2", "dnaberts"}  # tokenized with T, not U


def spans_pool(e, cuts):
    """(L, H), cut points (b1, b2) -> (3, H) means of [0,b1), [b1,b2), [b2,L); zeros for an empty span."""
    L = e.shape[0]
    cs = torch.cat([torch.zeros(1, e.shape[1], dtype=e.dtype, device=e.device), e.cumsum(0)])
    out = []
    for a, b in ((0, cuts[0]), (cuts[0], cuts[1]), (cuts[1], L)):
        a, b = max(0, min(L, a)), max(0, min(L, b))
        out.append((cs[b] - cs[a]) / (b - a) if b > a else torch.zeros(e.shape[1], dtype=e.dtype, device=e.device))
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("table")
    ap.add_argument("out")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget", type=int, default=65536, help="padded tokens per forward")
    ap.add_argument("--group", type=int, default=2_000_000, help="tokens per row group")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--isolate", action="store_true")
    ap.add_argument("--parquet", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--window", type=int, default=0, help="overlapping windows of this many nt instead of the chunk rule")
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--tokens", action="store_true", help="also keep every base's vector (bf16, tokens.bf16 + tokens_index.npy)")
    a = ap.parse_args()
    from multimolecule import RnaTokenizer  # noqa: F401  (registers the model classes)
    from transformers import AutoModel, AutoTokenizer

    mid, chunk, mb, tmap = MODELS[a.model]
    dev = torch.device(a.device)
    # EMBED_MEM_FRACTION caps this process's share of the card's memory (a card shared with other work)
    frac = os.environ.get("EMBED_MEM_FRACTION")
    if frac and dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(frac), dev)
    tok = AutoTokenizer.from_pretrained(mid) if tmap == "bpe" else RnaTokenizer.from_pretrained(mid)
    bk = getattr(tok, "_tokenizer", None)
    if tmap == "bpe" and bk is not None and bk.model.unk_token and bk.token_to_id(bk.model.unk_token) is None:
        bk.model.unk_token = tok.unk_token  # multimolecule's DNABERT-2/-S: the BPE names "[UNK]", the vocab has it as "<unk>" (id 0)
    try:  # eager attention materializes L x L per head; 8,190-nt chunks need SDPA
        model = AutoModel.from_pretrained(mid, attn_implementation="sdpa")
    except (ValueError, ImportError):
        model = AutoModel.from_pretrained(mid)
    model = model.to(dev).eval()
    print(f"{a.model}: attention {model.config._attn_implementation}, {a.dtype}", flush=True)
    if a.parquet:
        df = pd.read_parquet(a.parquet)
        order = np.arange(len(df))
    else:
        df = pd.read_parquet(DATA / f"{KEYS[a.table]}.parquet")
        order = np.concatenate([df.index[df["split"] == s].to_numpy() for s in ("train", "val", "test")])
    if a.limit:
        order = order[:a.limit]
    u5, cds, u3 = (df[c].fillna("").to_numpy()[order] for c in ("UTR5_sequence", "CDS_sequence", "UTR3_sequence"))
    clean = (lambda s: s.upper().replace("U", "T")) if a.model in DNA else (lambda s: s.upper().replace("T", "U"))  # noqa: E731
    u5, cds, u3 = ([clean(x) for x in v] for v in (u5, cds, u3))
    l5 = np.array([len(x) for x in u5]); lc = np.array([len(x) for x in cds]); l3 = np.array([len(x) for x in u3])
    L = l5 + lc + l3
    n = len(order)
    extra = (not a.parquet) and a.table in REGRESSION6 and not a.isolate
    k = 6 if tmap == "kmer6" else 1
    rng = np.random.default_rng(0)
    donors = []
    for _ in range(DRAWS):
        p = rng.permutation(n)
        donors.append(np.where(p == np.arange(n), np.roll(p, 1), p))

    H = model.config.hidden_size
    shapes = {"mbmean": (n, H), "bases": (n, H), "regions": (n, 3, H), "split3": (n, 3, H)}
    if extra:
        shapes |= {"shift": (n, len(SHIFTS), 3 * H), "random": (n, DRAWS, 3 * H)}
    feats = Path(a.out) / "features"
    feats.mkdir(parents=True, exist_ok=True)
    arr = {nm: np.lib.format.open_memmap(feats / f"{nm}.npy", mode="w+", dtype=np.float32, shape=shp) for nm, shp in shapes.items()}

    def segments(r):
        return [u5[r], cds[r], u3[r]] if a.isolate else [u5[r] + cds[r] + u3[r]]

    def starts_of(s):
        if not s:
            return []
        if a.window and k == 1:
            st = list(range(0, max(len(s) - a.window, 0) + 1, a.stride))
            if st[-1] + a.window < len(s):
                st.append(len(s) - a.window)
            return st
        return list(range(0, max(len(s), 1), chunk)) if k == 1 else list(range(0, max(len(s) - k + 1, 1), chunk))

    def chunks_of(s):
        if a.window and k == 1:
            return [s[st:st + a.window] for st in starts_of(s)]
        if k == 1:
            return [s[st:st + chunk] for st in range(0, max(len(s), 1), chunk)] if s else []
        ntok = max(len(s) - k + 1, 1)
        return [s[st:st + chunk + k - 1] for st in range(0, ntok, chunk)]

    if a.tokens:
        tok_file = open(feats / "tokens.bf16", "wb")
        tok_index = np.zeros((n, 2), np.int64)  # row -> (first base in the file, length)
        tok_at = 0
    autocast = torch.autocast(dev.type, dtype=torch.bfloat16) if a.dtype == "bf16" else torch.autocast(dev.type, enabled=False)
    by_len = np.argsort(-L, kind="stable")
    started = time.monotonic()
    tokens_done = rows_done = 0
    with torch.inference_mode(), autocast:
        g0 = 0
        while g0 < n:
            g1, acc = g0, 0
            while g1 < n and (acc == 0 or acc + L[by_len[g1]] <= a.group):
                acc += L[by_len[g1]] + 2
                g1 += 1
            rows = by_len[g0:g1]
            g0 = g1
            pieces = []  # (row, segment, chunk index, string)
            for r in rows:
                for si, s in enumerate(segments(r)):
                    for ci, (c, st) in enumerate(zip(chunks_of(s), starts_of(s))):
                        pieces.append((r, si, ci, c, st))
            outs, spans = {}, {}
            order_p = sorted(range(len(pieces)), key=lambda j: len(pieces[j][3]))
            j = 0
            while j < len(order_p):
                take = max(1, a.budget // (len(pieces[order_p[min(j + 1, len(order_p) - 1)]][3]) + 2))
                batch = order_p[j:j + take]
                longest = max(len(pieces[b][3]) for b in batch) + 2
                while len(batch) > 1 and len(batch) * longest > a.budget:
                    batch = batch[:-1]
                j += len(batch)
                enc = tok([pieces[b][3] for b in batch], return_tensors="pt", padding=True,
                          **({"return_offsets_mapping": True} if tmap == "bpe" else {}))
                offs = enc.pop("offset_mapping", None)
                enc = {kk: v.to(dev) for kk, v in enc.items() if kk in ("input_ids", "attention_mask")}
                h = model(**enc).last_hidden_state
                for bi, b in enumerate(batch):
                    m = int(enc["attention_mask"][bi].sum())
                    outs[b] = h[bi, :m].float()
                    if offs is not None:
                        spans[b] = offs[bi, :m].tolist()
                    tokens_done += m
            index = {}
            for b, (r, si, ci, _, _st) in enumerate(pieces):
                index.setdefault(r, {}).setdefault(si, []).append((ci, b))
            for r in rows:
                segs = index.get(r, {})
                full, nts = [], []
                for si in range(len(segments(r))):
                    parts = [outs[b] for _, b in sorted(segs.get(si, []))]
                    if not parts:
                        continue
                    if mb == "keep_all":
                        full.append(torch.cat(parts))
                    elif mb == "strip_inner":
                        full.append(torch.cat([p[(0 if q == 0 else 1):(len(p) if q == len(parts) - 1 else len(p) - 1)]
                                               for q, p in enumerate(parts)]))
                    else:
                        full.append(torch.stack([p[0] for p in parts]))
                    if tmap == "bpe":  # each base takes the vector of the token whose offset span covers it
                        rows_ = []
                        for (_, b) in sorted(segs.get(si, [])):
                            sp = spans[b][1:-1]
                            ln = len(pieces[b][3])
                            owner = torch.zeros(ln, dtype=torch.long)
                            for ti, (s0, s1) in enumerate(sp):
                                owner[s0:s1] = ti
                            rows_.append(outs[b][1:-1][owner.to(dev)])
                        inner = torch.cat(rows_)
                    elif a.window:  # overlapping windows: each base averages the windows that cover it
                        seg_len = len(segments(r)[si])
                        acc = torch.zeros(seg_len, H, device=dev)
                        cnt = torch.zeros(seg_len, 1, device=dev)
                        for (_, b) in sorted(segs.get(si, [])):
                            st, v = pieces[b][4], outs[b][1:-1]
                            acc[st:st + v.shape[0]] += v
                            cnt[st:st + v.shape[0]] += 1
                        inner = acc / cnt
                    else:
                        inner = torch.cat([p[1:-1] for p in parts])
                    if k > 1 and inner.shape[0] == 0:  # --isolate: a region shorter than one k-mer has no token; zeros, as an absent region
                        inner = torch.zeros(len(segments(r)[si]), H, device=dev)
                    elif k > 1:
                        cs = torch.cat([torch.zeros(1, H, device=dev), inner.cumsum(0)])
                        idx = torch.arange(len(segments(r)[si]), device=dev)
                        lo, hi = (idx - k + 1).clamp(min=0), idx.clamp(max=inner.shape[0] - 1) + 1
                        inner = (cs[hi] - cs[lo]) / (hi - lo).unsqueeze(1)
                    nts.append(inner)
                e = torch.cat(nts)
                assert e.shape[0] == L[r], (a.model, int(r), tuple(e.shape), int(L[r]))
                if a.tokens:
                    tok_file.write(e.to(torch.bfloat16).view(torch.int16).cpu().numpy().tobytes())
                    tok_index[r] = (tok_at, e.shape[0])
                    tok_at += e.shape[0]
                arr["mbmean"][r] = torch.cat(full).mean(0).cpu().numpy()
                arr["bases"][r] = e.mean(0).cpu().numpy()
                arr["regions"][r] = spans_pool(e, (int(l5[r]), int(l5[r] + lc[r]))).cpu().numpy()
                arr["split3"][r] = spans_pool(e, (int(round(L[r] / 3)), int(round(2 * L[r] / 3)))).cpu().numpy()
                if extra:
                    arr["shift"][r] = torch.stack([spans_pool(e, (int(l5[r] + d), int(l5[r] + lc[r] + d))).reshape(-1)
                                                   for d in SHIFTS]).cpu().numpy()
                    rand = []
                    for p in donors:
                        dn = p[r]
                        f5, fc = l5[dn] / max(L[dn], 1), lc[dn] / max(L[dn], 1)
                        rand.append(spans_pool(e, (int(round(f5 * L[r])), int(round((f5 + fc) * L[r])))).reshape(-1))
                    arr["random"][r] = torch.stack(rand).cpu().numpy()
            rows_done += len(rows)
            del outs
            spent = time.monotonic() - started
            print(f"  {rows_done}/{n} rows, {tokens_done / spent:,.0f} tok/s, {spent / 60:.1f} min", flush=True)
    for v in arr.values():
        v.flush()
    if a.tokens:
        tok_file.close()
        np.save(feats / "tokens_index.npy", tok_index)
    meta = {"window": a.window, "stride": a.stride, "model": a.model, "hf": mid, "table": a.table, "parquet": a.parquet, "isolate": a.isolate, "dtype": a.dtype,
            "chunk": chunk, "mb": mb, "tokens": tmap, "hidden": H, "rows": n, "order": order.tolist(),
            "sets": df["split"].to_numpy()[order].tolist() if "split" in df else None,
            "targets": {c: df[c].to_numpy(float)[order].tolist() for c in df.columns if c.startswith("target")},
            "shifts": SHIFTS, "draws": DRAWS, "l5": l5.tolist(), "lc": lc.tolist(), "l3": l3.tolist(),
            "seconds": time.monotonic() - started}
    (feats / "meta.json").write_text(json.dumps(meta))
    print(f"wrote {feats}: {n} rows, {tokens_done:,} tokens, {(time.monotonic() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
