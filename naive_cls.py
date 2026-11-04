"""naive_cls.py --dump|--score [--six] TABLE... — mRNABench's naive-4 / naive-6 rows on the classification tables.

Features as naive_ridge.py (3-7-mer counts over ACGT + GC; naive-6 adds CDS length and splice-site count). The readout
is mRNABench's linear_probe.py: sklearn's LogisticRegression(max_iter=5000) per label, fit by naive_cls_gpu.py (scipy's
L-BFGS-B, objective on the card; naive_cls_sk.py runs the same fits through sklearn itself, on the CPU).
  --dump    results/naive/cls-dump/<table>[.six]/: X.npy (float32), Y.npy, cuts.npy (the ten seeds' splits, SEEDS order)
  --score   naive_cls_gpu.py's P.npy -> results/naive/<table>[.six].cls.json; metrics (micro `ap`; `ap_macro` for eCLIP)
            and the ten seeds as fit_cls.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from paths import CODE, WORK

spec = importlib.util.spec_from_file_location("nr", CODE / "naive_ridge.py")
nr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nr)
cp = nr.cp
from metrics import score_logistic  # noqa: E402

DUMP = nr.OUT / "cls-dump"
METRIC = {"vep-traitgym-complex": "ap", "vep-traitgym-mendelian": "ap", "eclip-binding-hepg2": "ap_macro",
          "eclip-binding-k562": "ap_macro", "rna-loc-fazal": "ap", "rna-lifecycle-ietswaart": "ap", "prot-loc": "ap",
          "go-bp": "ap", "go-cc": "ap", "go-mf": "ap"}


def dump(table, six, d):
    df = pd.read_parquet(cp.DATA / f"{table}.parquet")
    order = np.concatenate([df.index[df["split"] == s].to_numpy() for s in ("train", "val", "test")])
    seqs = [(a + b + c).upper() for a, b, c in zip(*(df[col].fillna("").to_numpy()[order] for col in ("UTR5_sequence", "CDS_sequence", "UTR3_sequence")))]
    X = nr.kmer_features(seqs)
    if six:
        lc = df["CDS_sequence"].fillna("").str.len().to_numpy()[order].astype(np.float32)
        sites = sum(df[c].apply(lambda v: 0 if v is None else len(v)).to_numpy()[order].astype(np.float32)
                    for c in ("UTR5_splice_coord", "CDS_splice_coord", "UTR3_splice_coord"))
        X = np.concatenate([X, np.stack([lc, sites], 1)], 1)
    tcols = sorted(c for c in df.columns if c.startswith("target"))
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "X.npy", X)
    np.save(d / "Y.npy", np.stack([df[c].to_numpy(float)[order] for c in tcols], 1))
    np.save(d / "cuts.npy", np.stack(cp.seed_cuts(table, df, order)).astype("U5"))
    print(f"dumped {table}{' six' if six else ''}: {X.shape[0]} rows x {X.shape[1]} features, {len(tcols)} labels", flush=True)


def score(table, six, d):
    Y = torch.from_numpy(np.load(d / "Y.npy")).double()
    P = torch.from_numpy(np.load(d / "P.npy")).double()
    cuts = np.load(d / "cuts.npy")
    metric = METRIC[table]
    res = {"val": [], "test": [], "test_ap": [], "test_ap_macro": []}
    for cut, p in zip(cuts, P):
        va, te = (torch.from_numpy(cut == s) for s in ("val", "test"))
        sv, st = score_logistic(Y[va], p[va]), score_logistic(Y[te], p[te])
        res["val"].append(sv[metric]); res["test"].append(st[metric])
        res["test_ap"].append(st["ap"]); res["test_ap_macro"].append(st["ap_macro"])
    iters = np.load(d / "iters.npy")
    name = f"{table}{'.six' if six else ''}.cls.json"
    (nr.OUT / name).write_text(json.dumps({"table": table, "metric": metric, "features": int(np.load(d / "X.npy", mmap_mode="r").shape[1]),
                                           "solver": "naive_cls_gpu.py: scipy L-BFGS-B as sklearn 1.6.1 LogisticRegression(max_iter=5000)", "iters": iters.tolist(), "results": res}))
    print(f"naive-{6 if six else 4} {table} [{metric}]: test {np.mean(res['test']):.4f}  (iterations max {iters.max()}, "
          f"{(iters >= 5000).sum()} of {iters.size} fits at the cap)", flush=True)


def main():
    six = "--six" in sys.argv
    tables = [t for t in sys.argv[1:] if not t.startswith("--")]
    for table in tables:
        d = DUMP / f"{table}{'.six' if six else ''}"
        if "--dump" in sys.argv:
            dump(table, six, d)
        if "--score" in sys.argv:
            score(table, six, d)


if __name__ == "__main__":
    main()
