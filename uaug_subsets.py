"""uaug_subsets.py [TABLE...] — Table 4 without edits: the transcripts of each table grouped by the uAUG their 5'UTR
already carries (uaug_truth_tables.py's classes: kind x Kozak for one-uAUG 5'UTRs, "multi", "none"), and for each class
the difference to the uAUG-free transcripts in the label (truth) and in the two probes' predictions.

Per model the prediction of a row is the mean of its test-set predictions over the seeds that hold it out (every value
is out of sample; ~80% of rows are held out by at least one of the ten seeds); the truth is taken on the same rows. Raw:
(class mean − uAUG-free mean) / the label's SD on those rows. Adjusted: the class's coefficient in one OLS over all classes
(dummies, uAUG-free as the reference) with log1p length and GC of the three regions. Rows without a CDS are left out.
Writes results/elements/uaug_subsets.json.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from paths import CODE, WORK

P = WORK / "results"
spec = importlib.util.spec_from_file_location("ut", CODE / "uaug_truth_tables.py")
ut = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ut)
MODELS = ["rinalmo-giga", "ernierna", "rnaernie", "rnamsm", "utrlm-te_el", "rnafm", "splicebert", "orthrus-large6", "orthrus-base4",
          "helix-mrna", "helixnano", "rnabert", "utrbert-6mer", "dnabert2", "dnaberts", "hyenadna-medium-450k", "nt-2.5b-multi",
          "aido-rna-1.6b", "evo2-7b-base", "evo1-8k-base"]
TABLES = {"te-human": "batch.y", "te-mouse": "batch.y", "sugimoto": "batch.y", "mrl-egfp": "batch.unmod"}
CATS = [f"{k}/{z}" for k in ("extension", "overlap", "uORF") for z in ("strong", "weak")]
COV = ["l5", "lc", "l3", "gc5", "gcc", "gc3"]


def main():
    tables = [t for t in TABLES if not sys.argv[1:] or t in sys.argv[1:]]
    out = json.loads((P / "elements" / "uaug_subsets.json").read_text()) if (P / "elements" / "uaug_subsets.json").exists() else {}
    for table in tables:
        key, _ = ut.TABLES[table]
        df = pd.read_parquet(ut.DATA / f"{key}.parquet", columns=["UTR5_sequence", "CDS_sequence"])
        u5 = [s.upper().replace("U", "T") for s in df["UTR5_sequence"].fillna("")]
        cds = [s.upper().replace("U", "T") for s in df["CDS_sequence"].fillna("")]
        cls = np.array([("none" if n == 0 else "multi" if n > 1 else f"{k}/{z}") for n, k, z in (ut.classify(a, b) for a, b in zip(u5, cds))], dtype=object)
        has_cds = np.array([len(b) > 0 for b in cds])
        elem = pd.read_parquet(P / "elements" / f"{table}.parquet")
        cov = np.column_stack([np.log1p(elem[c]) if c in ("l5", "lc", "l3") else elem[c] for c in COV]).astype(float)
        cov = np.where(np.isnan(cov), np.nanmean(cov, 0), cov)
        tgt = TABLES[table]
        per_model = {}
        for m in MODELS:
            f = P / "preds" / m / f"{table}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            order = z["order"]
            y = z[f"y|{tgt}"]
            c, hc, cv = cls[order], has_cds[order], cov[order]
            preds = {r: np.nanmean(np.where(z["test"], z[f"{r}|{tgt}"].astype(float), np.nan), 0) for r in ("mbmean", "regions")}
            held = ~np.isnan(preds["mbmean"]) & ~np.isnan(preds["regions"]) & np.isfinite(y) & hc
            sd = y[held].std()
            ref = held & (c == "none")
            rec = {"n": {k: int((held & (c == k)).sum()) for k in CATS + ["none", "multi"]}}
            used = held  # every held-out row with a CDS is uAUG-free, multi or one of the kind/Kozak classes
            levels = sorted({k for k in c[used] if k != "none"})
            X = np.column_stack([np.ones(used.sum())] + [(c[used] == k).astype(float) for k in levels] + [cv[used]])
            for name, v in (("truth", y), ("mbmean", preds["mbmean"]), ("regions", preds["regions"])):
                coef = np.linalg.lstsq(X, v[used], rcond=None)[0]
                rec[name] = {k: float((v[held & (c == k)].mean() - v[ref].mean()) / sd) if (held & (c == k)).sum() else None for k in CATS + ["multi"]}
                rec[f"{name}_adj"] = {k: float(coef[1 + levels.index(k)] / sd) if k in levels else None for k in CATS + ["multi"]}
            per_model[m] = rec
        out[table] = per_model
        if not per_model:
            continue
        for adj in ("", "_adj"):
            print(f"{table} {'adjusted' if adj else 'raw'}  ({len(per_model)} models)")
            for k in CATS + ["multi"]:
                t = np.mean([r[f"truth{adj}"][k] for r in per_model.values() if r[f"truth{adj}"][k] is not None])
                mb = np.mean([r[f"mbmean{adj}"][k] for r in per_model.values() if r[f"mbmean{adj}"][k] is not None])
                rg = np.mean([r[f"regions{adj}"][k] for r in per_model.values() if r[f"regions{adj}"][k] is not None])
                closer = sum(abs(r[f"regions{adj}"][k] - r[f"truth{adj}"][k]) < abs(r[f"mbmean{adj}"][k] - r[f"truth{adj}"][k])
                             for r in per_model.values() if r[f"truth{adj}"][k] is not None)
                n = next(iter(per_model.values()))["n"][k]
                print(f"  {k:18} n {n:6}  truth {t:+.3f}  mbmean {mb:+.3f}  regions {rg:+.3f}  regions closer in {closer} of {len(per_model)}")
    (P / "elements" / "uaug_subsets.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
