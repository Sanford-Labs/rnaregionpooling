"""element_compare.py — test rows grouped by a sequence element: is the mbmean probe's prediction or the regions probe's
closer to the truth?

Per model, table and seed, on that seed's test rows: the difference between the element group and its reference group
in the label (the truth), in the mbmean probe's prediction and in the regions probe's, each divided by the label's SD
on those test rows. `adjusted`: the element's OLS coefficient with log1p length and GC of the three regions as
covariates, over the same rows, on the same scale. Per model: regions is closer (a win) when |regions − truth| is below
|mbmean − truth| with Wilcoxon p < 0.05 over the ten seeds, mbmean closer (a loss) the other way, a tie otherwise.

Reads results/preds/<model>/<table>.npz (harness/preds.py) and results/elements/<table>.parquet (elements.py); writes
results/elements/compare.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from paths import CODE, WORK

P = WORK / "results"
MODELS = ["rinalmo-giga", "ernierna", "rnaernie", "rnamsm", "utrlm-te_el", "rnafm", "splicebert", "orthrus-large6", "orthrus-base4",
          "helix-mrna", "helixnano", "rnabert", "utrbert-6mer", "dnabert2", "dnaberts", "hyenadna-medium-450k", "nt-2.5b-multi",
          "aido-rna-1.6b", "evo2-7b-base", "evo1-8k-base"]
TE_MRL = ["te-human", "te-mouse", "sugimoto"]
MPRA = ["mrl-egfp", "mrl-mcherry", "mrl-designed", "mrl-varying"]  # only the 5'UTR varies in these libraries
TARGET = {"mrl-egfp": "batch.unmod"}  # the eGFP library's unmodified-uridine MRL, as Figure 1a
HL = ["rnahl-human", "rnahl-mouse"]
# name -> (tables, element group, reference group, the effect's known sign on the label)
ELEMENTS = {
    "out-of-frame uAUG vs none": (TE_MRL + MPRA, lambda e: e["uaug"] == "out-of-frame", lambda e: e["uaug"] == "none", -1),
    "in-frame uAUG vs none": (TE_MRL + MPRA, lambda e: e["uaug"] == "in-frame", lambda e: e["uaug"] == "none", 0),
    "CAI top vs bottom quarter": (TE_MRL + HL, lambda e: e["cai"] >= e["cai"].quantile(.75), lambda e: e["cai"] <= e["cai"].quantile(.25), 1),
    "codon optimality (CSC) top vs bottom quarter": (TE_MRL + HL, lambda e: e["csc"] >= e["csc"].quantile(.75),
                                                     lambda e: e["csc"] <= e["csc"].quantile(.25), 1),
    "AUUUA ≥3 vs 0": (HL, lambda e: e["auuua"] >= 3, lambda e: e["auuua"] == 0, -1),
    "let-7 site vs none": (HL, lambda e: e["mir_let-7"] > 0, lambda e: e["mir_let-7"] == 0, -1),
    "miR-16 site vs none": (HL, lambda e: e["mir_miR-16"] > 0, lambda e: e["mir_miR-16"] == 0, -1),
}
COV = ["l5", "lc", "l3", "gc5", "gcc", "gc3"]


def diffs(v, a, b, covs, sd):
    raw = (v[a].mean() - v[b].mean()) / sd
    rows = a | b
    X = np.column_stack([np.ones(rows.sum()), a[rows].astype(float), covs[rows]])
    coef = np.linalg.lstsq(X, v[rows], rcond=None)[0][1] / sd
    return raw, coef


def main():
    out = {}
    for name, (tables, pick_a, pick_b, sign) in ELEMENTS.items():
        for table in tables:
            if not (P / "elements" / f"{table}.parquet").exists():
                continue
            elem = pd.read_parquet(P / "elements" / f"{table}.parquet")
            a_all, b_all = pick_a(elem).to_numpy(), pick_b(elem).to_numpy()
            cov = np.column_stack([np.log1p(elem[c]) if c in ("l5", "lc", "l3") else elem[c] for c in COV]).astype(float)
            cov = np.where(np.isnan(cov), np.nanmean(cov, 0), cov)
            per_model = {}
            for m in MODELS:
                f = P / "preds" / m / f"{table}.npz"
                if not f.exists():
                    continue
                z = np.load(f)
                order, test = z["order"], z["test"]
                tgt = TARGET.get(table, "batch.y")
                y = z[f"y|{tgt}"]
                a, b, cv = a_all[order], b_all[order], cov[order]
                rec = {k: [] for k in ("truth", "mbmean", "regions", "truth_adj", "mbmean_adj", "regions_adj")}
                for s in range(test.shape[0]):
                    rows = test[s] & np.isfinite(y)
                    sd = y[rows].std()
                    aa, bb = a & rows, b & rows
                    if aa.sum() < 10 or bb.sum() < 10:
                        continue
                    for k, v in (("truth", y), ("mbmean", z[f"mbmean|{tgt}"].astype(float)[s]), ("regions", z[f"regions|{tgt}"].astype(float)[s])):
                        raw, adj = diffs(np.nan_to_num(v), aa, bb, cv, sd)
                        rec[k].append(raw)
                        rec[f"{k}_adj"].append(adj)
                rec = {k: np.array(v) for k, v in rec.items()}
                res = {"n_a": int(a.sum()), "n_b": int(b.sum())}
                for adj in ("", "_adj"):
                    e_mb = np.abs(rec[f"mbmean{adj}"] - rec[f"truth{adj}"])
                    e_rg = np.abs(rec[f"regions{adj}"] - rec[f"truth{adj}"])
                    d = e_mb - e_rg
                    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                    res |= {f"truth{adj}": rec[f"truth{adj}"].mean(), f"mbmean{adj}": rec[f"mbmean{adj}"].mean(),
                            f"regions{adj}": rec[f"regions{adj}"].mean(), f"err_mbmean{adj}": e_mb.mean(), f"err_regions{adj}": e_rg.mean(),
                            f"verdict{adj}": "regions" if d.mean() > 0 and p < 0.05 else "mbmean" if d.mean() < 0 and p < 0.05 else "tie"}
                per_model[m] = res
            out.setdefault(name, {})[table] = {"sign": sign, "models": per_model}
            if not per_model:
                continue
            for adj in ("", "_adj"):
                t = np.mean([r[f"truth{adj}"] for r in per_model.values()])
                mb = np.mean([r[f"mbmean{adj}"] for r in per_model.values()])
                rg = np.mean([r[f"regions{adj}"] for r in per_model.values()])
                v = [r[f"verdict{adj}"] for r in per_model.values()]
                print(f"{name:28} {table:12} {'adjusted' if adj else 'raw     '} n {per_model[next(iter(per_model))]['n_a']:5}/{per_model[next(iter(per_model))]['n_b']:5} "
                      f"truth {t:+.3f}  mbmean {mb:+.3f}  regions {rg:+.3f}  | regions closer / mbmean closer / tie "
                      f"{v.count('regions')}/{v.count('mbmean')}/{v.count('tie')} of {len(v)}", flush=True)
    (P / "elements" / "compare.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
