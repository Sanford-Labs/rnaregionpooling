"""uaug_truth_tables.py — measured uAUG effects in every table the uAUG insertions run on, by uaug_truth.py's rules:
rows whose 5'UTR holds exactly one AUG, grouped by frame relative to the main AUG, whether an in-frame stop closes the
upstream ORF before the main AUG (uORF; otherwise extension when in frame, overlap when out of frame), and Kozak context
(-3 purine, +4 G: strong both, adequate one, weak neither; +4 read into the CDS when the AUG ends the 5'UTR). Reference:
the rows whose 5'UTR has no AUG; rows without a CDS are left out. delta_sd = (group mean - reference mean) / the
label's SD over the table. Writes results/perturb/uaug_truth_tables.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from paths import CODE, DATA, WORK

DATA = DATA
OUT = WORK / "results/perturb/uaug_truth_tables.json"
TABLES = {"te-human": ("translation-efficiency-human", "target"), "te-mouse": ("translation-efficiency-mouse", "target"),
          "sugimoto": ("mrl-sugimoto", "target"), "mrl-egfp": ("mrl-sample-egfp", "target_mrl_egfp_unmod")}
STOPS = {"TAA", "TAG", "TGA"}


def classify(u5, cds):
    pos = [p for p in range(len(u5) - 2) if u5[p:p + 3] == "ATG"]
    if not pos:
        return 0, "none", "none"
    if len(pos) > 1:
        return len(pos), "multi", "multi"
    p, l5 = pos[0], len(u5)
    seq = u5 + cds[:4]
    frame = "in" if (l5 - p) % 3 == 0 else "out"
    closed = any(seq[q:q + 3] in STOPS for q in range(p + 3, l5 - 2, 3))
    kind = "uORF" if closed else ("extension" if frame == "in" else "overlap")
    m3 = u5[p - 3] if p >= 3 else "N"
    p4 = seq[p + 3] if p + 3 < len(seq) else "N"
    return 1, kind, {2: "strong", 1: "adequate", 0: "weak"}[(m3 in "AG") + (p4 == "G")]


def main():
    out = {}
    for table, (key, col) in TABLES.items():
        df = pd.read_parquet(DATA / f"{key}.parquet", columns=["UTR5_sequence", "CDS_sequence", col])
        u5 = [s.upper().replace("U", "T") for s in df["UTR5_sequence"].fillna("")]
        cds = [s.upper().replace("U", "T") for s in df["CDS_sequence"].fillna("")]
        y = df[col].to_numpy(float)
        g = pd.DataFrame([classify(a, b) for a, b in zip(u5, cds)], columns=["n_uaug", "kind", "kozak"])
        keep = np.array([len(b) > 0 for b in cds]) & np.isfinite(y)
        base_rows = keep & (g["n_uaug"] == 0).to_numpy()
        base, sd = float(np.mean(y[base_rows])), float(np.std(y[keep]))
        res = {"label": col, "none_mean": base, "sd": sd, "n_none": int(base_rows.sum())}
        one = keep & (g["n_uaug"] == 1).to_numpy()
        for (kind, koz), sub in g[one].groupby(["kind", "kozak"]):
            vals = y[sub.index.to_numpy()]
            res[f"kind/kozak:{kind}/{koz}"] = {"n": int(len(sub)), "delta_sd": float((vals.mean() - base) / sd)}
        multi = keep & (g["n_uaug"] > 1).to_numpy()
        res["multi"] = {"n": int(multi.sum()), "delta_sd": float((y[multi].mean() - base) / sd)}
        out[table] = res
        cats = [f"{k}/{z}" for k in ("extension", "overlap", "uORF") for z in ("strong", "weak")]
        print(f"{table}: none {res['n_none']}, " + ", ".join(f"{c} {res.get('kind/kozak:' + c, {}).get('n', 0)} "
                                                           f"{res.get('kind/kozak:' + c, {}).get('delta_sd', float('nan')):+.2f}" for c in cats)
              + f", multi {res['multi']['n']} {res['multi']['delta_sd']:+.2f}", flush=True)
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
