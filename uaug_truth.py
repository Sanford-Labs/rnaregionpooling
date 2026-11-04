"""uaug_truth.py — measured uAUG effects in the Sample 2019 random 5'UTR library (mrl-sample-egfp), the
reference for the uAUG perturbations (perturb_design.py). 5'UTR = 25-nt constant + 50 random nt, then the eGFP AUG at
position 75. Rows with exactly one AUG wholly inside the 5'UTR, grouped by frame relative to the main AUG,
whether an in-frame stop closes the uORF before the main AUG, and Kozak context (-3 purine, +4 G).
"""
import json
import numpy as np
import pandas as pd
from paths import CODE, DATA, WORK

df = pd.read_parquet(DATA / "mrl-sample-egfp.parquet")
u5 = df["UTR5_sequence"].to_numpy()
full5 = [s + "ATGG" for s in u5]  # the CDS starts ATGGGC; +4 of an AUG at 72 reads into it
STOPS = {"TAA", "TAG", "TGA"}
rows = []
for i, s in enumerate(u5):
    seq = full5[i]
    pos = [p for p in range(len(s) - 2) if s[p:p + 3] == "ATG"]
    if len(pos) == 0:
        rows.append((i, 0, "none", "none", "none"))
        continue
    if len(pos) > 1:
        rows.append((i, len(pos), "multi", "multi", "multi"))
        continue
    p = pos[0]
    frame = "in" if (75 - p) % 3 == 0 else "out"
    closed = any(seq[q:q + 3] in STOPS for q in range(p + 3, 73, 3))
    kind = ("uORF" if closed else ("extension" if frame == "in" else "overlap"))
    m3 = s[p - 3] if p >= 3 else "N"
    p4 = seq[p + 3]
    koz = {2: "strong", 1: "adequate", 0: "weak"}[(m3 in "AG") + (p4 == "G")]
    rows.append((i, 1, frame, kind, koz))
g = pd.DataFrame(rows, columns=["row", "n_uaug", "frame", "kind", "kozak"])
out = {}
for col in ("target_mrl_egfp_unmod", "target_mrl_egfp_pseudo", "target_mrl_egfp_m1pseudo"):
    y = df[col].to_numpy(float)
    base = np.nanmean(y[g["n_uaug"] == 0])
    sd = np.nanstd(y)
    res = {"none_mean": base, "sd": sd, "n_none": int((g["n_uaug"] == 0).sum())}
    one = g["n_uaug"] == 1
    for keys in (("frame",), ("kind",), ("kind", "kozak")):
        for name, sub in g[one].groupby(list(keys)):
            name = name if isinstance(name, str) else "/".join(name)
            vals = y[sub["row"].to_numpy()]
            res["/".join(keys) + ":" + name] = {"n": int(len(sub)), "delta": float(np.nanmean(vals) - base),
                                                "delta_sd": float((np.nanmean(vals) - base) / sd)}
    vals = y[(g["n_uaug"] > 1).to_numpy()]
    res["multi"] = {"n": int((g["n_uaug"] > 1).sum()), "delta": float(np.nanmean(vals) - base)}
    out[col] = res
open(WORK / "results/perturb/uaug_truth.json", "w").write(json.dumps(out, indent=1))
r = out["target_mrl_egfp_unmod"]
print("no-uAUG rows", r["n_none"], "mean MRL", round(r["none_mean"], 3), "sd", round(r["sd"], 3))
for k, v in r.items():
    if isinstance(v, dict):
        print(f"  {k:32} n {v['n']:>6}  ΔMRL {v['delta']:+.3f}" + (f"  ({v['delta_sd']:+.2f} sd)" if 'delta_sd' in v else ""))
