"""calib_perseed.py — our mRNABench-protocol readout (mbmean, protocol A) against mRNABench's own per-seed results
(experiments/overall_df.parquet of morrislab/mRNABench main, canonical split rows), per model, table and target."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from paths import CODE, WORK

R = WORK / "results/external"
DF = pd.read_parquet(WORK / "dl/mrnabench-gh/overall_df.parquet")
SEEDS = [2541, 413, 411, 412, 2547, 321, 421, 311, 2516, 2515]
KEYS = {"mrl-egfp": "mrl-sample-egfp", "mrl-mcherry": "mrl-sample-mcherry", "mrl-varying": "mrl-sample-varying",
        "mrl-designed": "mrl-sample-designed", "sugimoto": "mrl-sugimoto", "mrl-hl": "mrl-hl-lbkwk",
        "te-human": "translation-efficiency-human", "te-mouse": "translation-efficiency-mouse"}
OURS2THEIRS = {"rinalmo-giga": "rinalmo", "ernierna": "ernierna", "rnaernie": "rnaernie", "rnamsm": "rnamsm",
               "utrlm-te_el": "utrlm-te-el", "rnafm": "rna-fm", "splicebert": "splicebert-v-1024nt",
               "orthrus-large6": "orthrus-large-6", "orthrus-base4": "orthrus-base-4", "rnabert": "rnabert",
               "utrbert-6mer": "utrbert-6mer", "dnabert2": "dnabert2", "dnaberts": "dnabert-s",
               "hyenadna-medium-450k": "hyenadna-medium-450k", "nt-2.5b-multi": "nt-2.5b-multi-species",
               "aido-rna-1.6b": "aido-rna-1b600m", "evo2-7b-base": "evo2-7b-base", "helix-mrna": "helix-mrna", "evo1-8k-base": "evo-1-8k-base"}
TGT = {"batch.mrl": "target_ribosome_load", "batch.hl": "target_in_cell_half_life", "batch.unmod": "target_mrl_egfp_unmod",
       "batch.pseudo": "target_mrl_egfp_pseudo", "batch.m1pseudo": "target_mrl_egfp_m1pseudo"}

def theirs(model, dataset):
    d = DF[(DF.model == model) & (DF.dataset == dataset) & (DF.split == DF.canonical_split)]
    return d

rows = []
for ours, mine in OURS2THEIRS.items():
    for f in sorted((R / ours).glob("*/fit*.json")) if (R / ours).exists() else []:
        if f.name not in ("fit.json", "fit_cls.json"):
            continue
        table = f.parent.name
        ds = KEYS.get(table, table)
        th = theirs(mine, ds)
        if th.empty:
            continue
        res = json.loads(f.read_text())["results"]
        if f.name == "fit.json":
            for t, per in res.items():
                tcol = TGT.get(t) or (th.target.unique()[0] if th.target.nunique() == 1 else None)
                if tcol is None or "mbmean|rawA" not in per:
                    continue
                v = th[th.target == tcol].set_index("seed")["test_r"].reindex(SEEDS).to_numpy()
                o = np.array(per["mbmean|rawA"]["test"], float)
                rows.append((ours, table, tcol.replace("target_", ""), o.mean(), np.nanmean(v), np.nanmax(np.abs(o - v))))
        else:
            if "mbmean|A" not in res:
                continue
            o = np.array(res["mbmean|A"]["test_ap"], float)
            # multilabel tables carry one row per seed with the micro AUPRC; eCLIP's column is the per-label mean
            g = th.groupby("seed")["test_auprc"].mean().reindex(SEEDS).to_numpy()
            if "eclip" in table:
                o = np.array(res["mbmean|A"]["test_ap_macro"], float)
            rows.append((ours, table, "auprc", o.mean(), np.nanmean(g), np.nanmax(np.abs(o - g))))
for r in rows:
    flag = "" if r[5] < 0.02 else "  <-- off"
    print(f"{r[0]:15s} {r[1]:26s} {r[2]:22s} ours {r[3]:.4f} theirs {r[4]:.4f} per-seed max|d| {r[5]:.4f}{flag}")
