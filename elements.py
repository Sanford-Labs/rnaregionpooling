"""elements.py — sequence elements of every row of the regression tables, for reading the probes' predictions by element.

  uAUG      ATGs in the 5'UTR; in frame with the main AUG when (5'UTR length − position) % 3 == 0. `uaug` = "none",
            "in-frame" (in-frame ones only), "out-of-frame" (at least one out of frame), or "no CDS" (no main AUG).
  CAI       Sharp & Li's geometric mean of relative adaptiveness, w = count / the largest count among the codon's
            synonyms, from the Kazusa Homo sapiens table (93,487 CDS; kazusa-human/showcodon.html). Met, Trp, stops
            and codons with other letters are left out; codons read from the CDS's first base.
  CSC       mean codon stability coefficient over the CDS's sense codons (results/perturb/csc.json: Pearson r of each codon's
            frequency with half-life over the rnahl-human train genes, the scale perturb_design.py recodes along).
  AUUUA     ATTTA occurrences in the 3'UTR, overlapping ones counted.
  miRNA     TargetScan site types per family in the 3'UTR, 15 nt or more after the stop: 8mer, 7mer-m8, 7mer-A1
            (an 8mer counted once). `mir_<family>` = the number of sites. FAMILIES holds the seeds (miRNA nt 2–8).
  covariates  l5, lc, l3 and the GC fraction of each region.

Writes results/elements/<table>.parquet, one row per parquet row (index = the parquet's row index); `elements.py TABLE...`
does those tables only.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from paths import CODE, DATA, WORK

DATA = DATA
OUT = WORK / "results/elements"
# the Codon Usage Database's Homo sapiens table, saved from
# https://www.kazusa.or.jp/codon/cgi-bin/showcodon.cgi?species=9606&aa=1&style=N
KAZUSA = WORK / "kazusa-human/showcodon.html"
CSC = WORK / "results/perturb/csc.json"
TABLES = {"sugimoto": "mrl-sugimoto", "te-human": "translation-efficiency-human", "te-mouse": "translation-efficiency-mouse",
          "rnahl-human": "rnahl-human", "rnahl-mouse": "rnahl-mouse", "mrl-hl": "mrl-hl-lbkwk",
          "mrl-egfp": "mrl-sample-egfp", "mrl-mcherry": "mrl-sample-mcherry", "mrl-designed": "mrl-sample-designed",
          "mrl-varying": "mrl-sample-varying"}
FAMILIES = {"let-7": "GAGGUAG", "miR-16": "AGCAGCA"}  # miRNA nucleotides 2-8 (let-7a-5p, miR-16-5p)
STOPS = {"TAA", "TAG", "TGA"}


def kazusa_weights():
    text = KAZUSA.read_text()
    body = text[text.index("<PRE>") + 5:text.index("</PRE>")]
    counts, aa = {}, {}
    for codon, a, n in re.findall(r"([ACGU]{3}) (\S) +[0-9.]+ +[0-9.]+ \( *(\d+)\)", body):
        c = codon.replace("U", "T")
        counts[c], aa[c] = int(n), a
    assert len(counts) == 64, len(counts)
    best = {}
    for c, a in aa.items():
        best[a] = max(best.get(a, 0), counts[c])
    return {c: counts[c] / best[aa[c]] for c in counts if aa[c] not in ("*", "M", "W")}


def cai(cds, w):
    logs = [np.log(w[c]) for c in (cds[i:i + 3] for i in range(0, len(cds) - len(cds) % 3, 3)) if c in w]
    return float(np.exp(np.mean(logs))) if logs else np.nan


def csc_mean(cds, csc):
    vals = [csc[c] for c in (cds[i:i + 3] for i in range(0, len(cds) - len(cds) % 3, 3)) if c in csc]
    return float(np.mean(vals)) if vals else np.nan


def uaug(u5):
    l5 = len(u5)
    frames = [(l5 - i) % 3 == 0 for i in range(l5 - 2) if u5[i:i + 3] == "ATG"]
    return sum(frames), len(frames) - sum(frames)


def overlapping(s, motif):
    return sum(1 for i in range(len(s) - len(motif) + 1) if s.startswith(motif, i))


def sites(u3, seed):
    """TargetScan 8mer / 7mer-m8 / 7mer-A1 sites of one seed (miRNA nt 2-8, RNA) in a 3'UTR, from nt 15 on."""
    comp = {"A": "T", "C": "G", "G": "C", "U": "A"}
    m8 = "".join(comp[b] for b in reversed(seed))  # pairs with nt 2-8
    a1 = m8[1:] + "A"  # pairs with nt 2-7, then an A opposite nt 1
    s = u3[15:]
    pos_m8 = {i for i in range(len(s) - 6) if s.startswith(m8, i)}
    pos_a1 = {i for i in range(len(s) - 6) if s.startswith(a1, i)}
    eight = sum(1 for i in pos_m8 if i + 1 in pos_a1)
    return len(pos_m8) + len(pos_a1) - eight


def gc(s):
    return (s.count("G") + s.count("C")) / len(s) if s else np.nan


def main():
    w = kazusa_weights()
    csc = {c: v for c, v in json.loads(CSC.read_text())["csc"].items() if c not in STOPS}
    OUT.mkdir(parents=True, exist_ok=True)
    for table, key in TABLES.items():
        if sys.argv[1:] and table not in sys.argv[1:]:
            continue
        df = pd.read_parquet(DATA / f"{key}.parquet", columns=["UTR5_sequence", "CDS_sequence", "UTR3_sequence"])
        u5, cds, u3 = ([s.upper().replace("U", "T") for s in df[c].fillna("")] for c in ("UTR5_sequence", "CDS_sequence", "UTR3_sequence"))
        rows = []
        for a, b, c in zip(u5, cds, u3):
            n_in, n_out = uaug(a)
            r = {"uaug_in": n_in, "uaug_out": n_out,
                 "uaug": "no CDS" if not b else "out-of-frame" if n_out else "in-frame" if n_in else "none",
                 "cai": cai(b, w) if b else np.nan, "csc": csc_mean(b, csc) if b else np.nan, "auuua": overlapping(c, "ATTTA"),
                 "l5": len(a), "lc": len(b), "l3": len(c), "gc5": gc(a), "gcc": gc(b), "gc3": gc(c)}
            r |= {f"mir_{f}": sites(c, seed) for f, seed in FAMILIES.items()}
            rows.append(r)
        out = pd.DataFrame(rows, index=df.index)
        out.to_parquet(OUT / f"{table}.parquet")
        print(f"{table}: {len(out)} rows; uAUG none / in-frame / out-of-frame {(out['uaug'] == 'none').sum()} / "
              f"{(out['uaug'] == 'in-frame').sum()} / {(out['uaug'] == 'out-of-frame').sum()}; CAI median {out['cai'].median():.3f} "
              f"(IQR {out['cai'].quantile(.25):.3f}–{out['cai'].quantile(.75):.3f}); AUUUA 0 / 1–2 / ≥3 "
              f"{(out['auuua'] == 0).sum()} / {out['auuua'].between(1, 2).sum()} / {(out['auuua'] >= 3).sum()}; "
              + ", ".join(f"{f} ≥1 site {(out[f'mir_{f}'] > 0).sum()}" for f in FAMILIES), flush=True)


if __name__ == "__main__":
    main()
