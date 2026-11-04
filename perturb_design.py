"""perturb_design.py — the CSC table and the perturbed sequence sets (length-preserving substitutions).

Rows: the default test split (parquet `split == "test"`) of each table, sampled with seed 0.
  te-human, sugimoto, mrl-egfp: uAUG insertion (extension / overlap / uORF x Kozak strong / weak, control =
      the same edits with a non-AUG triplet) and, where a row has exactly one uAUG, its removal (AUG -> AUC).
  rnahl-human: synonymous recoding of 25/50/100 % of codons to the max-CSC / min-CSC / a random synonym;
      let-7 sites (8mer, 7mer-m8, 7mer-A1, 6mer) x 1/2/4 copies, the 8mer's shuffle as control;
      ARE nonamer TTATTTATT x 1/2/4, its shuffle as control. 3'UTR edits start >= 15 nt after the stop codon.
CSC (Presnyak 2015): Pearson r across rnahl-human train genes between a codon's frequency in the CDS and the
half-life target. Writes results/perturb/csc.json and results/perturb/<table>.parquet.
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from paths import CODE, DATA, WORK

DATA = DATA
OUT = WORK / "results/perturb"
BASES = "ACGT"
CODE = {}
for i, a in enumerate("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"):
    CODE["TCAG"[i // 16] + "TCAG"[i // 4 % 4] + "TCAG"[i % 4]] = a
SYN = {}
for c, a in CODE.items():
    SYN.setdefault(a, []).append(c)
STOPS = {"TAA", "TAG", "TGA"}
LET7 = {"8mer": "CTACCTCA", "7mer-m8": "CTACCTC", "7mer-A1": "TACCTCA", "6mer": "TACCTC"}
ARE = "TTATTTATT"


def codons(cds):
    return [cds[i:i + 3] for i in range(0, len(cds) - len(cds) % 3, 3)]


def csc_table():
    df = pd.read_parquet(DATA / "rnahl-human.parquet", columns=["CDS_sequence", "target", "split"])
    tr = df[(df["split"] == "train") & df["CDS_sequence"].fillna("").str.len().gt(0)]
    sense = [c for c, a in CODE.items() if a != "*"]
    freq = np.zeros((len(tr), len(sense)))
    idx = {c: j for j, c in enumerate(sense)}
    for i, cds in enumerate(tr["CDS_sequence"]):
        cs = [c for c in codons(cds) if c in idx]
        for c in cs:
            freq[i, idx[c]] += 1
        freq[i] /= max(len(cs), 1)
    y = tr["target"].to_numpy(float)
    csc = {c: float(np.corrcoef(freq[:, j], y)[0, 1]) for c, j in idx.items()}
    best = {a: max(cs, key=csc.get) for a, cs in SYN.items() if a != "*" and len(cs) > 1}
    worst = {a: min(cs, key=csc.get) for a, cs in SYN.items() if a != "*" and len(cs) > 1}
    return csc, best, worst, len(tr)


def kozak_edit(s, p, strength):
    s = list(s)
    if p >= 3:
        s[p - 3] = "A" if strength == "strong" else "T"
    if p + 3 < len(s):
        s[p + 3] = "G" if strength == "strong" else "C"
    return s


def uorf_kind(full, p, main):
    frame_in = (main - p) % 3 == 0
    closed = any(full[q:q + 3] in STOPS for q in range(p + 3, main - 2, 3))
    return "uORF" if closed else ("extension" if frame_in else "overlap")


def uaug_variants(u5, cds, rng, first=5):
    """Insert one uAUG per (kind, kozak) where a position allows it; control = same edits, non-AUG triplet."""
    out = []
    main = len(u5)
    if "ATG" in u5 or main < 20:
        return out
    want = {(k, z) for k in ("extension", "overlap", "uORF") for z in ("strong", "weak")}
    positions = list(range(first, main - 6))
    rng.shuffle(positions)
    for p in positions:
        for z in ("strong", "weak"):
            s = kozak_edit(u5, p, z)
            s[p:p + 3] = "ATG"
            s = "".join(s)
            if s.count("ATG") != 1:
                continue
            k = uorf_kind(s + cds[:3], p, main)
            if (k, z) not in want:
                continue
            want.discard((k, z))
            trip = rng.choice([t for t in ("ACG", "ATC", "AAG", "CTG", "GTG", "TTG") if t != "ATG"])
            c = kozak_edit(u5, p, z)
            c[p:p + 3] = trip
            c = "".join(c)
            if "ATG" in c:
                continue
            out.append(("uAUG", f"{k}/{z}", 1, s, p))
            out.append(("uAUG-ctrl", f"{k}/{z}", 1, c, p))
        if not want:
            break
    return out


def uaug_removal(u5):
    pos = [p for p in range(len(u5) - 2) if u5[p:p + 3] == "ATG"]
    if len(pos) != 1:
        return []
    p = pos[0]
    return [("uAUG-remove", "AUG->AUC", 1, u5[:p] + "ATC" + u5[p + 3:], p)]


def recode(cds, frac, table, rng):
    cs = codons(cds)
    inner = [i for i in range(1, len(cs) - 1) if CODE.get(cs[i], "*") != "*" and len(SYN[CODE[cs[i]]]) > 1]
    pick = set(rng.sample(inner, int(round(frac * len(inner))))) if inner else set()
    new = []
    for i, c in enumerate(cs):
        if i in pick:
            a = CODE[c]
            new.append(table[a] if isinstance(table, dict) else rng.choice(SYN[a]))
        else:
            new.append(c)
    return "".join(new) + cds[len(cs) * 3:]


def place(u3, site, copies, rng, start=15):
    """Substitute `copies` non-overlapping copies of `site` into the 3'UTR; None if it does not fit."""
    span = len(site)
    free = list(range(start, len(u3) - span))
    if len(free) < copies * span * 3:
        return None, []
    s = list(u3)
    taken = []
    rng.shuffle(free)
    for p in free:
        if all(abs(p - q) >= span + 5 for q in taken):
            taken.append(p)
            s[p:p + span] = site
        if len(taken) == copies:
            break
    return ("".join(s), sorted(taken)) if len(taken) == copies else (None, [])


def shuffled(site, rng):
    while True:
        t = "".join(rng.sample(site, len(site)))
        if t != site:
            return t


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    csc, best, worst, ngenes = csc_table()
    (OUT / "csc.json").write_text(json.dumps({"genes": ngenes, "csc": csc, "optimal": best, "nonoptimal": worst}, indent=1))
    print(f"CSC on {ngenes} train genes; strongest +: {sorted(csc, key=csc.get)[-5:]}, strongest -: {sorted(csc, key=csc.get)[:5]}")
    plans = {"te-human": ("translation-efficiency-human", 300, "uaug"), "sugimoto": ("mrl-sugimoto", 300, "uaug"),
             "mrl-egfp": ("mrl-sample-egfp", 500, "uaug"), "rnahl-human": ("rnahl-human", 300, "hl")}
    for table, (key, n, kind) in plans.items():
        df = pd.read_parquet(DATA / f"{key}.parquet")
        test = df.index[df["split"] == "test"].to_numpy()
        rng = random.Random(0)
        u5s, cdss, u3s = (df[c].fillna("") for c in ("UTR5_sequence", "CDS_sequence", "UTR3_sequence"))
        rows = []
        pool = list(test)
        rng.shuffle(pool)
        used = 0
        for r in pool:
            u5, cds, u3 = u5s[r], cdss[r], u3s[r]
            if not cds:
                continue
            made = []
            if kind == "uaug":
                first = 25 if table == "mrl-egfp" else 5  # the egfp library varies positions 25-74 only
                made = [(k, v, d, s, p, cds, u3) for k, v, d, s, p in uaug_variants(u5, cds, rng, first) + uaug_removal(u5)]
            else:
                for frac in (0.25, 0.5, 1.0):
                    for name, tab in (("optimal", best), ("nonoptimal", worst), ("random", None)):
                        made.append(("codon", name, frac, u5, -1, recode(cds, frac, tab, rng), u3))
                for st, site in LET7.items():
                    for copies in (1, 2, 4):
                        s, pos = place(u3, site, copies, rng)
                        if s is not None:
                            made.append(("let7", st, copies, u5, -1, cds, s))
                            if st == "8mer":
                                made.append(("let7-ctrl", "shuffled-8mer", copies, u5, -1, cds,
                                             _sub(u3, pos, shuffled(site, rng))))
                for copies in (1, 2, 4):
                    s, pos = place(u3, ARE, copies, rng)
                    if s is not None:
                        made.append(("ARE", "TTATTTATT", copies, u5, -1, cds, s))
                        made.append(("ARE-ctrl", "shuffled", copies, u5, -1, cds, _sub(u3, pos, shuffled(ARE, rng))))
            if not made:
                continue
            rows.append((r, "original", "", 0, u5, -1, cds, u3))
            rows += [(r, k, v, d, a, p, c, b) for k, v, d, a, p, c, b in made]
            used += 1
            if used == n:
                break
        out = pd.DataFrame(rows, columns=["row", "kind", "variant", "dose", "UTR5_sequence", "edit_pos", "CDS_sequence", "UTR3_sequence"])
        out.to_parquet(OUT / f"{table}.parquet")
        nt = int((out["UTR5_sequence"].str.len() + out["CDS_sequence"].str.len() + out["UTR3_sequence"].str.len()).sum())
        print(f"{table}: {used} rows, {len(out)} sequences, {nt / 1e6:.1f}M nt; kinds {out.groupby(['kind']).size().to_dict()}")


def _sub(u3, pos, site):
    s = list(u3)
    for p in pos:
        s[p:p + len(site)] = site
    return "".join(s)


if __name__ == "__main__":
    main()
