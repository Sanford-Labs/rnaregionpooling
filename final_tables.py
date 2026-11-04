"""final_tables.py [OUT.txt] — every table of the regions paper from the fits under results/external.

Columns: mRNABench Table 2's ten (VEP, MRL MPRA, eCLIP, Loc-LR, Loc-SR, HL, MRL, MRL-HL-Pair, Prot-Loc, GO) + TE human,
TE mouse. mbmean and the naive rows are mRNABench's own, readout and probe as they ran them (key "A": ridge alphas
1e-3..10 on raw features, LOO; logistic C = 1). bases, split3, regions and every other readout are ours, probed with the
same estimators and the ridge alphas widened to 1e-6..1e6 (key "B"; classification stays at C = 1). A column's per-seed
value is the mean of its cells; tables print the ten-seed mean; paired lines print mean diff, seeds > 0 of 10, Wilcoxon p.
Loc-LR here is rna-lifecycle-ietswaart (V1.3), not the paper's rna-loc-ietswaart, so it is not compared to the paper.
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, wilcoxon
from paths import CODE, WORK

R = WORK / "results/external"
XML = WORK / "mrnabench-PMC12265608.xml"
REG = {"MRL MPRA": [("mrl-designed", "batch.y"), ("mrl-mcherry", "batch.y"), ("mrl-varying", "batch.y"),
                    ("mrl-egfp", "batch.unmod"), ("mrl-egfp", "batch.pseudo"), ("mrl-egfp", "batch.m1pseudo")],
       "HL": [("rnahl-human", "batch.y"), ("rnahl-mouse", "batch.y")], "MRL": [("sugimoto", "batch.y")],
       "MRL-HL-Pair": [("mrl-hl", "batch.mrl"), ("mrl-hl", "batch.hl")],
       "TE human": [("te-human", "batch.y")], "TE mouse": [("te-mouse", "batch.y")]}
CLS = {"VEP": ["vep-traitgym-complex", "vep-traitgym-mendelian"], "eCLIP": ["eclip-binding-hepg2", "eclip-binding-k562"],
       "Loc-LR": ["rna-lifecycle-ietswaart"], "Loc-SR": ["rna-loc-fazal"], "Prot-Loc": ["prot-loc"], "GO": ["go-bp", "go-cc", "go-mf"]}
COLS = ["VEP", "MRL MPRA", "eCLIP", "Loc-LR", "Loc-SR", "HL", "MRL", "MRL-HL-Pair", "Prot-Loc", "GO", "TE human", "TE mouse"]
REG6 = ["HL", "MRL", "MRL-HL-Pair", "TE human", "TE mouse"]
# label, results dir, the paper's name for the version, whether the model reads an annotation of the regions
MODELS = [("RiNALMo giga", "rinalmo-giga", "rinalmo", False), ("ERNIE-RNA", "ernierna", "ernierna", False),
          ("RNAErnie", "rnaernie", "rnaernie", False), ("RNA-MSM", "rnamsm", "rnamsm", False),
          ("UTR-LM te_el", "utrlm-te_el", "utrlm-te-el", False), ("RNA-FM", "rnafm", "rna-fm", False),
          ("SpliceBERT 1024", "splicebert", "splicebert-v-1024nt", False), ("Orthrus 6-track", "orthrus-large6", "orthrus-large-6", True),
          ("Orthrus 6-track (Nat. Methods release)", "orthrus-6track", "", True),
          ("Orthrus base-4", "orthrus-base4", "orthrus-base-4", False), ("Helix-mRNA", "helix-mrna", "helix-mrna", True), ("Helix-mRNA (no 'E')", "helix-mrna-noE", "", False), ("HelixNano-mRNA", "helixnano", "", False), ("RNABERT", "rnabert", "rnabert", False),
          ("3UTRBERT 6-mer", "utrbert-6mer", "utrbert-6mer", False), ("DNABERT2", "dnabert2", "dnabert2", False),
          ("DNABERT-S", "dnaberts", "dnabert-s", False), ("HyenaDNA medium-450k", "hyenadna-medium-450k", "hyenadna-medium-450k", False),
          ("NT 2.5B multi", "nt-2.5b-multi", "nt-2.5b-multi-species", False), ("AIDO.RNA 1.6B", "aido-rna-1.6b", "aido-rna-1b600m", False),
          ("Evo 2 7B base", "evo2-7b-base", "evo2-7b-base", False), ("Evo 1 8k base", "evo1-8k-base", "evo-1-8k-base", False)]
KEY = {"A": {"reg": "rawA", "cls": "A"}, "B": {"reg": "rawB", "cls": "A"}}  # B: mRNABench's probe, alpha grid widened


def paper_numbers():
    t = ET.parse(XML)
    out = {}
    names = {"T2": ["Loc-SR", "Loc-LR", "eCLIP", "MRL", "VEP"], "T3": ["HL", "MRL-HL-Pair", "MRL MPRA", "GO", "Prot-Loc"]}
    for tw in t.iter("table-wrap"):
        if tw.get("id") not in names:
            continue
        for tr in tw.iter("tr"):
            cells = [" ".join("".join(td.itertext()).split()) for td in tr if td.tag in ("td", "th")]
            if len(cells) != 6 or cells[0] == "Model":
                continue
            name = cells[0].replace("–", "-")
            for col, v in zip(names[tw.get("id")], cells[1:]):
                m = re.match(r"-?[0-9.]+", v)
                if m:
                    out.setdefault(name, {})[col] = float(m.group())
    return out


def load(d, table, cls=False):
    p = R / d / table / ("fit_cls.json" if cls else "fit.json")
    return json.loads(p.read_text())["results"] if p.exists() else None


def table_mean(col, vals):
    """A column's per-seed values: each table's targets averaged first, then the tables (mRNABench Table 2 counts
    mrl-egfp's three targets as one table)."""
    v = np.array(vals, float)
    if np.isnan(v).any():
        return None
    tables = [t for t, _ in REG[col]] if col in REG else CLS[col]
    groups = {}
    for t, x in zip(tables, v):
        groups.setdefault(t, []).append(x)
    return np.mean([np.mean(g, 0) for g in groups.values()], 0)


DCOLS = ["5'UTR", "CDS", "3'UTR", "cov", "r"]


def decomp_seeds(d, col, name="decomp.rawB.json"):
    """Per-seed shares of one regression column from results/external/<d>/<table>/<name>, or None when one is missing."""
    vals = []
    for table, target in REG[col]:
        p = R / d / table / name
        if not p.exists():
            return None
        r = json.loads(p.read_text())["results"].get(target)
        if r is None:
            return None
        sh = np.array(r["var_share"], float)
        vals.append(np.stack([sh[:, 0], sh[:, 1], sh[:, 2], np.array(r["cov_share"], float), np.array(r["test"], float)]))
    v = table_mean(col, vals)
    return None if v is None else dict(zip(DCOLS, v))


def seeds(d, col, fset, proto):
    """Per-seed test values of one column, or None when a cell is missing."""
    vals = []
    if col in REG:
        for table, target in REG[col]:
            res = load(d, table)
            key = fset if "|" in fset else f"{fset}|{KEY[proto]['reg']}"
            if res is None or target not in res or key not in res[target]:
                return None
            vals.append(res[target][key]["test"])
    else:
        for table in CLS[col]:
            res = load(d, table, cls=True)
            key = f"{fset}|{KEY[proto]['cls']}"
            if res is None or key not in res:
                return None
            vals.append(res[key]["test"])
    return table_mean(col, vals)


NAIVE = WORK / "results/naive"
TGT = {"batch.mrl": "target_ribosome_load", "batch.hl": "target_in_cell_half_life", "batch.unmod": "target_mrl_egfp_unmod",
       "batch.pseudo": "target_mrl_egfp_pseudo", "batch.m1pseudo": "target_mrl_egfp_m1pseudo"}


def naive_seeds(col, six, grid):
    """Per-seed test values of one column for a naive row (results/naive), or None when a table is missing.
    Classification columns carry mRNABench's C = 1 fit only, the same under either key."""
    sfx = ".six" if six else ""
    vals = []
    if col in REG:
        for table, target in REG[col]:
            p = NAIVE / f"{table}{sfx}.json"
            if not p.exists():
                return None
            res = json.loads(p.read_text())["results"]
            key = TGT.get(target) or (next(iter(res)) if len(res) == 1 else None)
            if key not in res:
                return None
            vals.append(res[key][grid]["test"])
    else:
        for table in CLS[col]:
            p = NAIVE / f"{table}{sfx}.cls.json"
            if not p.exists():
                return None
            vals.append(json.loads(p.read_text())["results"]["test"])
    return table_mean(col, vals)


def table(title, rows, cols, fmt="{:.4f}", bold=True):
    """rows: [(label, {col: per-seed array or None})]; bold marks each column's best mean."""
    print(f"\n{title}")
    print("| model | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    best = {c: max((np.mean(v[c]) for _, v in rows if v.get(c) is not None), default=None) for c in cols}
    for label, v in rows:
        cells = []
        for c in cols:
            x = v.get(c)
            if x is None:
                cells.append("–")
                continue
            s = fmt.format(np.mean(x))
            cells.append(f"**{s}**" if bold and best[c] is not None and np.isclose(np.mean(x), best[c]) else s)
        print(f"| {label} | " + " | ".join(cells) + " |")


def paired(title, rows, cols):
    print(f"\n{title}  (mean diff; seeds > 0 of 10; Wilcoxon p)")
    print("| model | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for label, a, b in rows:
        cells = []
        for c in cols:
            if a.get(c) is None or b.get(c) is None:
                cells.append("–")
                continue
            d = a[c] - b[c]
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            cells.append(f"{d.mean():+.4f} ({int((d > 0).sum())}/10, p {p:.3f})")
        print(f"| {label} | " + " | ".join(cells) + " |")


def main():
    if len(sys.argv) > 1:
        sys.stdout = open(sys.argv[1], "w")
    paper = paper_numbers()
    # Helix-mRNA (no 'E') is Helix-mRNA's weights on input without the codon marks: section 11 only, not a model of its own;
    # Orthrus 6-track is the version mRNABench evaluates (orthrus-large6), not the Nat. Methods release
    have = [(lab, d, pn, ann) for lab, d, pn, ann in MODELS if (R / d).exists() and d not in ("helix-mrna-noE", "orthrus-6track")]
    grab = lambda d, fset, proto: {c: seeds(d, c, fset, proto) for c in COLS}  # noqa: E731
    print("Regions-paper tables, built by papers/final_tables.py from results/external (-1 layer).")
    print("mbmean and naive are mRNABench's as published (readout and probe unchanged: ridge alpha 1e-3..10, classification "
          "C = 1); bases, split3, regions and the other readouts are ours, with the same estimators and the ridge alpha grid "
          "widened to 1e-6..1e6.")
    # calibration
    rows = []
    for lab, d, pn, _ in have:
        ours = grab(d, "mbmean", "A")
        rows.append((lab + " · ours", ours))
        if pn in paper:
            rows.append((lab + " · paper", {c: (np.full(10, paper[pn][c]) if c in paper[pn] and c != "Loc-LR" else None) for c in COLS}))
    table("0. Calibration: mbmean (mRNABench as published) against the same version in its paper's Tables 2/3 (Loc-LR labels differ, not compared)", rows, COLS, bold=False)
    # main scores, one table per readout (same width inside a table)
    for fset, name in (("bases", "bases (nucleotide mean)"), ("split3", "split3 (equal thirds, concatenated)"), ("regions", "regions (5'UTR/CDS/3'UTR, concatenated)")):
        table(f"1. {name}", [(lab, grab(d, fset, "B")) for lab, d, _, _ in have], COLS)
    paired("2. regions − mbmean", [(lab, grab(d, "regions", "B"), grab(d, "mbmean", "A")) for lab, d, _, _ in have], COLS)
    paired("2. regions − split3", [(lab, grab(d, "regions", "B"), grab(d, "split3", "B")) for lab, d, _, _ in have], COLS)
    # ranking
    print("\n3. Ranking: models ordered by mbmean and by regions in each column, Kendall tau; in parentheses the 95% interval of a bootstrap over seeds")
    print("| column | models | tau(mbmean, regions) |")
    print("|---|---|---|")
    rng = np.random.default_rng(0)
    for c in COLS:
        r = {lab: seeds(d, c, "regions", "B") for lab, d, _, _ in have}
        m = {lab: seeds(d, c, "mbmean", "A") for lab, d, _, _ in have}
        labs = [l for l in r if r[l] is not None and m[l] is not None]
        if len(labs) < 4:
            print(f"| {c} | {len(labs)} | – |")
            continue
        def tau(x, y, idx):
            return kendalltau([x[l][idx].mean() for l in labs], [y[l][idx].mean() for l in labs]).statistic
        full = np.arange(10)
        boots = [tau(m, r, i) for i in (rng.integers(0, 10, 10) for _ in range(1000))]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"| {c} | {len(labs)} | {tau(m, r, full):.2f} ({lo:.2f}, {hi:.2f}) |")
    print("\n3. Rank changes: rank under mbmean minus rank under regions per column (positive = places moved up); up / down / net summed over the 12 columns; the last row sums the absolute changes per column")
    print("| model | " + " | ".join(COLS) + " | up | down | net |")
    print("|---|" + "---|" * (len(COLS) + 3))
    move = {lab: {} for lab, _, _, _ in have}
    for c in COLS:
        r = {lab: seeds(d, c, "regions", "B") for lab, d, _, _ in have}
        m = {lab: seeds(d, c, "mbmean", "A") for lab, d, _, _ in have}
        labs = [l for l in r if r[l] is not None and m[l] is not None]
        rm = {l: i + 1 for i, l in enumerate(sorted(labs, key=lambda l: -m[l].mean()))}
        rr = {l: i + 1 for i, l in enumerate(sorted(labs, key=lambda l: -r[l].mean()))}
        for l in labs:
            move[l][c] = rm[l] - rr[l]
    for lab, mv in sorted(move.items(), key=lambda kv: -sum(kv[1].values())):
        up, down = sum(v for v in mv.values() if v > 0), -sum(v for v in mv.values() if v < 0)
        cells = [("–" if c not in mv else f"{mv[c]:+d}" if mv[c] else "0") for c in COLS]
        print(f"| {lab} | " + " | ".join(cells) + f" | {up} | {down} | {up - down:+d} |")
    tot = {c: sum(abs(mv[c]) for mv in move.values() if c in mv) for c in COLS}
    print("| total \\|Δ\\| | " + " | ".join(str(tot[c]) for c in COLS) + f" | {sum(tot.values())} | | |")
    # same width
    paired("4. Same width: PCA-d(regions) − PCA-d(split3)",
           [(lab, grab(d, "pca_regions|rawB", "B"), grab(d, "pca_split3|rawB", "B")) for lab, d, _, _ in have], [c for c in COLS if c in REG])
    paired("4. Same width: rbal − bases", [(lab, grab(d, "rbal", "B"), grab(d, "bases", "B")) for lab, d, _, _ in have], COLS)
    # single regions
    for fset, name in (("r5", "the 5'UTR block alone"), ("rc", "the CDS block alone"), ("r3", "the 3'UTR block alone")):
        table(f"5. {name}", [(lab, grab(d, fset, "B")) for lab, d, _, _ in have], COLS)
    # length control
    table("6. F6 (log lengths and GC fractions of the three regions)", [(lab, grab(d, "F6", "B")) for lab, d, _, _ in have[:1]], COLS)
    paired("6. (regions+F6) − (bases+F6)", [(lab, grab(d, "regions+F6", "B"), grab(d, "bases+F6", "B")) for lab, d, _, _ in have], COLS)
    # boundaries
    print("\n7. Boundaries: shifted by delta and random, minus the annotated boundaries (mean of HL, MRL, Pair, TE human, TE mouse)")
    shifts = (-300, -100, -30, -10, 10, 30, 100, 300)
    print("| model | " + " | ".join(f"δ {s:+d}" for s in shifts) + " | random | split3 |")
    print("|---|" + "---|" * (len(shifts) + 2))
    for lab, d, _, _ in have:
        base = grab(d, "regions", "B")
        if any(base[c] is None for c in REG6):
            continue
        cells = []
        for key in [f"shift{s:+d}|rawB" for s in shifts] + ["random", "split3|rawB"]:
            if key == "random":
                vs = [grab(d, f"random{j}|rawB", "B") for j in range(5)]
                if any(v[c] is None for v in vs for c in REG6):
                    cells.append("–"); continue
                diff = np.mean([np.mean([v[c].mean() for v in vs]) - base[c].mean() for c in REG6])
            else:
                v = grab(d, key, "B")
                if any(v[c] is None for c in REG6):
                    cells.append("–"); continue
                diff = np.mean([v[c].mean() - base[c].mean() for c in REG6])
            cells.append(f"{diff:+.4f}")
        print(f"| {lab} | " + " | ".join(cells) + " |")
    # isolated encoding
    rows = []
    for lab, d, _, _ in have:
        iso = d + "-iso"
        if (R / iso).exists():
            rows.append((lab, grab(d, "regions", "B"), grab(iso, "regions", "B")))
    paired("8. regions from the whole transcript − regions from each region encoded on its own: what the whole-transcript context adds", rows, REG6)
    # middle layer
    rows = []
    for lab, d, _, _ in have:
        if (R / (d + "-mid")).exists():
            rows.append((f"{lab} middle layer", grab(d + "-mid", "regions", "B"), grab(d + "-mid", "mbmean", "A")))
            rows.append((f"{lab} last layer", grab(d, "regions", "B"), grab(d, "mbmean", "A")))
    paired("8b. regions − mbmean at the middle layer, against the same model's last layer", rows, REG6)
    # annotation
    print("\n9. With or without region annotation in the input: regions − split3, column means averaged by group")
    groups = {True: [], False: []}
    for lab, d, _, ann in have:
        r_, s_ = grab(d, "regions", "B"), grab(d, "split3", "B")
        vals = [r_[c].mean() - s_[c].mean() for c in ("MRL", "TE human", "TE mouse") if r_[c] is not None and s_[c] is not None]
        if vals:
            groups[ann].append((lab, float(np.mean(vals))))
    for ann, items in groups.items():
        print(f"- {'with annotation' if ann else 'without annotation'} ({len(items)} models): mean of the MRL, TE human and TE mouse columns "
              + ", ".join(f"{l} {v:+.4f}" for l, v in items))
    # chunking rule
    if (R / "rinalmo-giga-w1022").exists():
        rows = [(f"RiNALMo giga {fs}", grab("rinalmo-giga-w1022", fs, pr), grab("rinalmo-giga", fs, pr))
                for fs, pr in (("mbmean", "A"), ("bases", "B"), ("regions", "B"))]
        paired("10. Chunking: RiNALMo giga in 1,022-token windows (stride 511) − mRNABench's 8,190-token chunks", rows, REG6)
    # codon marks
    if (R / "helix-mrna-noE").exists():
        rows = [(f"{fs}: with 'E' − without 'E'", grab("helix-mrna", fs, "B"), grab("helix-mrna-noE", fs, "B")) for fs in ("bases", "regions")]
        rows += [(f"{'with' if d == 'helix-mrna' else 'without'} 'E': regions − split3", grab(d, "regions", "B"), grab(d, "split3", "B"))
                 for d in ("helix-mrna", "helix-mrna-noE")]
        paired("11. Helix-mRNA's codon mark 'E' (same weights)", rows, COLS)
    # naive rows
    rows = []
    for six in (False, True):
        pn = f"naive-{6 if six else 4}"
        rows.append((pn + " · ours", {c: naive_seeds(c, six, "A") for c in COLS}))
        if pn in paper:
            rows.append((pn + " · paper", {c: (np.full(10, paper[pn][c]) if c in paper[pn] and c != "Loc-LR" else None) for c in COLS}))
    table("12. Naive baselines (mRNABench's 3-7-mers and GC, as published) against its paper", rows, COLS, bold=False)
    # region decomposition (harness/decomp_ext.py)
    for col in REG6:
        rows = [(lab, v) for lab, d, _, _ in have if (v := decomp_seeds(d, col)) is not None]
        table(f"13. Region decomposition, {col}: the variance of each block's contribution x_k·w_k over the test rows as a share of "
              "the prediction's variance; cov = the share of the covariance between blocks (the four sum to 1); r = the test r of regions", rows, DCOLS, fmt="{:.3f}", bold=False)
    rows = [(lab, v) for lab, d, _, _ in have if (v := decomp_seeds(d, "MRL", "decomp.complete.rawB.json")) is not None]
    table("13. Region decomposition, MRL, the rows with all three regions only (8,787 of the 11,693 sugimoto rows, used for fit and decomposition)", rows, DCOLS, fmt="{:.3f}", bold=False)
    # main-text summary across models
    print(f"\n14. Main table: mean over {len(have)} models per column (regression columns test Pearson r, classification AUPRC); "
          "mbmean is mRNABench as published, the rest ours; better / worse / tied = models with regions − mbmean > 0 / < 0 at "
          "Wilcoxon p < 0.05 over the ten seeds / the rest (not significant)")
    print("| column | mbmean | bases | split3 | regions | regions − mbmean | better / worse / tied |")
    print("|---|---|---|---|---|---|---|")
    for c in COLS:
        got = {k: [seeds(d, c, fset, proto) for _, d, _, _ in have] for k, fset, proto in
               (("mb", "mbmean", "A"), ("b", "bases", "B"), ("s3", "split3", "B"), ("r", "regions", "B"))}
        keep = [i for i in range(len(have)) if all(got[k][i] is not None for k in got)]
        mean = {k: np.mean([got[k][i].mean() for i in keep]) for k in got}
        cells = [f"{mean[k]:.4f}" for k in ("mb", "b", "s3", "r")]
        for ref in ("mb",):
            ds = [got["r"][i] - got[ref][i] for i in keep]
            ps = [wilcoxon(d).pvalue if np.any(d != 0) else 1.0 for d in ds]
            win = sum(d.mean() > 0 and pv < 0.05 for d, pv in zip(ds, ps))
            loss = sum(d.mean() < 0 and pv < 0.05 for d, pv in zip(ds, ps))
            cells += [f"{np.mean([d.mean() for d in ds]):+.4f}", f"{win} / {loss} / {len(ds) - win - loss}"]
        print(f"| {c} ({len(keep)}) | " + " | ".join(cells) + " |")
    # controls summary across models (regression columns)
    def mean_random(d):
        vs = [grab(d, f"random{j}|rawB", "B") for j in range(5)]
        return {c: None if any(v[c] is None for v in vs) else np.mean([v[c] for v in vs], 0) for c in REG6}

    controls = [("regions − split3", lambda d: (grab(d, "regions", "B"), grab(d, "split3", "B"))),
                ("rbal − bases (same width)", lambda d: (grab(d, "rbal", "B"), grab(d, "bases", "B"))),
                ("PCA-d(regions) − PCA-d(split3) (same width)", lambda d: (grab(d, "pca_regions|rawB", "B"), grab(d, "pca_split3|rawB", "B"))),
                ("(regions+F6) − (bases+F6) (length and GC)", lambda d: (grab(d, "regions+F6", "B"), grab(d, "bases+F6", "B"))),
                ("random boundaries − annotated boundaries (mean of 5 draws)", lambda d: (mean_random(d), grab(d, "regions", "B"))),
                ("boundaries shifted −100 nt − annotated", lambda d: (grab(d, "shift-100|rawB", "B"), grab(d, "regions", "B"))),
                ("boundaries shifted +100 nt − annotated", lambda d: (grab(d, "shift+100|rawB", "B"), grab(d, "regions", "B"))),
                (f"whole-transcript − per-region encoding (regions, {sum((R / (d + '-iso')).exists() for _, d, _, _ in have)} models)", lambda d: (grab(d, "regions", "B"), grab(d + "-iso", "regions", "B")) if (R / (d + "-iso")).exists() else None)]
    print("\n15. Controls (regression columns): mean difference over models [better / worse / tied], better / worse / tied = "
          "difference > 0 / < 0 at Wilcoxon p < 0.05 over the ten seeds / the rest (not significant)")
    print("| control | " + " | ".join(REG6) + " |")
    print("|---|" + "---|" * len(REG6))
    for name, pick in controls:
        cells = []
        pairs = [pick(d) for _, d, _, _ in have]
        for c in REG6:
            ds = [a_[c] - b_[c] for pr in pairs if pr is not None for a_, b_ in [pr] if a_.get(c) is not None and b_.get(c) is not None]
            if not ds:
                cells.append("–"); continue
            ps = [wilcoxon(x).pvalue if np.any(x != 0) else 1.0 for x in ds]
            win = sum(x.mean() > 0 and pv < 0.05 for x, pv in zip(ds, ps))
            loss = sum(x.mean() < 0 and pv < 0.05 for x, pv in zip(ds, ps))
            cells.append(f"{np.mean([x.mean() for x in ds]):+.4f} [{win}/{loss}/{len(ds) - win - loss}]")
        print(f"| {name} | " + " | ".join(cells) + " |")
    # learned pooling (harness/attn_pool.py)
    ap = R / "rinalmo-giga-tokens" / "te-human" / "attn_pool.json"
    if ap.exists():
        res = json.loads(ap.read_text())
        print("\n16. Learned pooling: RiNALMo giga, TE human, last-layer tokens, test r over the ten seeds (attention pooling stopped early on val)")
        print("| readout | test r | sd |")
        print("|---|---|---|")
        rows = [("attention pooling, no annotation", res["results"]["none"]["test"]),
                ("attention pooling, region-type embedding on the keys", res["results"]["regemb"]["test"]),
                ("attention pooling, one query per region (three pools concatenated)", res["results"]["perregion"]["test"]),
                *((f"ridge {k.split('|')[0]}", load("rinalmo-giga", "te-human")["batch.y"][k]["test"])
                  for k in ("bases|rawB", "split3|rawB", "regions|rawB"))]
        for name, v in rows:
            print(f"| {name} | {np.mean(v):.4f} | {np.std(v):.4f} |")


if __name__ == "__main__":
    main()
