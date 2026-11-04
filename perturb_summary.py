"""perturb_summary.py — the numbers the paper quotes from the perturbation scores, over every model scored
(story_html.PMODELS): per uAUG table the slope and Spearman against the measured class effects, class means, controls,
removal and the 5' block's share; on HL human the codon recoding and 3'UTR insertion contrasts."""
import importlib.util
import json

import numpy as np
from paths import CODE, WORK

spec = importlib.util.spec_from_file_location("sh", CODE / "story_html.py")
sh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sh)
scores = json.loads((sh.PERT / "scores.json").read_text())
truths = json.loads(sh.TRUTHS.read_text())
M = sh.PMODELS
print(f"{len(M)} models:", ", ".join(lab for _, lab in M))


def rng(v):
    v = np.asarray(v, float)
    return f"{v.min():+.3f} .. {v.max():+.3f} (median {np.median(v):+.3f})"


for table, tname in sh.UTABLES:
    ref = np.array([truths[table][f"kind/kozak:{c}"]["delta_sd"] for c in sh.UCATS])
    rec = {f: [sh._uaug(scores, m, table, f, ref) for m, _ in M] for f in ("mbmean", "regions")}
    print(f"\n== {tname}: measured " + " ".join(f"{c} {v:+.2f}" for c, v in zip(sh.UCATS, ref)))
    for f in ("mbmean", "regions"):
        r = rec[f]
        print(f"  {f:8} slope {rng([x['slope'] for x in r])}")
        print(f"  {'':8} rho   {rng([x['rho'] for x in r])}")
        print(f"  {'':8} most negative class mean {rng([x['cats'].min() for x in r])}; largest |class mean| {rng([np.abs(x['cats']).max() for x in r])}")
        print(f"  {'':8} control {rng([x['ctrl'] for x in r])}; removal {rng([x['rem'] for x in r])}")
        for c in ("extension/strong", "overlap/strong", "uORF/strong", "extension/weak", "overlap/weak", "uORF/weak"):
            print(f"  {'':8} {c:17} {rng([x['cats'][sh.UCATS.index(c)] for x in r])}")
        if f == "regions":
            print(f"  {'':8} 5' share {rng([x['share'] for x in r])}")
    closer = [(lab, b["slope"], g["slope"]) for (m, lab), b, g in zip(M, rec["mbmean"], rec["regions"])]
    print(f"  regions slope nearer 1: {sum(abs(g - 1) < abs(b - 1) for _, b, g in closer)}/{len(M)}; "
          f"regions slope above mbmean: {sum(g > b for _, b, g in closer)}/{len(M)}")
    print("  not nearer:", [(lab, round(b, 3), round(g, 3)) for lab, b, g in closer if not abs(g - 1) < abs(b - 1)])

print("\n== HL human edits")
keys = {"opt": ("codon", "optimal", 1.0), "non": ("codon", "nonoptimal", 1.0), "rnd": ("codon", "random", 1.0),
        "are": ("ARE", "TTATTTATT", 4.0), "are0": ("ARE-ctrl", "shuffled", 4.0), "let7": ("let7", "8mer", 4.0),
        "let70": ("let7-ctrl", "shuffled-8mer", 4.0)}
for f in ("mbmean", "regions"):
    v = {k: [sh._pick(scores[m]["rnahl-human"][f]["summary"], *keys[k]) for m, _ in M] for k in keys}
    mean = {k: np.array([x["mean"] for x in v[k]]) for k in keys}
    print(f"  {f}")
    for k in ("opt", "non", "rnd"):
        print(f"    {k:5} {rng(mean[k])}  positive in {(mean[k] > 0).sum()}/{len(M)}")
    c = mean["opt"] - mean["non"]
    print(f"    opt-non {rng(c)}  > 0 in {(c > 0).sum()}/{len(M)}")
    for a, b in (("are", "are0"), ("let7", "let70")):
        d = mean[a] - mean[b]
        print(f"    {a}-{b} {rng(d)}  < 0 in {(d < 0).sum()}/{len(M)}; |.| max {np.abs(d).max():.3f}")
    if f == "regions":
        share = np.array([x["bc"] / x["mean"] for x in v["non"]])
        print(f"    CDS share of the non-optimal change {rng(share)}; under 0.5: {[lab for (m, lab), s in zip(M, share) if s < 0.5]}")

print("\n== HL human codon recoding by dose (optimal − least optimal, label SD)")
for d in (0.25, 0.5, 1.0):
    c = {}
    for f in ("mbmean", "regions"):
        o = np.array([sh._pick(scores[m]["rnahl-human"][f]["summary"], "codon", "optimal", d)["mean"] for m, _ in M])
        n = np.array([sh._pick(scores[m]["rnahl-human"][f]["summary"], "codon", "nonoptimal", d)["mean"] for m, _ in M])
        c[f] = o - n
    print(f"  dose {d}: global mean median {np.median(c['mbmean']):+.3f} (> 0 in {(c['mbmean'] > 0).sum()}/{len(M)}), "
          f"region median {np.median(c['regions']):+.3f} (> 0 in {(c['regions'] > 0).sum()}/{len(M)}), "
          f"ratio of medians {np.median(c['mbmean']) / np.median(c['regions']):.2f}; "
          f"wrong sign: global {[lab for (m, lab), x in zip(M, c['mbmean']) if x <= 0]}, region {[lab for (m, lab), x in zip(M, c['regions']) if x <= 0]}")

import pandas as pd  # noqa: E402

e = pd.read_parquet(WORK / "results/elements/rnahl-human.parquet")
tot = e[["l5", "lc", "l3"]].sum(1)
print(f"\n== HL human composition: median CDS share of nucleotides {(e['lc'] / tot).median():.2f}; "
      f"r(mean CSC, 3'UTR GC) {e[['csc', 'gc3']].dropna().corr().iloc[0, 1]:.2f}; r(CDS GC, 3'UTR GC) {e[['gcc', 'gc3']].dropna().corr().iloc[0, 1]:.2f}")
