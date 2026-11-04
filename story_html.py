"""story_html.py — the paper's results as one English page, WORK/regions-paper-story.html, with its tables computed from
the fits through final_tables.py. The builders here also give the paper's figures and tables (draft_render.py,
extra_figs.py).

mbmean is mRNABench's readout and probe as they ran them (key "A"); bases, split3, regions and the controls are ours
(key "B"). Per model: a win / loss when the ten-seed paired difference is positive / negative with Wilcoxon p < 0.05, a
tie otherwise.
"""
import html
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr, wilcoxon
from paths import CODE, WORK

spec = importlib.util.spec_from_file_location("ft", CODE / "final_tables.py")
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)
OUT = WORK / "regions-paper-story.html"
# mRNABench Table 2 (PMC12265608, /table/T9/): its column order, column groups, names and metric row, its rows in
# alphabetical order of family with the baselines on top; TE human / TE mouse (not in it) close the Global (RNA) group.
COLS = ["VEP", "MRL MPRA", "eCLIP", "Loc-LR", "Loc-SR", "HL", "MRL", "MRL-HL-Pair", "TE human", "TE mouse", "Prot-Loc", "GO"]
GROUPS = [("Local", 3), ("Global (RNA)", 7), ("Global (protein)", 2)]
HEAD = {"Loc-LR": "mRNA Loc-LR", "Loc-SR": "mRNA Loc-SR", "Prot-Loc": "Prot Loc"}
PAPER_COL = {"MRL-HL-Pair": "MRL–HL pair"}  # the paper's names, those of its column table
METRIC = {c: "Pearson r" if c in ft.REG else "AUPRC" for c in COLS}
PAPER_ROWS = ["3UTRBERT 6-mer", "AIDO.RNA 1.6B", "DNABERT-S", "DNABERT2", "ERNIE-RNA", "Evo 1 8k base", "Evo 2 7B base",
              "Helix-mRNA", "HelixNano-mRNA", "HyenaDNA medium-450k", "NT 2.5B multi", "Orthrus 6-track", "Orthrus base-4",
              "RNA-FM", "RNA-MSM", "RNABERT", "RNAErnie", "RiNALMo giga", "SpliceBERT 1024", "UTR-LM te_el"]
HAVE = [(lab, d) for lab, d, _, _ in ft.MODELS if (ft.R / d).exists() and d not in ("helix-mrna-noE", "orthrus-6track")]
ORDERED = sorted(HAVE, key=lambda x: PAPER_ROWS.index(x[0]))
PAPER = False  # draft_render.py sets it: readouts, models and labels under the paper's names
RN = {"mbmean": "global mean", "regions": "region-aware", "split3": "equal thirds", "bases": "nucleotide mean", "truth": "measured"}
MN = {"3UTRBERT 6-mer": "3UTRBERT", "DNABERT2": "DNABERT-2", "Evo 1 8k base": "Evo 1", "Evo 2 7B base": "Evo 2 7B",
      "HyenaDNA medium-450k": "HyenaDNA", "NT 2.5B multi": "NT 2.5B", "SpliceBERT 1024": "SpliceBERT",
      "UTR-LM te_el": "UTR-LM",
      # the names the Orthrus release gives the two checkpoints
      "Orthrus 6-track": "Orthrus large 6-track", "Orthrus base-4": "Orthrus base 4-track"}


def P(story, paper):
    """A label as the story page writes it, or as the paper does."""
    return paper if PAPER else story


def rn(key, cap=False):
    """A readout's name."""
    if not PAPER:
        return key
    name = RN.get(key, key)
    return name[0].upper() + name[1:] if cap else name


def mn(lab):
    """A model's name."""
    return MN.get(lab, lab) if PAPER else lab


def signed(x, nd=4):
    return f"{x:+.{nd}f}".replace("-", "−")


def seed_p(d):
    return wilcoxon(d).pvalue if np.any(d != 0) else 1.0


def tally(pairs):
    """pairs: [(a, b)] per-seed arrays per model -> (mean diff, wins, losses, ties, n)."""
    ds = [a - b for a, b in pairs]
    means = np.array([x.mean() for x in ds])
    ps = [seed_p(x) for x in ds]
    win = sum(m > 0 and p < 0.05 for m, p in zip(means, ps))
    loss = sum(m < 0 and p < 0.05 for m, p in zip(means, ps))
    return means.mean(), win, loss, len(ds) - win - loss, len(ds)


def wide_head(first, extra=()):
    """Table 2's three header rows: column group, column, metric."""
    groups = "".join(f'<th colspan="{n}" class="grp">{g}</th>' for g, n in GROUPS)
    return (f'<thead><tr><th rowspan="3">{first}</th>{groups}' + "".join(f'<th rowspan="3">{e}</th>' for e in extra) + "</tr>"
            "<tr>" + "".join(f"<th>{HEAD.get(c, c)}</th>" for c in COLS) + "</tr>"
            "<tr>" + "".join(f'<th class="metric">{METRIC[c]}</th>' for c in COLS) + "</tr></thead>")


def main_table():
    lead = ["Column", "Metric", rn("mbmean", True), rn("bases", True), rn("split3", True), rn("regions", True)]
    pair = [P("regions − mbmean", "difference"), P("wins / losses / ties", "better / worse / tied")]
    out = ["<table><thead><tr>" + "".join(f'<th rowspan="2">{h}</th>' for h in lead)
           + f'<th colspan="2">{P("regions vs mbmean", "Region-aware vs global mean")}</th></tr><tr>'
           + "".join(f"<th>{h}</th>" for h in pair) + "</tr></thead><tbody>"]
    at = 0
    for g, n in GROUPS:
        out.append(f'<tr class="group"><td colspan="8">{g}</td></tr>')
        for c in COLS[at:at + n]:
            got = {k: [ft.seeds(d, c, fs, pr) for _, d in HAVE] for k, fs, pr in
                   (("mb", "mbmean", "A"), ("b", "bases", "B"), ("s3", "split3", "B"), ("r", "regions", "B"))}
            mean = {k: np.mean([x.mean() for x in v]) for k, v in got.items()}
            diff, win, loss, tie, _ = tally(list(zip(got["r"], got["mb"])))
            cls = "up" if win > loss else "down" if loss > win else ""
            out.append(f'<tr><td>{P(HEAD.get(c, c), PAPER_COL.get(c, c))}</td><td class="ns">{METRIC[c]}</td>'
                       + "".join(f"<td>{mean[k]:.4f}</td>" for k in ("mb", "b", "s3", "r"))
                       + f'<td class="{cls}">{signed(diff)}</td><td>{win} / {loss} / {tie}</td></tr>')
        at += n
    out.append("</tbody></table>")
    return "\n".join(out)


def per_model_table():
    out = [f'<table class="wide">{wide_head("Model", (P("wins", "Wins"), P("losses", "Losses"), P("ties", "Ties")))}<tbody>']
    for lab, d in ORDERED:
        cells, up, down = [], 0, 0
        for c in COLS:
            x = ft.seeds(d, c, "regions", "B") - ft.seeds(d, c, "mbmean", "A")
            p, m = seed_p(x), x.mean()
            if p < 0.05:
                up += m > 0
                down += m < 0
                cells.append(f'<td class="{"up" if m > 0 else "down"}">{signed(m, 3)}</td>')
            else:
                cells.append(f'<td class="ns">({signed(m, 3)})</td>')
        out.append(f"<tr><td>{html.escape(mn(lab))}</td>{''.join(cells)}<td>{up}</td><td>{down}</td><td>{len(COLS) - up - down}</td></tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def rank_table():
    move = {lab: {} for lab, _ in HAVE}
    tau = {}
    for c in COLS:
        r = {lab: ft.seeds(d, c, "regions", "B").mean() for lab, d in HAVE}
        m = {lab: ft.seeds(d, c, "mbmean", "A").mean() for lab, d in HAVE}
        rm = {l: i + 1 for i, l in enumerate(sorted(m, key=lambda l: -m[l]))}
        rr = {l: i + 1 for i, l in enumerate(sorted(r, key=lambda l: -r[l]))}
        for l in m:
            move[l][c] = rm[l] - rr[l]
        tau[c] = kendalltau([m[l] for l in m], [r[l] for l in m]).statistic
    out = [f'<table class="wide">{wide_head("Model", (P("up", "Up"), P("down", "Down"), P("net", "Net")))}<tbody>']
    for lab, mv in sorted(move.items(), key=lambda kv: -sum(kv[1].values())):
        up = sum(v for v in mv.values() if v > 0)
        down = -sum(v for v in mv.values() if v < 0)
        cells = "".join(f'<td class="{"up" if mv[c] > 0 else "down" if mv[c] < 0 else "ns"}">'
                        f'{signed(mv[c], 0) if mv[c] else "0"}</td>' for c in COLS)
        out.append(f"<tr><td>{html.escape(mn(lab))}</td>{cells}<td>{up}</td><td>{down}</td>"
                   f'<td class="{"up" if up > down else "down" if up < down else ""}">{signed(up - down, 0) if up != down else "0"}</td></tr>')
    tot = {c: sum(abs(mv[c]) for mv in move.values()) for c in COLS}
    out.append('<tr class="foot"><td>Σ |Δ rank|</td>' + "".join(f"<td>{tot[c]}</td>" for c in COLS)
               + f'<td colspan="3">{sum(tot.values())}</td></tr>')
    out.append('<tr class="foot"><td>Kendall τ</td>' + "".join(f"<td>{tau[c]:.2f}</td>" for c in COLS) + '<td colspan="3"></td></tr>')
    out.append("</tbody></table>")
    return "\n".join(out)


READOUTS = [("mbmean", "A"), ("regions", "B"), ("split3", "B"), ("bases", "B")]


def scores_table():
    """Every model's ten-seed means, one row per readout under the model; the column's best model within each readout is
    bold, in that readout's colour (`b.best-<readout>`). mRNABench's two naive baselines on top, as Table 2 has them."""
    vals = {(lab, fs): {c: ft.seeds(d, c, fs, pr).mean() for c in COLS} for lab, d in ORDERED for fs, pr in READOUTS}
    # one ranking per column over every model and readout: first, second, third
    rank = {}
    for c in COLS:
        order = sorted(vals, key=lambda k: -vals[k][c])
        for i, k in enumerate(order[:3]):
            rank[(k, c)] = i
    out = [f'<table class="wide scores">{wide_head("Model", ()).replace("<th rowspan=\"3\">Model</th>", "<th rowspan=\"3\">Model</th><th rowspan=\"3\">Readout</th>")}<tbody>']
    for six in (False, True):
        nv = {c: ft.naive_seeds(c, six, "A") for c in COLS}
        cells = "".join(f'<td class="ns">{nv[c].mean():.4f}</td>' if nv[c] is not None else '<td class="ns">–</td>' for c in COLS)
        out.append(f'<tr class="first"><td class="ns">{P("naive", "Naive")}-{6 if six else 4}</td><td class="ns">baseline</td>{cells}</tr>')
    for lab, _ in ORDERED:
        for i, (fs, _) in enumerate(READOUTS):
            v = vals[(lab, fs)]
            mark = {0: '<b class="rank1">{0}</b>', 1: '<span class="rank2">{0}</span>',
                    2: '<span class="rank3">{0}</span>'}
            cells = "".join("<td>" + mark.get(rank.get(((lab, fs), c)), "{0}").format(f"{v[c]:.4f}") + "</td>"
                            for c in COLS)
            name = f'<td rowspan="{len(READOUTS)}" class="model">{html.escape(mn(lab))}</td>' if i == 0 else ""
            out.append(f'<tr{" class=\"first\"" if i == 0 else ""}>{name}<td class="readout">{rn(fs, True)}</td>{cells}</tr>')
    out.append("</tbody></table>")
    return "\n".join(out)



def region_scores_table():
    """mRNABench's Table 2 laid out the same way, with every model read out region by region: ten-seed means, the
    column's best model in bold, the models in the benchmark's order."""
    fs, _ = next(r for r in READOUTS if r[0] == "regions")
    pr = dict(READOUTS)[fs]
    vals = {lab: {c: ft.seeds(d, c, fs, pr).mean() for c in COLS} for lab, d in ORDERED}
    best = {c: max(vals[lab][c] for lab, _ in ORDERED) for c in COLS}
    out = [f'<table class="wide scores">{wide_head("Model", ())}<tbody>']
    for lab, _ in ORDERED:
        v = vals[lab]
        cells = "".join(f"<td><b>{v[c]:.4f}</b></td>" if np.isclose(v[c], best[c]) else f"<td>{v[c]:.4f}</td>"
                        for c in COLS)
        out.append(f'<tr><td class="model">{html.escape(mn(lab))}</td>{cells}</tr>')
    out.append("</tbody></table>")
    return "\n".join(out)

PERT = WORK / "results/perturb"
PTABLES = ("te-human", "sugimoto", "mrl-egfp", "rnahl-human")


def _pmodels():
    """(dir, label) of every model with all four perturbation tables scored, in Table 2 order."""
    sc = json.loads((PERT / "scores.json").read_text()) if (PERT / "scores.json").exists() else {}
    return [(d, lab) for lab, d in ORDERED if all(t in sc.get(d, {}) for t in PTABLES)]


PMODELS = _pmodels()
UCATS = ["extension/strong", "extension/weak", "overlap/strong", "overlap/weak", "uORF/strong", "uORF/weak"]
# the paper writes an out-of-frame uAUG with no stop as an oORF and an in-frame one as an N-terminal extension
UNAME = lambda c: (c.replace("extension", "N-terminal extension").replace("overlap", "oORF")
                   .replace("/strong", ", strong Kozak").replace("/weak", ", weak Kozak"))  # noqa: E731


def _pick(summary, kind, variant, dose=None):
    for r in summary:
        if r["kind"] == kind and r["variant"] == variant and (dose is None or float(r["dose"]) == dose):
            return r
    return None


def _cell(x, cls=""):
    return f'<td class="{cls}">{signed(x, 3)}</td>' if x is not None else "<td>–</td>"


TRUTHS = WORK / "results/perturb/uaug_truth_tables.json"
UTABLES = (("te-human", "TE human"), ("sugimoto", "MRL"), ("mrl-egfp", "MRL MPRA eGFP"))


def _uaug(scores, m, table, feat, ref):
    """One model's insertion response on one table: the six class means, control, removal, rho and slope against the
    measured class effects, and (regions) the 5' block's share of the change."""
    sm = scores[m][table][feat]["summary"]
    ins = [_pick(sm, "uAUG", c) for c in UCATS]
    ctrl = [x["mean"] for x in (_pick(sm, "uAUG-ctrl", c) for c in UCATS) if x]
    rem = _pick(sm, "uAUG-remove", "AUG->AUC")
    got = np.array([x["mean"] for x in ins])
    rec = {"cats": got, "ctrl": float(np.mean(ctrl)) if ctrl else None, "rem": rem["mean"] if rem else None,
           "rho": float(spearmanr(ref, got).statistic), "slope": float(got @ ref / (ref @ ref))}
    if feat == "regions":
        rec["share"] = float(sum(x["b5"] for x in ins) / got.sum()) if got.sum() else None
    return rec


def uaug_table():
    """In silico uAUG insertion (perturb_score.py), mean over the models with perturbation scores: change of the
    prediction in label SD per kind and Kozak context, the same edit with a non-AUG triplet, AUG -> AUC removal; against
    each table's own measured uAUG effects (uaug_truth_tables.py): Spearman of the six and the slope of predicted on
    measured through the origin; models whose regions slope is nearer 1 than their mbmean slope."""
    scores = json.loads((PERT / "scores.json").read_text())
    truths = json.loads(TRUTHS.read_text())
    n = len(PMODELS)
    head = "".join(f"<th>{P(c.replace('/', ' '), UNAME(c))}</th>" for c in UCATS)
    out = [f'<table class="wide"><thead><tr><th>{P("table", "Dataset")}</th><th>Readout</th>{head}<th>{P("control", "Control edit")}</th>'
           f'<th>{P("removal", "Ablation")}</th><th>ρ vs measured</th><th>{P("slope vs measured", "Slope vs measured")}</th>'
           f'<th>{P("5' block share", "5′ UTR share")}</th><th>{P("regions closer", "Region-aware closer")}</th></tr></thead><tbody>']
    for table, tname in UTABLES:
        ref = np.array([truths[table][f"kind/kozak:{c}"]["delta_sd"] for c in UCATS])
        out.append(f'<tr class="first foot"><td rowspan="3">{tname}</td><td>{P("measured (n)", "Measured (n)")}</td>'
                   + "".join(f'<td>{signed(truths[table][f"kind/kozak:{c}"]["delta_sd"], 2)} <small>({truths[table][f"kind/kozak:{c}"]["n"]:,})</small></td>'
                             for c in UCATS) + '<td colspan="6"></td></tr>')
        recs = {feat: [_uaug(scores, m, table, feat, ref) for m, _ in PMODELS] for feat in ("mbmean", "regions")}
        closer = sum(abs(r["slope"] - 1) < abs(b["slope"] - 1) for b, r in zip(recs["mbmean"], recs["regions"]))
        for feat in ("mbmean", "regions"):
            rs = recs[feat]
            got = np.mean([r["cats"] for r in rs], 0)
            cells = "".join(_cell(float(v), "down" if v < 0 else "up") for v in got)
            mean = lambda k: float(np.mean([r[k] for r in rs if r.get(k) is not None])) if any(r.get(k) is not None for r in rs) else None  # noqa: E731
            share = f"{mean('share'):.2f}" if feat == "regions" and mean("share") is not None else ""
            tail = f"{closer} / {n}" if feat == "regions" else ""
            out.append(f'<tr><td class="readout">{P(f"{feat}, mean of {n}", f"{rn(feat, True)}, mean of {n} models")}</td>{cells}'
                       f'{_cell(mean("ctrl"))}{_cell(mean("rem"))}'
                       f"<td>{signed(mean('rho'), 2)}</td><td>{signed(mean('slope'), 2)}</td><td>{share}</td><td>{tail}</td></tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def uaug_models_table():
    """Per model: Spearman and slope of the insertion response against each table's measured uAUG effects."""
    scores = json.loads((PERT / "scores.json").read_text())
    truths = json.loads(TRUTHS.read_text())
    refs = {t: np.array([truths[t][f"kind/kozak:{c}"]["delta_sd"] for c in UCATS]) for t, _ in UTABLES}
    out = ['<table class="wide"><thead><tr><th rowspan="2">Model</th>'
           + "".join(f'<th colspan="4" class="grp">{tname}</th>' for _, tname in UTABLES) + "</tr><tr>"
           + (f"<th>{P('ρ mbmean', 'ρ, global')}</th><th>{P('ρ regions', 'ρ, region')}</th><th>{P('slope mbmean', 'Slope, global')}</th>"
              f"<th>{P('slope regions', 'Slope, region')}</th>") * len(UTABLES) + "</tr></thead><tbody>"]
    for m, mlab in PMODELS:
        cells = []
        for t, _ in UTABLES:
            b, r = (_uaug(scores, m, t, feat, refs[t]) for feat in ("mbmean", "regions"))
            cells.append(f"<td>{signed(b['rho'], 2)}</td><td>{signed(r['rho'], 2)}</td><td>{signed(b['slope'], 2)}</td><td>{signed(r['slope'], 2)}</td>")
        out.append(f'<tr><td class="model">{html.escape(mn(mlab))}</td>{"".join(cells)}</tr>')
    out.append("</tbody></table>")
    return "\n".join(out)


def hl_pert_table():
    """In silico edits on rnahl-human: synonymous recoding of every codon, ARE nonamer and let-7 8mer x 4 in the 3'UTR,
    each beside its control; mean change of the predicted half-life in label SD."""
    scores = json.loads((PERT / "scores.json").read_text())
    codon = P([("codon", "optimal", 1.0, "codon optimal"), ("codon", "nonoptimal", 1.0, "codon non-optimal"), ("codon", "random", 1.0, "codon random")],
              [("codon", v, d, f"{lab}, {pct}") for d, pct in ((0.25, "25%"), (1.0, "all")) for v, lab in
               (("optimal", "Optimal"), ("nonoptimal", "Non-optimal"), ("random", "Random"))])
    cols = codon + [
            ("ARE", "TTATTTATT", 4.0, "ARE ×4"), ("ARE-ctrl", "shuffled", 4.0, P("shuffled ×4", "Shuffled ARE ×4")),
            ("let7", "8mer", 4.0, "let-7 8mer ×4"), ("let7-ctrl", "shuffled-8mer", 4.0, P("shuffled 8mer ×4", "Shuffled let-7 ×4"))]
    head = "".join(f"<th>{c[3]}</th>" for c in cols)
    out = [f'<table class="wide"><thead><tr><th>Model</th><th>Readout</th>{head}<th>{P("CDS share, non-optimal", "CDS share (non-optimal, 25%)")}</th>'
           '</tr></thead><tbody>']
    for m, mlab in PMODELS:
        for feat in ("mbmean", "regions"):
            sm = scores[m]["rnahl-human"][feat]["summary"]
            recs = [_pick(sm, k, v, d) for k, v, d, _ in cols]
            cells = "".join(_cell(x["mean"] if x else None, "down" if x and x["mean"] < 0 else "up" if x else "") for x in recs)
            share = ""
            if feat == "regions" and recs[1]:
                share = f"{recs[1]['bc'] / recs[1]['mean']:.2f}"
            mcell = f'<td rowspan="2">{html.escape(mn(mlab))}</td>' if feat == "mbmean" else ""
            out.append(f'<tr{" class=\"first\"" if feat == "mbmean" else ""}>{mcell}<td class="readout">{rn(feat, True)}</td>{cells}<td>{share}</td></tr>')
    out.append("</tbody></table>")
    return "\n".join(out)


def _nice(lo, hi, n=4):
    span = hi - lo
    raw = span / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    a, b = np.floor(lo / step) * step, np.ceil(hi / step) * step
    return a, b, [round(v, 6) for v in np.arange(a, b + step / 2, step)]


def small_multiples(rows, cols, get, desc, series=("mbmean", "regions"), legend=True, small=False):
    """Dot-plot small multiples as inline SVG. rows: [(row title, [(item label, item key)], row key)]; cols: [(col title,
    col key)]; get(row, col, item, readout) -> value or None. The x scale is shared along a row. mbmean = red dot, regions =
    blue dot; each dot carries a 24 px hover target whose <title> leads with the value."""
    # print units: WIDTH units span the 6.5 in text block, F units of text print at 9 pt
    WIDTH, F = 936, 18
    em = 0.52 * F
    labels = [ilabel for _, items, _ in rows for ilabel, _ in items] + [r for r, _, _ in rows]
    L = max(len(s) for s in labels) * em + F
    G = 1.8 * F
    W = (WIDTH - L - (len(cols) - 1) * G) / len(cols)
    per_line = max(8, int(W / em))

    def wrap(s):
        lines, cur = [], ""
        for word in s.split():
            if cur and len(cur) + 1 + len(word) > per_line:
                lines.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}".strip()
        return lines + [cur]

    heads = [wrap(c) for c, _ in cols]
    TOP = (max(len(h) for h in heads) + 0.6) * 1.15 * F
    ROWT, AX = 1.7 * F, 2.4 * F
    width = WIDTH
    y = TOP
    parts = []
    for i, lines in enumerate(heads):
        for j, line in enumerate(lines):
            parts.append(f'<text class="coltitle" x="{L + i * (W + G) + W / 2:.1f}" '
                         f'y="{TOP - (len(lines) - j - 0.15) * 1.15 * F:.1f}" text-anchor="middle">{html.escape(line)}</text>')
    for rtitle, items, rkey in rows:
        vals = [v for _, ck in cols for _, ik in items for rd in series if (v := get(rkey, ck, ik, rd)) is not None]
        lo, hi = min(vals + [0.0]), max(vals + [0.0])
        pad = (hi - lo) * 0.08 or 0.05
        a, b, ticks = _nice(lo - pad, hi + pad, max(3, int(W / (4.5 * F))))
        a, b = a - 0.07 * (b - a), b + 0.07 * (b - a)  # the outer ticks sit inside the panel, clear of the neighbour's labels
        y += ROWT
        if rtitle:
            parts.append(f'<text class="rowtitle" x="0" y="{y - 0.45 * F:.1f}">{html.escape(rtitle)}</text>')
        H = (1.6 if len(series) == 3 else 1.3) * F
        ph = len(items) * H + 6
        for ci, (ctitle, ckey) in enumerate(cols):
            x0 = L + ci * (W + G)
            X = lambda v: x0 + (v - a) / (b - a) * W  # noqa: E731
            parts.append(f'<rect class="panel" x="{x0:.1f}" y="{y:.1f}" width="{W:.1f}" height="{ph:.1f}"/>')
            for t in ticks:
                cls = "zero" if abs(t) < 1e-12 else "grid"
                parts.append(f'<line class="{cls}" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{y}" y2="{y + ph}"/>')
                parts.append(f'<text class="tick" x="{X(t):.1f}" y="{y + ph + 1.1 * F:.1f}" text-anchor="middle">{signed(t, 2) if t else "0"}</text>')
            for ii, (ilabel, ikey) in enumerate(items):
                cy = y + 3 + ii * H + H / 2
                if ci == 0:
                    parts.append(f'<text class="item" x="{L - 0.5 * F:.1f}" y="{cy + 0.35 * F:.1f}" text-anchor="end">{html.escape(ilabel)}</text>')
                offs = {1: (0.0,), 2: (-0.3 * F, 0.3 * F), 3: (-0.42 * F, 0.0, 0.42 * F)}[len(series)]
                for rd, dy in zip(series, offs):
                    v = get(rkey, ckey, ikey, rd)
                    if v is None:
                        continue
                    tip = html.escape(f"{signed(v, 3)} SD · {rd} · {ctitle} · {rtitle} · {ilabel}")
                    parts.append(f'<g class="pt"><circle class="m-{rd}" cx="{X(v):.1f}" cy="{cy + dy:.1f}" r="{0.27 * F:.1f}"/>'
                                 f'<circle class="hit" cx="{X(v):.1f}" cy="{cy + dy:.1f}" r="12"><title>{tip}</title></circle></g>')
        y += ph + AX
    names = {"truth": "truth (measured label)", "mbmean": "mbmean (mRNABench's readout and probe)", "regions": "regions (ours)"}
    legend = "" if not legend else ('<div class="legend">' + "".join(f'<span><svg width="14" height="14"><circle class="m-{rd}" cx="7" cy="7" r="4.5"/></svg>'
                                               f'{html.escape(names[rd])}</span>' for rd in series) + "</div>")
    return (legend + f'<svg class="fig{" small" if small else ""}" viewBox="0 0 {width} {y}" role="img" aria-label="{html.escape(desc)}">'
            + "".join(parts) + "</svg>")


def single_dots(items, values, desc, xlabel):
    """One series (the measured MPRA effects) as a dot plot with its own axis."""
    L, W, H = 132, 380, 19
    lo, hi = min(values + [0.0]), max(values + [0.0])
    pad = (hi - lo) * 0.08
    a, b, ticks = _nice(lo - pad, hi + pad)
    a, b = a - 0.05 * (b - a), b + 0.05 * (b - a)
    ph = len(items) * H + 6
    X = lambda v: L + (v - a) / (b - a) * W  # noqa: E731
    parts = [f'<rect class="panel" x="{L}" y="4" width="{W}" height="{ph}"/>']
    for t in ticks:
        parts.append(f'<line class="{"zero" if abs(t) < 1e-12 else "grid"}" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="4" y2="{4 + ph}"/>')
        parts.append(f'<text class="tick" x="{X(t):.1f}" y="{ph + 17}" text-anchor="middle">{signed(t, 1) if t else "0"}</text>')
    for i, (lab, v) in enumerate(zip(items, values)):
        cy = 4 + 3 + i * H + H / 2
        parts.append(f'<text class="item" x="{L - 8}" y="{cy + 4}" text-anchor="end">{html.escape(lab)}</text>')
        parts.append(f'<g class="pt"><circle class="m-truth" cx="{X(v):.1f}" cy="{cy:.1f}" r="4.5"/><circle class="hit" cx="{X(v):.1f}" cy="{cy:.1f}" r="12">'
                     f'<title>{html.escape(f"{signed(v, 2)} SD · measured · {lab}")}</title></circle></g>')
    parts.append(f'<text class="tick" x="{L + W / 2}" y="{ph + 32}" text-anchor="middle">{html.escape(xlabel)}</text>')
    return f'<svg class="fig small" viewBox="0 0 {L + W + 12} {ph + 38}" role="img" aria-label="{html.escape(desc)}">' + "".join(parts) + "</svg>"


UITEMS = [("N-terminal extension, strong Kozak", "extension/strong"), ("N-terminal extension, weak Kozak", "extension/weak"),
          ("oORF, strong Kozak", "overlap/strong"), ("oORF, weak Kozak", "overlap/weak"),
          ("uORF, strong Kozak", "uORF/strong"), ("uORF, weak Kozak", "uORF/weak"),
          ("control (non-AUG)", "control"), ("ablation (AUG→AUC)", "removal")]


def fig_uaug():
    scores = json.loads((PERT / "scores.json").read_text())
    truths = json.loads(TRUTHS.read_text())

    def get(table, model, item, readout):
        sm = scores[model][table][readout]["summary"]
        if item == "control":
            xs = [_pick(sm, "uAUG-ctrl", c) for c in UCATS]
            return float(np.mean([x["mean"] for x in xs if x])) if any(xs) else None
        if item == "removal":
            r = _pick(sm, "uAUG-remove", "AUG->AUC")
            return r["mean"] if r else None
        r = _pick(sm, "uAUG", item)
        return r["mean"] if r else None

    def get_truth(table, _, item, __):
        return truths[table][f"kind/kozak:{item}"]["delta_sd"]

    titles = {"te-human": "TE human (endogenous transcripts)", "sugimoto": "MRL, sugimoto (endogenous transcripts)",
              "mrl-egfp": "MRL MPRA eGFP (50-nt random 5'UTRs)"}
    mrows = [(titles[t], [(f"{lab} (n {truths[t]['kind/kozak:' + k]['n']:,})", k) for lab, k in UITEMS[:6]], t) for t, _ in UTABLES]
    measured = small_multiples(mrows, [("measured, label SD", "m")], get_truth,
                               "measured difference by uAUG kind in each table", series=("truth",), legend=False, small=True)
    rows = [(titles[t], UITEMS, t) for t, _ in UTABLES]
    mean = lambda table, _, item, readout: float(np.mean([get(table, m, item, readout) for m, _ in PMODELS]))  # noqa: E731
    grid = small_multiples(rows, [(f"mean of {len(PMODELS)} models", "mean")], mean, "predicted change after uAUG insertion, by table, mean over the models")
    return measured, grid


def fig_hl():
    scores = json.loads((PERT / "scores.json").read_text())
    keys = {"opt": ("codon", "optimal", 1.0), "non": ("codon", "nonoptimal", 1.0), "rnd": ("codon", "random", 1.0),
            "are": ("ARE", "TTATTTATT", 4.0), "are0": ("ARE-ctrl", "shuffled", 4.0), "let7": ("let7", "8mer", 4.0),
            "let70": ("let7-ctrl", "shuffled-8mer", 4.0)}

    def get(_, model, item, readout):
        r = _pick(scores[model]["rnahl-human"][readout]["summary"], *keys[item])
        return r["mean"] if r else None

    rows = [("Codon recoding, every codon (CDS)", [("most optimal synonym", "opt"), ("least optimal synonym", "non"), ("random synonym", "rnd")], "c"),
            ("3'UTR elements, four copies", [("ARE UUAUUUAUU", "are"), ("ARE, shuffled", "are0"), ("let-7 8mer site", "let7"), ("let-7 8mer, shuffled", "let70")], "u")]
    mean = lambda row, _, item, readout: float(np.mean([get(row, m, item, readout) for m, _ in PMODELS]))  # noqa: E731
    return small_multiples(rows, [(f"mean of {len(PMODELS)} models", "mean")], mean, "predicted change of HL human after edits, mean over the models")


ECMP = WORK / "results/elements/compare.json"
ENAMES = {"out-of-frame uAUG vs none": "out-of-frame uAUG vs none", "in-frame uAUG vs none": "in-frame uAUG vs none",
          "CAI top vs bottom quarter": "CAI, top vs bottom quarter",
          "codon optimality (CSC) top vs bottom quarter": "codon optimality (CSC), top vs bottom quarter", "AUUUA ≥3 vs 0": "AUUUA, ≥3 vs none",
          "let-7 site vs none": "let-7 site vs none", "miR-16 site vs none": "miR-16 site vs none"}
TNAMES = {"te-human": "TE human", "te-mouse": "TE mouse", "sugimoto": "MRL", "rnahl-human": "HL human", "rnahl-mouse": "HL mouse",
          "mrl-egfp": "MPRA eGFP", "mrl-mcherry": "MPRA mCherry", "mrl-designed": "MPRA designed", "mrl-varying": "MPRA varying"}


def element_table():
    cmp_ = json.loads(ECMP.read_text())
    sub = (f"<th>{P('truth', 'Measured')}</th><th>{rn('mbmean', True)}</th><th>{rn('regions', True)}</th>"
           f"<th>{P('regions / mbmean / tie', 'Region / global / tie')}</th>")
    out = [f'<table class="wide"><thead><tr><th rowspan="2">Element</th><th rowspan="2">{P("table", "Dataset")}</th><th rowspan="2">{P("expected", "Expected")}</th>'
           f'<th rowspan="2">{P("rows (element / reference)", "n (with / without)")}</th><th colspan="4" class="grp">{P("difference", "Difference")}</th>'
           f'<th colspan="4" class="grp">{P("adjusted for length and GC", "Adjusted for length and GC")}</th></tr><tr>{sub}{sub}</tr></thead><tbody>']
    for name, per_table in cmp_.items():
        first = True
        for table, rec in per_table.items():
            ms = list(rec["models"].values())
            if len(ms) < len(HAVE):
                continue
            cells = []
            for adj in ("", "_adj"):
                t, mb, rg = (np.mean([m[f"{k}{adj}"] for m in ms]) for k in ("truth", "mbmean", "regions"))
                v = [m[f"verdict{adj}"] for m in ms]
                cells.append(f"<td>{signed(t, 3)}</td><td>{signed(mb, 3)}</td><td>{signed(rg, 3)}</td>"
                             f"<td>{v.count('regions')} / {v.count('mbmean')} / {v.count('tie')}</td>")
            sign = {-1: "lower", 1: "higher", 0: "–"}[rec["sign"]]
            ecell = f'<td rowspan="{sum(1 for r in per_table.values() if len(r["models"]) >= len(HAVE))}">{html.escape(ENAMES[name])}</td>' if first else ""
            out.append(f'<tr{" class=\"first\"" if first else ""}>{ecell}<td>{TNAMES[table]}</td><td class="ns">{sign}</td>'
                       f"<td>{ms[0]['n_a']:,} / {ms[0]['n_b']:,}</td>{''.join(cells)}</tr>")
            first = False
    out.append("</tbody></table>")
    return "\n".join(out)


def fig_elements():
    cmp_ = json.loads(ECMP.read_text())

    def get(name, col, table, readout):
        ms = list(cmp_[name][table]["models"].values())
        return float(np.mean([m[f"{readout}{col}"] for m in ms])) if ms else None

    rows = [(ENAMES[n], [(TNAMES[t], t) for t in cmp_[n] if len(cmp_[n][t]["models"]) >= len(HAVE)], n) for n in cmp_]
    return small_multiples(rows, [("difference", ""), ("adjusted for length and GC", "_adj")], get,
                           "element group minus reference group, truth and the two probes", series=("truth", "mbmean", "regions"))


R2_TARGET = {"mrl-egfp": "batch.unmod"}
_R2 = {}


def r2(table, readout):
    """Mean over the 20 models of the squared ten-seed mean test r (mbmean with mRNABench's grid, regions with ours): the
    share of any group difference a probe would reproduce if it caught every part of the label alike."""
    if (table, readout) not in _R2:
        vals = []
        for _, d in HAVE:
            f = ft.R / d / table / "fit.json"
            if f.exists():
                r = json.loads(f.read_text())["results"][R2_TARGET.get(table, "batch.y")][f"{readout}|raw{'A' if readout == 'mbmean' else 'B'}"]["test"]
                vals.append(float(np.mean(r)) ** 2)
        _R2[(table, readout)] = float(np.mean(vals)) if vals else None
    return _R2[(table, readout)]


def dumbbell(panels, rows_by_group, desc, ref_label=None, ref_one=True, kozak=False, xlabel=""):
    """Headline dumbbells: one line per row, the mbmean and regions dots joined, reference lines at 0 and 1.
    panels: [(panel title, key)]; rows_by_group: [(group title, [(row label, {key: (mb, rg, note)})])], a value may be
    None (not drawn; the note says why)."""
    # print units: WIDTH units span the 6.5 in text block, F units of text print at 9 pt
    WIDTH, F = 936, 18                                               # 18 of 936 units prints at 9 pt
    em = 0.52 * F                                                    # the width of an average character
    labels = [row[0] for _, rows in rows_by_group for row in rows] + [g for g, _ in rows_by_group]
    notes = [row[1][k][2] for _, rows in rows_by_group for row in rows for _, k in panels if row[1][k][2]]
    L = max(len(s) for s in labels) * em + F
    R = max([len(s) for s in notes] + [len(P("regions closer", "closer"))]) * em + F if notes else 0
    G, H, GT = 2.0 * F, 1.00 * F, 1.5 * F
    RPAD = 2.2 * 0.52 * F                                        # the last tick label stays inside
    W = (WIDTH - L - len(panels) * R - (len(panels) - 1) * G - RPAD) / len(panels)
    per_line = max(8, int((W + R) / (0.56 * F)))                     # a title may run over its panel's note column

    def wrap(s):
        lines, cur = [], ""
        for word in s.split():
            if cur and len(cur) + 1 + len(word) > per_line:
                lines.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}".strip()
        return lines + [cur]

    heads = [wrap(title) for title, _ in panels]
    TOP = (max(len(h) for h in heads) * 1.2 + 2.3) * F
    ref_label = ref_label or P("= truth", "measured")
    vals = [v for _, rows in rows_by_group for row in rows for k in row[1] for v in row[1][k][:2] if v is not None]
    lo, hi = min(vals), max(vals + ([1.0] if ref_one else []))
    pad = (hi - lo) * 0.06
    a, b, ticks = _nice(lo - pad, hi + pad, max(5, int(W / (2.6 * F))))
    a = max(a, lo - pad)                                           # no empty step below the data
    b = min(b, max(hi, 1.0 if ref_one else hi) + pad)
    a, b = a - 0.05 * (b - a), b + 0.05 * (b - a)                  # the outer tick labels stay inside the panel
    ticks = [v for v in ticks if a <= v <= b]
    width = WIDTH
    nrows = sum(len(r) for _, r in rows_by_group)
    has_base = any(len(row) > 2 for _, rows in rows_by_group for row in rows)
    height = TOP + sum(GT for g, _ in rows_by_group if g) + nrows * H + 3.4 * F
    parts = []
    for pi, (ptitle, key) in enumerate(panels):
        x0 = L + pi * (W + R + G)
        X = lambda v: x0 + (v - a) / (b - a) * W  # noqa: E731
        for j, line in enumerate(heads[pi]):
            tx = min(max(x0 + W / 2, len(line) * 0.3 * F), width - len(line) * 0.3 * F)
            parts.append(f'<text class="coltitle" x="{tx:.1f}" y="{(j + 1) * 1.2 * F:.1f}" text-anchor="middle">{html.escape(line)}</text>')
        if any(row[1][key][2] for _, rows in rows_by_group for row in rows):
            parts.append(f'<text class="tick" x="{x0 + W + 0.6 * F:.1f}" y="{TOP - 0.7 * F:.1f}">{P("regions closer", "closer")}</text>')
        parts.append(f'<rect class="panel" x="{x0:.1f}" y="{TOP - 4}" width="{W:.1f}" height="{height - TOP - 3.8 * F + 4:.1f}"/>')
        step = (X(ticks[1]) - X(ticks[0])) if len(ticks) > 1 else W
        every = max(1, int(np.ceil(3.4 * 0.52 * F / step)))          # label only the ticks that have room
        for i, t in enumerate(ticks):
            cls = "zero" if abs(t) < 1e-12 else "one" if ref_one and abs(t - 1) < 1e-12 else "grid"
            parts.append(f'<line class="{cls}" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{TOP - 4}" y2="{height - 3.8 * F:.1f}"/>')
            if i % every == 0 or abs(t) < 1e-12 or (ref_one and abs(t - 1) < 1e-12):
                parts.append(f'<text class="tick" x="{X(t):.1f}" y="{height - 1.4 * F:.1f}" text-anchor="middle">{t:g}</text>')

        if ref_one:
            if not any(abs(t - 1) < 1e-12 for t in ticks):
                parts.append(f'<line class="one" x1="{X(1):.1f}" x2="{X(1):.1f}" y1="{TOP - 4}" y2="{height - 3.8 * F:.1f}"/>')
            parts.append(f'<text class="tick" x="{X(1) - 0.3 * F:.1f}" y="{TOP - 0.7 * F:.1f}" text-anchor="end">{html.escape(ref_label)}</text>')
        y = TOP
        for gtitle, rows in rows_by_group:
            y += GT if gtitle else 0
            if pi == 0:
                parts.append(f'<text class="rowtitle" x="0" y="{y - 0.45 * F:.1f}">{html.escape(gtitle)}</text>')
            for row in rows:
                rlabel, d = row[0], row[1]
                base = row[2] if len(row) > 2 else None
                cy = y + H / 2
                if base:
                    for rd, v in zip(("mbmean", "regions"), base):
                        if v is not None:
                            parts.append(f'<line class="b-{rd}" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{cy - 0.42 * H:.1f}" y2="{cy + 0.42 * H:.1f}">'
                                         f'<title>{html.escape(f"r² {v:.2f} · {rn(rd)} · {rlabel}")}</title></line>')
                if pi == 0:
                    parts.append(f'<text class="item" x="{L - 0.5 * F:.1f}" y="{cy + 0.35 * F:.1f}" text-anchor="end">{html.escape(rlabel)}</text>')
                mb, rg, note = d[key]
                if mb is not None and rg is not None:
                    parts.append(f'<line class="bar" x1="{X(mb):.1f}" x2="{X(rg):.1f}" y1="{cy:.1f}" y2="{cy:.1f}"/>')
                    for rd, v in (("mbmean", mb), ("regions", rg)):
                        tip = html.escape((f"{v:.2f} of the " + P("truth", "measured effect") if ref_one else f"{signed(v, 3)} SD")
                                          + f" · {rn(rd)} · {gtitle} · {rlabel} · {ptitle}")
                        parts.append(f'<g class="pt"><circle class="m-{rd}" cx="{X(v):.1f}" cy="{cy:.1f}" r="{0.3 * H:.1f}"/>'
                                     f'<circle class="hit" cx="{X(v):.1f}" cy="{cy:.1f}" r="12"><title>{tip}</title></circle></g>')
                if note:
                    parts.append(f'<text class="tick" x="{x0 + W + 0.6 * F:.1f}" y="{cy + 0.35 * F:.1f}">{html.escape(note)}</text>')
                y += H
    if xlabel:
        parts.append(f'<text class="tick" x="{L + (width - L) / 2:.1f}" y="{height - 0.2 * F:.1f}" '
                     f'text-anchor="middle">{html.escape(xlabel)}</text>')
    legend = ('<div class="legend"><span><svg width="14" height="14"><circle class="m-mbmean" cx="7" cy="7" r="4.5"/></svg>'
              + P("mbmean (mRNABench's readout and probe)", "Global mean")
              + '</span><span><svg width="14" height="14"><circle class="m-regions" cx="7" cy="7" r="4.5"/></svg>'
              + P("regions (ours)", "Region-aware") + "</span>"
              + ('<span>' + P("strong/weak: Kozak context", "strong Kozak: a purine at \u22123 and a G at +4; weak: neither") + "</span>" if kozak else "")
              + ('<span><svg width="14" height="14"><line class="b-mbmean" x1="5" x2="5" y1="1" y2="13"/><line class="b-regions" x1="10" x2="10" y1="1" y2="13"/></svg>'
                 + P("r² of each probe: the share its accuracy alone would reproduce", "Tick: the probe's r², the share accuracy alone would reproduce") + "</span>" if has_base else "") + "</div>")
    return (legend + f'<svg class="fig" viewBox="0 0 {width} {height:.0f}" role="img" '
            f'aria-label="{html.escape(desc)}">' + "".join(parts) + "</svg>")


SUBS = WORK / "results/elements/uaug_subsets.json"
SCLASSES = [("oORF, strong Kozak", "overlap/strong"), ("oORF, weak Kozak", "overlap/weak"), ("uORF, strong Kozak", "uORF/strong"),
            ("uORF, weak Kozak", "uORF/weak"), ("several uAUGs", "multi")]


def head_subsets():
    """Share of the measured uAUG-class difference the probes' held-out predictions reproduce, mean over 20 models."""
    subs = json.loads(SUBS.read_text())
    groups = []
    for table, tname in (("te-human", "TE human"), ("te-mouse", "TE mouse"), ("sugimoto", "MRL"), ("mrl-egfp", "MRL MPRA eGFP")):
        if table not in subs or len(subs[table]) < len(HAVE):
            continue
        ms = list(subs[table].values())
        rows = []
        for lab, k in SCLASSES:
            n = ms[0]["n"][k]
            d = {}
            for adj in ("", "_adj"):
                t = np.mean([m[f"truth{adj}"][k] for m in ms])
                if n < 30 or abs(t) < 0.15:
                    d[adj] = (None, None, P(f"n {n}", f"n = {n}") if n < 30 else P("truth ≈ 0", "measured ≈ 0"))
                    continue
                mb = np.mean([m[f"mbmean{adj}"][k] for m in ms]) / t
                rg = np.mean([m[f"regions{adj}"][k] for m in ms]) / t
                closer = sum(abs(m[f"regions{adj}"][k] - m[f"truth{adj}"][k]) < abs(m[f"mbmean{adj}"][k] - m[f"truth{adj}"][k]) for m in ms)
                d[adj] = (mb, rg, f"{closer}/{len(ms)}")
            rows.append((P(f"{lab} (n {n:,})", f"{lab} (n = {n:,})"), d, (r2(table, "mbmean"), r2(table, "regions"))))
        groups.append((tname, rows))
    return dumbbell([(P("difference", "Difference"), ""), (P("adjusted for length and GC", "Adjusted for length and GC"), "_adj")], groups,
                    "share of the measured uAUG-class difference reproduced by the two probes", kozak=True,
                    xlabel="share of the measured difference reproduced")


def head_insertion():
    """Slope of the insertion response on the table's measured class effects: one panel per table, one row per model;
    the panel title counts the models whose regions slope is nearer 1."""
    scores = json.loads((PERT / "scores.json").read_text())
    truths = json.loads(TRUTHS.read_text())
    panels, rows = [], {}
    for table, tname in UTABLES:
        ref = np.array([truths[table][f"kind/kozak:{c}"]["delta_sd"] for c in UCATS])
        closer = 0
        for m, mlab in PMODELS:
            mb, rg = (_uaug(scores, m, table, feat, ref)["slope"] for feat in ("mbmean", "regions"))
            rows.setdefault(mn(mlab), {})[table] = (mb, rg, "")
            closer += abs(rg - 1) < abs(mb - 1)
        panels.append((P(f"{tname} · regions closer {closer}/{len(PMODELS)}", f"{tname} · {closer}/{len(PMODELS)}"), table))
    return dumbbell(panels, [("", list(rows.items()))],
                    "slope of the predicted change on the measured effect, per dataset and model", kozak=True,
                    xlabel="slope on the measured class effects")


def head_elements():
    """Share of the truth's element difference the probes reproduce, mean over 20 models; per-model verdicts."""
    cmp_ = json.loads(ECMP.read_text())
    groups = []
    sets = ((("5'UTR", ["out-of-frame uAUG vs none", "in-frame uAUG vs none"]),
             ("CDS", ["codon optimality (CSC) top vs bottom quarter", "CAI top vs bottom quarter"]),
             ("3'UTR", ["AUUUA ≥3 vs 0", "let-7 site vs none", "miR-16 site vs none"])),
            (("5′ UTR", ["out-of-frame uAUG vs none", "in-frame uAUG vs none"]),
             ("CDS", ["codon optimality (CSC) top vs bottom quarter"])))  # the paper's figure: uAUGs and CSC; the rest in Table S7
    for gname, names in P(*sets):
        rows = []
        for name in names:
            for table, rec in cmp_[name].items():
                ms = list(rec["models"].values())
                if len(ms) < len(HAVE):  # a table still being computed stays out
                    continue
                d = {}
                for adj in ("", "_adj"):
                    t = np.mean([m[f"truth{adj}"] for m in ms])
                    v = [m[f"verdict{adj}"] for m in ms]
                    if abs(t) < 0.05:
                        d[adj] = (None, None, P("truth ≈ 0", "measured ≈ 0"))
                        continue
                    d[adj] = (np.mean([m[f"mbmean{adj}"] for m in ms]) / t, np.mean([m[f"regions{adj}"] for m in ms]) / t,
                              f"{v.count('regions')}/{len(ms)}")
                rows.append((f"{ENAMES[name].replace(' vs none', '').replace(', top vs bottom quarter', '').replace(', ≥3 vs none', ' ≥3')} · {TNAMES.get(table, table)}", d,
                             (r2(table, "mbmean"), r2(table, "regions"))))
        groups.append((gname, rows))
    return dumbbell([(P("difference", "Difference"), ""), (P("adjusted for length and GC", "Adjusted for length and GC"), "_adj")], groups,
                    "share of the measured element difference reproduced by the two probes",
                    xlabel="share of the measured difference reproduced")


def head_hl():
    """Contrasts of the HL edits, one panel each, one row per model: optimal minus least optimal codons (expected above
    zero), ARE and let-7 minus their shuffles (expected below zero), in label SD."""
    scores = json.loads((PERT / "scores.json").read_text())
    panels, rows = [], {}
    dose = P(1.0, 0.25)
    for key, (title, a, b) in enumerate(((P("codons: optimal − least optimal (> 0)", "Optimal minus least optimal codons, 25% recoded (expected > 0)"),
                                          ("codon", "optimal", dose), ("codon", "nonoptimal", dose)),
                                         (P("ARE ×4 − shuffled (< 0)", "Four AREs minus shuffled controls (expected < 0)"), ("ARE", "TTATTTATT", 4.0), ("ARE-ctrl", "shuffled", 4.0)),
                                         (P("let-7 8mer ×4 − shuffled (< 0)", "Four let-7 sites minus shuffled controls (expected < 0)"),
                                          ("let7", "8mer", 4.0), ("let7-ctrl", "shuffled-8mer", 4.0)))):
        panels.append((title, key))
        for m, mlab in PMODELS:
            v = {}
            for feat in ("mbmean", "regions"):
                sm = scores[m]["rnahl-human"][feat]["summary"]
                v[feat] = _pick(sm, *a)["mean"] - _pick(sm, *b)["mean"]
            rows.setdefault(mn(mlab), {})[key] = (v["mbmean"], v["regions"], "")
    return dumbbell(panels, [("", list(rows.items()))], "contrasts of the HL edits per model", ref_one=False,
                    xlabel="change in the predicted half-life (label SD)")


def subsets_table():
    """Table 4a: the uAUG classes as the tables have them, no edits; mean over 20 models of held-out predictions."""
    subs = json.loads(SUBS.read_text())
    sub = (f"<th>{P('measured', 'Measured')}</th><th>{rn('mbmean', True)}</th><th>{rn('regions', True)}</th>"
           f"<th>{P('regions closer', 'Region-aware closer')}</th>")
    head = (f'<table class="wide"><thead><tr><th rowspan="2">{P("table", "Dataset")}</th><th rowspan="2">{P("class", "Class")}</th><th rowspan="2">n</th>'
            f'<th colspan="4" class="grp">{P("difference to uAUG-free, label SD", "Difference to uAUG-free transcripts (label SD)")}</th>'
            f'<th colspan="4" class="grp">{P("adjusted for length and GC", "Adjusted for length and GC")}</th></tr>'
            f"<tr>{sub}{sub}</tr></thead><tbody>")
    out = [head]
    classes = [(P(c.replace("/", ", "), UNAME(c)), c) for c in UCATS] + [("several uAUGs", "multi")]
    for table, tname in (("te-human", "TE human"), ("te-mouse", "TE mouse"), ("sugimoto", "MRL"), ("mrl-egfp", "MRL MPRA eGFP")):
        if table not in subs or len(subs[table]) < len(HAVE):
            continue
        ms = list(subs[table].values())
        first = True
        for lab, k in classes:
            cells = []
            for adj in ("", "_adj"):
                t = np.mean([m[f"truth{adj}"][k] for m in ms])
                mb = np.mean([m[f"mbmean{adj}"][k] for m in ms])
                rg = np.mean([m[f"regions{adj}"][k] for m in ms])
                closer = sum(abs(m[f"regions{adj}"][k] - m[f"truth{adj}"][k]) < abs(m[f"mbmean{adj}"][k] - m[f"truth{adj}"][k]) for m in ms)
                cells.append(f"<td>{signed(t, 3)}</td><td>{signed(mb, 3)}</td><td>{signed(rg, 3)}</td><td>{closer} / {len(ms)}</td>")
            tcell = f'<td rowspan="{len(classes)}">{tname}</td>' if first else ""
            out.append(f'<tr{" class=\"first\"" if first else ""}>{tcell}<td>{lab}</td><td>{ms[0]["n"][k]:,}</td>{"".join(cells)}</tr>')
            first = False
    out.append("</tbody></table>")
    return "\n".join(out)


def controls_table():
    def mean_random(d, c):
        vs = [ft.seeds(d, c, f"random{j}|rawB", "B") for j in range(5)]
        return None if any(v is None for v in vs) else np.mean(vs, 0)

    ctl = [
        (P("against split3 (same width)", "Difference from equal thirds"), [
            (P("regions", "Region-aware"), lambda d, c: (ft.seeds(d, c, "regions", "B"), ft.seeds(d, c, "split3", "B"))),
            (P("PCA-d(regions) vs PCA-d(split3)", "Region-aware, both reduced to d by PCA"),
             lambda d, c: (ft.seeds(d, c, "pca_regions|rawB", "B"), ft.seeds(d, c, "pca_split3|rawB", "B")))]),
        (P("against bases (same width)", "Difference from the nucleotide mean"), [
            (P("rbal", "Balanced region mean"), lambda d, c: (ft.seeds(d, c, "rbal", "B"), ft.seeds(d, c, "bases", "B"))),
            (P("regions+F6 vs bases+F6 (length, GC)", "Region-aware, both with length and GC"),
             lambda d, c: (ft.seeds(d, c, "regions+F6", "B"), ft.seeds(d, c, "bases+F6", "B")))]),
        (P("against true boundaries", "Difference from annotated boundaries"), [
            (P("random boundaries", "Random boundaries"), lambda d, c: (mean_random(d, c), ft.seeds(d, c, "regions", "B"))),
            (P("shifted −100 nt", "Shifted 100 nt toward the 5′ end"),
             lambda d, c: (ft.seeds(d, c, "shift-100|rawB", "B"), ft.seeds(d, c, "regions", "B"))),
            (P("shifted +100 nt", "Shifted 100 nt toward the 3′ end"),
             lambda d, c: (ft.seeds(d, c, "shift+100|rawB", "B"), ft.seeds(d, c, "regions", "B")))]),
        (P("against per-region encoding", "Difference from per-region encoding"), [
            (P("whole-transcript", "Whole-transcript encoding"),
             lambda d, c: (ft.seeds(d, c, "regions", "B"), ft.seeds(d + "-iso", c, "regions", "B"))
             if (ft.R / (d + "-iso")).exists() else (None, None))])]
    cols = ["HL", "MRL", "MRL-HL-Pair", "TE human", "TE mouse"]
    head = "".join(f"<th>{P(c, PAPER_COL.get(c, c))}</th>" for c in cols)
    out = [f'<table class="wide"><thead><tr><th>Control</th>{head}</tr></thead><tbody>']
    for group, rows in ctl:
        out.append(f'<tr class="group"><td colspan="{len(cols) + 1}" class="left">{group}</td></tr>')
        for name, pick in rows:
            cells = []
            for c in cols:
                pairs = [pick(d, c) for _, d in HAVE]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                diff, win, loss, tie, n = tally(pairs)
                cls = "up" if win > loss else "down" if loss > win else ""
                cells.append(f'<td class="{cls}">{signed(diff)}<br><small>{win} / {loss} / {tie}'
                             + (f" (n = {n})" if n != len(HAVE) and not PAPER else "") + "</small></td>")
            out.append(f"<tr><td>{html.escape(name)}</td>{''.join(cells)}</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


CSS = """
/* The page's colour tokens. */
:root {
  --ink: #50616d; --ink-deep: #161823; --gao: #f2ecde; --paper: #fbf7ee; --paper-2: #f6efdf;
  --line: #ded6c4; --muted: #7c8896; --faint: #9aa4ae; --accent: #177cb0; --accent-deep: #065279;
  --good: #2a6e3f; --warn: #705438; --bad: #960018; --info: #2e59a7; --special: #801dae; --c-rg: #2e59a7; --c-mb: #b35c44;
  --code-bg: #e9e1cd; --code-fg: #20242c; --code-border: #d8cfb8; --warn-soft: #f6ebd4;
}
@media (prefers-color-scheme: dark) {
  :root {
    --ink: #e6e0d2; --ink-deep: #f2ecde; --gao: #161823; --paper: #1e2233; --paper-2: #262b3d;
    --line: #2f3548; --muted: #9aa4ae; --faint: #7c8896; --accent: #3fa3d8; --accent-deep: #8fd0f0;
    --good: #8fce9e; --warn: #e6c293; --bad: #db5a6b; --info: #5fb0d8; --special: #b97fd8; --c-rg: #177cb0; --c-mb: #ed5736;
    --code-bg: #0f1019; --code-fg: #e6e0d2; --code-border: #2f3548; --warn-soft: #2b2620;
  }
}
* { box-sizing: border-box; }
html { font-size: 19px; }
body { margin: 0; background: var(--gao); color: var(--ink);
       font-family: "EB Garamond", "Noto Serif SC", "Source Han Serif SC", Georgia, serif;
       font-feature-settings: "onum" 1, "liga" 1; line-height: 1.55; }
main { max-width: 60rem; margin: 0 auto; padding: 2.5rem 1.4rem 4rem; }
h1, h2, h3 { color: var(--ink-deep); font-weight: 600; line-height: 1.25; }
h1 { font-size: 2rem; margin: 0 0 .3rem; }
h2 { font-size: 1.35rem; margin: 2.4rem 0 .6rem; padding-bottom: .25rem; border-bottom: 1px solid var(--line); }
h3 { font-size: 1.08rem; margin: 1.5rem 0 .4rem; }
p { margin: .55rem 0; } ul, ol { margin: .4rem 0 .8rem; padding-left: 1.3rem; } li { margin: .25rem 0; }
.sub { color: var(--muted); font-style: italic; margin: 0 0 1.6rem; }
.thesis { background: var(--paper); border: 1px solid var(--line); border-left: 4px solid var(--accent); padding: .9rem 1.2rem; margin: 1rem 0 1.4rem; }
.thesis p:first-child { margin-top: 0; } .thesis p:last-child { margin-bottom: 0; }
code { font-family: "JetBrains Mono", Menlo, monospace; font-size: .8em; background: var(--code-bg); color: var(--code-fg);
       border: 1px solid var(--code-border); padding: 0 .25em; border-radius: 3px; }
.tablewrap { overflow-x: auto; margin: .8rem 0 .4rem; }
table { border-collapse: collapse; width: 100%; background: var(--paper); border: 1px solid var(--line); font-size: .86rem;
        font-variant-numeric: lining-nums tabular-nums; }
table.wide { font-size: .74rem; }
th, td { padding: .3rem .5rem; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
th { background: var(--paper-2); color: var(--ink-deep); font-weight: 600; }
th:first-child, td:first-child { text-align: left; }
td small { color: var(--muted); font-size: .9em; }
tr.group td { background: var(--paper-2); color: var(--muted); font-style: italic; text-align: left; }
tr.foot td { background: var(--paper-2); font-weight: 600; }
td.up { color: var(--good); } td.down { color: var(--bad); } td.ns { color: var(--faint); }
table.scores tr.first td { border-top: 2px solid var(--line); } td.model { vertical-align: top; color: var(--ink-deep); }
td.readout { text-align: left; color: var(--muted); font-style: italic; }
b.best-mbmean { color: var(--ink-deep); } b.best-regions { color: var(--info); } b.best-split3 { color: var(--special); }
b.best-bases { color: var(--warn); }
dl.defs { margin: .4rem 0 .8rem; } dl.defs dt { font-weight: 600; color: var(--ink-deep); margin-top: .35rem; }
dl.defs dd { margin: 0 0 0 1.2rem; }
th.grp { text-align: center; }
.howto { background: var(--paper); border: 1px solid var(--line); border-left: 4px solid var(--c-rg); padding: .55rem 1.1rem .65rem; margin: .4rem 0 1.4rem; }
.howto h4 { margin: .1rem 0 .3rem; color: var(--ink-deep); font-size: .95rem; }
.howto p { margin: .35rem 0; font-size: .92rem; }
.howto ul { margin: .3rem 0 .4rem; font-size: .92rem; }
.howto.expect { border-left-color: var(--c-mb); }
svg.fig { width: 100%; height: auto; display: block; margin: .2rem 0 .3rem; font-family: inherit; }
svg.fig.small { max-width: 30rem; }
svg.fig .panel { fill: var(--paper); stroke: var(--line); stroke-width: 1; }
svg.fig .grid { stroke: var(--line); stroke-width: 1; }
svg.fig .zero { stroke: var(--muted); stroke-width: 1; }
svg.fig text { fill: var(--muted); font-size: 11px; }
svg.fig text.item { fill: var(--ink); }
svg.fig text.rowtitle, svg.fig text.coltitle { fill: var(--ink-deep); font-size: 12.5px; font-weight: 600; }
svg.fig .m-regions { fill: var(--c-rg); stroke: var(--paper); stroke-width: 2; }
.scores .rank1 { color: var(--c-rg); font-weight: 700; }
.scores .rank2 { color: var(--c-mb); font-weight: 700; }
.scores .rank3 { text-decoration: underline; }
svg.fig .m-mbmean, .legend .m-mbmean { fill: var(--c-mb); stroke: var(--paper); stroke-width: 2; }
.legend .m-regions { fill: var(--c-rg); }
svg.fig .m-truth, .legend .m-truth { fill: var(--ink); stroke: var(--paper); stroke-width: 2; }
svg.fig .hit { fill: transparent; pointer-events: all; }
svg.fig .bar { stroke: var(--muted); stroke-width: 2; stroke-linecap: round; opacity: .55; }
svg.fig .one { stroke: var(--ink-deep); stroke-width: 1; }
svg.fig .b-mbmean, .legend .b-mbmean { stroke: var(--c-mb); stroke-width: 2; opacity: .7; }
svg.fig .b-regions, .legend .b-regions { stroke: var(--c-rg); stroke-width: 2; opacity: .7; }
svg.fig g.pt:hover circle:first-child { stroke: var(--ink-deep); stroke-width: 2; }
.legend { display: flex; gap: 1.4rem; font-size: .8rem; color: var(--muted); margin: .6rem 0 0; }
.legend span { display: inline-flex; align-items: center; gap: .35rem; } th.metric { font-weight: 400; font-style: italic; color: var(--muted); }
.caption { color: var(--muted); font-size: .86rem; margin: .2rem 0 1.2rem; }
.risk { background: var(--warn-soft); border: 1px solid var(--line); padding: .7rem 1.1rem .8rem; margin: 1rem 0; }
.risk h3 { margin-top: .2rem; color: var(--warn); }
footer { margin-top: 3rem; color: var(--faint); font-size: .8rem; border-top: 1px solid var(--line); padding-top: .6rem; }
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Regional Signals Matter: the story so far</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400..700;1,400..600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{css}</style>
</head>
<body>
<main>

<h1>Regional Signals Matter</h1>
<p class="sub">Rethinking global pooling for mRNA foundation models: the story so far (RECOMB 2027 draft notes, 2 Nov 2026)</p>

<div class="thesis">
<p>Benchmarks such as mRNABench read a frozen model by averaging every token vector of the last layer into one vector and fitting a linear probe on it. That global average mixes the signals of the 5'UTR, the CDS and the 3'UTR.</p>
<p>We keep the same forward pass and the same probe estimator and replace the global average by one average per region, concatenated (<em>regions</em>). On mRNA translation and stability tasks, 20 models gain 0.05–0.09 Pearson r. The gain follows the true region boundaries, and the regions the probe relies on match the known roles of the UTRs and the CDS. The global average underestimates these models.</p>
</div>

<h2>Setup</h2>
<ul>
<li><em>mbmean</em> (mRNABench's global mean) and the naive k-mer baselines are mRNABench's own, reproduced with their readout, splits and probe: across 380 cells the median absolute difference from the published numbers is 0.0009.</li>
<li><em>regions</em> adds no parameters. The region coordinates come from the sequence annotation in the benchmark tables.</li>
<li>20 models, ten mRNABench seeds per table. A model wins or loses a column when its ten-seed paired difference is positive or negative with Wilcoxon p &lt; 0.05, and ties otherwise.</li>
</ul>

<h2>Result 1. Regions beats the global mean</h2>
<ul>
<li>Regression tasks on endogenous transcripts: TE human +0.087 (20 wins / 0 losses / 0 ties), TE mouse +0.076 (20/0/0), MRL +0.072 (19/0/1), HL +0.054 (18/0/2), MRL–HL pair +0.092 (14/0/6). MRL MPRA +0.059 (20/0/0). No model drops on any of the six regression columns.</li>
<li>Classification tasks: VEP +0.028, eCLIP +0.018, Loc-SR, Loc-LR and GO between +0.006 and +0.008, Prot-Loc −0.009. The significant drops sit in the classification columns, in NT, Evo 2, RiNALMo and AIDO (5–6 columns each) and in RNAErnie and DNABERT-S (4 each).</li>
</ul>
<div class="tablewrap">{t1}</div>
<p class="caption">Table 1. Mean over 20 models. mbmean is mRNABench's readout and probe as published; bases (nucleotide mean), split3 (equal thirds) and regions are ours. wins / losses / ties: models whose ten-seed difference is positive / negative with Wilcoxon p &lt; 0.05 / not significant.</p>

<div class="tablewrap">{t2}</div>
<p class="caption">Table 2. regions − mbmean for each model, ten-seed mean. Colored: Wilcoxon p &lt; 0.05 over the ten seeds; in parentheses: not significant. wins / losses / ties: columns with a significant gain / a significant drop / no significant difference.</p>

<h2>Result 2. The gain comes from the true boundaries</h2>
<p>Table 3 compares our readouts with each other.</p>
<ul>
<li>Equal width. Regions beats an equal-thirds split (split3, the same 3d width) by 0.017–0.054 (TE human 19/1/0). With the top d principal components of each, the gap is 0.018–0.054.</li>
<li>Random boundaries, cut at another transcript's region proportions: HL, MRL, TE human and TE mouse drop by 0.053–0.085, with 20 of 20 models significant in each column.</li>
<li>Shifted boundaries. Moving both boundaries 100 nt toward the 5' end costs MRL 0.041 (0/20/0); moving them 100 nt toward the 3' end costs 0.008. The 5' shift puts the 100 nt before the start codon into the CDS block. The 3' shift leaves the 5'UTR intact. This fits MRL depending on the 5'UTR next to the start codon.</li>
<li>Length and GC. With the log lengths and GC fractions of the three regions (F6) appended to both readouts, regions beats bases by 0.021–0.055 (TE human 20/0/0).</li>
</ul>

<h3>The controls in Table 3</h3>
<dl class="defs">
<dt>split3</dt><dd>The transcript cut into three equal-length thirds, whatever the annotation says; each third averaged, the three concatenated. The same 3d width as regions.</dd>
<dt>PCA-d</dt><dd>regions and split3 each reduced to their top d principal components (d = the model's hidden size, the width of bases), taken on the training rows, before the ridge.</dd>
<dt>rbal</dt><dd>The mean of the three region averages, each region weighted equally; width d. bases weights the regions by their length. rbal − bases asks whether reweighting the regions, without keeping them apart, explains the gain.</dd>
<dt>F6</dt><dd>Six scalars: the log length and the GC fraction of the 5'UTR, the CDS and the 3'UTR. Appended to both readouts, so (regions+F6) − (bases+F6) is the gain left once region length and GC are given to the probe.</dd>
<dt>Random boundaries</dt><dd>The two boundaries placed at the region proportions of another transcript drawn at random; five draws, averaged. The same width as regions.</dd>
<dt>Shifted boundaries</dt><dd>Both boundaries moved 100 nt toward the 5' end (−100) or toward the 3' end (+100).</dd>
<dt>Per-region encoding</dt><dd>Each region fed to the model on its own (three forward passes) instead of the whole transcript once, and regions taken from those outputs. Run for 3 models.</dd>
</dl>
<div class="tablewrap">{t4}</div>
<p class="caption">Table 3. Controls among our readouts, regression columns. Each cell: mean difference over models, then wins / losses / ties as in Table 1. Per-region encoding (each region through the model on its own) exists for 3 models.</p>

<h2>Result 3. Which regions the probe uses</h2>
<ul>
<li>Splitting the probe's prediction variance by block (median over 20 models): MRL 5'UTR 0.56; HL CDS 0.54 and 3'UTR 0.36; TE human CDS 0.51 and 5'UTR 0.27. This matches the known roles: the 5'UTR controls initiation, the 3'UTR and codon usage control stability, the CDS carries translation efficiency. On the sugimoto rows with all three regions present, the MRL 5'UTR share is 0.48.</li>
<li>In silico perturbation (RiNALMo, ERNIE-RNA, Orthrus). An inserted uAUG in endogenous transcripts lowers the predicted TE and MRL, and the drop lands in the 5' block (Table 4). Optimal or non-optimal codons raise or lower the predicted HL; ERNIE-RNA and Orthrus attribute this to the CDS block. ARE and let-7 sites are indistinguishable from shuffled controls (Table 5).</li>
</ul>

<h3>uAUGs: in the data and after insertion (Figure 1, Tables 4a and 4b)</h3>
<p>Each test transcript gets one AUG written into its 5'UTR by a length-preserving substitution, in one of three positions relative to the main ORF. <em>Extension</em>: in frame with the main AUG and no stop codon in between, so a ribosome starting there makes the protein with an N-terminal extension. <em>Overlap</em>: out of frame, with no stop before the main AUG, so the upstream ORF runs past the main start. <em>uORF</em>: followed by an in-frame stop inside the 5'UTR. Each comes with a strong or a weak Kozak context. The control writes the same substitution with a non-AUG triplet; removal turns AUG into AUC in the rows that carry one uAUG. The number is the mean change of the probe's prediction over the edited rows, in SD of the label: below zero, the probe predicts lower TE or MRL.</p>
<p>What a readout should do here: make the probe sensitive to elements with a known effect, in the measured direction and size, and leave it unmoved by matched neutral changes. Insensitivity to a functional element means the readout does not carry it; a response to a neutral change means the probe reads something other than the element. Two ways to ask whether a probe reads uAUGs. Without edits: take the transcripts of a table as they are, group them by the uAUG their 5'UTR carries, and compare each group's predicted label with its measured one (Figure 1a, Table 4a). With edits: write a uAUG into transcripts that had none and see how far the prediction moves (Figure 1b, Table 4b). Both are scored as the share of the measured effect the probe reproduces: 1 is all of it, 0 none.</p>
{h1a}
<p class="caption">Figure 1a. No edits. For each class of 5'UTR (one uAUG of a kind and Kozak context, or several uAUGs), the class's predicted label minus that of uAUG-free transcripts, divided by the same difference in the measured label; mean over 20 models, from held-out predictions. Right: models in which regions is closer to the measured difference than mbmean. Classes with fewer than 30 transcripts or a measured difference under 0.15 SD are left out (listed in Table 4a).</p>
<div class="howto"><h4>How to read Figure 1a</h4>
<p>Each line is one class of 5'UTR in one table; the red dot is where the mbmean probe lands, the blue dot where the regions probe lands, the dark vertical line the measured difference (1). A dot at 0.3 means the probe's predictions separate that class from uAUG-free transcripts by 30% of the measured gap. Left panel: the plain difference; right panel: the class's coefficient once region lengths and GC are in the regression, the part a uAUG explains beyond length and composition.</p>
<p>Reading: on every line the blue dot sits right of the red one, so regions reproduces more of the measured uAUG effect than mbmean, in 13–20 of the 20 models per line before adjustment and 18–20 after. After adjustment mbmean keeps 7–22% of the measured effect and regions 24–63%.</p>
<p>The short vertical marks are each probe's r² on that table (mbmean 0.39 and regions 0.50 on TE human, 0.34 and 0.43 on TE mouse, 0.09 and 0.13 on MRL): a probe that caught every part of the label alike would reproduce about r² of any group difference, so the mark is what accuracy alone accounts for. After adjustment mbmean lands well below its mark on TE human and TE mouse (0.07–0.22 against 0.34–0.39) and at it on MRL; regions comes to 0.26–0.63 against marks of 0.43–0.50 on the TE tables and passes its mark on MRL (0.24–0.37 against 0.13). Measured against its own accuracy, regions keeps more of the uAUG effect than mbmean in every table.</p></div>
{h1b}
<p class="caption">Figure 1b. With edits. For each table and model, the slope of the predicted change after inserting a uAUG on the table's measured class effects (the six kinds and Kozak contexts of Table 4b): the share of the measured effect the probe reproduces.</p>
<p>The reference is each table's own measurement (Figure 1c): transcripts whose 5'UTR carries exactly one uAUG, grouped the same way, against transcripts with none. In every table overlap and uORF uAUGs go with a lower label, a strong Kozak context more than a weak one: by 0.47–0.94 SD (strong) on TE human, 0.21–0.63 on MRL, 0.76–2.2 on the MPRA. Extension uAUGs have no effect in the MPRA; in the endogenous tables only 7–39 transcripts carry one, too few to fix their effect.</p>
<p>On the endogenous transcripts (Figure 1d, top two rows) mbmean moves by at most 0.06 SD and reproduces 0–11% of the measured effect (slope of predicted on measured, Table 4). regions drops by up to 0.04–0.26 SD at a strong Kozak context, more than at a weak one, reproduces 7–36% of the measured effect, and carries 74–104% of its change in the 5'UTR block. The controls stay within 0.02 SD of zero and removal raises the regions prediction by 0.04–0.12 SD. regions drops about as much for extension uAUGs as for overlap and uORF ones, where the MPRA puts the extension effect at zero: the probe reads an AUG in a strong context in the 5'UTR, not its frame. On the MPRA library (bottom row) both readouts respond, by 0.05–1.8 SD, and reproduce 36–76% (mbmean) and 44–83% (regions) of the measured effect: there the 5'UTR is the only part that varies.</p>
{f1a}
<p class="caption">Figure 1c. Measured, per table: the label of transcripts whose 5'UTR carries exactly one uAUG of each kind and Kozak context, minus that of transcripts with no uAUG, in SD of the table's label; n in brackets.</p>
<div class="howto"><h4>How to read Figure 1c</h4>
<p>No model is involved: this is the measurement the insertions are judged against, taken from each table's labels by the rules of the MPRA analysis. A 5'UTR with exactly one AUG is classed by the AUG's relation to the main ORF (extension, overlap, uORF, as defined above) and its Kozak context (strong: a purine at −3 and G at +4; weak: neither; the one-of-two "adequate" class is left out of the figure). Each dot is the class's mean label minus the mean label of the uAUG-free transcripts of the same table (5,699 on TE human, 5,039 on MRL, 143,310 in the MPRA), divided by the label's SD. The MPRA's 5'UTRs are a 25-nt constant part plus 50 random nucleotides in front of the eGFP AUG, so one AUG changes nothing else; the endogenous 5'UTRs differ in many other ways too, so their dots also carry whatever travels with a uAUG.</p>
<p>Reading: overlap and uORF fall below zero in all three tables, strong below weak. The extension dots rest on 7–39 transcripts in the endogenous tables, against 4,506–9,443 in the MPRA, where they sit at zero. A probe that reads uAUGs should repeat this shape in the same row of Figure 1d.</p></div>
{f1b}
<p class="caption">Figure 1d. Predicted change after the insertion, per table (rows; one x scale per row) and model (columns). Hover a dot for its value; Table 4 lists every number.</p>
<div class="howto expect"><h4>Expected result for Figure 1d and Table 4</h4>
<p>Each row of Figure 1d against the same row of Figure 1c. A probe that reads uAUGs should show:</p>
<ul>
<li>overlap and uORF insertions: the prediction drops, more with a strong Kozak context than with a weak one, toward the measured values of that table;</li>
<li>extension insertions: no change (the MPRA's measurement; the endogenous extension classes are too small to say);</li>
<li>control (a non-AUG triplet at the same place): no change;</li>
<li>removal (AUG → AUC): the prediction rises;</li>
<li>regions probe: the change carried by the 5'UTR block (share near 1);</li>
<li>against the table's measured six: ρ near +1 and a slope as far toward 1 as the probe's predictions allow.</li>
</ul>
<p>Observed on the endogenous tables (top two rows). mbmean meets the controls and nothing else: everything stays within 0.06 SD of zero, a slope of −0.01 to +0.11. regions meets the drops (strong below weak), the controls, the removal and the 5'UTR share, with a slope of +0.07 to +0.36; it misses extension, which drops with the others. ρ ranks the six without their size, so mbmean's near-zero values in the right order score as high as regions' (0.54–1.00 against 0.26–0.77). On the MPRA (bottom row) both readouts meet the expectations except extension in Orthrus, ρ 0.83–1.00, slope 0.36–0.76 for mbmean and 0.44–0.83 for regions.</p></div>
<div class="howto"><h4>How to read Figure 1d</h4>
<p>Each panel is one table (row) and one model (column): RiNALMo giga, ERNIE-RNA and Orthrus 6-track, the three models the perturbations were run on. Inside a panel each line is one edit; the red dot is the mbmean probe and the blue dot the regions probe, both fitted on the table's default train split. A dot's x position is the mean, over the edited test transcripts (139–224 per edit on TE human, 150–211 on MRL, 269–300 on the MPRA), of the prediction after the edit minus the prediction before it, in SD of that table's label. Left of the zero line: the edit lowers the predicted TE or MRL. The three panels of a row share one x scale, so the models can be compared along a row; the rows have their own scales, since the MPRA responses are ten times larger.</p>
<p>Reading the top two rows (endogenous transcripts): the red dots stay at zero, so the global mean barely registers an inserted AUG; the blue dots sit left of zero for the strong-context insertions, at zero for the controls and right of zero for the removals, so the regions probe registers the AUG and its context. Bottom row (MPRA): red and blue dots follow the measured order, because in that library the 5'UTR is the only part that varies and any readout carries it.</p></div>
<div class="tablewrap">{t4a}</div>
<p class="caption">Table 4a. No edits: the transcripts of each table grouped by the uAUG their 5'UTR carries (one uAUG by kind and Kozak context, or several), against uAUG-free ones. measured: the label's difference; mbmean, regions: the same difference in each probe's held-out predictions (the mean over the seeds that hold a transcript out), mean over 20 models, in SD of the label. regions closer: models in which regions is nearer the measured difference. Adjusted: the class coefficient with region lengths and GC in the regression.</p>
<div class="tablewrap">{t6}</div>
<p class="caption">Table 4b. In silico uAUG insertion into test transcripts of each table: mean change of the probe's prediction, in label SD, per uAUG kind (extension: in frame with the main ORF, no stop before it; overlap: out of frame, running into the main ORF; uORF: ending before it) and Kozak context. measured: the table's own one-uAUG 5'UTRs minus uAUG-free ones (Figure 1c), n in brackets. control: the same edits with a non-AUG triplet (mean of the six); removal: AUG → AUC where a row has one uAUG. ρ: Spearman of the six categories against the measured row; slope: predicted on measured through the origin, the share of the measured effect the probe reproduces. 5' block share: the part of the regions probe's change carried by the 5'UTR block.</p>
<h3>Codon, ARE and let-7 edits (Figure 2b, Table 5)</h3>
<p>On HL human test transcripts every codon is recoded to its most optimal, its least optimal or a random synonym, ranked by the codon stability coefficient (the correlation between a codon's frequency and half-life over the training genes). In the 3'UTR, 15 nt or more past the stop, four copies of the ARE nonamer UUAUUUAUU or of a let-7 8mer site go in, each beside a shuffled control of the same composition. Expected: optimal codons lengthen the half-life and non-optimal ones shorten it; ARE and let-7 sites shorten it relative to their shuffles.</p>
<p>Both readouts follow the codon direction in all three models: optimal +0.23 to +1.0 SD, non-optimal −0.45 to −1.26 SD. For ERNIE-RNA and Orthrus the regions probe carries 68–77% of the non-optimal change in the CDS block; RiNALMo carries it in the 3'UTR block (CDS share −0.23), the whole-transcript encoding passing the CDS edit into the 3'UTR vectors. ARE and let-7 sites differ from their shuffles by at most 0.02 SD in both readouts: neither probe registers them.</p>
{h2a}
<p class="caption">Figure 2a. The three contrasts of the HL edits, per model: optimal minus least optimal codons (a probe that reads codon optimality puts it above zero), ARE and let-7 sites minus their shuffles (below zero). The full set is Figure 2b.</p>
{f2}
<p class="caption">Figure 2b. Predicted change of HL human after each edit, per model; codon recoding and 3'UTR elements on their own x scales. Hover a dot for its value; Table 5 lists every number.</p>
<div class="howto expect"><h4>Expected result for Figure 2b and Table 5</h4>
<ul>
<li>most optimal synonyms: the predicted half-life rises; least optimal: it falls; random synonyms: it falls a little, since a random synonym is on average less optimal than the one in place;</li>
<li>ARE and let-7 sites: the prediction falls below that of the shuffled control;</li>
<li>regions probe: the codon change carried by the CDS block, the ARE and let-7 change by the 3'UTR block.</li>
</ul>
<p>Observed: the codon directions hold in both readouts and all three models. The CDS block carries the codon change for ERNIE-RNA and Orthrus (68–77%) and not for RiNALMo (3'UTR block). Neither readout puts the ARE or the let-7 site below its shuffled control (within 0.02 SD).</p></div>
<div class="howto"><h4>How to read Figure 2b</h4>
<p>Columns are the three models; the top row is codon recoding of the whole CDS, the bottom row the 3'UTR insertions, each row on its own x scale. A dot is the mean change of the predicted half-life over the edited HL human test transcripts (300 for codon recoding, 271–294 for the 3'UTR edits), in SD of the label; red = mbmean, blue = regions.</p>
<p>Reading the top row: right of zero for the most optimal synonyms, left for the least optimal and the random ones, in both readouts and all three models, which is the direction measured codon optimality predicts. Bottom row: an element and its shuffled control sit next to each other; a probe that saw the ARE or the let-7 site would put the element left of its control. The pairs land within 0.02 SD of each other, so neither readout separates the sites from the shuffles.</p></div>
<div class="tablewrap">{t7}</div>
<p class="caption">Table 5. In silico edits on HL human test transcripts: every codon recoded to its most optimal, least optimal or a random synonym (codon stability coefficients from the training genes), four copies of the ARE nonamer UUAUUUAUU or of a let-7 8mer site in the 3'UTR, each beside a shuffled control. Mean change of the predicted half-life, in label SD. CDS share: the part of the regions probe's change for non-optimal recoding carried by the CDS block.</p>

<h2>Result 4. Natural sequence elements: which probe tracks the truth</h2>
<p>The perturbations write elements into transcripts. Here the transcripts are left as they are, and the test rows of each seed are split by an element they already carry: 5'UTRs with an out-of-frame uAUG (expected to lower TE and MRL) or with in-frame uAUGs only, against uAUG-free 5'UTRs; the top against the bottom quarter of CDS codon optimality, measured two ways: the mean codon stability coefficient (CSC, the scale the codon recoding of Table 5 uses) and CAI (standard human codon usage, Kazusa; both expected higher); 3'UTRs with three or more AUUUA against none (expected lower half-life); 3'UTRs with a let-7 or a miR-16 site (TargetScan 8mer, 7mer-m8 or 7mer-A1) against none (expected lower). For each group pair, three differences on the same rows, in SD of the label: the measured label (truth), the mbmean probe's prediction, the regions probe's prediction. The adjusted version takes the element's coefficient in a regression with the log length and GC of the three regions, since uAUG-carrying 5'UTRs are longer and CAI follows GC. Per model, regions is closer when its distance to the truth is below mbmean's with Wilcoxon p &lt; 0.05 over the ten seeds, and the other way for mbmean.</p>
<ul>
<li>5'UTR. For out-of-frame uAUGs the truth is −0.41 to −0.82 SD on the endogenous tables and −1.4 to −1.6 SD in the four MPRA libraries. mbmean recovers −0.06 to −0.29 and −0.56 to −0.70, regions −0.13 to −0.54 and −0.71 to −0.86, and regions is closer in every model on all seven tables (19/0/1 for MRL before adjustment). In-frame uAUGs give the same picture on the endogenous tables; in the MPRA, where they are milder (−0.33 to −0.42 SD), regions predicts more repression than measured (107–129% against mbmean's 90–105%) and mbmean is closer in 10–13 of the 20 models.</li>
<li>CDS. Codon optimality by CSC has the strongest measured effects of the six elements: +0.24 to +0.93 SD after adjustment, on every table. Both probes are sensitive to it: after adjustment mbmean reproduces 59–74% of the effect and regions 70–99%, well above either probe's r². regions is closer in 9–11 of the 20 models on the HL and TE tables (3 on MRL), mbmean in at most one, the rest tie. CAI is a weaker measure of the same thing (on TE human its unadjusted difference has the wrong sign); there the two probes follow it to a similar degree. The CSC comes from the half-lives of the HL human training genes, so on HL human its grouping carries a little of the label itself.</li>
<li>3'UTR. Both probes recover most of the AUUUA difference (truth −0.53 to −0.56; mbmean −0.43 to −0.44; regions −0.45 to −0.46). After adjustment the truth falls to −0.23 to −0.26 and mbmean is closer on HL mouse (17 models). The let-7 and miR-16 differences shrink to −0.01 to −0.10 after adjustment, and neither probe follows them; mbmean is closer for let-7 in 8–11 models.</li>
</ul>
<p>Read as sensitivity: for uAUGs regions makes the probe sensitive where mbmean leaves it nearly blind; for codon optimality both probes are sensitive and regions somewhat more; for AUUUA and miRNA sites both probes follow the difference that region length and GC carry and not the element (after adjustment, and in the insertions of Table 5, neither responds to the site). The 5'UTR, where regions gains most, holds 5–10% of a transcript's nucleotides; the CDS and the 3'UTR hold most of the rest.</p>
{h3a}
<p class="caption">Figure 3a. Share of the element's measured difference reproduced by the two probes, mean over 20 models; right: models in which regions is closer (Wilcoxon over the ten seeds). Lines whose measured difference is under 0.05 SD are not drawn. The full numbers are Figure 3b and Table 8.</p>
{f3}
<p class="caption">Figure 3b. Element group minus reference group, mean over 20 models: truth, mbmean and regions per table, before (left) and after (right) adjusting for region length and GC. Hover a dot for its value.</p>
<div class="howto expect"><h4>Expected result for Figure 3b and Table 8</h4>
<p>The truth should carry the known signs: an out-of-frame uAUG lowers TE and MRL; a higher CAI raises TE and half-life; AUUUA and miRNA sites lower half-life. A probe that has learned an element should reproduce its truth dot, before and after adjustment. Observed: the truth carries the expected sign for out-of-frame uAUGs (−0.41 to −0.82 SD), codon optimality by CSC (+0.23 to +0.93), AUUUA (−0.53 to −0.56) and the miRNA sites (−0.21 to −0.30), and for CAI on four of five tables (before adjustment TE human is −0.14, after adjustment MRL is −0.03). For out-of-frame uAUGs regions is closer to the truth than mbmean in 19 or 20 of the 20 models on every table, before and after adjustment. For CSC both probes reproduce most of the effect, regions closer in about half the models and tied in the rest. For CAI, AUUUA and the miRNA sites the two probes land at about the same place, short of the truth.</p></div>
<div class="howto"><h4>How to read Figure 3b</h4>
<p>Each row block is one element, each line one table. Three dots per line: the dark dot is the truth, the difference of the measured label between the element group and its reference group; the red and the blue dot are the same difference in the mbmean and the regions probe's predictions. All three come from the same test rows: per model and seed the difference is taken on that seed's test rows and divided by the label's SD there, then averaged over the ten seeds and over the 20 models. The left column is the plain difference; the right column is the element's coefficient after the log length and GC of the three regions enter the regression, which removes what length and composition explain.</p>
<p>Figure 3a shows the same numbers as shares of the truth, with each probe's r² on the line's table as a short mark: what accuracy alone would reproduce. On the uAUG lines regions sits right of mbmean; on the CAI, AUUUA and miRNA lines the two sit together, at or above their marks, so the probes carry these elements as far as their accuracy goes, and regions' extra width adds nothing there.</p>
<p>Reading: the closer a probe's dot to the dark dot, the more of the element's real effect the probe's predictions carry. In the uAUG blocks the blue dots sit between the red dots and the truth on every line, before and after adjustment. In the CAI, AUUUA and miRNA blocks the red and the blue dot sit together; after adjustment both stay short of the truth for AUUUA, and the miRNA truth shrinks toward zero. Table 8 gives the numbers and the per-model verdicts.</p></div>
<div class="tablewrap">{t8}</div>
<p class="caption">Table 8. The numbers behind Figure 3b, with per-model verdicts: regions closer / mbmean closer / tie, by Wilcoxon over the ten seeds. miRNA families: let-7 and miR-16 stand in until a cell type is chosen; HL human and mouse are Saluki's half-lives combined over many datasets.</p>

<h2>What it means for the benchmark</h2>
<ul>
<li>Rankings on the regression columns stay close: Kendall τ(mbmean, regions) is 0.86 on TE human, 0.81 on TE mouse, 0.84 on HL, 0.82 on MRL and 0.80 on MRL MPRA. The gain belongs to all models, not to a few.</li>
<li>Rank changes add up to 584 places, 396 of them in the classification columns. Largest net gains: 3UTRBERT +43, RNA-FM +35, UTR-LM +34. Largest net losses: NT −49, RiNALMo −43, DNABERT-S −38, AIDO −33.</li>
</ul>
<div class="tablewrap">{t3}</div>
<p class="caption">Table 6. Rank of each model by mbmean minus its rank by regions, per column (positive = moved up); up / down / net summed over the 12 columns. Σ |Δ rank|: the column's total movement; Kendall τ between the two orders.</p>



<h2>Per-model scores</h2>
<p>Ten-seed means for every model and column, laid out as in mRNABench's Table 2, one row per readout under each model. Regression columns are test Pearson r, classification columns AUPRC.</p>
<div class="tablewrap">{t5}</div>
<p class="caption">Table 7. mbmean: mRNABench's readout and probe, as reproduced (median |difference| from their published cells 0.0009). regions (5'UTR, CDS and 3'UTR averages, concatenated), split3 (three equal thirds) and bases (mean over the whole transcript's nucleotides) are ours. The column's best model within each readout is bold: <b class="best-mbmean">mbmean</b> in black, <b class="best-regions">regions</b> in blue, <b class="best-split3">split3</b> in purple, <b class="best-bases">bases</b> in brown. The naive k-mer baselines are mRNABench's.</p>

<h2>Supplement</h2>
<ul>
<li>Learned attention pooling (RiNALMo, TE human): no region information 0.652, a region-type embedding on the keys 0.658, one query per region 0.712; ridge on regions 0.732.</li>
<li>RiNALMo with 1,022-nt windows in place of mRNABench's 8,190-nt chunks: HL +0.04–0.06, TE +0.02–0.08, MRL −0.005 to −0.037.</li>
</ul>

<div class="risk">
<h3>Questions reviewers will ask</h3>
<ol>
<li>Classification tasks show no gain. The claim covers regression tasks on translation and stability.</li>
<li>Encoding the three regions in three separate passes through the model beats whole-transcript regions in most cells for the 3 models tested, by 0.03–0.06 for RiNALMo. Reviewers will compare with mRNA-LM (one model per region, trained).</li>
<li>Novelty. Averaging per region is a simple idea. The novelty rests on the systematic measurement over 20 models, the boundary controls, the decomposition and perturbation, and the point that the benchmark's default readout underestimates the models.</li>
<li>Perturbation covers 3 models. RiNALMo puts the codon effect in the 3' block, because whole-transcript encoding carries the CDS change into the 3'UTR vectors. ARE and let-7 effects are not detected.</li>
<li>Calibration exceptions. Evo 1 on MPRA and the pair column (bf16 mean) and RNAErnie (weights repository rebuilt in May 2026) do not match the paper. Loc-LR uses the V1.3 substitute labels, not the paper's.</li>
</ol>
</div>

<footer>Tables generated by <code>story_html.py</code> from the fits that <code>final_tables.py</code> summarizes.</footer>
</main>
</body>
</html>
"""


def main():
    page = PAGE.format(css=CSS, t1=main_table(), t2=per_model_table(), t3=rank_table(), t4=controls_table(),
                       t5=scores_table(), t6=uaug_table(), t7=hl_pert_table(), f1a=fig_uaug()[0], f1b=fig_uaug()[1],
                       f2=fig_hl(), f3=fig_elements(), t8=element_table(), h1a=head_subsets(), h1b=head_insertion(),
                       h2a=head_hl(), h3a=head_elements(), t4a=subsets_table())
    OUT.write_text(page)
    print(f"wrote {OUT}: {len(page):,} bytes")


if __name__ == "__main__":
    main()
