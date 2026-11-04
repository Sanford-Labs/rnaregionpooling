"""extra_figs.py — the floats the paper's text pages carry: a teaser, the models, the tables, the rankings.

Built in story_html's style and exported the same way (draft_figs_export.py); `main()` writes the two figure
PDFs into the LaTeX project and the two tabulars beside them.
"""
import importlib.util
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from paths import CODE, PAPER, WORK

P = WORK
FIGS = PAPER / "figures"
OUT = PAPER

spec = importlib.util.spec_from_file_location("dr", CODE / "draft_render.py")
dr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dr)
sh, ft = dr.sh, dr.ft
C5, CC, C3 = dr.C5, dr.CC, dr.C3
# the readouts keep one colour each: the region-aware blue and the global-mean terracotta of the other figures
GAIN, DROP, MB, GREY = "#2e59a7", "#b35c44", "#b35c44", "#7c8896"

REG5 = ["HL", "MRL", "MRL-HL-Pair", "TE human", "TE mouse"]
SHOWN = {"MRL-HL-Pair": "MRL–HL pair"}
# the tables whose transcripts have all three regions: the MPRA libraries vary the 5' UTR of one construct
NATURAL = ["rnahl-human", "rnahl-mouse", "sugimoto", "te-human", "te-mouse"]


def models():
    """(label, directory, annotation input) for the models of the paper, in the paper's order."""
    have = [(lab, d, ann) for lab, d, _, ann in ft.MODELS
            if (ft.R / d).exists() and d not in ("helix-mrna-noE", "orthrus-6track")]
    return sorted(have, key=lambda x: sh.mn(x[0]).lower())


def width(d):
    """The model's embedding width, from the global mean it was pooled into."""
    for t in NATURAL:
        m = ft.R / d / t / "features/meta.json"
        if m.exists():
            meta = json.loads(m.read_text())
            for k in ("dim", "width", "d"):
                if k in meta:
                    return int(meta[k])
            f = ft.R / d / t / "features/mbmean.npy"
            if f.exists():
                return int(np.load(f, mmap_mode="r").shape[1])
    return None


def nucleotide_shares():
    """Median share of a transcript's nucleotides in each region, over the natural tables."""
    rows = []
    for t in NATURAL:
        f = P / "results/elements" / f"{t}.parquet"
        if not f.exists():
            continue
        e = pq.read_table(f, columns=["l5", "lc", "l3"]).to_pandas()
        tot = e[["l5", "lc", "l3"]].sum(axis=1)
        keep = tot > 0
        rows.append([float(np.median(e.loc[keep, c] / tot[keep])) for c in ("l5", "lc", "l3")])
    v = np.mean(rows, 0)
    return v / v.sum()


def prediction_shares():
    """Median share of the region probe's prediction variance per block, per column."""
    out = {}
    for c in REG5:
        v = [x for _, d, _ in models() if (x := ft.decomp_seeds(d, c)) is not None]
        s = np.array([np.median([np.mean(x[k]) for x in v]) for k in ("5'UTR", "CDS", "3'UTR")])
        out[c] = s / s.sum()
    return out


def gains():
    """Region-aware minus global mean per regression column: the mean, the wins, and the per-model values."""
    grab = lambda d, fs, pr: {c: ft.seeds(d, c, fs, pr) for c in ft.COLS}  # noqa: E731
    from scipy.stats import wilcoxon
    out = {}
    for c in REG5:
        ds = []
        for _, d, _ in models():
            r, m = grab(d, "regions", "B")[c], grab(d, "mbmean", "A")[c]
            if r is not None and m is not None:
                ds.append(r - m)
        p = [wilcoxon(x).pvalue if np.any(x != 0) else 1.0 for x in ds]
        out[c] = (float(np.mean([x.mean() for x in ds])),
                  sum(x.mean() > 0 and q < 0.05 for x, q in zip(ds, p)), len(ds),
                  np.array([x.mean() for x in ds]))
    return out


def violin(v, x0, cy, w, h, col):
    """One horizontal violin: a Gaussian kernel density of `v` mirrored about `cy`, with its quartile box."""
    lo, hi = v.min(), v.max()
    pad = (hi - lo) * 0.25 or 0.01
    grid = np.linspace(lo - pad, hi + pad, 72)
    bw = 1.06 * v.std(ddof=1) * len(v) ** -0.2 or (hi - lo) / 8 or 0.01
    dens = np.exp(-0.5 * ((grid[:, None] - v[None, :]) / bw) ** 2).sum(1) / (len(v) * bw * np.sqrt(2 * np.pi))
    dens = dens / dens.max() * h / 2
    X = lambda u: x0(u)  # noqa: E731
    up = " ".join(f"{X(u):.1f},{cy - d:.1f}" for u, d in zip(grid, dens))
    down = " ".join(f"{X(u):.1f},{cy + d:.1f}" for u, d in reversed(list(zip(grid, dens))))
    q1, med, q3 = (float(np.percentile(v, q)) for q in (25, 50, 75))
    return (f'<polygon points="{up} {down}" fill="{col}" fill-opacity=".22" stroke="{col}" stroke-width="1"/>'
            f'<line x1="{X(v.min()):.1f}" x2="{X(v.max()):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" stroke="{col}" '
            f'stroke-width="1" stroke-opacity=".5"/>'
            f'<rect x="{X(q1):.1f}" y="{cy - h * 0.16:.1f}" width="{max(X(q3) - X(q1), 1):.1f}" '
            f'height="{h * 0.32:.1f}" fill="{col}" fill-opacity=".75"/>'
            f'<circle cx="{X(med):.1f}" cy="{cy:.1f}" r="{h * 0.11:.1f}" fill="#fff"/>')


def teaser():
    """The readout and what it buys, drawn for the half column it is printed in (VW units across 0.46 of the text
    block, F units of text at 9 pt)."""
    nuc, gain = nucleotide_shares(), gains()
    VW = 432
    F = TEXT_PT * VW / PRINT_PT["teaser"]
    em = 0.5 * F
    parts, x0, y = [], 2, 1.05 * F
    parts.append(f'<text class="rowtitle" x="{x0}" y="{y:.1f}">(a) The readout</text>')

    # the transcript, its three regions at their share of the nucleotides
    W, at = VW - 2 * x0, x0
    y += 1.5 * F
    bar = 1.25 * F
    for s, col, name in zip(nuc, (C5, CC, C3), ("5′ UTR", "CDS", "3′ UTR")):
        bw = s * W
        parts.append(f'<rect x="{at:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bar:.1f}" fill="{col}" '
                     f'fill-opacity=".22" stroke="{col}" stroke-width="1.5" rx="2"/>')
        if bw > 4 * em:
            parts.append(f'<text x="{at + bw / 2:.1f}" y="{y - 0.3 * F:.1f}" text-anchor="middle">{name}</text>')
            parts.append(f'<text x="{at + bw / 2:.1f}" y="{y + 0.9 * F:.1f}" text-anchor="middle" '
                         f'style="fill:{col}">{s * 100:.0f}%</text>')
        else:
            parts.append(f'<text x="{at:.1f}" y="{y + bar + 1.0 * F:.1f}">{name} {s * 100:.0f}%</text>')
        at += bw
    y += bar + 1.0 * F

    # the two readouts, one row of d values against one of 3d
    D, pitch = 9, W / 27
    for label, cols in (("global mean: d values", [MB] * D),
                        ("region-aware: 3d values", [C5] * D + [CC] * D + [C3] * D)):
        y += 1.55 * F
        parts.append(f'<text x="{x0}" y="{y:.1f}">{label}</text>')
        for c, col in enumerate(cols):                               # drawn as the transcript above them
            parts.append(f'<rect x="{x0 + c * pitch:.1f}" y="{y + 0.4 * F:.1f}" width="{pitch - 1.5:.1f}" '
                         f'height="{0.95 * F:.1f}" rx="1.5" fill="{col}" fill-opacity=".22" stroke="{col}" '
                         f'stroke-width="1.2"/>')
        y += 1.35 * F
    y += 1.4 * F
    parts.append(f'<text x="{x0}" y="{y:.1f}">same model, forward pass and probe</text>')

    # the gain on the five transcript columns
    y += 2.1 * F
    parts.append(f'<text class="rowtitle" x="{x0}" y="{y:.1f}">(b) Test r over the global mean</text>')
    L = max(len(SHOWN.get(c, c)) for c in REG5) * em + 0.6 * F
    NUM, CNT = 6.6 * em, 5.4 * em
    BW = VW - x0 - L - NUM - CNT - 1.2 * F
    allv = np.concatenate([v[3] for v in gain.values()])
    pad = (allv.max() - allv.min()) * 0.3
    lo, hi = min(0.0, allv.min() - pad), allv.max() + pad
    X = lambda u: x0 + L + (u - lo) / (hi - lo) * BW  # noqa: E731
    y += 0.5 * F
    top = y
    for c in REG5:
        mean, win, n, per = gain[c]
        row = 1.58 * F
        cy = y + row / 2
        parts.append(f'<text x="{x0 + L - 0.4 * F:.1f}" y="{cy + 0.35 * F:.1f}" text-anchor="end">'
                     f'{SHOWN.get(c, c)}</text>')
        parts.append(violin(per, X, cy, row * 0.78, row * 0.78, GAIN))
        parts.append(f'<text x="{x0 + L + BW + 0.6 * F:.1f}" y="{cy + 0.35 * F:.1f}">+{mean:.3f}</text>')
        parts.append(f'<text x="{VW - x0:.1f}" y="{cy + 0.35 * F:.1f}" text-anchor="end">{win}/{n}</text>')
        y += row
    parts.append(f'<line class="zero" x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{top:.1f}" y2="{y:.1f}"/>')
    y += 1.25 * F
    parts.append(f'<text class="tick" x="{x0 + L + BW / 2:.1f}" y="{y:.1f}" text-anchor="middle">'
                 f'test r, region-aware − global mean</text>')
    y += 1.25 * F
    parts.append(f'<text class="tick" x="{x0:.1f}" y="{y:.1f}">one model each; box: quartiles; '
                 f'dot: median</text>')
    h = y + 0.6 * F
    return (f'<svg class="fig" viewBox="0 0 {VW} {h:.0f}" role="img" '
            f'aria-label="the readout and the gain">{"".join(parts)}</svg>')


def ranks():
    """How far each model moves in a column's ranking when the readout changes."""
    from scipy.stats import kendalltau
    grab = lambda d, fs, pr: {c: ft.seeds(d, c, fs, pr) for c in ft.COLS}  # noqa: E731
    cols = [c for c in ft.COLS]
    shift, tau = {}, {}
    for c in cols:
        a, b, names = [], [], []
        for lab, d, _ in models():
            x, y = grab(d, "mbmean", "A")[c], grab(d, "regions", "B")[c]
            if x is None or y is None:
                continue
            names.append(sh.mn(lab))
            a.append(x.mean())
            b.append(y.mean())
        lo = {n: i for i, n in enumerate(sorted(names, key=lambda n: -a[names.index(n)]))}
        hi = {n: i for i, n in enumerate(sorted(names, key=lambda n: -b[names.index(n)]))}
        shift[c] = {n: lo[n] - hi[n] for n in names}
        tau[c] = kendalltau([lo[n] for n in names], [hi[n] for n in names]).statistic
    rows = sorted({n for s in shift.values() for n in s},
                  key=lambda n: -np.mean([s[n] for s in shift.values() if n in s]))
    biggest = max(abs(v) for s in shift.values() for v in s.values())

    L, CW, RH, TOP = 152, 33, 17, 80
    W = L + len(cols) * CW
    parts = []
    for ci, c in enumerate(cols):
        x = L + ci * CW + CW / 2
        parts.append(f'<text class="tick" x="{x - CW / 2 + 2}" y="{TOP - 7}" '
                     f'transform="rotate(-45 {x - CW / 2 + 2} {TOP - 7})">{SHOWN.get(c, c)}</text>')
    for ri, n in enumerate(rows):
        y = TOP + ri * RH
        parts.append(f'<text class="tick" x="{L - 6}" y="{y + RH - 4}" text-anchor="end">{n}</text>')
        for ci, c in enumerate(cols):
            v = shift[c].get(n)
            x = L + ci * CW
            if v is None:
                continue
            col = GAIN if v > 0 else (DROP if v < 0 else "#d7dce2")
            o = 0.14 + 0.86 * abs(v) / biggest if v else 0.25
            parts.append(f'<rect x="{x + 1}" y="{y + 1}" width="{CW - 2}" height="{RH - 2}" rx="2" fill="{col}" '
                         f'fill-opacity="{o:.2f}"><title>{n} · {c}: {v:+d}</title></rect>')
            if abs(v) >= 3:
                parts.append(f'<text class="tick" x="{x + CW / 2}" y="{y + RH - 4}" text-anchor="middle" '
                             f'fill="#fff">{v:+d}</text>')
    y = TOP + len(rows) * RH
    parts.append(f'<text class="tick" x="{L - 6}" y="{y + 12}" text-anchor="end">Kendall $\tau$</text>'.replace("$\tau$", "τ"))
    for ci, c in enumerate(cols):
        parts.append(f'<text class="tick" x="{L + ci * CW + CW / 2}" y="{y + 12}" text-anchor="middle">{tau[c]:.2f}</text>')
    parts.append(f'<text class="tick" x="{L}" y="{y + 30}">blue: the region readout places the model higher; '
                 f'red: lower</text>')
    return (f'<svg class="fig" viewBox="0 0 {W + 76} {y + 44}" role="img" '
            f'aria-label="rank change per column">{"".join(parts)}</svg>')


def models_table():
    """One row per model: its group, its width, and whether the model is given the region annotation."""
    group = {}
    for lab, d, ann in models():
        g = ("mRNA, annotated" if ann else
             "RNA, one region" if d in ("utrlm-te_el", "utrbert-6mer") else
             "DNA" if d in ("dnabert2", "dnaberts", "nt-2.5b-multi", "hyenadna-medium-450k",
                            "evo1-8k-base", "evo2-7b-base") else ("mRNA" if d in ("orthrus-base4", "helixnano") else "RNA"))
        group[sh.mn(lab)] = (g, width(d))
    rows = ['<table><thead><tr><th>Model</th><th>Kind</th><th>Hidden size d</th>'
            '<th>Model</th><th>Kind</th><th>Hidden size d</th></tr></thead><tbody>']
    items = sorted(group.items())
    half = (len(items) + 1) // 2
    for left, right in zip(items[:half], items[half:] + [("", ("", None))] * (half - len(items[half:]))):
        cells = []
        for name, (g, w) in (left, right):
            cells += [name, g, f"{w:,}" if w else ""]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return "".join(rows) + "</tbody></table>"


# the names mRNABench's catalog gives the datasets our short keys stand for
CATALOG = {"sugimoto": "mrl-sugimoto", "mrl-designed": "mrl-sample-designed", "mrl-egfp": "mrl-sample-egfp",
           "mrl-mcherry": "mrl-sample-mcherry", "mrl-varying": "mrl-sample-varying", "mrl-hl": "mrl-hl-lbkwk",
           "te-human": "translation-efficiency-human", "te-mouse": "translation-efficiency-mouse"}


def tables_table():
    """One row per column of the paper: the benchmark datasets behind it, their size and their target."""
    kind = {"HL": "half-life", "MRL": "ribosome load", "MRL MPRA": "ribosome load",
            "MRL-HL-Pair": "ribosome load, half-life", "TE human": "translation efficiency",
            "TE mouse": "translation efficiency"}
    rows = ['<table><thead><tr><th>Column</th><th>Benchmark table</th><th>Transcripts</th>'
            '<th>Measured property</th><th>Score</th></tr></thead><tbody>']
    for c in ("HL", "MRL", "MRL MPRA", "MRL-HL-Pair", "TE human", "TE mouse"):
        names, n = sorted({table for table, _ in ft.REG[c]}), 0   # a dataset once, whatever its targets
        for table in names:
            f = P / "results/elements" / f"{table}.parquet"
            if f.exists():
                n += pq.read_metadata(f).num_rows
        rows.append(f"<tr><td>{SHOWN.get(c, c)}</td><td>{', '.join(CATALOG.get(n, n) for n in names)}</td>"
                    f"<td>{n:,}</td><td>{kind[c]}</td><td>Pearson r</td></tr>")
    cls = [("VEP", "vep-traitgym-complex, vep-traitgym-mendelian", "variant effect"),
           ("eCLIP", "eclip-binding-hepg2, eclip-binding-k562", "protein binding"),
           ("Loc-LR, Loc-SR", "rna-lifecycle-ietswaart, rna-loc-fazal", "localization"),
           ("Prot-Loc", "prot-loc", "protein localization"),
           ("GO", "go-bp, go-cc, go-mf", "gene ontology")]
    for name, tables, what in cls:
        rows.append(f"<tr><td>{name}</td><td>{tables}</td><td></td><td>{what}</td><td>AUPRC</td></tr>")
    return "".join(rows) + "</tbody></table>"


# the printed size of every figure's text, and the width each figure is printed at (the text block, or the
# introduction's half column)
TEXT_PT, TITLE_PT = 9.0, 10.0
PRINT_PT = {"teaser": 0.46 * 468}


def sized(frag, width_pt):
    """Each SVG's text scaled to its own viewBox, so that it prints at TEXT_PT whatever the drawing's width."""
    def one(m):
        vb = float(m.group(2))
        return f'{m.group(1)} style="font-size:{TEXT_PT * vb / width_pt:.2f}px"'
    return re.sub(r'(<svg class="fig[^"]*" viewBox="0 0 ([0-9.]+) [0-9.]+")', one, frag)


def export(figures):
    """Each figure to its own PDF, through the same headless Chromium the other figures use."""
    FIGS.mkdir(parents=True, exist_ok=True)
    html_px = TEXT_PT / (468 / 750) * (1000 / 750)         # HTML text on the 1000 px page, printed at TEXT_PT
    with tempfile.TemporaryDirectory() as td:
        for name, frag in figures.items():
            frag = sized(frag, PRINT_PT.get(name, 468))
            page = (f'<!doctype html><html><head><meta charset="utf-8"><link href="https://fonts.googleapis.com/css2?'
                    f'family=Tinos:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">'
                    f'<style>{dr.light_css()}{dr.PAPER_CSS} :root{{--paper:#fff;--gao:#fff}} body{{background:#fff}} '
                    f'#fig{{width:1000px;padding:6px;background:#fff}} '
                    f'svg.fig text, .legend, .paneltag {{font-family: Tinos, "Liberation Serif", "Times New Roman", serif}} '
                    f'svg.fig text {{font-size: 1em; fill: #2b3038}} '
                    f'svg.fig text.rowtitle, svg.fig text.coltitle {{font-size: {TITLE_PT / TEXT_PT:.3f}em; fill: #14171d}} '
                    f'.legend, .paneltag {{font-size: {html_px:.1f}px; color: #14171d}} '
                    f'.paneltag {{font-weight: 700}} .legend {{flex-wrap: wrap; row-gap: 4px}} .legend span {{white-space: nowrap}}</style></head><body><div id="fig">{frag}</div></body></html>')
            Path(td, f"{name}.html").write_text(page)
        subprocess.run([__import__("sys").executable, str(dr.EXPORT), td, str(FIGS)], check=True)


def main():
    export({"teaser": teaser(), "ranks": ranks()})
    for name, build in (("models", models_table), ("tables", tables_table)):
        (OUT / f"{name}.tex").write_text(dr.table_tex(build()))
        print(f"{name}.tex written")
    for name in ("teaser", "ranks"):
        print(f"figures/{name}.pdf: {(FIGS / f'{name}.pdf').stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
