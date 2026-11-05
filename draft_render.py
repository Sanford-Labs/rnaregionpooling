"""draft_render.py — the paper's figures and tables, and a plain-text draft (PAPER/regions-paper-draft.txt)
rendered as an HTML page to read and a LaTeX bundle (tables as tabular, figures as PDF) into PAPER/draft. Light
colours only: the page is a paper.

The text's shape: a title line; a note paragraph; "ABSTRACT"; numbered section headings ("1. INTRODUCTION",
"2.1 Foundation models ..."), "SUPPLEMENTARY MATERIAL"; paragraphs separated by blank lines; a block indented by four
spaces is a display formula; list items "  (n) ..."; a block "[[FIGURE id]]" or "[[TABLE id]]" followed by its
caption; "REFERENCES ..." with entries whose label is separated from the text by two or more spaces. Citations are
bracket groups of labels separated by ";", a label may carry a nested "[verify]"; a bracket naming no label is a note.

Figures and tables come from story_html.py's builders (the story page's numbers), plus the schematic, the region
decomposition and the attention-pooling table built here.
"""
import html
import importlib.util
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from paths import CODE, PAPER, WORK

SRC = PAPER / "regions-paper-draft.txt"
DOCS = PAPER / "draft"
NAME = "regions-paper-draft"
P = WORK
spec = importlib.util.spec_from_file_location("sh", CODE / "story_html.py")
sh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sh)
sh.PAPER = True  # the paper's names for readouts, models and labels
ft = sh.ft
EXPORT = CODE / "draft_figs_export.py"
PDFLATEX = next((Path(c) for c in (__import__("shutil").which("pdflatex"), Path.home() / ".TinyTeX/bin/x86_64-linux/pdflatex") if c and Path(c).exists()),
                Path.home() / ".TinyTeX/bin/x86_64-linux/pdflatex")
# region colours: purple, green and amber, checked for colour-vision separation on the light surface #FBF7EE
MB, GREY = "#b35c44", "#7c8896"   # the global mean and the model output
C5, CC, C3 = "#801dae", "#2a6e3f", "#a8740e"


# ---------------------------------------------------------------------------------------------------------- parsing

def parse(text):
    lines = text.split("\n")
    doc = {"title": lines[0].strip(), "note": "", "abstract": [], "sections": [], "refs": {}}
    blocks, cur = [], []
    for ln in lines[1:]:
        if ln.strip():
            cur.append(ln)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    where = "note"
    for b in blocks:
        head = b[0].strip()
        if head == "ABSTRACT":
            where, b = "abstract", b[1:]
            if not b:
                continue
        m1 = re.match(r"^(\d+)\.\s+([A-Z][A-Z ]+)$", head)
        m2 = re.match(r"^(\d+\.\d+)\s+(.+)$", head)
        mf = re.match(r"^\[\[(FIGURE|TABLE) ([\w-]+)\]\]$", head)
        if head.startswith("REFERENCES"):
            where, b = "refs", b[1:]
        elif head == "SUPPLEMENTARY MATERIAL":
            doc["sections"].append({"level": 1, "num": "", "title": "Supplementary material", "body": [], "supp": True})
            where = "body"
            continue
        elif m1:
            doc["sections"].append({"level": 1, "num": m1.group(1), "title": m1.group(2).title(), "body": []})
            where, b = "body", b[1:]
        elif m2 and len(b) == 1:
            doc["sections"].append({"level": 2, "num": m2.group(1), "title": m2.group(2), "body": []})
            where = "body"
            continue
        elif mf:
            doc["sections"][-1]["body"].append((mf.group(1).lower(), mf.group(2), " ".join(x.strip() for x in b[1:])))
            continue
        if not b:
            continue
        if where == "note":
            doc["note"] = " ".join(x.strip() for x in b)
        elif where == "abstract":
            doc["abstract"].append(" ".join(x.strip() for x in b))
        elif where == "refs":
            label = None
            for ln in b:
                if ln and not ln.startswith(" "):
                    parts = re.split(r"\s{2,}", ln.strip(), maxsplit=1)
                    label = parts[0]
                    doc["refs"][label] = parts[1] if len(parts) > 1 else ""
                elif label:
                    doc["refs"][label] += " " + ln.strip()
        elif all(ln.startswith("    ") for ln in b):
            doc["sections"][-1]["body"].append(("display", " ".join(x.strip() for x in b), ""))
        elif re.match(r"^\s*\(\d+\)", b[0]):
            items, cur_item = [], []
            for ln in b:
                if re.match(r"^\s*\(\d+\)", ln):
                    if cur_item:
                        items.append(" ".join(cur_item))
                    cur_item = [re.sub(r"^\s*\(\d+\)\s*", "", ln)]
                else:
                    cur_item.append(ln.strip())
            items.append(" ".join(cur_item))
            doc["sections"][-1]["body"].append(("list", items, ""))
        else:
            doc["sections"][-1]["body"].append(("p", " ".join(x.strip() for x in b), ""))
    return doc


def groups(s):
    out, depth, start = [], 0, None
    for i, ch in enumerate(s):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]" and depth:
            depth -= 1
            if depth == 0:
                out.append((start, i + 1, s[start + 1:i]))
    return out


def resolve(inner, labels):
    res = []
    for item in inner.split(";"):
        raw = item.strip()
        verify = "[verify" in raw
        clean = re.sub(r"\[[^\]]*\]", "", raw).strip(" ,")
        hit = max((lab for lab in labels if clean == lab or clean.startswith(lab + ",") or clean.startswith(lab + " ")),
                  key=len, default=None)
        res.append((hit, verify, raw))
    return res


def texts_of(doc):
    out = list(doc["abstract"])
    for sec in doc["sections"]:
        for kind, x, cap in sec["body"]:
            out += [x] if kind == "p" else x if kind == "list" else [cap] if kind in ("figure", "table") else []
    return out


def number_refs(doc):
    labels, order = list(doc["refs"]), []
    for t in texts_of(doc):
        for _, _, inner in groups(t):
            for hit, _, _ in resolve(inner, labels):
                if hit and hit not in order:
                    order.append(hit)
    order += [lab for lab in labels if lab not in order]
    return {lab: i + 1 for i, lab in enumerate(order)}, order


# ---------------------------------------------------------------------------------------------------- new figures

def schematic():
    """Figure: the same transcript pooled globally (left) and by region (right), drawn in print units."""
    WIDTH, F = 936, 18                                               # 18 of 936 units prints at 9 pt
    half = WIDTH / 2
    W, segs = half - 5 * F, ((0.07, C5, "5′ UTR"), (0.53, CC, "CDS"), (0.40, C3, "3′ UTR"))
    D, cell = 8, 0.78 * F                                            # d values, one cell each
    vec = D * cell
    parts = []
    for side, (x0, title) in enumerate(((2 * F, "Global mean"), (half + 2 * F, "Region-aware"))):
        parts.append(f'<text class="rowtitle" x="{x0}" y="{1.1 * F:.1f}">{title}</text>')
        x, bounds = x0, []
        for frac, col, lab in segs:
            w = frac * W
            parts.append(f'<rect x="{x:.1f}" y="{2.9 * F:.1f}" width="{w:.1f}" height="{1.15 * F:.1f}" fill="{col}" '
                         f'fill-opacity=".22" stroke="{col}" stroke-width="1.5"/>')
            parts.append(f'<text class="tick" x="{x + (w / 2 if w > 4 * F else 0):.1f}" y="{2.6 * F:.1f}" '
                         f'text-anchor="{"middle" if w > 4 * F else "start"}">{lab}</text>')
            bounds.append((x, x + w, col))
            x += w
        y = 4.6 * F
        for i in range(0, int(W), int(0.33 * F)):                    # one vector per nucleotide
            parts.append(f'<line x1="{x0 + i + 1:.1f}" x2="{x0 + i + 1:.1f}" y1="{y:.1f}" y2="{y + 1.0 * F:.1f}" '
                         f'stroke="{GREY}" stroke-width="1"/>')
        top = y + 1.35 * F
        vy, vh = top + 1.8 * F, 1.0 * F
        if side == 0:
            vx = x0 + (W - vec) / 2
            parts.append(f'<polygon points="{x0},{top:.1f} {x0 + W:.1f},{top:.1f} {vx + vec:.1f},{vy - 0.1 * F:.1f} '
                         f'{vx:.1f},{vy - 0.1 * F:.1f}" fill="{MB}" fill-opacity=".10"/>')
            for c in range(D):
                parts.append(f'<rect x="{vx + c * cell:.1f}" y="{vy:.1f}" width="{cell - 2:.1f}" height="{vh:.1f}" '
                             f'fill="{MB}" fill-opacity=".22" stroke="{MB}" stroke-width="1.2" rx="1.5"/>')
            after = vy + vh + 0.2 * F
        else:
            # the three means side by side in one row, equal widths, each in its region's colour
            wide = 3 * vec
            vx = x0 + (W - wide) / 2
            for k, (a, b, col) in enumerate(bounds):
                gx = vx + k * vec
                parts.append(f'<polygon points="{a:.1f},{top:.1f} {b:.1f},{top:.1f} {gx + vec:.1f},{vy - 0.1 * F:.1f} '
                             f'{gx:.1f},{vy - 0.1 * F:.1f}" fill="{col}" fill-opacity=".10"/>')
                for c in range(D):
                    parts.append(f'<rect x="{gx + c * cell:.1f}" y="{vy:.1f}" width="{cell - 2:.1f}" '
                                 f'height="{vh:.1f}" fill="{col}" fill-opacity=".22" stroke="{col}" '
                                 f'stroke-width="1.2" rx="1.5"/>')
            after = vy + vh + 0.2 * F
        bottom = after
        px, by = x0 + W / 2, bottom + 0.35 * F
        parts.append(f'<line x1="{px:.1f}" x2="{px:.1f}" y1="{by:.1f}" y2="{by + 1.3 * F:.1f}" stroke="{GREY}" '
                     f'stroke-width="1.5" marker-end="url(#arr)"/>')
        label = "same linear probe → ŷ"
        bw = (len(label) + 2) * 0.52 * F
        parts.append(f'<rect x="{px - bw / 2:.1f}" y="{by + 1.6 * F:.1f}" width="{bw:.1f}" height="{1.7 * F:.1f}" '
                     f'rx="4" fill="none" stroke="{GREY}"/>')
        parts.append(f'<text x="{px:.1f}" y="{by + 2.75 * F:.1f}" text-anchor="middle">{label}</text>')
        height = by + 3.45 * F
    defs = ('<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{GREY}"/></marker></defs>')
    return (f'<svg class="fig" viewBox="0 0 {WIDTH} {height:.0f}" role="img" '
            f'aria-label="global versus region-aware pooling">{defs}{"".join(parts)}</svg>')


def decomposition():
    """Figure: the share of the region probe's prediction variance each segment carries, per regression column,
    the median over the 20 models with the interquartile range."""
    have = [(lab, d) for lab, d, _, _ in ft.MODELS if (ft.R / d).exists() and d not in ("helix-mrna-noE", "orthrus-6track")]
    cols = ["HL", "MRL", "MRL-HL-Pair", "TE human", "TE mouse"]
    names = {"MRL-HL-Pair": "MRL–HL pair"}
    share = {}
    for c in cols:
        v = [x for _, d in have if (x := ft.decomp_seeds(d, c)) is not None]
        share[c] = {k: np.array([np.mean(x[k]) for x in v]) for k in ("5'UTR", "CDS", "3'UTR")}

    WIDTH, F = 936, 18                                               # 18 of 936 units prints at 9 pt
    em = 0.52 * F
    L = max(len(names.get(c, c)) for c in cols) * em + F
    TOP, BH, GAP = 1.6 * F, 0.95 * F, 0.9 * F
    W = WIDTH - L - 3 * F
    top = max(float(np.percentile(v, 75)) for s in share.values() for v in s.values())
    ticks = [round(0.1 * i, 2) for i in range(0, int(top * 10) + 2)]
    X = lambda v: L + v / ticks[-1] * W  # noqa: E731
    rows = len(cols) * (3 * BH + 2 + GAP) - GAP
    parts = []
    for tk in ticks:
        parts.append(f'<line class="{"zero" if tk == 0 else "grid"}" x1="{X(tk):.1f}" x2="{X(tk):.1f}" '
                     f'y1="{TOP - 0.4 * F:.1f}" y2="{TOP + rows + 0.3 * F:.1f}"/>')
        parts.append(f'<text class="tick" x="{X(tk):.1f}" y="{TOP + rows + 1.4 * F:.1f}" '
                     f'text-anchor="middle">{tk:g}</text>')
    parts.append(f'<text class="tick" x="{L + W / 2:.1f}" y="{TOP + rows + 2.8 * F:.1f}" text-anchor="middle">'
                 f'share of the prediction variance</text>')
    y = TOP
    for c in cols:
        parts.append(f'<text class="item" x="{L - 0.6 * F:.1f}" y="{y + 1.5 * BH + 0.35 * F:.1f}" '
                     f'text-anchor="end">{names.get(c, c)}</text>')
        for k, (key, col) in enumerate((("5'UTR", C5), ("CDS", CC), ("3'UTR", C3))):
            v = share[c][key]
            med, q1, q3 = (float(np.percentile(v, q)) for q in (50, 25, 75))
            yy = y + k * (BH + 1)
            cy = yy + BH / 2
            parts.append(f'<rect x="{L}" y="{yy:.1f}" width="{max(X(med) - L, 0):.1f}" height="{BH:.1f}" rx="2" '
                         f'fill="{col}" fill-opacity=".22" stroke="{col}" stroke-width="1.5"/>')
            parts.append(f'<line x1="{X(q1):.1f}" x2="{X(q3):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" stroke="{col}" '
                         f'stroke-width="1.4"/>')
            for q in (q1, q3):                                       # the whisker caps
                parts.append(f'<line x1="{X(q):.1f}" x2="{X(q):.1f}" y1="{cy - BH * 0.3:.1f}" '
                             f'y2="{cy + BH * 0.3:.1f}" stroke="{col}" stroke-width="1.4"/>')
            parts.append(f'<text class="tick" x="{X(q3) + 0.4 * F:.1f}" y="{cy + 0.35 * F:.1f}">{med:.2f}</text>')
        y += 3 * BH + 2 + GAP
    height = TOP + rows + 3.4 * F
    legend = ('<div class="legend">' + "".join(
        f'<span><svg width="14" height="14"><rect x="1" y="3" width="12" height="9" rx="2" fill="{col}" '
        f'fill-opacity=".22" stroke="{col}" stroke-width="1.5"/></svg>{lab}</span>' for lab, col in (("5′ UTR", C5), ("CDS", CC), ("3′ UTR", C3)))
        + "</div>")
    return (legend + f'<svg class="fig" viewBox="0 0 {WIDTH} {height:.0f}" role="img" '
            f'aria-label="region decomposition">{"".join(parts)}</svg>')


def attention_table():
    ap = json.loads((ft.R / "rinalmo-giga-tokens" / "te-human" / "attn_pool.json").read_text())["results"]
    ridge = ft.load("rinalmo-giga", "te-human")["batch.y"]
    rows = [("Attention pooling, no region information", ap["none"]["test"]),
            ("Attention pooling, region-type embedding on the keys", ap["regemb"]["test"]),
            ("Attention pooling, one query per region", ap["perregion"]["test"]),
            ("Ridge, nucleotide mean", ridge["bases|rawB"]["test"]), ("Ridge, equal thirds", ridge["split3|rawB"]["test"]),
            ("Ridge, region-aware", ridge["regions|rawB"]["test"])]
    # the share of the pooling weight on each region, mean over a split's test rows and then over the ten splits;
    # under the nucleotide mean, the regions' shares of the nucleotides
    d = ft.R / "rinalmo-giga-tokens" / "te-human"
    share, nt = {}, []
    for v in ("none", "regemb"):
        per = []
        for part in ("a", "b"):
            r = json.loads((d / f"attn_shares.{v}-{part}.json").read_text())["results"][v]
            per += [np.mean(s, 0) for s in r["shares"]]
            nt += [np.mean(s, 0) for s in r["nt"]] if v == "none" else []
        share[v] = np.mean(per, 0)
    share["bases"] = np.mean(nt, 0)
    keys = ["none", "regemb", None, "bases", None, None]
    out = ['<table><thead><tr><th rowspan="2">Readout</th><th rowspan="2">Test r</th><th rowspan="2">SD over splits</th>'
           '<th colspan="3">Share of the pooling weight</th></tr><tr><th>5′ UTR</th><th>CDS</th><th>3′ UTR</th></tr>'
           "</thead><tbody>"]
    for (n, v), k in zip(rows, keys):
        cells = "".join(f"<td>{x:.3f}</td>" for x in share[k]) if k else "<td></td>" * 3
        out.append(f"<tr><td>{n}</td><td>{np.mean(v):.4f}</td><td>{np.std(v):.4f}</td>{cells}</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def fig_uaug():
    # (b) shares the legend (a) opens with
    import re as _re
    insertion = _re.sub(r'^<div class="legend">.*?</div>', '', sh.head_insertion(), count=1, flags=_re.S)
    return (f'<div class="paneltag">(a) Natural transcripts grouped by uAUG class</div>{sh.head_subsets()}'
            f'<div class="paneltag">(b) After inserting a uAUG into transcripts</div>{insertion}')


FIGURES = {"schematic": schematic, "decomposition": decomposition, "uaug": fig_uaug, "elements": sh.head_elements, "hledits": sh.head_hl}
TABLES = {"main": sh.main_table, "permodel": sh.per_model_table, "controls": sh.controls_table, "ranks": sh.rank_table,
          "scores": sh.scores_table, "subsets": sh.subsets_table, "insertion": sh.uaug_table, "insertion_models": sh.uaug_models_table, "hledits": sh.hl_pert_table,
          "elements": sh.element_table, "attention": attention_table}


# ------------------------------------------------------------------------------------------------------------- HTML

def inline_html(t):
    t = html.escape(t)
    t = re.sub(r"\bmu_(\w)\b", r"μ<sub>\1</sub>", t)
    t = re.sub(r"\b([A-Za-z])_(\w+)\b", r"\1<sub>\2</sub>", t)
    t = re.sub(r"10\^(-?\d+)", lambda m: f"10<sup>{m.group(1).replace('-', '−')}</sup>", t)
    return t


def cite_html(t, nums, labels):
    out, last = [], 0
    for a, b, inner in groups(t):
        out.append(inline_html(t[last:a]))
        res = resolve(inner, labels)
        if all(h is None for h, _, _ in res):
            out.append(f'<span class="ednote">[{html.escape(inner)}]</span>')
        else:
            parts = [f'<span class="ednote">{html.escape(raw)}</span>' if h is None else
                     f'<a href="#ref-{nums[h]}" title="{html.escape(h)}">{nums[h]}</a>' + ('<sup class="vf">?</sup>' if v else "") for h, v, raw in res]
            out.append("[" + ", ".join(parts) + "]")
        last = b
    out.append(inline_html(t[last:]))
    return "".join(out)


PAPER_CSS = """
main { max-width: 52rem; }
p { text-align: justify; hyphens: auto; }
.bar { display:flex; gap:1rem; flex-wrap:wrap; font-size:.85rem; margin:.4rem 0 1.2rem; }
.bar a { border:1px solid var(--line); background:var(--paper); padding:.15rem .6rem; border-radius:3px; color: var(--accent-deep); text-decoration: none; }
.note { background:var(--warn-soft); border:1px solid var(--line); padding:.6rem 1rem; font-size:.85rem; color:var(--warn); margin:0 0 1.4rem; }
.abstract { background:var(--paper); border:1px solid var(--line); border-left:4px solid var(--c-rg); padding:.8rem 1.2rem; }
.abstract h2 { border:0; margin:.1rem 0 .3rem; font-size:1.05rem; }
ol.contrib li { margin:.35rem 0; text-align: justify; }
.display { text-align:center; font-style: italic; margin: .8rem 0; }
.ednote { background: var(--warn-soft); color: var(--warn); font-size: .85em; padding: 0 .2em; border-radius: 2px; }
sup.vf { color: var(--warn); font-size: .65em; }
ol.refs { font-size: .85rem; padding-left: 1.8rem; } ol.refs li { margin: .25rem 0; }
.vtag { color: var(--warn); background: var(--warn-soft); font-size: .8em; padding: 0 .3em; border-radius: 2px; margin-left: .3em; }
figure { margin: 1.2rem 0 1.4rem; } figcaption, .tcap { font-size: .86rem; color: var(--muted); margin: .3rem 0 0; text-align: justify; }
figcaption b, .tcap b { color: var(--ink-deep); }
.paneltag { font-size: .9rem; color: var(--ink-deep); font-weight: 600; margin: .8rem 0 0; }
svg.fig .panel { fill: none; }
"""


def light_css():
    css = sh.CSS
    i = css.index("@media (prefers-color-scheme: dark)")
    depth, j = 0, css.index("{", i)
    for k in range(j, len(css)):
        depth += css[k] == "{"
        depth -= css[k] == "}"
        if depth == 0:
            return css[:i] + css[k + 1:]
    return css


def caption_html(cap, nums, labels):
    m = re.match(r"^((?:Figure|Table) S?\d+[a-z]?\.)\s*(.*)$", cap)
    return (f"<b>{m.group(1)}</b> {cite_html(m.group(2), nums, labels)}" if m else cite_html(cap, nums, labels))


def render_html(doc, built):
    nums, order = number_refs(doc)
    labels = list(doc["refs"])
    out = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
           f"<title>{html.escape(doc['title'])}</title>",
           '<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400..700;1,400..600&display=swap" rel="stylesheet">',
           f"<style>{light_css()}{PAPER_CSS}</style></head><body><main>", f"<h1>{html.escape(doc['title'])}</h1>",
           f'<div class="bar">' + (f'<a href="{NAME}.pdf" download>PDF</a>' if (DOCS / "en" / f"{NAME}.pdf").exists() else "")
           + f'<a href="{NAME}-latex.zip" download>Download LaTeX (.tex + figure PDFs, zip)</a><a href="{NAME}.tex" download>.tex only</a>'
           f'<a href="{NAME}.txt" download>plain text</a><a href="regions-paper-story.html">story page</a></div>']
    if doc["note"]:
        out.append(f"<!-- {html.escape(doc['note'])} -->")
    out.append('<div class="abstract"><h2>Abstract</h2>' + "".join(f"<p>{cite_html(p, nums, labels)}</p>" for p in doc["abstract"]) + "</div>")
    for sec in doc["sections"]:
        tag = "h2" if sec["level"] == 1 else "h3"
        num = f"{sec['num']}{'.' if sec['level'] == 1 else ''} " if sec["num"] else ""
        out.append(f"<{tag}>{num}{html.escape(sec['title'])}</{tag}>")
        for kind, x, cap in sec["body"]:
            if kind == "p":
                out.append(f"<p>{cite_html(x, nums, labels)}</p>")
            elif kind == "display":
                out.append(f'<div class="display">{inline_html(x)}</div>')
            elif kind == "list":
                out.append('<ol class="contrib">' + "".join(f"<li>{cite_html(i, nums, labels)}</li>" for i in x) + "</ol>")
            elif kind == "figure":
                out.append(f'<figure>{built[("figure", x)]}<figcaption>{caption_html(cap, nums, labels)}</figcaption></figure>')
            elif kind == "table":
                out.append(f'<div class="tcap">{caption_html(cap, nums, labels)}</div><div class="tablewrap">{built[("table", x)]}</div>')
    out.append('<h2>References</h2><ol class="refs">')
    for lab in order:
        txt = doc["refs"][lab]
        clean = re.sub(r"\s*\[verify[^\]]*\]", "", txt)
        out.append(f'<li id="ref-{nums[lab]}">{html.escape(clean)}' + ('<span class="vtag">verify</span>' if "[verify" in txt else "") + "</li>")
    out.append("</ol></main></body></html>")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------------------ LaTeX

TEX_CHARS = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}")]
UNI = [("′", "$'$"), ("−", "$-$"), ("–", "--"), ("—", "---"), ("τ", r"$\tau$"), ("×", r"$\times$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"),
       ("≈", r"$\approx$"), ("…", r"\ldots{}"), ("Σ", r"$\Sigma$"), ("Δ", r"$\Delta$"), ("ρ", r"$\rho$"), ("²", r"$^2$"), ("→", r"$\rightarrow$"),
       ("·", r"$\cdot$"), ("ŷ", r"$\hat{y}$"), ("±", r"$\pm$"), (" > ", " $>$ "), (" < ", " $<$ "), ("|", r"$|$")]
MATH = [(r"\bmu_(\w)\b", r"$\\mu_{\1}$"), (r"\b([A-Za-z])_(\w+)\b", r"$\1_{\2}$"), (r"10\^(-?\d+)", r"$10^{\1}$")]


def esc(t):
    keep = []

    def stash(m):
        keep.append(m.group(0) if not isinstance(m, str) else m)
        return f"\x00{len(keep) - 1}\x00"

    for pat, rep in MATH:
        t = re.sub(pat, lambda m, rep=rep, pat=pat: stash_str(keep, re.sub(pat, rep, m.group(0))), t)
    for a, b in TEX_CHARS:
        t = t.replace(a, b)
    for a, b in UNI:
        t = t.replace(a, b)
    return re.sub("\x00(\\d+)\x00", lambda m: keep[int(m.group(1))], t)


def stash_str(keep, s):
    keep.append(s)
    return f"\x00{len(keep) - 1}\x00"


def key(lab):
    return re.sub(r"[^A-Za-z0-9]+", "", lab.replace(" and ", "")).lower()


def cite_tex(t, labels):
    out, last = [], 0
    for a, b, inner in groups(t):
        out.append(esc(t[last:a]))
        res = resolve(inner, labels)
        hits = [h for h, _, _ in res if h]
        if hits:
            out.append(r"~\cite{" + ",".join(key(h) for h in hits) + "}")
        out += [r" \textbf{[" + esc(raw) + "]}" for h, _, raw in res if h is None]
        last = b
    out.append(esc(t[last:]))
    return "".join(out)


def cell_tex(inner):
    s = re.sub(r"<br\s*/?>", " ", inner)
    s = re.sub(r"<sup[^>]*>.*?</sup>", "", s)
    s = re.sub(r'<b class="rank1">(.*?)</b>', lambda m: "\x05" + m.group(1) + "\x02", s)
    s = re.sub(r'<span class="rank2">(.*?)</span>', lambda m: "\x06" + m.group(1) + "\x02", s)
    s = re.sub(r'<span class="rank3">(.*?)</span>', lambda m: "\x07" + m.group(1) + "\x02", s)
    s = re.sub(r"<b[^>]*>(.*?)</b>", lambda m: "\x01" + m.group(1) + "\x02", s)
    s = re.sub(r"<small[^>]*>(.*?)</small>", lambda m: "\x03" + m.group(1) + "\x04", s)
    s = html.unescape(re.sub(r"<[^>]+>", "", s)).strip()
    s = esc(s)
    return (s.replace("\x01", r"\textbf{").replace("\x02", "}").replace("\x03", r"{\tiny ").replace("\x04", "}")
             .replace("\x05", r"\textbf{\color{rankone}").replace("\x06", r"\textbf{\color{ranktwo}")
             .replace("\x07", r"\underline{"))


SHORT_TEX = {"mRNA Loc-LR": "Loc-LR", "mRNA Loc-SR": "Loc-SR", "MRL-HL-Pair": "Pair", "MRL–HL pair": "Pair", "MRL MPRA": "MPRA",
             "TE human": "TE h", "TE mouse": "TE m", "Prot Loc": "Prot", "Global (protein)": "Protein", "Wins": "better", "Losses": "worse",
             "Ties": "tied", "Region-aware closer": "Closer", "Models better / worse / tied": "better / worse / tied", "Region-aware − global mean": "Region − global"}


def table_tex(tab_html, short=True):
    """An HTML table (rowspan / colspan) as a LaTeX tabular; with `short`, header cells take the short names of
    SHORT_TEX."""
    head = tab_html.split("<tbody>")[0] if "<tbody>" in tab_html else ""
    for a, b in SHORT_TEX.items() if short else ():
        head = head.replace(f">{a}<", f">{b}<")
    if head:
        tab_html = head + "<tbody>" + tab_html.split("<tbody>", 1)[1]
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tab_html, re.S):
        cells = []
        for tag, attrs, inner in re.findall(r"<(t[hd])([^>]*)>(.*?)</t[hd]>", tr, re.S):
            cs = int(re.search(r'colspan="(\d+)"', attrs).group(1)) if "colspan" in attrs else 1
            align = "l" if 'class="left"' in attrs else "c"
            rs = int(re.search(r'rowspan="(\d+)"', attrs).group(1)) if "rowspan" in attrs else 1
            cells.append((tag == "th", cs, rs, cell_tex(inner), align))
        rows.append(cells)
    ncol = max(sum(c[1] for c in r) for r in rows)
    taken = {}  # (row, col) -> rows still covered by a \multirow above
    lines = []
    for ri, cells in enumerate(rows):
        out, col, it = [], 0, iter(cells)
        pending = list(cells)
        while col < ncol:
            if taken.get((ri, col)):
                out.append("")
                col += 1
                continue
            if not pending:
                out.append("")
                col += 1
                continue
            th, cs, rs, txt, align = pending.pop(0)
            txt = r"\textbf{" + txt + "}" if th and not txt.startswith(r"\textbf") else txt
            if rs > 1:
                for r2 in range(ri + 1, ri + rs):
                    for c2 in range(col, col + cs):
                        taken[(r2, c2)] = True
                txt = r"\multirow{" + str(rs) + "}{*}{" + txt + "}"
            if cs > 1:
                txt = r"\multicolumn{" + str(cs) + "}{" + align + "}{" + (r"\textit{" + txt + "}" if align == "l" else txt) + "}"
            out.append(txt)
            col += cs
        lines.append(" & ".join(out) + r" \\")
    head_rows = len(re.findall(r"<tr", tab_html.split("<tbody>")[0])) if "<tbody>" in tab_html else 1
    body = "\n".join(lines[:head_rows]) + "\n\\hline\n" + "\n".join(lines[head_rows:])
    return "\\begin{tabular}{l" + "r" * (ncol - 1) + "}\n\\hline\n" + body + "\n\\hline\n\\end{tabular}"


def caption_tex(cap, labels):
    return cite_tex(re.sub(r"^(?:Figure|Table) S?\d+[a-z]?\.\s*", "", cap), labels)


CHANGE_MARKS = ['\\usepackage{xcolor}', '\\usepackage[normalem]{ulem}', '% Change marks; \\showchangesfalse prints the text with every change applied and no marks.', "%   \\cmt[who]{old}{new}  replacement: old struck through in red, new in the author's colour", '%   \\add[who]{new}       insertion       \\del[who]{old}   deletion', '%   \\cnote[who]{text}    comment in the margin', '%   \\defauthor{who}{colour}, e.g. \\defauthor{SN}{teal}; authors without one are blue', '\\newif\\ifshowchanges \\showchangestrue', '\\makeatletter', '\\colorlet{chg@default}{blue}', '\\newcommand{\\defauthor}[2]{\\colorlet{chg@#1}{#2}}', '\\newcommand{\\chg@col}[1]{\\ifcsname\\string\\color@chg@#1\\endcsname chg@#1\\else chg@default\\fi}', '\\newcommand{\\chg@tag}[1]{\\if\\relax\\detokenize{#1}\\relax\\else\\textsuperscript{\\sffamily\\tiny #1}\\fi}', '\\newcommand{\\cmt}[3][]{\\ifshowchanges{\\color{red}\\sout{#2}}\\,{\\color{\\chg@col{#1}}#3}\\chg@tag{#1}\\else#3\\fi}', '\\newcommand{\\add}[2][]{\\ifshowchanges{\\color{\\chg@col{#1}}#2}\\chg@tag{#1}\\else#2\\fi}', '\\newcommand{\\del}[2][]{\\ifshowchanges{\\color{red}\\sout{#2}}\\chg@tag{#1}\\fi}', '\\newcommand{\\cnote}[2][]{\\ifshowchanges\\marginpar{\\raggedright\\sffamily\\tiny{\\color{\\chg@col{#1}}\\if\\relax\\detokenize{#1}\\relax\\else #1: \\fi}#2}\\fi}', '\\makeatother']


def render_tex(doc, built):
    labels = list(doc["refs"])
    _, order = number_refs(doc)
    out = [r"% " + doc["note"], r"% Generated from regions-paper-draft.txt by papers/draft_render.py; edit the text file and render again.",
           r"\documentclass[runningheads]{llncs}", r"\usepackage[T1]{fontenc}", r"\usepackage[utf8]{inputenc}", r"\usepackage{lmodern}",
           r"\usepackage{amsmath,graphicx,multirow,adjustbox}", r"\usepackage{hyperref}", *CHANGE_MARKS, r"\begin{document}",
           r"\title{" + esc(doc["title"]) + "}", r"\titlerunning{Regional Signals Matter}", r"\author{Authors}",
           r"\authorrunning{Authors}", r"\institute{Institutions}", r"\maketitle", r"\begin{abstract}"]
    out += [cite_tex(p, labels) + "\n" for p in doc["abstract"]]
    out += [r"\keywords{mRNA \and foundation models \and pooling \and benchmarking \and untranslated regions}", r"\end{abstract}", ""]
    def bibliography():
        out.append(r"\begin{thebibliography}{99}")
        for lab in order:
            txt = re.sub(r"\s*\[verify[^\]]*\]", "", doc["refs"][lab])
            out.append(r"\bibitem{" + key(lab) + "} " + esc(txt) + (r" \textbf{[verify]}" if "[verify" in doc["refs"][lab] else ""))
        out.append(r"\end{thebibliography}")

    supp = False
    for sec in doc["sections"]:
        if sec.get("supp"):
            supp = True
            bibliography()
            out += [r"\clearpage", r"\section*{Supplementary material}", r"\setcounter{figure}{0}\setcounter{table}{0}",
                    r"\renewcommand{\thefigure}{S\arabic{figure}}\renewcommand{\thetable}{S\arabic{table}}"]
        else:
            out.append(("\\section{" if sec["level"] == 1 else "\\subsection{") + esc(sec["title"]) + "}")
        for kind, x, cap in sec["body"]:
            if kind == "p":
                out.append(cite_tex(x, labels) + "\n")
            elif kind == "display":
                out.append(r"\[ z = \bigl[\,\bar h_{R_5};\ \bar h_{R_C};\ \bar h_{R_3}\,\bigr], \qquad \bar h_{R} = \frac{1}{|R|}\sum_{i\in R} h_i \]" + "\n"
                           if x.startswith("z = [") else r"\[ " + esc(x) + r" \]" + "\n")
            elif kind == "list":
                out += [r"\begin{enumerate}"] + [r"\item " + cite_tex(i, labels) for i in x] + [r"\end{enumerate}" + "\n"]
            elif kind == "figure":
                out += [r"\begin{figure}[" + ("p" if supp else "!tbp") + "]", r"\centering",
                        r"\includegraphics[width=\linewidth,height=0.8\textheight,keepaspectratio]{figures/" + x + ".pdf}",
                        r"\caption{" + caption_tex(cap, labels) + "}", r"\label{fig:" + x + "}", r"\end{figure}" + ("\n\\clearpage\n" if supp else "\n")]
            elif kind == "table":
                out += [r"\begin{table}[" + ("p" if supp else "!tbp") + "]", r"\centering", r"\caption{" + caption_tex(cap, labels) + "}",
                        r"\label{tab:" + x + "}", r"\scriptsize", r"\begin{adjustbox}{max width=\linewidth}", table_tex(built[("table", x)]),
                        r"\end{adjustbox}", r"\end{table}" + ("\n\\clearpage\n" if supp else "\n")]
    if not supp:
        bibliography()
    out += [r"\end{document}", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------- figures

def export_pdfs(built, dest):
    """Each figure on its own page at its own size, light colours, through headless Chromium (draft_figs_export.py)."""
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for (kind, fid), frag in built.items():
            if kind != "figure":
                continue
            page = (f'<!doctype html><html><head><meta charset="utf-8"><link href="https://fonts.googleapis.com/css2?family=Tinos:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet"><style>{light_css()}{PAPER_CSS} :root{{--paper:#fff;--gao:#fff}} '
                    f'body{{background:#fff}} #fig{{width:1000px;padding:6px;background:#fff}} svg.fig text, .legend, .paneltag {{font-family: Tinos, "Liberation Serif", "Times New Roman", serif}} svg.fig text {{font-size: 13.5px; fill: #2b3038}} svg.fig text.rowtitle, svg.fig text.coltitle {{font-size: 15px; fill: #14171d}} .legend, .paneltag {{font-size: 14px; color: #14171d}}</style></head><body><div id="fig">{frag}</div></body></html>')
            Path(td, f"{fid}.html").write_text(page)
        subprocess.run([__import__("sys").executable, str(EXPORT), td, str(dest)], check=True)


def compile_pdf(tex, figdir):
    """pdflatex twice (citations) in a scratch directory; the PDF, or None with the log's first error printed."""
    if not PDFLATEX.exists():
        print(f"no pdflatex at {PDFLATEX}: LaTeX not compiled")
        return None
    with tempfile.TemporaryDirectory() as td:
        Path(td, f"{NAME}.tex").write_text(tex)
        (Path(td) / "figures").mkdir()
        for f in figdir.glob("*.pdf"):
            Path(td, "figures", f.name).write_bytes(f.read_bytes())
        for _ in range(2):
            r = subprocess.run([str(PDFLATEX), "-interaction=nonstopmode", "-halt-on-error", f"{NAME}.tex"], cwd=td,
                               capture_output=True, encoding="utf-8", errors="replace")
        out = Path(td, f"{NAME}.pdf")
        if r.returncode or not out.exists():
            log = Path(td, f"{NAME}.log").read_text(errors="ignore") if Path(td, f"{NAME}.log").exists() else r.stdout
            err = [ln for ln in log.splitlines() if ln.startswith("!")][:3]
            print("pdflatex failed:", " | ".join(err) or r.stdout[-400:])
            return None
        return out.read_bytes()


def main():
    doc = parse(SRC.read_text())
    built = {}
    for sec in doc["sections"]:
        for kind, x, _ in sec["body"]:
            if kind == "figure":
                built[(kind, x)] = FIGURES[x]()
            elif kind == "table":
                built[(kind, x)] = TABLES[x]()
    tex = render_tex(doc, built)
    figdir = DOCS / "en" / f"{NAME}-figures"
    export_pdfs(built, figdir)
    pdf = compile_pdf(tex, figdir)
    if pdf:
        for sub in ("en", "site"):
            (DOCS / sub / f"{NAME}.pdf").write_bytes(pdf)
    page = render_html(doc, built)
    for sub in ("en", "site"):
        d = DOCS / sub
        (d / f"{NAME}.html").write_text(page)
        (d / f"{NAME}.tex").write_text(tex)
        (d / f"{NAME}.txt").write_text(SRC.read_text())
        with zipfile.ZipFile(d / f"{NAME}-latex.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"{NAME}/{NAME}.tex", tex)
            for f in sorted(figdir.glob("*.pdf")):
                z.write(f, f"{NAME}/figures/{f.name}")
            if pdf:
                z.writestr(f"{NAME}/{NAME}.pdf", pdf)
    nums, order = number_refs(doc)
    nf = sum(1 for k in built if k[0] == "figure")
    print(f"{len(doc['sections'])} sections, {nf} figures, {len(built) - nf} tables, {len(order)} references "
          f"({sum(1 for t in doc['refs'].values() if '[verify' in t)} to verify); PDFs in {figdir}")


if __name__ == "__main__":
    main()
