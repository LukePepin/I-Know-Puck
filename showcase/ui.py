"""Shared look, data loading and page pieces for the public showcase: a dark, phone-first sports page.
Reads only showcase/data/."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"
AUTHOR = "Luke Pepin"
GITHUB = "https://github.com/LukePepin/I-Know-Puck"

# 5-slot categorical order checked on the dark background: contrast at least 6:1, and every pair stays apart under
# deuteranopia, protanopia and tritanopia (minimum CIELAB distance 13.5; 21 among the first three). Never cycled past 5.
CAT = ["#6EA8EC", "#E07B53", "#F0D27A", "#4EC69C", "#E58FC8"]
BG, SURFACE, BORDER = "#0F1216", "#171B21", "#2A3038"
INK, MUTED, FAINT, GRID = "#E8E4DA", "#9AA1AA", "#5C636D", "#232830"
ACCENT = CAT[0]
SEQ = [[0.0, "#18202B"], [0.5, "#3D6EA8"], [1.0, "#BFD9F7"]]  # brighter = more
GROUP = {"F": ("Forwards", CAT[0]), "D": ("Defense", CAT[1]), "G": ("Goalies", CAT[2])}
SERIF = '"Source Serif", Georgia, "Times New Roman", serif'

pio.templates["showcase"] = go.layout.Template(layout=dict(
    font=dict(family=SERIF, size=13, color=INK), colorway=CAT, paper_bgcolor=BG, plot_bgcolor=BG,
    xaxis=dict(showgrid=False, linecolor=FAINT, tickcolor=FAINT, ticks="outside", zeroline=False, automargin=True, title_standoff=8),
    yaxis=dict(gridcolor=GRID, linecolor=FAINT, tickcolor=FAINT, ticks="outside", zeroline=False, automargin=True, title_standoff=8),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", x=0, y=1.02, yanchor="bottom"), margin=dict(l=8, r=12, t=16, b=8),
    hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(family=SERIF, color=INK, size=13)),
))
pio.templates.default = "showcase"

CSS = f"""
<style>
[data-testid="stMainBlockContainer"] {{ padding-top: 4.4rem; padding-bottom: 5rem; max-width: 1180px; }}  /* clears the 60px app header */
@media (max-width: 640px) {{ [data-testid="stMainBlockContainer"] {{ padding: 4.1rem 1rem 4rem; }} }}
h3 {{ border-top: 2px solid {INK}; padding-top: .55rem !important; margin-top: 1.8rem !important; }}
.kp-mast {{ border-bottom: 4px double {INK}; padding-bottom: .5rem; }}
.kp-name {{ font-size: 2.7rem; font-weight: 800; line-height: 1; letter-spacing: .03em; text-transform: uppercase; }}
.kp-dek {{ color: {MUTED}; font-size: .88rem; margin-top: .4rem; line-height: 1.45; }}
.kp-dek a {{ color: {ACCENT}; }}
.kp-kicker {{ font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: {ACCENT}; }}
@media (max-width: 640px) {{ .kp-name {{ font-size: 1.9rem; }} }}
.kp-wrap {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
table.kp {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: .93rem; }}
table.kp th {{ text-align: left; font-size: .68rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: {MUTED};
  padding: .35rem .5rem; border-bottom: 2px solid {INK}; white-space: nowrap; }}
table.kp td {{ padding: .45rem .5rem; border-bottom: 1px solid {BORDER}; vertical-align: middle; }}
table.kp .n {{ text-align: right; white-space: nowrap; }}
table.kp .sub {{ display: block; color: {MUTED}; font-size: .76rem; line-height: 1.25; }}
table.kp .warn {{ color: {CAT[1]}; }}
table.kp td.cell {{ text-align: center; font-weight: 600; }}
.kp-bar {{ display: inline-block; width: 56px; height: 6px; background: {BORDER}; vertical-align: middle; margin-right: .45rem; }}
.kp-bar i {{ display: block; height: 100%; background: {ACCENT}; }}
@media (max-width: 640px) {{ table.kp {{ font-size: .86rem; }} table.kp td, table.kp th {{ padding: .4rem .3rem; }} .kp-bar {{ width: 30px; }} }}
.kp-board {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .6rem; }}
@media (max-width: 900px) {{ .kp-board {{ grid-template-columns: repeat(2, 1fr); }} }}
@media (max-width: 520px) {{ .kp-board {{ grid-template-columns: 1fr; }} }}
.kp-game {{ border: 1px solid {BORDER}; background: {SURFACE}; padding: .55rem .75rem .45rem; }}
.kp-game .row {{ display: flex; justify-content: space-between; font-size: 1.05rem; line-height: 1.5; color: {MUTED}; }}
.kp-game .row.lead {{ color: {INK}; font-weight: 700; }}
.kp-game .row b {{ font-variant-numeric: tabular-nums; }}
.kp-game .foot {{ font-size: .76rem; color: {MUTED}; border-top: 1px solid {BORDER}; margin-top: .35rem; padding-top: .3rem; }}
.kp-flow {{ display: flex; align-items: stretch; gap: .3rem; margin: .3rem 0 .9rem; }}
.kp-step {{ flex: 1 1 0; min-width: 0; border: 1px solid {BORDER}; border-top: 3px solid {ACCENT}; background: {SURFACE}; padding: .45rem .55rem; }}
.kp-step b {{ display: block; font-size: .86rem; }}
.kp-step span {{ display: block; font-size: .76rem; color: {MUTED}; line-height: 1.3; }}
.kp-arrow {{ align-self: center; color: {MUTED}; }}
@media (max-width: 760px) {{ .kp-flow {{ flex-direction: column; }} .kp-arrow {{ transform: rotate(90deg); }} }}
.kp-sys {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: .5rem; margin: .3rem 0 1rem; }}
.kp-col {{ border: 1px solid {BORDER}; border-top: 3px solid {ACCENT}; background: {SURFACE}; padding: .5rem .65rem; }}
.kp-col .t {{ font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: {ACCENT}; margin-bottom: .3rem; }}
.kp-col div.i {{ font-size: .84rem; padding: .3rem 0; border-top: 1px solid {BORDER}; line-height: 1.3; }}
.kp-col div.i span {{ display: block; color: {MUTED}; font-size: .74rem; }}
@media (max-width: 900px) {{ .kp-sys {{ grid-template-columns: 1fr 1fr; }} }}
@media (max-width: 520px) {{ .kp-sys {{ grid-template-columns: 1fr; }} }}
</style>
"""


def style() -> None:
    st.html(CSS)


def group_of(pos: str) -> str:
    return "G" if pos == "G" else ("D" if pos == "D" else "F")


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def esc(x) -> str:
    return html.escape("" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x))


@st.cache_data
def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


@st.cache_data
def js(name: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def snapshot_date() -> str:
    d = datetime.strptime(js("meta.json")["snapshot"], "%Y-%m-%d %H:%M")
    return f"{d:%b} {d.day}, {d.year}"


def masthead(pages: list) -> None:
    meta = js("meta.json")
    st.html(f'<div class="kp-mast"><div class="kp-name">I Know Puck</div><div class="kp-dek">Models and data from my {meta["teams"]}-team ESPN '
            f'fantasy hockey league · by {AUTHOR} · <a href="{GITHUB}" target="_blank">code on GitHub</a> · data as of {snapshot_date()}</div></div>')
    with st.container(horizontal=True, gap="small"):
        for p in pages:
            st.page_link(p)


def header(title: str, dek: str) -> None:
    st.markdown(f"## {title}")
    st.markdown(dek)


def fig_show(fig, height: int = 380, key: str | None = None):
    """Chart title as page text above the chart (it wraps on a phone instead of being clipped). No toolbar and no zoom:
    a swipe scrolls the page and a tap shows the values."""
    title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else None
    if title:
        st.markdown(f"**{title}**")
        fig.update_layout(title=None)
    # Streamlit swaps plotly's default template for its own when its chart module loads, so name ours explicitly
    fig.update_layout(template="showcase", paper_bgcolor=BG, plot_bgcolor=BG, font=dict(family=SERIF, color=INK),
                      height=height, margin=dict(t=16), dragmode=False)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return st.plotly_chart(fig, key=key, theme=None, config={"displayModeBar": False, "scrollZoom": False})


def note(text: str) -> None:
    with st.container(border=True):
        st.markdown(text)


def pick_one(label: str, options: list, key: str, default=None):
    """Tap-friendly single choice that always has a value."""
    default = options[0] if default is None else default
    return st.pills(label, options, default=default, required=True, key=key) or default


def table(rows: pd.DataFrame, spec: list[tuple]) -> None:
    """Compact standings-style table that fits a phone.
    spec: (column, header, fmt[, sub_column]). fmt None = text (left), a format string or a function = number (right),
    ("bar", max, fmt) = number with a small bar. sub_column adds a second, muted line under the value."""
    head = "".join(f'<th class="{"" if s[2] is None else "n"}">{esc(s[1])}</th>' for s in spec)
    body = []
    for _, r in rows.iterrows():
        cells = []
        for s in spec:
            col, fmt = s[0], s[2]
            v = r[col]
            sub = f'<span class="sub">{esc(r[s[3]])}</span>' if len(s) > 3 and s[3] and pd.notna(r[s[3]]) and str(r[s[3]]) else ""
            if fmt is None:
                cells.append(f"<td>{esc(v)}{sub}</td>")
            elif isinstance(fmt, tuple):
                _, vmax, f = fmt
                w = 0 if pd.isna(v) else max(0.0, min(1.0, float(v) / vmax))
                cells.append(f'<td class="n"><span class="kp-bar"><i style="width:{100 * w:.0f}%"></i></span>{f.format(v)}{sub}</td>')
            else:
                txt = "" if pd.isna(v) else (fmt(v) if callable(fmt) else fmt.format(v))
                cells.append(f'<td class="n">{txt}{sub}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    st.html(f'<div class="kp-wrap"><table class="kp"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def flow(steps: list[tuple[str, str]]) -> None:
    """Left-to-right pipeline on a wide screen, top-to-bottom on a phone."""
    parts = []
    for i, (label, sub) in enumerate(steps):
        if i:
            parts.append('<div class="kp-arrow">→</div>')
        parts.append(f'<div class="kp-step"><b>{i + 1}. {esc(label)}</b><span>{esc(sub)}</span></div>')
    st.html('<div class="kp-flow">' + "".join(parts) + "</div>")


def system_map(columns: list[tuple[str, list[tuple[str, str]]]]) -> None:
    cols = []
    for title, items in columns:
        cols.append(f'<div class="kp-col"><div class="t">{esc(title)}</div>'
                    + "".join(f'<div class="i">{esc(a)}<span>{esc(b)}</span></div>' for a, b in items) + "</div>")
    st.html('<div class="kp-sys">' + "".join(cols) + "</div>")


def math(*items: tuple[str, str]) -> None:
    """Short math box: (latex, one-line explanation) pairs."""
    with st.expander("The math", icon=":material/functions:"):
        for tex, text in items:
            st.latex(tex)
            st.caption(text)


def how_built(steps: list[tuple[str, str]], details: list[str]) -> None:
    """Systems overview of one page: where its data comes from and what is done to it."""
    st.markdown("### How this page is built")
    flow(steps)
    st.markdown("\n".join(f"{i + 1}. {d}" for i, d in enumerate(details)))


def definitions(terms: list[tuple[str, str]]) -> None:
    st.markdown("### Definitions")
    cols = st.columns(2)
    half = (len(terms) + 1) // 2  # fill down, so a phone's stacked columns keep the reading order
    for i, (term, meaning) in enumerate(terms):
        cols[i >= half].markdown(f"**{term}.** {meaning}")


BADGE = {"Holds": ("green", ":material/check_circle:"), "Fixed": ("blue", ":material/build:"), "Partly holds": ("orange", ":material/adjust:"),
         "Roughly holds": ("orange", ":material/adjust:"), "Does not hold": ("red", ":material/cancel:"), "Not supported": ("red", ":material/cancel:"),
         "Yes": ("green", ":material/check_circle:"), "No clear difference": ("gray", ":material/remove:"), "No, it does worse": ("red", ":material/cancel:")}


def badge(label: str) -> None:
    colr, icon = BADGE.get(label, ("gray", ":material/help:"))
    st.badge(label, icon=icon, color=colr)


def label_positions(x, y, texts, width_px: int = 900, height_px: int = 420, font_px: int = 12) -> list[str]:
    """Plotly textposition per point so direct labels avoid each other (greedy: above, below, right, left)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    sx = (x - x.min()) / max(np.ptp(x), 1e-9) * width_px * 0.8 + width_px * 0.1
    sy = (y - y.min()) / max(np.ptp(y), 1e-9) * height_px * 0.8 + height_px * 0.1
    placed, out, dot = [], [""] * len(x), 14.0

    def box(i, where):
        w, h = 0.56 * font_px * len(texts[i]), font_px * 1.3
        cx, cy = sx[i], sy[i]
        return {"top center": (cx - w / 2, cy + dot, cx + w / 2, cy + dot + h), "bottom center": (cx - w / 2, cy - dot - h, cx + w / 2, cy - dot),
                "middle right": (cx + dot, cy - h / 2, cx + dot + w, cy + h / 2), "middle left": (cx - dot - w, cy - h / 2, cx - dot, cy + h / 2)}[where]

    def overlap(a, b):
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

    dots = [(sx[j] - dot / 2, sy[j] - dot / 2, sx[j] + dot / 2, sy[j] + dot / 2) for j in range(len(x))]
    for i in np.argsort(-sy):
        scores = {}
        for where in ("top center", "bottom center", "middle right", "middle left"):
            bx = box(i, where)
            scores[where] = sum(overlap(bx, p) for p in placed) + sum(overlap(bx, d) for j, d in enumerate(dots) if j != i)
            if scores[where] == 0:
                break
        out[i] = min(scores, key=scores.get)
        placed.append(box(i, out[i]))
    return out
