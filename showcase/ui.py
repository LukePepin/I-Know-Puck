"""Shared look, data loading and page sections for the public showcase. Reads only showcase/data/."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"

# 5-slot categorical order validated for colour-vision deficiency, chroma and contrast on white; never cycled past 5
CAT = ["#2A6BB0", "#C0563B", "#8460B0", "#1F9A6E", "#B08A1E"]
INK, MUTED, FAINT = "#1B1B1B", "#8A8A8A", "#C9C7C0"
ACCENT = CAT[0]
SEQ = [[0.0, "#F3F6FA"], [0.5, "#7FA3CC"], [1.0, "#1F3A5F"]]
DIV = [[0.0, "#C0563B"], [0.5, "#EEEDEA"], [1.0, "#2A6BB0"]]
GROUP = {"F": ("Forwards", CAT[0]), "D": ("Defense", CAT[1]), "G": ("Goalies", CAT[2])}

pio.templates["showcase"] = go.layout.Template(layout=dict(
    font=dict(family="Georgia, 'Times New Roman', serif", size=13, color=INK), colorway=CAT, paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(showgrid=False, linecolor="#444", ticks="outside", zeroline=False, automargin=True, title_standoff=8),
    yaxis=dict(gridcolor="#E6E4DE", linecolor="#444", ticks="outside", zeroline=False, automargin=True, title_standoff=8),
    legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=12, r=16, t=16, b=12), hoverlabel=dict(font_family="Georgia, serif"),
))
pio.templates.default = "showcase"


def group_of(pos: str) -> str:
    return "G" if pos == "G" else ("D" if pos == "D" else "F")


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


@st.cache_data
def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


@st.cache_data
def js(name: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def header(title: str, subtitle: str) -> None:
    meta = js("meta.json")
    st.markdown(f"## {title}")
    st.caption(f"{subtitle}  ·  I Know Puck, a fantasy hockey analytics project  ·  data as of {meta['snapshot']}")


def fig_show(fig, height: int = 380, key: str | None = None):
    """Chart title as page text above the chart (it wraps instead of being clipped)."""
    title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else None
    if title:
        st.markdown(f"**{title}**")
        fig.update_layout(title=None)
    fig.update_layout(height=height, margin=dict(t=16))
    return st.plotly_chart(fig, key=key, theme=None)


def note(text: str) -> None:
    with st.container(border=True):
        st.markdown(text)


def math(*items: tuple[str, str]) -> None:
    """Short math box: (latex, one-line explanation) pairs."""
    with st.expander("The math", icon=":material/functions:"):
        for tex, text in items:
            st.latex(tex)
            st.caption(text)


def how_built(steps: list[tuple[str, str]], details: list[str]) -> None:
    """Systems overview of one page: where its data comes from and what is done to it."""
    st.markdown("### How this page is built")
    nodes = "; ".join(f'n{i} [label="{label}\\n{sub}"]' for i, (label, sub) in enumerate(steps))
    edges = " -> ".join(f"n{i}" for i in range(len(steps)))
    st.graphviz_chart(f"""
    digraph G {{
      rankdir=LR; bgcolor="transparent"; nodesep=0.25;
      node [shape=box, style="rounded,filled", fillcolor="#F4F3EF", color="#2A6BB0", fontname="Georgia", fontsize=11, margin="0.15,0.08"];
      edge [color="#555555"];
      {nodes}; {edges};
    }}""", width="stretch")
    st.markdown("\n".join(f"{i + 1}. {d}" for i, d in enumerate(details)))


def definitions(terms: list[tuple[str, str]]) -> None:
    st.markdown("### Definitions")
    cols = st.columns(2)
    for i, (term, meaning) in enumerate(terms):
        cols[i % 2].markdown(f"**{term}.** {meaning}")


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
