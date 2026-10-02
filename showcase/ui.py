"""Shared look and data loading for the public showcase. Reads only the files in showcase/data/."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"
AUTHOR = "Luke (author)"

# 5-slot categorical order validated for colour-vision deficiency, chroma and contrast on white; never cycled past 5
CAT = ["#2A6BB0", "#C0563B", "#8460B0", "#1F9A6E", "#B08A1E"]
INK, MUTED, FAINT = "#1B1B1B", "#8A8A8A", "#C9C7C0"
ME = CAT[1]  # the author's team is the one highlighted colour; everyone else is muted
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


BADGE = {"Holds": ("green", ":material/check_circle:"), "Fixed": ("blue", ":material/build:"), "Partly holds": ("orange", ":material/adjust:"),
         "Roughly holds": ("orange", ":material/adjust:"), "Does not hold": ("red", ":material/cancel:"), "Not supported": ("red", ":material/cancel:"),
         "Yes": ("green", ":material/check_circle:"), "No clear difference": ("gray", ":material/remove:"), "No, it does worse": ("red", ":material/cancel:")}


def badge(label: str) -> None:
    colr, icon = BADGE.get(label, ("gray", ":material/help:"))
    st.badge(label, icon=icon, color=colr)
