"""Shared helpers for every page: data loading, chart styling, small layout helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iknowpuck.config import RUNS_DIR  # noqa: E402
from iknowpuck.injuries import DEFAULT_GAMES_MISSED  # noqa: E402
from iknowpuck.pipeline import build  # noqa: E402

NAVY, RUST, SAGE, SAND, PLUM, GREY = "#1F3A5F", "#A5452B", "#5E7D5B", "#C9A35B", "#6B5B8C", "#8A8A8A"
PALETTE = [NAVY, RUST, SAGE, SAND, PLUM, GREY, "#3E7C8C", "#B07AA1", "#7F6A4C", "#4F6D7A", "#9C755F", "#59636E"]

pio.templates["academic"] = go.layout.Template(
    layout=dict(
        font=dict(family="Georgia, 'Times New Roman', serif", size=13, color="#1B1B1B"),
        colorway=PALETTE,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, linecolor="#444", ticks="outside", zeroline=False, automargin=True, title_standoff=8),
        yaxis=dict(gridcolor="#E6E4DE", linecolor="#444", ticks="outside", zeroline=False, automargin=True, title_standoff=8),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=12, r=16, t=16, b=12),  # axes grow the margins to fit their labels (automargin)
        hoverlabel=dict(font_family="Georgia, serif"),
    )
)
pio.templates.default = "academic"


def note(text: str) -> None:
    """A short 'how to read this' or takeaway box."""
    with st.container(border=True):
        st.markdown(text)


def fig_show(fig, height: int = 380, key: str | None = None, on_select: str = "ignore", selection_mode="points"):
    """Render a Plotly figure. The chart title is shown as page text above the chart (it wraps instead of
    being clipped), and axes size their margins to their labels so long names never cover the plot."""
    title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else None
    if title:
        st.markdown(f"**{title}**")
        fig.update_layout(title=None)
    left = fig.layout.margin.l if fig.layout.margin and fig.layout.margin.l is not None else None
    fig.update_layout(height=height, margin=dict(t=16, l=left if left is not None and left > 12 else 12))
    if on_select == "ignore":
        return st.plotly_chart(fig, key=key, theme=None)
    return st.plotly_chart(fig, key=key, on_select=on_select, selection_mode=selection_mode, theme=None)


def selected_custom(event) -> str | None:
    """customdata of the first clicked point in a Plotly selection event (or None)."""
    try:
        pts = event["selection"]["points"]
    except (KeyError, TypeError):
        return None
    if not pts:
        return None
    cd = pts[0].get("customdata")
    if isinstance(cd, (list, tuple)):
        cd = cd[0] if cd else None
    return cd


@st.cache_resource(show_spinner="Building projections, opponent models and league history (first run takes a few minutes)...")
def get_bundle(season: int, refresh_token: int):
    return build(season, refresh=refresh_token > 0)


@st.cache_resource(max_entries=8, show_spinner=False)
def injured_context(_bundle, season: int, overrides_key: tuple, defaults_key: tuple, order: tuple, my_team: int):
    """Pool and draft context with current injuries applied (cached per injury settings)."""
    pool = _bundle.injured_pool(dict(overrides_key), dict(defaults_key))
    return pool, _bundle.context(list(order), my_team, pool=pool)


def load_runs(n: int = 2) -> list[dict]:
    paths = sorted(RUNS_DIR.glob("*_ikp/results.json"), reverse=True) if RUNS_DIR.exists() else []
    return [json.loads(p.read_text()) for p in paths[:n]]


def app_state() -> dict:
    """Everything the entry script prepared for this run (bundle, context, names, settings)."""
    return st.session_state["app"]


DEFAULT_MISSED = dict(DEFAULT_GAMES_MISSED)
