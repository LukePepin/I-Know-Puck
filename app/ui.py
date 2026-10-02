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

from iknowpuck.config import RUNS_DIR, load_credentials  # noqa: E402
from iknowpuck.data.dataset import build_panel  # noqa: E402
from iknowpuck.data.espn import EspnClient  # noqa: E402
from iknowpuck.diagnostics import matchups  # noqa: E402
from iknowpuck.injuries import DEFAULT_GAMES_MISSED  # noqa: E402
from iknowpuck.pipeline import FIRST_PANEL_SEASON, build  # noqa: E402

NAVY, RUST, SAGE, SAND, PLUM, GREY = "#1F3A5F", "#A5452B", "#5E7D5B", "#C9A35B", "#6B5B8C", "#8A8A8A"
PALETTE = [NAVY, RUST, SAGE, SAND, PLUM, GREY, "#3E7C8C", "#B07AA1", "#7F6A4C", "#4F6D7A", "#9C755F", "#59636E"]
# Model pages: a 5-slot categorical order validated for colour-vision deficiency, chroma and contrast
# (dataviz validate_palette.js: all checks pass on white). Use in this order; never cycle past 5.
CAT = ["#2A6BB0", "#C0563B", "#8460B0", "#1F9A6E", "#B08A1E"]
INK, MUTED, FAINT = "#1B1B1B", "#8A8A8A", "#D9D7D0"
SEQ = [[0.0, "#F3F6FA"], [0.5, "#7FA3CC"], [1.0, "#1F3A5F"]]  # magnitude: one hue, light -> dark
DIV = [[0.0, "#C0563B"], [0.5, "#EEEDEA"], [1.0, "#2A6BB0"]]  # polarity: rust <- neutral grey -> blue

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


@st.cache_resource(show_spinner="Loading ten seasons of player data...")
def get_panel(season: int):
    """Every player-season 2018..season (ESPN + MoneyPuck), from the disk cache."""
    return build_panel(list(range(FIRST_PANEL_SEASON, season + 1)), EspnClient(load_credentials()))


@st.cache_resource(show_spinner="Loading weekly matchup scores from ESPN...")
def get_matchups(seasons: tuple[int, ...]):
    """(team-week matchup scores, scoring period -> week) for past seasons."""
    return matchups(EspnClient(load_credentials()), list(seasons))


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def label_positions(x, y, texts, x_range=None, y_range=None, width_px: int = 900, height_px: int = 480, font_px: int = 12) -> list[str]:
    """Plotly textposition per point so direct labels don't collide with each other or with other dots.

    Greedy: points from top to bottom, each tries above, below, right, left and keeps the first spot whose
    estimated text box is clear (falls back to the least-overlapping spot). Pass the axis ranges so pixel
    distances match the drawn chart."""
    import numpy as np

    x, y = np.asarray(x, float), np.asarray(y, float)
    x0, x1 = x_range or (x.min(), x.max())
    y0, y1 = y_range or (y.min(), y.max())
    sx = (x - x0) / max(x1 - x0, 1e-9) * width_px
    sy = (y - y0) / max(y1 - y0, 1e-9) * height_px
    placed: list[tuple[float, float, float, float]] = []
    out = [""] * len(x)
    dot = 12.0

    def box(i: int, where: str) -> tuple[float, float, float, float]:
        w, h = 0.56 * font_px * len(texts[i]), font_px * 1.3
        cx, cy = sx[i], sy[i]
        return {"top center": (cx - w / 2, cy + dot, cx + w / 2, cy + dot + h), "bottom center": (cx - w / 2, cy - dot - h, cx + w / 2, cy - dot),
                "middle right": (cx + dot, cy - h / 2, cx + dot + w, cy + h / 2), "middle left": (cx - dot - w, cy - h / 2, cx - dot, cy + h / 2)}[where]

    def overlap(a, b) -> float:
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

    dots = [(sx[j] - dot / 2, sy[j] - dot / 2, sx[j] + dot / 2, sy[j] + dot / 2) for j in range(len(x))]
    for i in np.argsort(-sy):
        scores = {}
        for where in ("top center", "bottom center", "middle right", "middle left"):
            bx = box(i, where)
            scores[where] = sum(overlap(bx, p) for p in placed) + sum(overlap(bx, d) for j, d in enumerate(dots) if j != i)
            if scores[where] == 0:
                break
        best = min(scores, key=scores.get)
        out[i] = best
        placed.append(box(i, best))
    return out


def load_runs(n: int = 2) -> list[dict]:
    paths = sorted(RUNS_DIR.glob("*_ikp/results.json"), reverse=True) if RUNS_DIR.exists() else []
    return [json.loads(p.read_text()) for p in paths[:n]]


def app_state() -> dict:
    """Everything the entry script prepared for this run (bundle, context, names, settings)."""
    return st.session_state["app"]


DEFAULT_MISSED = dict(DEFAULT_GAMES_MISSED)
