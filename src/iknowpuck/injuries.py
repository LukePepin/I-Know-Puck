"""Current injuries: default games missed by ESPN status, user overrides, and injury-risk history.

ESPN gives a status (DAY_TO_DAY, OUT, INJURY_RESERVE, SUSPENSION) but no return date, so each
status maps to a default number of games missed that the user can override per player. The
projection is scaled by (projected games - games missed) / projected games.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT

DEFAULT_GAMES_MISSED = {"DAY_TO_DAY": 3, "OUT": 15, "INJURY_RESERVE": 30, "SUSPENSION": 10}
OVERRIDES_PATH = ROOT / "data" / "injury_overrides.json"
RISK_THRESHOLD = 15.0  # average games missed per season over the last three seasons


def load_overrides(path: Path = OVERRIDES_PATH) -> dict[int, int]:
    try:
        return {int(k): int(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    except (FileNotFoundError, ValueError):
        return {}


def save_overrides(overrides: dict[int, int], path: Path = OVERRIDES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): int(v) for k, v in sorted(overrides.items())}, indent=1), encoding="utf-8")


def games_missed(frame: pd.DataFrame, overrides: dict[int, int], defaults: dict[str, int] | None = None) -> pd.Series:
    d = DEFAULT_GAMES_MISSED if defaults is None else defaults
    base = frame["injury_status"].map(lambda s: d.get(str(s), 0)).astype(float)
    ov = frame["player_id"].map(lambda p: overrides.get(int(p), np.nan))
    return ov.fillna(base)


def apply_injuries(frame: pd.DataFrame, stats: list[int], overrides: dict[int, int], defaults: dict[str, int] | None = None) -> pd.DataFrame:
    """Scale every projected stat by the share of projected games the player is expected to play."""
    out = frame.copy()
    missed = games_missed(out, overrides, defaults)
    gp = out["p_30"].where(out["p_30"] > 0)
    factor = ((gp - missed) / gp).clip(0, 1).fillna(1.0)
    for k in [*stats, 30]:
        c = f"p_{k}"
        if c in out:
            out[c] = out[c] * factor
    out["games_missed_assumed"] = missed
    out["injury_factor"] = factor
    return out


def injury_history(avail: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """Average games missed per season (absence spells of 3+ games) over the given seasons."""
    if avail is None or avail.empty:
        return pd.DataFrame(columns=["player_id", "avg_missed", "seasons_seen", "injury_risk"])
    a = avail[avail.season.isin(seasons) & (avail.gp >= 10)]
    g = a.groupby("player_id").agg(avg_missed=("missed", "mean"), seasons_seen=("season", "nunique")).reset_index()
    g["injury_risk"] = np.where(g.avg_missed >= RISK_THRESHOLD, "high", np.where(g.avg_missed >= RISK_THRESHOLD / 2, "moderate", "low"))
    return g
