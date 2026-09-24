"""In-season waiver-wire recommendations.

For my roster, every (add free agent, drop one of my players) swap is scored by how much it raises
expected weekly fantasy points and P(win a week) against the league's average roster. The same
optimal-slotting valuation as the draft is used, so bench-only upgrades get little credit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data.espn import EspnClient, EspnError
from .valuation import PlayerPool, Valuator


def current_rosters(client: EspnClient, season: int) -> dict[int, list[int]]:
    """team id -> ESPN player ids on the roster right now (empty before the draft)."""
    try:
        raw = client.league_raw(season, ["mRoster"], max_age_s=300)
    except EspnError:
        return {}
    out = {}
    for t in raw.get("teams", []):
        entries = (t.get("roster") or {}).get("entries", [])
        out[int(t["id"])] = [int(e["playerId"]) for e in entries if e.get("playerId")]
    return out


def waiver_suggestions(pool: PlayerPool, valuator: Valuator, rosters: dict[int, list[int]], my_team: int,
                       n_free_agents: int = 60, n_drop_candidates: int = 8, top: int = 15) -> pd.DataFrame:
    idx_of = {int(p): i for i, p in enumerate(pool.frame.player_id)}
    mine = [idx_of[p] for p in rosters.get(my_team, []) if p in idx_of]
    if not mine:
        return pd.DataFrame()
    rostered = {idx_of[p] for r in rosters.values() for p in r if p in idx_of}
    value = valuator.pts if valuator.points_mode else valuator._z_weight
    fa = [i for i in np.argsort(-value) if i not in rostered][:n_free_agents]
    others = [[idx_of[p] for p in r if p in idx_of] for t, r in rosters.items() if t != my_team and r]
    u = valuator.usage(mine)
    # drop candidates: my least useful players (lowest usage-weighted value)
    drop = [mine[i] for i in np.argsort(u * value[mine])[:n_drop_candidates]]
    base_pts = valuator.weekly_points(mine)
    base_win = valuator.score(mine, others) if others else np.nan
    rows = []
    for a in fa:
        for d in drop:
            new = [p for p in mine if p != d] + [a]
            pts = valuator.weekly_points(new)
            if pts <= base_pts + 0.05:
                continue
            rows.append({"add": a, "drop": d, "gain_pts": pts - base_pts})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("gain_pts", ascending=False)
    df = df.drop_duplicates("add").head(top)  # best drop for each free agent
    if others:
        df["gain_win"] = [valuator.score([p for p in mine if p != d] + [a], others) - base_win for a, d in zip(df["add"], df["drop"])]
    f = pool.frame
    df["add_player"] = f.loc[df["add"], "name"].to_numpy()
    df["add_pos"] = f.loc[df["add"], "pos"].to_numpy()
    df["add_status"] = f.loc[df["add"], "injury_status"].to_numpy()
    df["drop_player"] = f.loc[df["drop"], "name"].to_numpy()
    df["drop_pos"] = f.loc[df["drop"], "pos"].to_numpy()
    return df.reset_index(drop=True)
