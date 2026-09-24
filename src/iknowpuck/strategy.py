"""Which draft strategies have worked in this league?

For every manager-season we join draft behaviour to outcomes:
  behaviour : reach vs market, first goalie round, goalies in rounds 1-3, D share in rounds 1-6,
              market adherence (Spearman of pick number vs market rank)
  draft     : haul = actual fantasy points of the drafted players; value_added = haul minus the
              league-wide expectation for those pick slots (regression of actual points on log pick)
  outcome   : regular-season win %, points for, final rank
Associations are Spearman correlations with bootstrap 95% CIs. With ~36 manager-seasons these are
descriptive evidence, not causal estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .config import LeagueSettings
from .data.espn import EspnClient
from .opponents import group_of


def season_outcomes(client: EspnClient, seasons: list[int]) -> pd.DataFrame:
    rows = []
    for s in seasons:
        raw = client.league_raw(s, ["mTeam"])
        for t in raw.get("teams", []):
            ov = (t.get("record") or {}).get("overall", {})
            owner = (t.get("owners") or [t.get("primaryOwner")])[0]
            rows.append({
                "season": s, "team_id": t["id"], "owner_id": owner,
                "win_pct": ov.get("percentage"), "points_for": ov.get("pointsFor") or t.get("points"),
                "final_rank": t.get("rankCalculatedFinal") or t.get("rankFinal") or None,
                "playoff_seed": t.get("playoffSeed"),
            })
    return pd.DataFrame(rows)


def manager_seasons(drafts: pd.DataFrame, panel: pd.DataFrame, outcomes: pd.DataFrame, settings: LeagueSettings) -> pd.DataFrame:
    pl = panel[["season", "player_id", "adp", "pos"] + [c for c in panel.columns if c.startswith("act_")]].drop_duplicates(["season", "player_id"])
    d = drafts.merge(pl, on=["season", "player_id"], how="left")
    d["act_pts"] = sum(d.get(f"act_{c.stat_id}", 0).fillna(0) * c.points for c in settings.scoring_categories)
    d["grp"] = d["pos"].map(group_of)
    d["mkt"] = d["adp"].fillna(d["overall"].max() + 30)
    d["reach"] = np.log(d["mkt"]) - np.log(d["overall"])
    # expected actual points for a pick slot, league-wide
    X = np.column_stack([np.ones(len(d)), np.log(d["overall"])])
    beta = np.linalg.lstsq(X, d["act_pts"].to_numpy(float), rcond=None)[0]
    d["value_added"] = d["act_pts"] - X @ beta
    rows = []
    for (s, oid), g in d.groupby(["season", "owner_id"]):
        early = g[g["round"] <= 6]
        goalies = g[g.grp == 2]
        rows.append({
            "season": s, "owner_id": oid,
            "reach_early": early["reach"].mean(),
            "market_adherence": sps.spearmanr(g["overall"], g["mkt"]).statistic,
            "first_goalie_round": goalies["round"].min() if len(goalies) else g["round"].max() + 1,
            "goalies_r1_3": int((goalies["round"] <= 3).sum()),
            "d_share_r1_6": (early.grp == 1).mean(),
            "haul_top16": g.nlargest(16, "act_pts")["act_pts"].sum(),
            "value_added_r1_6": early["value_added"].sum(),
            "value_added_all": g["value_added"].sum(),
        })
    ms = pd.DataFrame(rows).merge(outcomes, on=["season", "owner_id"], how="left")
    return ms


STRATEGY_COLS = {
    "first_goalie_round": "First goalie round (higher = waited)",
    "goalies_r1_3": "Goalies taken in rounds 1-3",
    "d_share_r1_6": "Defense share of rounds 1-6",
    "reach_early": "Reach vs market, rounds 1-6 (>0 = early)",
    "market_adherence": "Follows market order (Spearman)",
    "value_added_r1_6": "Draft value added, rounds 1-6 (actual pts vs slot)",
    "value_added_all": "Draft value added, all rounds",
    "haul_top16": "Actual pts of best 16 drafted players",
}
OUTCOME_COLS = {"win_pct": "Win %", "points_for": "Points for", "final_rank": "Final rank (1 = best)"}


def strategy_correlations(ms: pd.DataFrame, n_boot: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for x, xl in STRATEGY_COLS.items():
        for y, yl in OUTCOME_COLS.items():
            m = ms[[x, y]].dropna()
            if len(m) < 8:
                continue
            r, p = sps.spearmanr(m[x], m[y])
            idx = rng.integers(0, len(m), (n_boot, len(m)))
            boots = [sps.spearmanr(m[x].to_numpy()[i], m[y].to_numpy()[i]).statistic for i in idx]
            lo, hi = np.nanquantile(boots, [0.025, 0.975])
            rows.append({"strategy": xl, "outcome": yl, "rho": r, "ci_low": lo, "ci_high": hi, "p": p, "n": len(m)})
    out = pd.DataFrame(rows)
    # orient rank so that positive rho always means "better outcome"
    flip = out["outcome"] == OUTCOME_COLS["final_rank"]
    out.loc[flip, ["rho", "ci_low", "ci_high"]] = -out.loc[flip, ["rho", "ci_high", "ci_low"]].to_numpy()
    out["outcome"] = out["outcome"].replace({OUTCOME_COLS["final_rank"]: "Final standing (higher = better)"})
    return out
