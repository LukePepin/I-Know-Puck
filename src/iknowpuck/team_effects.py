"""Theory test: when an NHL team beats its preseason expectations, do all of its players benefit?

Expectation = ESPN's preseason projections. Every player's projected fantasy points per game is his
projected season points / projected games. A player's surprise on a team is

    surprise = actual points / (games on that team x projected points per game) - 1

and his team's surprise is the same ratio pooled over his *teammates* (skaters, leave one out), so the
player's own points never sit on both sides of the comparison. Teams come from the game logs (the team
he actually played each game for), which is why the test uses the seasons with league game logs.

Players on one team share one team surprise, so inference is at the team level: the permutation null
reassigns team-season surprises among teams within the same season, and the bootstrap resamples whole
team-seasons.

The in-season version asks whether a team's FIRST-half surprise predicts its players' SECOND half beyond
each player's own first half; that is experiment H5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import STAT_IDS, LeagueSettings
from .pipeline import fantasy_points

W_ID = STAT_IDS["W"]
KEYS = ["season", "pro_team_id"]


def projected_rates(panel: pd.DataFrame, settings: LeagueSettings, seasons: list[int], min_proj_games: float = 10) -> pd.DataFrame:
    """ESPN preseason projection per game: fantasy points, and wins for goalies."""
    p = panel[panel.season.isin(seasons)].drop_duplicates(["season", "player_id"]).copy()
    p = p[p["proj_30"].fillna(0) >= min_proj_games]
    p["proj_rate"] = fantasy_points(p, settings, "proj") / p["proj_30"]
    p["proj_w_rate"] = p.get(f"proj_{W_ID}", pd.Series(0.0, index=p.index)).fillna(0) / p["proj_30"]
    p["grp"] = np.where(p.pos == "G", "G", np.where(p.pos == "D", "D", "F"))
    return p[["season", "player_id", "name", "pos", "grp", "proj_rate", "proj_w_rate"]]


def stints(gamelogs: pd.DataFrame, rates: pd.DataFrame, split: pd.Series | None = None) -> pd.DataFrame:
    """One row per (season, player, NHL team[, half]) with games, actual and expected points (and wins).

    split: season -> last scoring period of the first half (adds a ``half`` column of 1 / 2)."""
    g = gamelogs.merge(rates, on=["season", "player_id"], how="inner")
    keys = ["season", "player_id", "pro_team_id"]
    if split is not None:
        g = g.assign(half=np.where(g.scoring_period <= g.season.map(split), 1, 2))
        keys.append("half")
    w_col = f"g_{W_ID}"
    g["w"] = g[w_col].fillna(0) if w_col in g else 0.0
    out = g.groupby(keys).agg(games=("fp", "size"), fp=("fp", "sum"), w=("w", "sum"), proj_rate=("proj_rate", "first"),
                              proj_w_rate=("proj_w_rate", "first"), name=("name", "first"), pos=("pos", "first"), grp=("grp", "first")).reset_index()
    out["exp"] = out.games * out.proj_rate
    out["exp_w"] = out.games * out.proj_w_rate
    return out


def team_table(st: pd.DataFrame) -> pd.DataFrame:
    """Per team-season: skater scoring vs projection, and wins vs projected wins (from its goalies)."""
    sk = st[st.grp != "G"].groupby(KEYS).agg(t_fp=("fp", "sum"), t_exp=("exp", "sum"), skaters=("fp", "size"))
    gk = st[st.grp == "G"].groupby(KEYS).agg(wins=("w", "sum"), exp_wins=("exp_w", "sum"))
    t = sk.join(gk, how="left")
    t["scoring_surprise"] = t.t_fp / t.t_exp - 1
    t["wins_surprise"] = t.wins / t.exp_wins - 1
    return t.reset_index()


def add_team_surprise(st: pd.DataFrame, min_games: int = 20, measure: str = "scoring") -> pd.DataFrame:
    """Player surprise and his teammates' (leave-one-out) team surprise.

    measure 'scoring': teammates' skater points vs projection (goalies get their whole team's skaters).
    measure 'wins'   : team wins vs projected wins; skaters only, since a goalie's own wins are the measure."""
    t = team_table(st).set_index(KEYS)
    d = st.join(t[["t_fp", "t_exp", "wins", "exp_wins", "scoring_surprise", "wins_surprise"]], on=KEYS)
    d = d[d.games >= min_games].copy()
    d["surprise"] = d.fp / d.exp - 1
    sk = d.grp != "G"
    if measure == "wins":
        d = d[sk & d.exp_wins.gt(0)].copy()
        d["team_surprise"] = d.wins_surprise
        d["team_full"] = d.wins_surprise
    else:
        d["team_surprise"] = np.where(sk, (d.t_fp - d.fp) / (d.t_exp - d.exp), d.t_fp / d.t_exp) - 1
        d["team_full"] = d.scoring_surprise
    d["beat"] = d.surprise > 0
    return d.replace([np.inf, -np.inf], np.nan).dropna(subset=["surprise", "team_surprise"]).reset_index(drop=True)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / max((xc**2).sum(), 1e-12))


def team_effect(d: pd.DataFrame, n_perm: int = 2000, n_boot: int = 2000, seed: int = 0) -> dict:
    """OLS slope of player surprise on teammates' surprise with team-level inference.

    Permutation: within each season, team-season surprises are shuffled among teams (every player on a
    team receives the same other team's surprise). Bootstrap: resample team-seasons with replacement."""
    x, y = d.team_surprise.to_numpy(float), d.surprise.to_numpy(float)
    obs = _slope(x, y)
    rng = np.random.default_rng(seed)
    team_val = d.groupby(KEYS).team_full.first()
    codes = pd.MultiIndex.from_frame(d[KEYS])
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = pd.concat([s.set_axis(rng.permutation(s.index.to_numpy())) for _, s in team_val.groupby(level=0)])
        null[i] = _slope(perm.reindex(codes).to_numpy(float), y)
    groups = list(d.groupby(KEYS).indices.values())  # row positions of each team-season
    boots = np.empty(n_boot)
    for i in range(n_boot):
        rows = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        boots[i] = _slope(x[rows], y[rows])
    return {"slope": obs, "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)),
            "r": float(np.corrcoef(x, y)[0, 1]), "p": float((np.sum(null >= obs) + 1) / (n_perm + 1)), "null": null,
            "n": len(d), "teams": d.groupby(KEYS).ngroups}


def icc(d: pd.DataFrame) -> float:
    """Share of the variance in player surprise that is shared within a team-season (one-way ICC(1))."""
    g = d.groupby(KEYS).surprise
    k = g.size()
    n, m = len(d), len(k)
    k0 = (n - (k**2).sum() / n) / (m - 1)
    msb = (k * (g.mean() - d.surprise.mean()) ** 2).sum() / (m - 1)
    msw = ((d.surprise - g.transform("mean")) ** 2).sum() / (n - m)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def by_quantile(d: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    """Players grouped by their team's surprise: average surprise and share who beat their projection."""
    b = pd.qcut(d.team_surprise, q, labels=False, duplicates="drop")
    out = d.groupby(b).agg(team=("team_surprise", "mean"), surprise=("surprise", "mean"), beat=("beat", "mean"), n=("beat", "size")).reset_index(drop=True)
    z = 1.96
    den = 1 + z**2 / out.n
    mid = (out.beat + z**2 / (2 * out.n)) / den
    half = z * np.sqrt(out.beat * (1 - out.beat) / out.n + z**2 / (4 * out.n**2)) / den
    out["lo"], out["hi"] = mid - half, mid + half
    return out


# --- in-season: does a hot first half carry over? ---------------------------------------------------
def halves(gamelogs: pd.DataFrame, schedules: pd.DataFrame, rates: pd.DataFrame, min_games: int = 15) -> pd.DataFrame:
    """Skater-seasons on one team with first-half and second-half surprise plus teammates' first-half surprise."""
    split = schedules.groupby("season").scoring_period.median()
    st = stints(gamelogs, rates, split)
    sk = st[st.grp != "G"]
    w = sk.pivot_table(index=["season", "player_id", "pro_team_id"], columns="half", values=["games", "fp", "exp"], aggfunc="sum")
    w.columns = [f"{a}{b}" for a, b in w.columns]
    w = w.dropna().reset_index()
    t1 = sk[sk.half == 1].groupby(KEYS)[["fp", "exp"]].sum().rename(columns={"fp": "t_fp1", "exp": "t_exp1"})
    w = w.join(t1, on=KEYS)
    w = w[(w.games1 >= min_games) & (w.games2 >= min_games)].copy()
    w["team1"] = (w.t_fp1 - w.fp1) / (w.t_exp1 - w.exp1) - 1
    w["team_full"] = w.t_fp1 / w.t_exp1 - 1
    w["own1"] = w.fp1 / w.exp1 - 1
    w["surprise2"] = w.fp2 / w.exp2 - 1
    w["rate2"] = w.exp2 / w.games2  # projected points per game
    w["actual2"] = w.fp2 / w.games2
    meta = rates.drop_duplicates(["season", "player_id"]).set_index(["season", "player_id"])[["name", "pos", "grp"]]
    return w.join(meta, on=["season", "player_id"]).replace([np.inf, -np.inf], np.nan).dropna(subset=["team1", "own1", "surprise2"]).reset_index(drop=True)


def persistence(w: pd.DataFrame, n_perm: int = 2000, seed: int = 0) -> dict:
    """surprise2 = a + b * own1 + c * team1. Is c > 0 once the player's own first half is known?"""
    X = np.column_stack([np.ones(len(w)), w.own1, w.team1])
    beta = np.linalg.lstsq(X, w.surprise2, rcond=None)[0]
    rng = np.random.default_rng(seed)
    team_val = w.groupby(KEYS).team_full.first()
    codes = pd.MultiIndex.from_frame(w[KEYS])
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = pd.concat([s.set_axis(rng.permutation(s.index.to_numpy())) for _, s in team_val.groupby(level=0)])
        Xp = np.column_stack([np.ones(len(w)), w.own1, perm.reindex(codes).to_numpy(float)])
        null[i] = np.linalg.lstsq(Xp, w.surprise2, rcond=None)[0][2]
    return {"const": float(beta[0]), "own": float(beta[1]), "team": float(beta[2]), "team_alone": _slope(w.team1.to_numpy(float), w.surprise2.to_numpy(float)),
            "p": float((np.sum(null >= beta[2]) + 1) / (n_perm + 1)), "null": null, "n": len(w)}


def loso_errors(w: pd.DataFrame) -> dict[str, np.ndarray]:
    """Second-half points-per-game error, each season predicted from a model fit on the other seasons.

    projection : ESPN's per-game projection, corrected for its average miss
    own        : + the player's own first-half surprise
    own + team : + his teammates' first-half surprise"""
    out = {"projection": [], "own": [], "own + team": []}
    for s in sorted(w.season.unique()):
        tr, te = w[w.season != s], w[w.season == s]
        a = np.linalg.lstsq(np.column_stack([np.ones(len(tr)), tr.own1]), tr.surprise2, rcond=None)[0]
        b = np.linalg.lstsq(np.column_stack([np.ones(len(tr)), tr.own1, tr.team1]), tr.surprise2, rcond=None)[0]
        out["projection"].append(np.abs(te.rate2 * (1 + tr.surprise2.mean()) - te.actual2))
        out["own"].append(np.abs(te.rate2 * (1 + a[0] + a[1] * te.own1) - te.actual2))
        out["own + team"].append(np.abs(te.rate2 * (1 + b[0] + b[1] * te.own1 + b[2] * te.team1) - te.actual2))
    return {k: np.concatenate(v).astype(float) for k, v in out.items()}
