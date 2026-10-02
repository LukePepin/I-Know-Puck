"""Assumption tests: does the data behave the way the models assume?

Every model rests on assumptions that were priors, not facts. This module checks them against the
league's own history:

  weekly team scores ~ Normal          actual ESPN matchup scores, standardised within team-season:
                                       skewness, excess kurtosis, Shapiro-Wilk, D'Agostino K^2
  P(win) = Phi(dmu / sqrt(var_a+var_b)) every past regular-season matchup, predicted from each team's
                                       other weeks (leave-one-week-out) -> reliability, Brier score
  stat counts ~ over-dispersed Poisson  weekly player stat counts: variance / mean per stat vs the
                                       dispersion constants in valuation.py (bootstrap over players)
  covariance inflation c               observed weekly fantasy-point variance / the independent
                                       over-dispersed-Poisson variance the model implies
  projections are calibrated           actual ~ a + b * projected (b = 1 means calibrated), binned
                                       calibration, Breusch-Pagan test for non-constant error variance
  opponent pick logit is calibrated    leave-one-season-out choice probabilities vs what was picked

Everything here is descriptive and fast enough to run inside the app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.special import ndtr

from .config import STAT_IDS, LeagueSettings
from .data.espn import EspnClient, EspnError
from .opponents import OpponentModel, build_observations
from .valuation import COVAR_INFLATION, DISPERSION

GOALIE_STATS = {STAT_IDS[s] for s in ("W", "L", "SA", "GA", "SV", "SO", "OTL", "G_TOI", "GS")}


# --- weekly matchups -----------------------------------------------------------------------------
def matchups(client: EspnClient, seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per team per decided matchup, scoring period -> matchup week).

    Rows: season, week, team_id, opp_id, points, opp_points, result (1 win, 0 loss, 0.5 tie), playoff."""
    rows, periods = [], []
    for s in seasons:
        try:
            raw = client.league_raw(s, ["mMatchupScore", "mMatchup"])
        except EspnError:
            continue
        for m in raw.get("schedule", []):
            home, away = m.get("home"), m.get("away")
            if not home or not away or m.get("winner") in (None, "UNDECIDED"):
                continue
            week = int(m["matchupPeriodId"])
            playoff = m.get("playoffTierType", "NONE") != "NONE"
            for side, other, won in ((home, away, "HOME"), (away, home, "AWAY")):
                res = 0.5 if m["winner"] == "TIE" else float(m["winner"] == won)
                rows.append({"season": s, "week": week, "team_id": int(side["teamId"]), "opp_id": int(other["teamId"]),
                             "points": float(side.get("totalPoints", 0.0)), "opp_points": float(other.get("totalPoints", 0.0)),
                             "result": res, "playoff": playoff})
                for sp in (side.get("pointsByScoringPeriod") or {}):
                    periods.append({"season": s, "scoring_period": int(sp), "week": week})
    per = pd.DataFrame(periods).drop_duplicates(["season", "scoring_period"]) if periods else pd.DataFrame(columns=["season", "scoring_period", "week"])
    return pd.DataFrame(rows), per


def standardised_scores(m: pd.DataFrame) -> pd.Series:
    """Weekly score as a z-score within its own team-season (removes team strength)."""
    g = m.groupby(["season", "team_id"]).points
    return (m.points - g.transform("mean")) / g.transform("std")


def normality(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    out = {"n": len(x), "skew": float(sps.skew(x)), "excess_kurtosis": float(sps.kurtosis(x))}
    out["p_shapiro"] = float(sps.shapiro(x).pvalue) if 3 <= len(x) <= 5000 else float("nan")
    out["p_dagostino"] = float(sps.normaltest(x).pvalue) if len(x) >= 20 else float("nan")
    return out


def qq(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(theoretical Normal quantiles, sorted standardised sample) for a Q-Q plot."""
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    z = (x - x.mean()) / x.std(ddof=1)
    p = (np.arange(1, len(x) + 1) - 0.5) / len(x)
    return sps.norm.ppf(p), z


def win_prob_backtest(m: pd.DataFrame) -> pd.DataFrame:
    """Predict every regular-season matchup with the model's Normal formula.

    Each team's mean and SD come from its OTHER regular-season weeks that season (leave one week out),
    so a matchup never predicts itself. One row per matchup (home side)."""
    reg = m[~m.playoff].copy()
    g = reg.groupby(["season", "team_id"]).points
    n, s1, s2 = g.transform("size"), g.transform("sum"), g.transform(lambda x: (x**2).sum())
    reg["mu_loo"] = (s1 - reg.points) / (n - 1)
    var = ((s2 - reg.points**2) - (n - 1) * reg.mu_loo**2) / (n - 2)
    reg["var_loo"] = var.clip(lower=1.0)
    key = reg.set_index(["season", "week", "team_id"])[["mu_loo", "var_loo"]]
    reg = reg[reg.team_id < reg.opp_id].copy()  # one row per matchup
    opp = key.reindex(pd.MultiIndex.from_arrays([reg.season, reg.week, reg.opp_id])).to_numpy()
    reg["opp_mu"], reg["opp_var"] = opp[:, 0], opp[:, 1]
    reg["p_model"] = ndtr((reg.mu_loo - reg.opp_mu) / np.sqrt(reg.var_loo + reg.opp_var))
    return reg.dropna(subset=["p_model"]).reset_index(drop=True)


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Binned predicted probability vs observed frequency with Wilson 95% intervals."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        sel = idx == b
        k, n = y[sel].sum(), sel.sum()
        if n == 0:
            continue
        ph, z = k / n, 1.96
        den = 1 + z**2 / n
        mid = (ph + z**2 / (2 * n)) / den
        half = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / den
        rows.append({"predicted": float(p[sel].mean()), "observed": float(ph), "n": int(n), "lo": float(mid - half), "hi": float(mid + half)})
    return pd.DataFrame(rows)


def scores(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return {"brier": float(np.mean((p - y) ** 2)), "brier_coin": 0.25,
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), "log_loss_coin": float(np.log(2)),
            "accuracy": float(np.mean((p > 0.5) == (y > 0.5)))}


# --- player stat variance ------------------------------------------------------------------------
def weekly_player_stats(gamelogs: pd.DataFrame, periods: pd.DataFrame, settings: LeagueSettings) -> pd.DataFrame:
    """One row per (season, player, matchup week) with at least one game: games, each scoring stat, fp."""
    cols = [f"g_{c.stat_id}" for c in settings.scoring_categories if f"g_{c.stat_id}" in gamelogs]
    g = gamelogs.merge(periods, on=["season", "scoring_period"], how="inner")
    agg = {c: "sum" for c in cols} | {"fp": "sum", "scoring_period": "size"}
    return g.groupby(["season", "player_id", "week"]).agg(agg).rename(columns={"scoring_period": "games"}).reset_index()


def _pooled_ratio(num: np.ndarray, den: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    """sum(num) / sum(den) with a percentile bootstrap over units."""
    est = num.sum() / den.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    boots = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return float(est), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def dispersion_table(weekly: pd.DataFrame, settings: LeagueSettings, positions: dict[int, str],
                     min_weeks: int = 10, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Estimated dispersion (weekly variance / weekly mean) per scoring stat vs the value assumed.

    Pooled over player-seasons: d = sum (n-1) s^2 / sum (n-1) mean, bootstrap CI over player-seasons.
    Poisson is d = 1; above 1 is over-dispersed (streakier than Poisson)."""
    w = weekly.assign(is_g=weekly.player_id.map(positions).eq("G"))
    rows = []
    for c in settings.scoring_categories:
        col = f"g_{c.stat_id}"
        if col not in w:
            continue
        sub = w[w.is_g == (c.stat_id in GOALIE_STATS)]
        g = sub.groupby(["season", "player_id"])[col]
        t = pd.DataFrame({"n": g.size(), "mean": g.mean(), "var": g.var(ddof=1)})
        t = t[(t.n >= min_weeks) & (t["mean"] > 0)]
        if len(t) < 5:
            continue
        est, lo, hi = _pooled_ratio(((t.n - 1) * t["var"]).to_numpy(), ((t.n - 1) * t["mean"]).to_numpy(), n_boot, seed)
        rows.append({"stat": c.name, "stat_id": c.stat_id, "points": c.points, "assumed": DISPERSION.get(c.stat_id, 1.2),
                     "estimated": est, "lo": lo, "hi": hi, "player_seasons": len(t), "weekly_mean": float(t["mean"].mean()),
                     "goalie": c.stat_id in GOALIE_STATS})
    return pd.DataFrame(rows)


def covariance_inflation(weekly: pd.DataFrame, settings: LeagueSettings, positions: dict[int, str],
                         dispersion: dict[int, float] | None = None, min_weeks: int = 10, n_boot: int = 1000,
                         seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Observed weekly fantasy-point variance vs the variance of independent over-dispersed Poisson stats.

    implied = sum_k w_k^2 d_k mean_k (what valuation.py uses before inflation). The pooled ratio
    observed / implied is the covariance inflation the data supports (the model assumes COVAR_INFLATION)."""
    disp = dispersion or {c.stat_id: DISPERSION.get(c.stat_id, 1.2) for c in settings.scoring_categories}
    cats = [c for c in settings.scoring_categories if f"g_{c.stat_id}" in weekly]
    g = weekly.groupby(["season", "player_id"])
    t = pd.DataFrame({"n": g.size(), "fp_mean": g.fp.mean(), "fp_var": g.fp.var(ddof=1)})
    t["implied"] = sum((c.points**2) * disp.get(c.stat_id, 1.2) * g[f"g_{c.stat_id}"].mean().clip(lower=0) for c in cats)
    t = t[(t.n >= min_weeks) & (t.implied > 0)].reset_index()
    t["ratio"] = t.fp_var / t.implied
    t["pos"] = t.player_id.map(positions)
    est, lo, hi = _pooled_ratio(((t.n - 1) * t.fp_var).to_numpy(), ((t.n - 1) * t.implied).to_numpy(), n_boot, seed)
    return t, {"estimated": est, "lo": lo, "hi": hi, "assumed": COVAR_INFLATION, "player_seasons": len(t)}


# --- projections ----------------------------------------------------------------------------------
def calibration_line(pred: np.ndarray, actual: np.ndarray) -> dict:
    """OLS actual = a + b * pred with 95% CIs. b = 1 and a = 0 mean perfectly calibrated."""
    x, y = np.asarray(pred, float), np.asarray(actual, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    r = sps.linregress(x, y)
    t = sps.t.ppf(0.975, len(x) - 2)
    return {"slope": r.slope, "slope_lo": r.slope - t * r.stderr, "slope_hi": r.slope + t * r.stderr,
            "intercept": r.intercept, "intercept_lo": r.intercept - t * r.intercept_stderr,
            "intercept_hi": r.intercept + t * r.intercept_stderr, "r2": r.rvalue**2, "n": len(x)}


def breusch_pagan(pred: np.ndarray, resid: np.ndarray) -> dict:
    """LM test of whether error variance grows with the prediction (H0: constant variance)."""
    x, e = np.asarray(pred, float), np.asarray(resid, float)
    ok = np.isfinite(x) & np.isfinite(e)
    x, e2 = x[ok], e[ok] ** 2
    r = sps.linregress(x, e2 / e2.mean())
    lm = len(x) * r.rvalue**2
    return {"lm": float(lm), "p": float(sps.chi2.sf(lm, 1)), "slope_sign": float(np.sign(r.slope))}


def binned_means(pred: np.ndarray, actual: np.ndarray, bins: int = 10) -> pd.DataFrame:
    x, y = np.asarray(pred, float), np.asarray(actual, float)
    q = pd.qcut(pd.Series(x), bins, duplicates="drop")
    d = pd.DataFrame({"x": x, "y": y, "q": q}).groupby("q", observed=True)
    out = pd.DataFrame({"predicted": d.x.mean(), "actual": d.y.mean(), "sd": d.y.std(), "n": d.y.size()}).reset_index(drop=True)
    out["se"] = out.sd / np.sqrt(out.n)
    return out


# --- opponent pick model --------------------------------------------------------------------------
def pick_model_loso(drafts: pd.DataFrame, panel: pd.DataFrame, lineup: dict[int, int]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Leave-one-season-out choice probabilities of the league-wide pick logit.

    Returns (one row per held-out pick: season, round, p_chosen, model_rank, adp_rank, n_choices),
    and the flattened (probability, was-chosen) pairs over every available player at every pick."""
    rows, probs, chosen = [], [], []
    for held in sorted(drafts.season.unique()):
        train, test = drafts[drafts.season != held], drafts[drafts.season == held]
        m = OpponentModel().fit(build_observations(train, panel, lineup), per_manager=False)
        obs = build_observations(test, panel, lineup)
        P, A = m.choice_probs(obs)
        for i, o in enumerate(obs):
            c = int(A.mask[i].sum())
            p = P[i, :c]
            order = np.argsort(-p)
            rows.append({"season": held, "round": o.round, "p_chosen": float(p[o.chosen]),
                         "model_rank": int(np.flatnonzero(order == o.chosen)[0]) + 1,
                         "adp_rank": int(np.argsort(np.argsort(o.log_adp))[o.chosen]) + 1, "n_choices": c})
            probs.append(p)
            chosen.append(np.arange(c) == o.chosen)
    return pd.DataFrame(rows), np.concatenate(probs), np.concatenate(chosen).astype(float)


# --- lineup rules ---------------------------------------------------------------------------------
def bench_usage(m: pd.DataFrame, periods: pd.DataFrame, gamelogs: pd.DataFrame, stints: pd.DataFrame,
                positions: dict[int, str], lineup: dict[int, int]) -> dict:
    """How much of a bench player's points end up counting, from real weekly scores.

    For every regular-season team-week: the players on the roster that week, split into likely starters
    (the best players at each position by the manager's own season average, up to the starting slots)
    and bench. Then real score = a x starters + b x bench (no intercept); b is the bench usage."""
    slots = {"C": lineup.get(0, 0), "LW": lineup.get(1, 0), "RW": lineup.get(2, 0), "D": lineup.get(4, 0), "G": lineup.get(5, 0)}
    gl = gamelogs.merge(periods, on=["season", "scoring_period"])
    x = gl[["season", "player_id", "scoring_period", "week", "fp"]].merge(stints[["season", "team_id", "player_id", "start_sp", "end_sp"]],
                                                                          on=["season", "player_id"])
    x = x[(x.scoring_period >= x.start_sp) & (x.scoring_period < x.end_sp)]
    wk = x.groupby(["season", "team_id", "week", "player_id"]).fp.sum().reset_index()
    wk = wk.join(x.groupby(["season", "team_id", "player_id"]).fp.mean().rename("rate"), on=["season", "team_id", "player_id"])
    wk["pos"] = wk.player_id.map(positions).fillna("C")
    wk["starter"] = wk.groupby(["season", "team_id", "week", "pos"]).rate.rank(ascending=False, method="first") <= wk.pos.map(slots).fillna(0)
    wk["part"] = np.where(wk.starter, "starters", "bench")
    agg = wk.pivot_table(index=["season", "team_id", "week"], columns="part", values="fp", aggfunc="sum", fill_value=0).reset_index()
    j = m[~m.playoff].merge(agg, on=["season", "team_id", "week"])
    X = j[["starters", "bench"]].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, j.points.to_numpy(float), rcond=None)
    resid = j.points.to_numpy(float) - X @ beta
    cov = resid.var(ddof=2) * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {"starters": float(beta[0]), "bench": float(beta[1]), "bench_lo": float(beta[1] - 1.96 * se[1]), "bench_hi": float(beta[1] + 1.96 * se[1]),
            "team_weeks": len(j), "real_mean": float(j.points.mean()), "starters_mean": float(j.starters.mean()), "bench_mean": float(j.bench.mean())}
