import numpy as np
import pandas as pd

from iknowpuck.config import STAT_IDS, Category, LeagueSettings
from iknowpuck import diagnostics as D
from iknowpuck.draft import DraftContext, crn_experiment, pick_distribution
from iknowpuck.opponents import OpponentModel, PickObs
from iknowpuck.spectral import bootstrap_coclustering, eigengap_null, manager_features, manager_season_features
from iknowpuck.valuation import Valuator

from test_hockey import points_settings, synthetic_pool

G, A = STAT_IDS["G"], STAT_IDS["A"]


def _weekly_poisson(dispersion: float = 1.0, n_players: int = 60, n_weeks: int = 20, seed: int = 0) -> pd.DataFrame:
    """Weekly goal/assist counts; dispersion > 1 via a gamma-Poisson (negative binomial) mixture."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_players):
        mg, ma = rng.uniform(0.3, 1.5), rng.uniform(0.5, 2.0)
        for w in range(n_weeks):
            if dispersion > 1:  # gamma(mean/(d-1), scale d-1) mixed Poisson has variance = d * mean
                g = rng.poisson(rng.gamma(mg / (dispersion - 1), dispersion - 1))
                a = rng.poisson(rng.gamma(ma / (dispersion - 1), dispersion - 1))
            else:
                g, a = rng.poisson(mg), rng.poisson(ma)
            rows.append({"season": 2025, "player_id": p, "week": w, f"g_{G}": g, f"g_{A}": a, "fp": 2 * g + a, "games": 3})
    return pd.DataFrame(rows)


def _settings():
    return LeagueSettings(categories=[Category(G, points=2.0), Category(A, points=1.0)], scoring_type="H2H_POINTS")


def test_dispersion_recovers_poisson_and_overdispersion():
    st = _settings()
    pos = {p: "C" for p in range(60)}
    d1 = D.dispersion_table(_weekly_poisson(1.0), st, pos, n_boot=200)
    d2 = D.dispersion_table(_weekly_poisson(2.0, seed=1), st, pos, n_boot=200)
    assert d1.estimated.between(0.85, 1.15).all()
    assert d2.estimated.between(1.7, 2.3).all()


def test_covariance_inflation_is_one_for_independent_stats():
    st = _settings()
    wk = _weekly_poisson(1.0, n_players=120, seed=2)
    _, s = D.covariance_inflation(wk, st, {p: "C" for p in range(120)}, dispersion={G: 1.0, A: 1.0}, n_boot=200)
    assert 0.9 < s["estimated"] < 1.1 and s["lo"] < s["estimated"] < s["hi"]


def test_win_prob_backtest_is_calibrated_on_normal_scores():
    rng = np.random.default_rng(4)
    strength = rng.normal(100, 8, 10)
    rows = []
    for week in range(1, 41):
        order = rng.permutation(10)
        for a, b in zip(order[::2], order[1::2]):
            pa, pb = rng.normal(strength[a], 15), rng.normal(strength[b], 15)
            for t, o, x, y in ((a, b, pa, pb), (b, a, pb, pa)):
                rows.append({"season": 2025, "week": week, "team_id": int(t), "opp_id": int(o), "points": x, "opp_points": y,
                             "result": float(x > y), "playoff": False})
    wp = D.win_prob_backtest(pd.DataFrame(rows))
    assert len(wp) == 200 and wp.p_model.between(0, 1).all()
    s = D.scores(wp.p_model, wp.result)
    assert s["brier"] < 0.25 and s["accuracy"] > 0.55
    rel = D.reliability(wp.p_model, wp.result, 5)
    assert (rel.lo <= rel.hi).all() and rel.n.sum() == len(wp)


def test_normality_and_qq():
    x = np.random.default_rng(5).normal(size=500)
    nz = D.normality(x)
    assert nz["p_shapiro"] > 0.01 and abs(nz["skew"]) < 0.3
    th, sm = D.qq(x)
    assert len(th) == 500 and np.corrcoef(th, sm)[0, 1] > 0.99


def test_calibration_line_and_breusch_pagan():
    rng = np.random.default_rng(6)
    x = rng.uniform(20, 200, 800)
    y = x + rng.normal(0, 20, 800)  # calibrated, constant variance
    cl = D.calibration_line(x, y)
    assert cl["slope_lo"] < 1 < cl["slope_hi"]
    assert D.breusch_pagan(x, y - x)["p"] > 0.01
    y2 = x + rng.normal(0, 1, 800) * x / 4  # spread grows with x
    assert D.breusch_pagan(x, y2 - x)["p"] < 0.01


def test_choice_probs_sum_to_one():
    rng = np.random.default_rng(7)
    obs = [PickObs("m", 1 + i % 20, np.arange(30), int(rng.integers(30)), np.log(rng.uniform(1, 200, 30)), np.zeros(30, int), np.zeros(30))
           for i in range(50)]
    P, A_ = OpponentModel().fit(obs, per_manager=False).choice_probs(obs)
    assert np.allclose(P.sum(axis=1), 1) and (P[~A_.mask] == 0).all()


def test_eigengap_null_separates_real_groups_from_noise():
    rng = np.random.default_rng(8)
    grouped = pd.DataFrame(np.vstack([rng.normal(0, 0.3, (8, 5)), rng.normal(0, 0.3, (8, 5)) + np.array([3, 3, -3, 3, -3])]),
                           columns=list("abcde"))
    noise = pd.DataFrame(rng.normal(size=(16, 5)), columns=list("abcde"))
    assert eigengap_null(grouped, n_perm=200, k_max=2)[2] < 0.05
    assert eigengap_null(noise, n_perm=200, k_max=2)[2] > 0.05


def test_bootstrap_coclustering_is_stable_for_clear_groups():
    rng = np.random.default_rng(9)
    rows = []
    for m in range(10):
        centre = 0.0 if m < 5 else 4.0
        for s in range(3):
            rows.append({"manager": f"m{m}", "season": s, "x": centre + rng.normal(0, 0.3), "y": centre + rng.normal(0, 0.3)})
    co, ari = bootstrap_coclustering(pd.DataFrame(rows), ["x", "y"], k=2, n_boot=30)
    assert co.shape == (10, 10) and np.median(ari) > 0.9
    assert co.loc["m0", "m1"] > 0.9 and co.loc["m0", "m9"] < 0.1


def test_manager_features_is_mean_of_season_features():
    drafts = pd.DataFrame({"season": [1, 1, 2, 2], "overall": [1, 2, 1, 2], "round": [1, 1, 1, 1], "owner_id": ["a", "b", "a", "b"],
                           "player_id": [10, 11, 10, 11], "auto": [False, True, False, False]})
    players = pd.DataFrame({"season": [1, 1, 2, 2], "player_id": [10, 11, 10, 11], "adp": [1.0, 5.0, 3.0, 1.0], "pos": ["C", "G", "C", "D"], "pro_team_id": [1, 2, 1, 2]})
    sf = manager_season_features(drafts, players)
    mf = manager_features(drafts, players)
    assert len(sf) == 4 and mf.loc["a", "n_picks"] == 2
    assert np.isclose(mf.loc["b", "auto_share"], sf[sf.manager == "b"].auto_share.mean())


def test_assignment_matches_usage():
    st = points_settings()
    pool = synthetic_pool(st)
    v = Valuator(pool)
    roster = list(np.argsort(-v.pts)[:5])
    cost, rows, cols, starter = v.assignment(roster)
    assigned = {int(i) for i, j in zip(rows, cols) if cost[i, j] < 1e6}
    u = v.usage(roster)
    assert all((u[i] > 0) == (i in assigned) for i in range(len(roster)))


def test_crn_reduces_gap_noise_and_pick_distribution_shape():
    st = points_settings()
    pool = synthetic_pool(st)
    ctx = DraftContext(pool, Valuator(pool), [1, 2, 3, 4], my_team=1)
    cands, crn, ind = crn_experiment(ctx, [], n_candidates=3, n_rollouts=40)
    assert crn.shape == ind.shape == (3, 40)
    # shared futures make candidates' outcomes move together (that shared noise cancels in their gaps); independent ones don't
    corr_crn = np.mean([np.corrcoef(crn[0], crn[i])[0, 1] for i in (1, 2)])
    corr_ind = np.mean([np.corrcoef(ind[0], ind[i])[0, 1] for i in (1, 2)])
    assert corr_crn > 0.3 and corr_crn > corr_ind + 0.2
    dist = pick_distribution(ctx, n_rollouts=5)
    assert dist.shape == (5, len(pool)) and ((dist >= 0).sum(axis=1) == len(ctx.order)).all()
