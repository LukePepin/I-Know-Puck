import numpy as np
import pandas as pd

from iknowpuck.config import STAT_IDS, Category, LeagueSettings
from iknowpuck.draft import DraftContext, recommend, snake_order
from iknowpuck.opponents import OpponentModel, PickObs
from iknowpuck.spectral import spectral_clusters
from iknowpuck.valuation import PlayerPool, Valuator

G, A, W = STAT_IDS["G"], STAT_IDS["A"], STAT_IDS["W"]


def points_settings(n_teams=4):
    return LeagueSettings(
        n_teams=n_teams,
        categories=[Category(G, points=2.0), Category(A, points=1.0), Category(W, points=4.0)],
        lineup={0: 1, 1: 1, 4: 1, 5: 1, 7: 1},
        scoring_type="H2H_POINTS",
    )


def synthetic_pool(settings, n=80, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.choice(["C", "LW", "D", "G"], size=n, p=[0.3, 0.3, 0.25, 0.15])
    slots = {"C": [0, 6, 7], "LW": [1, 6, 7], "D": [4, 6, 7], "G": [5, 7]}
    skater = pos != "G"
    f = pd.DataFrame({
        "player_id": np.arange(n) + 1000,
        "name": [f"P{i}" for i in range(n)],
        "pos": pos,
        "eligible_slots": [slots[p] for p in pos],
        f"p_{G}": np.where(skater, rng.gamma(3, 6, n), 0),
        f"p_{A}": np.where(skater, rng.gamma(3, 9, n), 0),
        f"p_{W}": np.where(~skater, rng.gamma(4, 6, n), 0),
        "p_30": 70.0,
    })
    f["adp"] = (-(2 * f[f"p_{G}"] + f[f"p_{A}"] + 4 * f[f"p_{W}"])).rank() + rng.normal(0, 3, n)
    return PlayerPool(f, settings, [G, A, W])


def test_snake_order():
    assert snake_order([1, 2, 3], 3) == [1, 2, 3, 3, 2, 1, 1, 2, 3]


def test_points_valuation_prefers_better_roster():
    st = points_settings()
    pool = synthetic_pool(st)
    v = Valuator(pool)
    order = np.argsort(-v.pts)
    strong, weak = list(order[:5]), list(order[-5:])
    assert v.weekly_points(strong) > v.weekly_points(weak)
    assert v.score(strong, [weak]) > 0.5 > v.score(weak, [strong])


def test_usage_respects_slots():
    st = points_settings()
    pool = synthetic_pool(st)
    v = Valuator(pool)
    goalies = list(np.flatnonzero(pool.is_goalie)[:3])
    u = v.usage(goalies)  # 1 G slot + 1 bench: one starter, one bench, one unrostered
    assert sorted(u.tolist()) == sorted([1.0, v.goalie_bench_factor, 0.0])


def test_recommend_runs_and_ranks():
    st = points_settings()
    pool = synthetic_pool(st)
    ctx = DraftContext(pool, Valuator(pool), [1, 2, 3, 4], my_team=1)
    rec = recommend(ctx, [], n_candidates=5, n_rollouts=4)
    assert len(rec) >= 5 and rec.win_prob.is_monotonic_decreasing


def test_logit_recovers_adp_sensitivity():
    rng = np.random.default_rng(3)
    true_b = 4.0
    obs = []
    for i in range(600):
        la = np.log(rng.uniform(1, 200, 40))
        u = true_b * (-la)
        p = np.exp(u - u.max()); p /= p.sum()
        obs.append(PickObs("m", 1 + i % 20, np.arange(40), int(rng.choice(40, p=p)), la, np.zeros(40, int), np.zeros(40)))
    m = OpponentModel().fit(obs, per_manager=False)
    assert abs(m.global_[0] - true_b) < 0.6


def test_spectral_separates_two_groups():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.3, (6, 4)); b = rng.normal(3, 0.3, (6, 4))
    feats = pd.DataFrame(np.vstack([a, b]), columns=list("wxyz"), index=[f"m{i}" for i in range(12)])
    res = spectral_clusters(feats, k=2)
    assert len(set(res.labels[:6])) == 1 and len(set(res.labels[6:])) == 1 and res.labels[0] != res.labels[6]
