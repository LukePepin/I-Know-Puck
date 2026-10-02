import numpy as np
import pandas as pd

from iknowpuck import team_effects as T


def _league(team_sd: float, seed: int = 0, n_teams: int = 16, per_team: int = 12, games: int = 60, seasons=(2024, 2025)):
    """Synthetic game logs: each player's per-game points = projected rate x (1 + team factor + player factor) + noise."""
    rng = np.random.default_rng(seed)
    logs, rates = [], []
    pid = 0
    for s in seasons:
        for t in range(1, n_teams + 1):
            tau = rng.normal(0, team_sd)
            for _ in range(per_team):
                pid += 1
                rate = rng.uniform(0.8, 3.0)
                eps = rng.normal(0, 0.15)
                pts = np.clip(rate * (1 + tau + eps) + rng.normal(0, 1.0, games), 0, None)
                logs.append(pd.DataFrame({"season": s, "player_id": pid, "pro_team_id": t, "scoring_period": np.arange(1, games + 1), "fp": pts}))
                rates.append({"season": s, "player_id": pid, "name": f"P{pid}", "pos": "C", "grp": "F", "proj_rate": rate, "proj_w_rate": 0.0})
    sched = pd.DataFrame({"season": np.repeat(list(seasons), games), "scoring_period": np.tile(np.arange(1, games + 1), len(seasons))})
    return pd.concat(logs, ignore_index=True), pd.DataFrame(rates), sched


def test_team_effect_detects_a_real_team_factor():
    gl, rates, _ = _league(team_sd=0.12, seed=1)
    d = T.add_team_surprise(T.stints(gl, rates), min_games=20)
    e = T.team_effect(d, n_perm=300, n_boot=300)
    assert e["p"] < 0.01 and e["lo"] > 0.3 and T.icc(d) > 0.1


def test_team_effect_is_not_fooled_by_noise():
    ps = []
    for seed in range(4):
        gl, rates, _ = _league(team_sd=0.0, seed=10 + seed)
        ps.append(T.team_effect(T.add_team_surprise(T.stints(gl, rates), min_games=20), n_perm=300, n_boot=100)["p"])
    assert np.median(ps) > 0.1  # leave-one-out keeps a player's own points out of his team's surprise


def test_leave_one_out_excludes_the_player():
    gl, rates, _ = _league(team_sd=0.0, seed=3, n_teams=2, per_team=3, seasons=(2025,))
    d = T.add_team_surprise(T.stints(gl, rates), min_games=1)
    r = d.iloc[0]
    mates = d[(d.pro_team_id == r.pro_team_id) & (d.player_id != r.player_id)]
    assert np.isclose(r.team_surprise, mates.fp.sum() / mates.exp.sum() - 1)


def test_quantiles_and_halves_shapes():
    gl, rates, sched = _league(team_sd=0.1, seed=4)
    d = T.add_team_surprise(T.stints(gl, rates), min_games=20)
    q = T.by_quantile(d, 5)
    assert len(q) == 5 and (q.lo <= q.beat).all() and (q.beat <= q.hi).all() and q.team.is_monotonic_increasing
    w = T.halves(gl, sched, rates, min_games=10)
    assert len(w) > 0 and {"own1", "team1", "surprise2"} <= set(w.columns)
    pz = T.persistence(w, n_perm=100)
    er = T.loso_errors(w)
    assert 0 <= pz["p"] <= 1 and len({len(v) for v in er.values()}) == 1
