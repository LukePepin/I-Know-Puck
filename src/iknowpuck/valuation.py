"""Roster valuation for H2H points and H2H category leagues.

H2H POINTS: a roster's weekly fantasy points ~ Normal(mu, var) with
  mu  = sum_i u_i * pts_i / W
  var = COVAR_INFLATION * sum_i u_i * sum_k w_k^2 * disp_k * S_ik / W
(independent over-dispersed Poisson stats, inflated for within-player stat covariance, e.g. G and PPP)
and the objective is P(win the weekly matchup) = Phi((mu_me - mu_opp) / sqrt(var_me + var_opp)).

H2H CATEGORIES:

A roster's weekly total in category c is modelled as Normal(mu_c, var_c):
  counting stats : mu = sum_i u_i * S_ic / W,  var = disp_c * mu   (over-dispersed Poisson)
  +/-            : mu as above, var = sum_i u_i * GP_i / W * PM_VAR  (per-game variance)
  GAA            : 3600 * GA / TOI, delta method with GA ~ over-dispersed Poisson
  SV%            : SV / SA, binomial variance
where u_i is usage (1 for a starter, ``bench_factor`` for bench) from an optimal slot assignment
(linear_sum_assignment) and W is the number of weeks in the NHL season.

P(beat opponent in c) = Phi((mu_me - mu_opp) / sqrt(var_me + var_opp)), sign-flipped for reverse
categories. The objective is expected categories won per week, averaged over opponents.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import ndtr

from .config import BENCH_SLOT, GOALIE_SLOT, IR_SLOT, RATIO_STATS, STAT_IDS, LeagueSettings

WEEKS_IN_SEASON = 26.0
PM_VAR_PER_GAME = 1.0
COVAR_INFLATION = 1.6  # calibrated against league matchup history when available
DISPERSION = {
    STAT_IDS["G"]: 1.1, STAT_IDS["A"]: 1.1, STAT_IDS["PTS"]: 1.2, STAT_IDS["PIM"]: 3.0, STAT_IDS["PPP"]: 1.2,
    STAT_IDS["SOG"]: 1.3, STAT_IDS["HIT"]: 1.4, STAT_IDS["BLK"]: 1.3, STAT_IDS["W"]: 0.7, STAT_IDS["SO"]: 1.0,
    STAT_IDS["GA"]: 1.2, STAT_IDS["FOW"]: 1.5,
}
EPS = 1e-9


@dataclass
class PlayerPool:
    """Array view of the draftable player universe used by every optimiser."""

    frame: pd.DataFrame  # index 0..n-1, columns include player_id, name, pos, adp, p_<stat>
    settings: LeagueSettings
    stat_ids: list[int]
    S: np.ndarray = field(init=False)  # (n, n_stats) season totals
    gp: np.ndarray = field(init=False)
    elig: np.ndarray = field(init=False)  # (n, n_slot_types) bool over starting+bench slots
    slot_types: list[int] = field(init=False)
    is_goalie: np.ndarray = field(init=False)

    def __post_init__(self):
        f = self.frame.reset_index(drop=True)
        self.frame = f
        self.S = np.column_stack([f.get(f"p_{k}", pd.Series(0.0, index=f.index)).fillna(0).to_numpy(float) for k in self.stat_ids])
        self.gp = f.get("p_30", pd.Series(0.0, index=f.index)).fillna(0).to_numpy(float)
        self.slot_types = [s for s in self.settings.lineup if s != IR_SLOT]
        self.is_goalie = (f["pos"] == "G").to_numpy()
        elig = np.zeros((len(f), len(self.slot_types)), dtype=bool)
        for i, slots in enumerate(f["eligible_slots"]):
            ss = set(int(s) for s in slots)
            for j, st in enumerate(self.slot_types):
                elig[i, j] = st in ss or st == BENCH_SLOT
        self.elig = elig
        self.col = {k: i for i, k in enumerate(self.stat_ids)}

    def __len__(self) -> int:
        return len(self.frame)

    def idx_of(self, player_ids) -> list[int]:
        m = pd.Series(self.frame.index, index=self.frame.player_id)
        return [int(m[p]) for p in player_ids if p in m.index]


class Valuator:
    def __init__(self, pool: PlayerPool, bench_factor: float = 0.35, goalie_bench_factor: float = 0.2):
        self.pool = pool
        self.st = pool.settings
        self.bench_factor = bench_factor
        self.goalie_bench_factor = goalie_bench_factor
        self.points_mode = self.st.is_points
        self.cats = self.st.scoring_categories
        if self.points_mode:
            w = np.array([c.points for c in self.cats])
            cols = [pool.col[c.stat_id] for c in self.cats]
            disp = np.array([DISPERSION.get(c.stat_id, 1.2) for c in self.cats])
            self.pts = pool.S[:, cols] @ w  # season fantasy points per player
            self.pts_var = (pool.S[:, cols].clip(min=0) * disp) @ (w**2) * COVAR_INFLATION  # season var
        # starting slot list expanded by count (bench included for assignment, IR excluded)
        self.slot_list: list[int] = []
        for s in pool.slot_types:
            self.slot_list += [s] * self.st.lineup[s]
        self._z_weight = self._player_weights()

    # --- helpers ------------------------------------------------------------------------------
    def _player_weights(self) -> np.ndarray:
        """Scalar per-player value used only to decide who starts in the slot assignment:
        fantasy points (points leagues) or summed category z-scores (category leagues)."""
        if self.st.is_points:
            return self.pts.copy()
        n_top = min(len(self.pool), self.st.n_teams * self.st.rounds)
        order = np.argsort(-self.pool.S.sum(axis=1))[:n_top]
        z = np.zeros(len(self.pool))
        for c in self.cats:
            if c.stat_id in RATIO_STATS:
                num, den, scale = RATIO_STATS[c.stat_id]
                a = self.pool.S[:, self.pool.col[num]]
                b = self.pool.S[:, self.pool.col[den]]
                rate = np.divide(a * scale, b, out=np.zeros_like(a), where=b > 0)
                lg = a[order].sum() * scale / max(b[order].sum(), EPS)
                v = (rate - lg) * b  # volume-weighted contribution
                if c.reverse:
                    v = -v
            else:
                v = self.pool.S[:, self.pool.col[c.stat_id]] * (-1 if c.reverse else 1)
            sd = v[order].std() or 1.0
            z += (v - v[order].mean()) / sd
        return z

    def usage(self, roster: list[int]) -> np.ndarray:
        """Optimal starter assignment -> usage weight per rostered player."""
        if not roster:
            return np.zeros(0)
        r = np.asarray(roster)
        slot_idx = [self.pool.slot_types.index(s) for s in self.slot_list]
        elig = self.pool.elig[np.ix_(r, slot_idx)]
        w = self._z_weight[r] - self._z_weight.min() + 1.0
        starter = np.array([s != BENCH_SLOT for s in self.slot_list])
        cost = np.where(elig, -(w[:, None] * np.where(starter, 1.0, 1e-3)[None, :]), 1e6)
        rows, cols = linear_sum_assignment(cost)
        u = np.zeros(len(r))
        is_g = self.pool.is_goalie[r]
        for i, j in zip(rows, cols):
            if cost[i, j] >= 1e6:
                continue
            u[i] = 1.0 if starter[j] else (self.goalie_bench_factor if is_g[i] else self.bench_factor)
        return u

    def team_dist(self, roster: list[int], usage: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Weekly (mu, var): one dimension (points) or one per category."""
        r = np.asarray(roster, dtype=int)
        u = self.usage(roster) if usage is None else usage
        if self.points_mode:
            if not len(r):
                return np.zeros(1), np.full(1, 1.0)
            return (np.array([(u * self.pts[r]).sum() / WEEKS_IN_SEASON]), np.array([(u * self.pts_var[r]).sum() / WEEKS_IN_SEASON + 1.0]))
        tot = (u[:, None] * self.pool.S[r]).sum(axis=0) / WEEKS_IN_SEASON if len(r) else np.zeros(len(self.pool.stat_ids))
        gpw = float((u * self.pool.gp[r]).sum()) / WEEKS_IN_SEASON if len(r) else 0.0
        return self.dist_from_totals(tot, gpw)

    def dist_from_totals(self, tot: np.ndarray, gp_week: float) -> tuple[np.ndarray, np.ndarray]:
        col = self.pool.col
        mu = np.zeros(len(self.cats))
        var = np.zeros(len(self.cats))
        for ci, c in enumerate(self.cats):
            k = c.stat_id
            if k == 10:  # GAA
                ga, toi = tot[col[4]], tot[col[8]]
                mu[ci] = 3600 * ga / max(toi, EPS)
                var[ci] = (3600 / max(toi, EPS)) ** 2 * DISPERSION.get(4, 1.2) * ga + 0.05
            elif k == 11:  # SV%
                sv, sa = tot[col[6]], tot[col[3]]
                p = sv / max(sa, EPS)
                mu[ci] = p
                var[ci] = p * (1 - p) / max(sa, 1.0) + 1e-6
            elif k == 15:
                mu[ci] = tot[col[k]]
                var[ci] = PM_VAR_PER_GAME * gp_week + 0.5
            else:
                mu[ci] = tot[col[k]]
                var[ci] = DISPERSION.get(k, 1.2) * max(mu[ci], 0.0) + 0.25
        return mu, var

    def win_probs(self, mine: tuple[np.ndarray, np.ndarray], opp: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Per-dimension win probability (a single matchup probability in points mode)."""
        sign = np.ones(1) if self.points_mode else np.array([-1.0 if c.reverse else 1.0 for c in self.cats])
        z = sign * (mine[0] - opp[0]) / np.sqrt(mine[1] + opp[1])
        return ndtr(z)

    def score(self, roster: list[int], opponents: list[list[int]]) -> float:
        """The optimisation objective: P(win weekly matchup) averaged over opponents.
        Points: exact Normal. Categories: Poisson-binomial over categories."""
        if self.points_mode:
            me = self.team_dist(roster)
            return float(np.mean([self.win_probs(me, self.team_dist(o))[0] for o in opponents]))
        return self.matchup_win_prob(roster, opponents)

    def weekly_points(self, roster: list[int]) -> float:
        return float(self.team_dist(roster)[0][0]) if self.points_mode else float("nan")

    def expected_cat_wins(self, roster: list[int], opponents: list[list[int]]) -> float:
        """Mean expected categories won per week against each opponent roster."""
        me = self.team_dist(roster)
        return float(np.mean([self.win_probs(me, self.team_dist(o)).sum() for o in opponents]))

    def matchup_win_prob(self, roster: list[int], opponents: list[list[int]]) -> float:
        """P(win the weekly matchup, i.e. > half the categories), Poisson-binomial over categories."""
        me = self.team_dist(roster)
        n = len(self.cats)
        out = []
        for o in opponents:
            p = self.win_probs(me, self.team_dist(o))
            dist = np.zeros(n + 1)
            dist[0] = 1.0
            for pc in p:
                dist[1:] = dist[1:] * (1 - pc) + dist[:-1] * pc
                dist[0] *= 1 - pc
            out.append(dist[n // 2 + 1 :].sum() + 0.5 * (dist[n // 2] if n % 2 == 0 else 0.0))
        return float(np.mean(out))

    def season_record(self, rosters: list[list[int]]) -> np.ndarray:
        """Round-robin expected category win share for every team (ranking a whole league)."""
        dists = [self.team_dist(r) for r in rosters]
        n = len(rosters)
        share = np.zeros(n)
        for i in range(n):
            share[i] = np.mean([self.win_probs(dists[i], dists[j]).mean() for j in range(n) if j != i])
        return share
