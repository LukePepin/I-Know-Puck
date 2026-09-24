"""Snake-draft Monte Carlo simulator and pick recommender.

Recommendation = argmax over candidates c of  E[ score(my final roster) | I take c now ],
estimated by rolling the rest of the draft forward M times:
  - opponents pick by sampling their conditional-logit model (per-manager tendencies)
  - my later picks follow a fast greedy policy (value over next available, VONA)
  - the finished league is scored with the Valuator (P(win weekly matchup) vs the 11 real rosters)
Every candidate is evaluated on the same M random seeds (common random numbers), so the
comparison between candidates is paired and far less noisy than independent rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import BENCH_SLOT, IR_SLOT, LeagueSettings
from .opponents import OpponentModel, default_target_counts, group_of
from .valuation import PlayerPool, Valuator

OPP_CHOICE_SET = 150
SPECIFIC_FIRST = (0, 1, 2, 4, 5, 3, 6, BENCH_SLOT)  # fill C/LW/RW/D/G before F/UTIL/bench


def snake_order(first_round: list[int], rounds: int) -> list[int]:
    out: list[int] = []
    for r in range(rounds):
        out += first_round if r % 2 == 0 else first_round[::-1]
    return out


def effective_adp(frame: pd.DataFrame, value: np.ndarray) -> np.ndarray:
    """ADP where available; undrafted players are ordered after everyone by projected value."""
    adp = frame["adp"].to_numpy(float)
    have = np.isfinite(adp)
    base = np.nanmax(adp) if have.any() else 0.0
    rank_val = np.argsort(np.argsort(-value)).astype(float)
    return np.where(have, adp, base + 1 + rank_val)


@dataclass
class TeamState:
    roster: list[int] = field(default_factory=list)
    open: dict[int, int] = field(default_factory=dict)  # slot -> remaining capacity
    groups: list[int] = field(default_factory=list)


class DraftContext:
    """Everything that is fixed for one draft: pool, values, order, opponent models."""

    def __init__(
        self,
        pool: PlayerPool,
        valuator: Valuator,
        first_round: list[int],
        my_team: int,
        opp_model: OpponentModel | None = None,
        manager_of_team: dict[int, str] | None = None,
    ):
        self.pool = pool
        self.val = valuator
        self.st: LeagueSettings = pool.settings
        self.rounds = self.st.rounds
        self.order = snake_order(first_round, self.rounds)
        self.teams = list(first_round)
        self.my_team = my_team
        self.opp = opp_model or OpponentModel()
        self.mgr = manager_of_team or {}
        self.value = valuator.pts if valuator.points_mode else valuator._z_weight
        self.adp = effective_adp(pool.frame, self.value)
        self.neg_log_adp = -np.log(np.clip(self.adp, 1, None))
        self.adp_order = np.argsort(self.adp)
        self.grp = pool.frame["pos"].map(group_of).to_numpy()
        self.targets = default_target_counts(self.st.lineup)
        self.slot_cap = {s: c for s, c in self.st.lineup.items() if s != IR_SLOT}
        self.slot_col = {s: pool.slot_types.index(s) for s in self.slot_cap}
        self.max_goalies = self.st.lineup.get(5, 2) + 2

    # --- state helpers ------------------------------------------------------------------------
    def new_team(self) -> TeamState:
        return TeamState(open=dict(self.slot_cap))

    def place(self, t: TeamState, j: int) -> bool:
        """Put player j in the most specific open eligible slot. False if no room at all."""
        for s in SPECIFIC_FIRST:
            if self.slot_cap.get(s, 0) and t.open.get(s, 0) > 0 and self.pool.elig[j, self.slot_col[s]]:
                t.open[s] -= 1
                t.roster.append(j)
                t.groups.append(int(self.grp[j]))
                return True
        return False

    def starter_fit(self, t: TeamState, cand: np.ndarray) -> np.ndarray:
        """Usage weight per candidate: 1 if it fills an open starting slot, bench factor if only
        bench is open, 0 if the roster has no room for it."""
        start = np.zeros(len(cand), bool)
        for s, c in t.open.items():
            if s != BENCH_SLOT and c > 0:
                start |= self.pool.elig[cand, self.slot_col[s]]
        bench_open = t.open.get(BENCH_SLOT, 0) > 0
        is_g = self.pool.is_goalie[cand]
        bench_u = np.where(is_g, self.val.goalie_bench_factor, self.val.bench_factor) if bench_open else 0.0
        return np.where(start, 1.0, bench_u)

    def initial_state(self, picks: list[tuple[int, int]]) -> tuple[np.ndarray, dict[int, TeamState]]:
        """picks: [(team_id, pool_index)] already made, in order."""
        taken = np.zeros(len(self.pool), bool)
        teams = {tid: self.new_team() for tid in self.teams}
        for tid, j in picks:
            taken[j] = True
            if not self.place(teams[tid], j):
                teams[tid].roster.append(j)
                teams[tid].groups.append(int(self.grp[j]))
        return taken, teams

    # --- policies -----------------------------------------------------------------------------
    def greedy_scores(self, t: TeamState, taken: np.ndarray, pick_no: int, avail: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """VONA: usage-weighted value minus the value of the best same-group player expected to
        remain at my next pick (approximated as the k-th best remaining, k = picks until then)."""
        if avail is None:
            avail = np.flatnonzero(~taken)
        v = self.value[avail]
        u = self.starter_fit(t, avail)
        k = self.picks_until_next(pick_no)
        repl = np.zeros(3)
        for g in range(3):
            vals = np.sort(v[self.grp[avail] == g])[::-1]
            repl[g] = vals[min(k, len(vals) - 1)] if len(vals) else 0.0
        return avail, u * (v - repl[self.grp[avail]]) + 1e-6 * v

    def picks_until_next(self, pick_no: int) -> int:
        me = self.order[pick_no] if pick_no < len(self.order) else self.my_team
        for d, tid in enumerate(self.order[pick_no + 1 :], start=1):
            if tid == me:
                return d
        return 0

    def opponent_pick(self, tid: int, t: TeamState, taken: np.ndarray, pick_no: int, rng: np.random.Generator) -> int:
        avail = self.adp_order[~taken[self.adp_order]][:OPP_CHOICE_SET]
        g = self.grp[avail]
        have = np.bincount(np.asarray(t.groups, int), minlength=3) if t.groups else np.zeros(3, int)
        open_g = np.array([self.targets[x] - have[x] for x in range(3)])
        need = (open_g[g] > 0).astype(float)
        u = self.opp.utilities(self.mgr.get(tid), pick_no // len(self.teams) + 1, self.neg_log_adp[avail], g, need)
        fits = self.starter_fit(t, avail) > 0
        u = np.where(fits, u, -np.inf)
        if have[2] >= self.max_goalies:
            u = np.where(g == 2, -np.inf, u)
        if not np.isfinite(u).any():
            return int(avail[0])
        return int(avail[np.argmax(u + rng.gumbel(size=len(u)))])

    # --- rollouts -----------------------------------------------------------------------------
    def rollout(self, taken: np.ndarray, teams: dict[int, TeamState], pick_no: int, first: int | None, rng: np.random.Generator, record_next: bool = False):
        taken = taken.copy()
        teams = {k: TeamState(list(v.roster), dict(v.open), list(v.groups)) for k, v in teams.items()}
        avail_at_next = None
        first_done = False
        for p in range(pick_no, len(self.order)):
            tid = self.order[p]
            t = teams[tid]
            if tid == self.my_team:
                if record_next and p > pick_no and avail_at_next is None:
                    avail_at_next = ~taken
                if first is not None and not first_done:
                    j = first
                else:
                    avail, sc = self.greedy_scores(t, taken, p)
                    j = int(avail[np.argmax(sc)])
                first_done = True
            else:
                j = self.opponent_pick(tid, t, taken, p, rng)
            taken[j] = True
            if not self.place(t, j):
                t.roster.append(j)
                t.groups.append(int(self.grp[j]))
        mine = teams[self.my_team].roster
        others = [teams[k].roster for k in self.teams if k != self.my_team]
        return self.val.score(mine, others), avail_at_next, teams


def recommend(
    ctx: DraftContext,
    picks: list[tuple[int, int]],
    n_candidates: int = 10,
    n_rollouts: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """Rank candidate picks for the team on the clock by simulated P(win weekly matchup)."""
    pick_no = len(picks)
    if pick_no >= len(ctx.order):
        return pd.DataFrame()
    taken, teams = ctx.initial_state(picks)
    on_clock = ctx.order[pick_no]
    avail, sc = ctx.greedy_scores(teams[on_clock], taken, pick_no)
    top_greedy = avail[np.argsort(-sc)[:n_candidates]]
    top_adp = ctx.adp_order[~taken[ctx.adp_order]][: max(3, n_candidates // 3)]
    cands = list(dict.fromkeys([*top_greedy.tolist(), *[j for j in top_adp.tolist() if ctx.starter_fit(teams[on_clock], np.array([j]))[0] > 0]]))

    scores = np.zeros((len(cands), n_rollouts))
    avail_next = np.zeros(len(ctx.pool))
    my_ctx_team = ctx.my_team
    ctx.my_team = on_clock  # evaluate from the perspective of whoever is on the clock
    try:
        for r in range(n_rollouts):
            for ci, c in enumerate(cands):
                rng = np.random.default_rng([seed, r])  # common random numbers across candidates
                s, an, _ = ctx.rollout(taken, teams, pick_no, c, rng, record_next=(ci == 0))
                scores[ci, r] = s
                if an is not None:
                    avail_next += an
    finally:
        ctx.my_team = my_ctx_team

    mean = scores.mean(axis=1)
    se = scores.std(axis=1, ddof=1) / np.sqrt(n_rollouts) if n_rollouts > 1 else np.zeros(len(cands))
    best = scores.argmax(axis=0)
    p_best = np.bincount(best, minlength=len(cands)) / n_rollouts
    diff_vs_best = scores - scores[np.argmax(mean)]
    se_diff = diff_vs_best.std(axis=1, ddof=1) / np.sqrt(n_rollouts) if n_rollouts > 1 else np.zeros(len(cands))
    f = ctx.pool.frame
    vona_of = dict(zip(avail.tolist(), sc.tolist()))
    out = pd.DataFrame(
        {
            "pool_idx": cands,
            "player_id": f.loc[cands, "player_id"].to_numpy(),
            "name": f.loc[cands, "name"].to_numpy(),
            "pos": f.loc[cands, "pos"].to_numpy(),
            "proj_value": ctx.value[cands],
            "adp": f.loc[cands, "adp"].to_numpy(),
            "vona": [float(vona_of.get(c, np.nan)) for c in cands],
            "win_prob": mean,
            "se": se,
            "gap_to_best": mean - mean.max(),
            "gap_se": se_diff,
            "p_best": p_best,
            "p_avail_next_pick": avail_next[cands] / n_rollouts,
        }
    ).sort_values("win_prob", ascending=False)
    return out.reset_index(drop=True)


def availability(ctx: DraftContext, picks: list[tuple[int, int]], n_rollouts: int = 60, seed: int = 1) -> np.ndarray:
    """P(each player is still available when I next pick), from policy rollouts."""
    taken, teams = ctx.initial_state(picks)
    acc = np.zeros(len(ctx.pool))
    n = 0
    for r in range(n_rollouts):
        _, an, _ = ctx.rollout(taken, teams, len(picks), None, np.random.default_rng([seed, r]), record_next=True)
        if an is not None:
            acc += an
            n += 1
    return acc / max(n, 1)
