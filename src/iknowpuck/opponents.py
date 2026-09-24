"""Opponent pick model: a conditional (McFadden) logit over the available players.

    P(manager m picks j | available A, pick t, roster) = exp(u_mj) / sum_{k in A} exp(u_mk)
    u_mj = b_adp[m] * (-log ADP_j) + b_need * need_j(roster) + sum_g,phase b_pos[m, g, phase] * 1{pos_j = g}

Global coefficients are fit by maximum likelihood on every historical pick. Manager-specific
deviations are L2-shrunk toward a *group* mean, where groups come from spectral clustering of
manager behaviour (see ``spectral.py``). With only 2-4 drafts per manager, pooling is what makes
per-manager estimates stable; experiment H3/H4 tests whether it actually predicts better.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

GROUPS = ("F", "D", "G")
PHASES = ((1, 4), (5, 10), (11, 99))  # round ranges
N_POS = len(GROUPS) * len(PHASES)
ADP_MISSING = 250.0
CHOICE_SET = 300  # consider the top-N remaining players by ADP (+ the actual pick)


def phase_of(rnd: int) -> int:
    return next(i for i, (a, b) in enumerate(PHASES) if a <= rnd <= b)


def group_of(pos: str) -> int:
    return 2 if pos == "G" else (1 if pos == "D" else 0)


@dataclass
class PickObs:
    manager: str
    round: int
    avail: np.ndarray  # indices into the season's player table (choice set)
    chosen: int  # position of the chosen player within avail
    log_adp: np.ndarray
    grp: np.ndarray
    need: np.ndarray


def need_vector(grp: np.ndarray, roster_groups: list[int], target_counts: dict[int, int]) -> np.ndarray:
    have = np.bincount(np.asarray(roster_groups, dtype=int), minlength=3) if roster_groups else np.zeros(3, int)
    open_ = np.array([max(target_counts.get(g, 0) - have[g], 0) for g in range(3)], dtype=float)
    return (open_[grp] > 0).astype(float)


def default_target_counts(lineup: dict[int, int]) -> dict[int, int]:
    f = sum(lineup.get(s, 0) for s in (0, 1, 2, 3)) + lineup.get(6, 0)
    return {0: f, 1: lineup.get(4, 0), 2: lineup.get(5, 0)}


def build_observations(drafts: pd.DataFrame, players: pd.DataFrame, lineup: dict[int, int]) -> list[PickObs]:
    """drafts: season, overall, round, owner_id, player_id. players: season, player_id, adp, pos."""
    targets = default_target_counts(lineup)
    obs: list[PickObs] = []
    for season, d in drafts.groupby("season"):
        pl = players[players.season == season].drop_duplicates("player_id").set_index("player_id")
        ids = pl.index.to_numpy()
        pos_of = {pid: i for i, pid in enumerate(ids)}
        log_adp = np.log(pl["adp"].fillna(ADP_MISSING).clip(lower=1).to_numpy())
        grp = pl["pos"].map(group_of).to_numpy()
        taken = np.zeros(len(ids), bool)
        rosters: dict[str, list[int]] = {}
        order = np.argsort(log_adp)
        for _, p in d.sort_values("overall").iterrows():
            j = pos_of.get(p.player_id)
            mgr = str(p.owner_id)
            if j is None:
                continue
            if not p.get("keeper", False):
                avail_all = order[~taken[order]]
                cs = avail_all[:CHOICE_SET]
                if j not in cs:
                    cs = np.append(cs, j)
                obs.append(
                    PickObs(
                        manager=mgr,
                        round=int(p["round"]),
                        avail=cs,
                        chosen=int(np.where(cs == j)[0][0]),
                        log_adp=log_adp[cs],
                        grp=grp[cs],
                        need=need_vector(grp[cs], rosters.get(mgr, []), targets),
                    )
                )
            taken[j] = True
            rosters.setdefault(mgr, []).append(int(grp[j]))
    return obs


@dataclass
class ObsArrays:
    """Observations padded to a rectangle so the likelihood is fully vectorised."""

    la: np.ndarray  # (n, C) -log ADP
    grp: np.ndarray  # (n, C) int
    need: np.ndarray  # (n, C)
    mask: np.ndarray  # (n, C) bool, valid choice
    chosen: np.ndarray  # (n,)
    phase: np.ndarray  # (n,)
    mgr: np.ndarray  # (n,) index into managers, -1 = unknown

    @classmethod
    def from_obs(cls, obs: list[PickObs], managers: list[str]) -> "ObsArrays":
        n, C = len(obs), max(len(o.avail) for o in obs)
        la = np.zeros((n, C)); grp = np.zeros((n, C), int); need = np.zeros((n, C)); mask = np.zeros((n, C), bool)
        midx = {m: i for i, m in enumerate(managers)}
        for i, o in enumerate(obs):
            c = len(o.avail)
            la[i, :c], grp[i, :c], need[i, :c], mask[i, :c] = -o.log_adp, o.grp, o.need, True
        return cls(
            la, grp, need, mask,
            np.array([o.chosen for o in obs]),
            np.array([phase_of(o.round) for o in obs]),
            np.array([midx.get(o.manager, -1) for o in obs]),
        )


def _pick_loglik(theta_obs: np.ndarray, b_need: float, A: ObsArrays) -> np.ndarray:
    """theta_obs: (n, 1 + N_POS) coefficients for the manager making each pick."""
    pos = theta_obs[:, 1:].reshape(-1, len(GROUPS), len(PHASES))[np.arange(len(A.phase)), :, A.phase]  # (n, 3)
    pos = pos - pos[:, :1]  # forwards are the reference group (identifiability)
    u = theta_obs[:, :1] * A.la + b_need * A.need + np.take_along_axis(pos, A.grp, axis=1)
    u = np.where(A.mask, u, -np.inf)
    return u[np.arange(len(u)), A.chosen] - logsumexp(u, axis=1)


@dataclass
class OpponentModel:
    """theta per manager: [b_adp, b_pos(3 groups x 3 phases)] ; shared: b_need."""

    managers: list[str] = field(default_factory=list)
    group_of_manager: dict[str, int] = field(default_factory=dict)
    shrink: float = 2.0
    global_: np.ndarray = field(default_factory=lambda: np.r_[6.0, np.zeros(N_POS)])
    b_need: float = 1.0
    per_mgr: dict[str, np.ndarray] = field(default_factory=dict)

    def _thetas(self, g, grp_dev, mgr_dev) -> np.ndarray:
        out = np.tile(g, (len(self.managers), 1))
        for i, m in enumerate(self.managers):
            gi = self.group_of_manager.get(m, -1)
            if gi >= 0 and len(grp_dev):
                out[i] += grp_dev[gi]
            out[i] += mgr_dev[i]
        return out

    def fit(self, obs: list[PickObs], per_manager: bool = True) -> "OpponentModel":
        k = 1 + N_POS
        self.managers = sorted({o.manager for o in obs})
        A = ObsArrays.from_obs(obs, self.managers)

        def nll_global(x):
            return -_pick_loglik(np.tile(x[:k], (len(A.chosen), 1)), x[k], A).sum() + 1e-3 * float((x[1:k] ** 2).sum())

        res = minimize(nll_global, np.r_[self.global_, self.b_need], method="L-BFGS-B")
        self.global_, self.b_need = res.x[:k], float(res.x[k])
        self.per_mgr = {}
        if per_manager and self.managers:
            n_groups = max(self.group_of_manager.values(), default=-1) + 1

            def unpack(x):
                gd = x[: n_groups * k].reshape(n_groups, k) if n_groups else np.zeros((0, k))
                return gd, x[n_groups * k :].reshape(len(self.managers), k)

            def nll_mgr(x):
                gd, md = unpack(x)
                th = self._thetas(self.global_, gd, md)
                th_obs = np.where((A.mgr >= 0)[:, None], th[np.clip(A.mgr, 0, None)], self.global_)
                pen = self.shrink * float((md**2).sum()) + 0.5 * self.shrink * float((gd**2).sum())
                return -_pick_loglik(th_obs, self.b_need, A).sum() + pen

            res = minimize(nll_mgr, np.zeros((n_groups + len(self.managers)) * k), method="L-BFGS-B", options={"maxiter": 500})
            th = self._thetas(self.global_, *unpack(res.x))
            self.per_mgr = {m: th[i] for i, m in enumerate(self.managers)}
        return self

    def theta(self, manager: str | None) -> np.ndarray:
        return self.per_mgr.get(str(manager), self.global_) if manager is not None else self.global_

    def log_likelihood(self, obs: list[PickObs]) -> np.ndarray:
        """Per-pick log-likelihood (the unit of analysis in H3/H4)."""
        A = ObsArrays.from_obs(obs, self.managers)
        th_obs = np.array([self.theta(o.manager) for o in obs])
        return _pick_loglik(th_obs, self.b_need, A)

    def utilities(self, manager: str | None, rnd: int, neg_log_adp: np.ndarray, grp: np.ndarray, need: np.ndarray) -> np.ndarray:
        th = self.theta(manager)
        pos = th[1:].reshape(len(GROUPS), len(PHASES))[:, phase_of(rnd)]
        pos = pos - pos[0]
        return th[0] * neg_log_adp + self.b_need * need + pos[grp]

    def describe(self) -> pd.DataFrame:
        rows = []
        for m, th in {"(league)": self.global_, **self.per_mgr}.items():
            pos = th[1:].reshape(len(GROUPS), len(PHASES))
            pos = pos - pos[0]
            row = {"manager": m, "adp_sensitivity": th[0]}
            for gi, g in enumerate(GROUPS[1:], start=1):
                for pi, (a, b) in enumerate(PHASES):
                    row[f"{g} bias R{a}-{b if b < 99 else '+'}"] = pos[gi, pi]
            rows.append(row)
        return pd.DataFrame(rows)
