"""Pre-registered experiment suite.   Run:  python -m iknowpuck.experiments [--quick]

H1  blend projections beat ESPN projections        unit: player-season (held-out 2024-26), |fantasy-pt error|
H1b MoneyPuck features improve the own model        unit: player-season, |fantasy-pt error|
H2  VONA draft policy beats ADP drafting            unit: (season, draft slot, seed); final roster scored on
                                                    ACTUAL season stats -> P(win weekly matchup) vs the league
H3  per-manager logit predicts picks better than a  unit: held-out pick (leave-one-season-out), log-likelihood
    league-wide ADP logit
H4  spectral-cluster pooling improves per-manager   unit: held-out pick, log-likelihood
    models over independent shrinkage
"""

from __future__ import annotations

import argparse
from functools import lru_cache

import numpy as np
import pandas as pd

from puckcore.experiment import Experiment, ExperimentOutput, run_suite

from .config import RUNS_DIR, load_credentials
from .data.dataset import build_panel
from .data.espn import EspnClient
from .draft import DraftContext
from .opponents import OpponentModel, build_observations
from .pipeline import FIRST_PANEL_SEASON, fantasy_points
from .projections import Blender, OwnModel, fill_projection, modeled_stats, season_frame
from .spectral import manager_features, spectral_clusters
from .valuation import PlayerPool, Valuator

TEST_SEASONS = (2024, 2025, 2026)


@lru_cache(maxsize=1)
def _data():
    creds = load_credentials()
    client = EspnClient(creds)
    settings = client.settings(creds.season)
    panel = build_panel(list(range(FIRST_PANEL_SEASON, creds.season + 1)), client)
    past = [s for s in client.seasons_available() if s < creds.season]
    drafts = pd.concat([client.draft(s) for s in past], ignore_index=True)
    return settings, panel, drafts


@lru_cache(maxsize=8)
def _season_projection(target: int, use_moneypuck: bool = True) -> pd.DataFrame:
    """Own + blend projections for ``target`` using only data from earlier seasons."""
    settings, panel, _ = _data()
    stats = modeled_stats(settings)
    train_targets = list(range(FIRST_PANEL_SEASON + 3, target))
    oos = []
    for t in range(target - 3, target):
        if t <= FIRST_PANEL_SEASON + 3:
            continue
        m = OwnModel(stats, use_moneypuck).fit(panel, list(range(FIRST_PANEL_SEASON + 3, t)))
        oos.append(season_frame(panel, m, t))
    blender = Blender(stats).fit(pd.concat(oos))
    model = OwnModel(stats, use_moneypuck).fit(panel, train_targets)
    return blender.transform(season_frame(panel, model, target)).reset_index()


def _eval_frame(use_moneypuck: bool = True) -> pd.DataFrame:
    settings, _, _ = _data()
    out = []
    for t in TEST_SEASONS:
        f = _season_projection(t, use_moneypuck)
        f = f[f.proj_30.notna() & f.act_30.notna() & (f.act_30 > 0)].copy()
        for src in ("proj", "own", "blend", "act"):
            f[f"fp_{src}"] = fantasy_points(f, settings, src)
        out.append(f)
    return pd.concat(out, ignore_index=True)


# --- H1 / H1b ----------------------------------------------------------------------------------
def h1(seed: int) -> ExperimentOutput:
    f = _eval_frame()
    return ExperimentOutput(
        (f.fp_proj - f.fp_act).abs().to_numpy(),
        (f.fp_blend - f.fp_act).abs().to_numpy(),
        unit="player-season (ESPN-projected players, 2024-26)",
        extras={"mae_espn": float((f.fp_proj - f.fp_act).abs().mean()), "mae_own": float((f.fp_own - f.fp_act).abs().mean()),
                "corr_espn": float(f.fp_proj.corr(f.fp_act)), "corr_blend": float(f.fp_blend.corr(f.fp_act))},
    )


def h1b(seed: int) -> ExperimentOutput:
    a = _eval_frame(True).set_index(["season", "player_id"])
    b = _eval_frame(False).set_index(["season", "player_id"])
    j = a[["fp_own", "fp_act"]].join(b[["fp_own"]], rsuffix="_nomp", how="inner")
    return ExperimentOutput((j.fp_own_nomp - j.fp_act).abs().to_numpy(), (j.fp_own - j.fp_act).abs().to_numpy(), unit="player-season")


# --- H2 ----------------------------------------------------------------------------------------
def _backtest_context(season: int, opp: OpponentModel, mgr: dict[int, str]) -> tuple[DraftContext, Valuator]:
    settings, panel, _ = _data()
    stats = modeled_stats(settings)
    f = _season_projection(season)
    f = f[(f.adp.notna()) | (f.blend_30.fillna(0) > 0)].copy()
    proj = fill_projection(f, stats, "blend")
    proj = proj.assign(_k=np.where(proj.adp.notna(), proj.adp, 1000)).nsmallest(700, "_k").drop(columns="_k").reset_index(drop=True)
    pool_proj = PlayerPool(proj, settings, stats)
    truth = fill_projection(proj, stats, "act")
    pool_true = PlayerPool(truth, settings, stats)
    order = settings.pick_order or list(range(1, settings.n_teams + 1))
    ctx = DraftContext(pool_proj, Valuator(pool_proj), order, order[0], opp, mgr)
    return ctx, Valuator(pool_true)


def _adp_policy(ctx: DraftContext):
    def choose(t, taken, p):
        avail = ctx.adp_order[~taken[ctx.adp_order]][:60]
        fits = ctx.starter_fit(t, avail)
        return int(avail[np.argmax(fits > 0.5)]) if (fits > 0.5).any() else int(avail[np.argmax(fits > 0)])
    return choose


def _run_draft(ctx: DraftContext, me: int, policy, rng) -> dict[int, list[int]]:
    taken, teams = ctx.initial_state([])
    for p, tid in enumerate(ctx.order):
        t = teams[tid]
        if tid == me:
            j = policy(t, taken, p)
        else:
            j = ctx.opponent_pick(tid, t, taken, p, rng)
        taken[j] = True
        if not ctx.place(t, j):
            t.roster.append(j)
    return {k: v.roster for k, v in teams.items()}


def h2(seed: int, n_seeds: int = 4) -> ExperimentOutput:
    settings, panel, drafts = _data()
    opp = OpponentModel().fit(build_observations(drafts, panel, settings.lineup), per_manager=False)
    base, treat, rows = [], [], []
    for season in TEST_SEASONS:
        ctx, v_true = _backtest_context(season, opp, {})
        adp_pol = _adp_policy(ctx)

        def vona_pol(t, taken, p):
            avail, sc = ctx.greedy_scores(t, taken, p)
            return int(avail[np.argmax(sc)])

        for me in ctx.teams:
            ctx.my_team = me
            for s in range(n_seeds):
                out = []
                for pol in (adp_pol, vona_pol):
                    rosters = _run_draft(ctx, me, pol, np.random.default_rng([seed, season, me, s]))
                    others = [r for k, r in rosters.items() if k != me]
                    out.append(v_true.score(rosters[me], others))
                base.append(out[0]); treat.append(out[1])
                rows.append((season, me, s))
    return ExperimentOutput(np.array(base), np.array(treat), unit="(season, slot, seed) scored on actual stats", extras={"n_seeds": n_seeds})


# --- H3 / H4 -----------------------------------------------------------------------------------
def _loso_loglik(mode: str) -> np.ndarray:
    settings, panel, drafts = _data()
    out = []
    for held in sorted(drafts.season.unique()):
        train, test = drafts[drafts.season != held], drafts[drafts.season == held]
        obs_tr = build_observations(train, panel, settings.lineup)
        obs_te = build_observations(test, panel, settings.lineup)
        m = OpponentModel()
        if mode == "spectral":
            m.group_of_manager = spectral_clusters(manager_features(train, panel)).groups()
        m.fit(obs_tr, per_manager=mode != "global")
        out.append(m.log_likelihood(obs_te))
    return np.concatenate(out)


def h3(seed: int) -> ExperimentOutput:
    return ExperimentOutput(_loso_loglik("global"), _loso_loglik("spectral"), unit="held-out pick (leave-one-season-out)")


def h4(seed: int) -> ExperimentOutput:
    return ExperimentOutput(_loso_loglik("independent"), _loso_loglik("spectral"), unit="held-out pick (leave-one-season-out)")


def suite(quick: bool = False) -> list[Experiment]:
    exps = [
        Experiment("H1", "Blended projections have lower fantasy-point error than ESPN projections",
                   "Blend error = ESPN error", "abs fantasy-pt error", "ESPN projection", "blend (own+ESPN)",
                   h1, higher_is_better=False, config={"test_seasons": TEST_SEASONS}),
        Experiment("H1b", "Adding MoneyPuck features lowers own-model error", "No difference",
                   "abs fantasy-pt error", "own model w/o MoneyPuck", "own model + MoneyPuck", h1b, higher_is_better=False),
        Experiment("H3", "Per-manager (spectral-pooled) pick model beats a league-wide ADP logit on held-out drafts",
                   "Equal predictive log-likelihood", "pick log-likelihood", "league-wide logit", "per-manager + spectral", h3),
        Experiment("H4", "Pooling managers by spectral cluster beats independent shrinkage", "Equal predictive log-likelihood",
                   "pick log-likelihood", "independent per-manager", "spectral-pooled per-manager", h4),
    ]
    if not quick:
        exps.append(Experiment("H2", "VONA draft policy yields higher actual P(win weekly matchup) than drafting by ADP",
                               "Equal P(win)", "P(win weekly matchup), actual stats", "ADP drafting", "VONA policy", h2,
                               config={"seasons": TEST_SEASONS, "n_seeds": 4}))
    return exps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the draft backtest (H2)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    results, run_dir = run_suite(suite(a.quick), out_dir=RUNS_DIR, seed=a.seed, suite_name="ikp")
    print((run_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
