"""End-to-end build: data -> projections -> valuation -> opponent models -> draft context.

The fitted bundle is pickled so the Streamlit app starts instantly on draft day.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CACHE_DIR, LeagueSettings, load_credentials
from .data.dataset import build_panel
from .data.espn import EspnClient, EspnError
from .draft import DraftContext
from .opponents import OpponentModel, build_observations
from .projections import Blender, MarketAdjuster, OwnModel, fill_projection, modeled_stats, season_frame
from .spectral import SpectralResult, manager_features, spectral_clusters
from .strategy import manager_seasons, season_outcomes, strategy_correlations
from .valuation import PlayerPool, Valuator

FIRST_PANEL_SEASON = 2018
POOL_SIZE = 750


@dataclass
class Bundle:
    season: int
    settings: LeagueSettings
    stats: list[int]
    projections: pd.DataFrame  # current-season players with own/proj/blend + p_ columns
    pool: PlayerPool
    valuator: Valuator
    opp_model: OpponentModel
    manager_of_team: dict[int, str]
    teams: pd.DataFrame
    drafts: pd.DataFrame
    spectral: SpectralResult | None
    blend_weights: dict[int, float]
    managers: pd.DataFrame = field(default_factory=pd.DataFrame)  # all managers, all seasons
    sim_opp: OpponentModel | None = None  # opponent model used in simulation (league-wide, per H3)
    market: MarketAdjuster | None = None
    proj_eval: pd.DataFrame = field(default_factory=pd.DataFrame)  # out-of-sample projection vs actual
    strategy: pd.DataFrame = field(default_factory=pd.DataFrame)  # manager-season behaviour + outcomes
    strategy_cor: pd.DataFrame = field(default_factory=pd.DataFrame)
    source: str = "blend"
    notes: list[str] = field(default_factory=list)

    @property
    def manager_names(self) -> dict[str, str]:
        return dict(zip(self.managers.owner_id.astype(str), self.managers.manager)) if len(self.managers) else {}

    def team_labels(self) -> dict[int, str]:
        """Current-season team id -> manager name (falls back to the ESPN team name)."""
        names = self.manager_names
        out = dict(self.settings.team_names)
        for tid, oid in self.manager_of_team.items():
            out[tid] = names.get(oid, out.get(tid, f"Team {tid}"))
        return out

    def context(self, first_round: list[int] | None = None, my_team: int | None = None) -> DraftContext:
        order = first_round or self.settings.pick_order or list(range(1, self.settings.n_teams + 1))
        me = my_team or self.settings.my_team_id or order[0]
        return DraftContext(self.pool, self.valuator, order, me, self.sim_opp or self.opp_model, self.manager_of_team)


def fantasy_points(frame: pd.DataFrame, settings: LeagueSettings, prefix: str) -> pd.Series:
    return sum(frame.get(f"{prefix}_{c.stat_id}", pd.Series(0.0, index=frame.index)).fillna(0) * c.points for c in settings.scoring_categories)


def fit_projections(panel: pd.DataFrame, stats: list[int], season: int, use_moneypuck: bool = True) -> tuple[pd.DataFrame, Blender, OwnModel, pd.DataFrame]:
    """Blend weights are fit on *out-of-sample* own-model predictions for the 3 prior seasons."""
    oos = []
    for t in range(season - 3, season):
        if t <= FIRST_PANEL_SEASON + 3:  # no earlier target seasons to train on
            continue
        m = OwnModel(stats, use_moneypuck).fit(panel, list(range(FIRST_PANEL_SEASON + 3, t)))
        oos.append(season_frame(panel, m, t))
    blender = Blender(stats).fit(pd.concat(oos))
    model = OwnModel(stats, use_moneypuck).fit(panel, list(range(FIRST_PANEL_SEASON + 3, season)))
    cur = blender.transform(season_frame(panel, model, season))
    return cur, blender, model, blender.transform(pd.concat(oos))


def build(season: int | None = None, refresh: bool = False, source: str = "blend", n_pool: int = POOL_SIZE) -> Bundle:
    creds = load_credentials()
    season = season or creds.season
    path = CACHE_DIR / f"bundle_{season}_{source}.pkl"
    if path.exists() and not refresh:
        # Safe: this pickle is only ever written by build() below on this machine, never downloaded.
        with open(path, "rb") as fh:
            return pickle.load(fh)

    client = EspnClient(creds)
    notes: list[str] = []
    try:
        settings = client.settings(season)
    except EspnError as e:
        notes.append(f"League settings unavailable ({e}); using defaults.")
        settings = LeagueSettings()
    stats = modeled_stats(settings)
    panel = build_panel(list(range(FIRST_PANEL_SEASON, season + 1)), client)

    cur, blender, _, oos = fit_projections(panel, stats, season)
    cur = fill_projection(cur.reset_index(), stats, source)
    cur["fpts_model"] = fantasy_points(cur, settings, "p")
    # market adjustment: regress past actual points on (our projection, log ADP), apply to this season
    oos = oos.copy()
    oos["bp"] = fantasy_points(oos, settings, source)
    oos["ap"] = fantasy_points(oos, settings, "act").where(oos["act_30"].notna())
    market = MarketAdjuster().fit(oos, "bp", "ap")
    oos["mp"] = market.adjusted_points(oos, "bp")
    oos["ep"] = fantasy_points(oos, settings, "proj")
    oos["op"] = fantasy_points(oos, settings, "own")
    proj_eval = oos.loc[oos["ap"].notna() & oos["proj_30"].notna(), ["season", "name", "pos", "adp", "ep", "op", "bp", "mp", "ap"]].rename(
        columns={"ep": "espn", "op": "own", "bp": "blend", "mp": "market_adj", "ap": "actual"}).reset_index(drop=True)
    cur = market.apply(cur, stats, "fpts_model")
    cur["fpts"] = fantasy_points(cur, settings, "p")
    cur["fpts_espn"] = fantasy_points(cur, settings, "proj")
    cur["fpts_own"] = fantasy_points(cur, settings, "own")
    rank_key = np.where(cur["adp"].notna(), cur["adp"], 1000) - 0.01 * cur["fpts"]
    in_pool = (cur["fpts"] > 0) | cur["adp"].notna()
    pool_frame = cur[in_pool].assign(_k=rank_key[in_pool]).nsmallest(n_pool, "_k").drop(columns="_k")
    # keep the top projected players even if ESPN's ADP ignores them
    extra = cur[~cur.player_id.isin(pool_frame.player_id)].nlargest(50, "fpts")
    pool_frame = pd.concat([pool_frame, extra], ignore_index=True)
    pool = PlayerPool(pool_frame, settings, stats)
    valuator = Valuator(pool)

    # league history -> opponent models + spectral clusters
    drafts = pd.DataFrame()
    teams = pd.DataFrame()
    managers = pd.DataFrame()
    spec = None
    opp = OpponentModel()
    manager_of_team: dict[int, str] = {}
    if creds.has_league:
        try:
            past = [s for s in client.seasons_available() if s < season]
            drafts = pd.concat([client.draft(s) for s in past], ignore_index=True) if past else pd.DataFrame()
            teams = client.teams(season)
            managers = client.managers([*past, season])
            manager_of_team = {int(r.team_id): str(r.owner_id) for r in teams.itertuples()}
        except EspnError as e:
            notes.append(f"League history unavailable: {e}")
    if len(drafts):
        feats = manager_features(drafts, panel)
        spec = spectral_clusters(feats)
        opp.group_of_manager = spec.groups()
        obs = build_observations(drafts, panel, settings.lineup)
        opp.fit(obs, per_manager=True)
        # H3/H4: per-manager models did not beat the league-wide logit out of sample, so the
        # simulator uses the league-wide model; per-manager fits are kept for the League intel tab.
        sim_opp = OpponentModel(global_=opp.global_.copy(), b_need=opp.b_need)
    else:
        notes.append("No draft history: opponents follow ADP with default noise.")

    strategy = strategy_cor = pd.DataFrame()
    if len(drafts):
        try:
            past = sorted(drafts.season.unique().tolist())
            strategy = manager_seasons(drafts, panel, season_outcomes(client, past), settings)
            strategy_cor = strategy_correlations(strategy)
        except Exception as e:  # descriptive extra; never block the draft tool
            notes.append(f"Strategy analysis skipped: {e}")

    b = Bundle(season, settings, stats, cur, pool, valuator, opp, manager_of_team, teams, drafts, spec, blender.w_, managers, source, notes)
    b.sim_opp = sim_opp if len(drafts) else opp
    b.market = market
    b.proj_eval, b.strategy, b.strategy_cor = proj_eval, strategy, strategy_cor
    with open(path, "wb") as fh:
        pickle.dump(b, fh)
    return b
