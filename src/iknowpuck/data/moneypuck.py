"""MoneyPuck season-summary CSVs (free, no auth). Season = the year the season STARTS."""

from __future__ import annotations

import io

import pandas as pd
import requests

from puckcore.cache import DiskCache

from ..config import CACHE_DIR

URL = "https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/{kind}.csv"

SKATER_COLS = {
    "icetime": "mp_toi",
    "games_played": "mp_gp",
    "I_F_xGoals": "mp_ixg",
    "I_F_goals": "mp_goals",
    "I_F_shotsOnGoal": "mp_sog",
    "I_F_shotAttempts": "mp_isa",
    "I_F_points": "mp_pts",
    "I_F_hits": "mp_hits",
    "shotsBlockedByPlayer": "mp_blk",
    "I_F_penalityMinutes": "mp_pim",
    "OnIce_F_xGoals": "mp_onxgf",
    "OnIce_A_xGoals": "mp_onxga",
    "OnIce_F_goals": "mp_ongf",
    "OnIce_A_goals": "mp_onga",
    "gameScore": "mp_gamescore",
}
GOALIE_COLS = {"icetime": "mp_toi", "games_played": "mp_gp", "xGoals": "mp_xga", "goals": "mp_ga", "ongoal": "mp_sa"}


class MoneyPuck:
    name = "moneypuck"

    def __init__(self, cache: DiskCache | None = None):
        self.cache = cache or DiskCache(CACHE_DIR)

    def _raw(self, season: int, kind: str) -> pd.DataFrame:
        def fetch() -> pd.DataFrame:
            r = requests.get(URL.format(season=season, kind=kind), timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))

        return self.cache.frame("moneypuck", {"season": season, "kind": kind}, fetch)

    def skaters(self, season: int) -> pd.DataFrame:
        """One row per skater with all-situations stats plus power-play (5on4) usage."""
        raw = self._raw(season, "skaters")
        allsit = raw[raw.situation == "all"][["playerId", "name", "team", "position", *SKATER_COLS]].rename(columns=SKATER_COLS)
        pp = raw[raw.situation == "5on4"][["playerId", "icetime", "I_F_points", "I_F_xGoals"]].rename(
            columns={"icetime": "mp_pp_toi", "I_F_points": "mp_pp_pts", "I_F_xGoals": "mp_pp_ixg"}
        )
        df = allsit.merge(pp, on="playerId", how="left").rename(columns={"playerId": "nhl_id", "name": "mp_name", "team": "mp_team"})
        df["mp_season"] = season
        return df

    def goalies(self, season: int) -> pd.DataFrame:
        raw = self._raw(season, "goalies")
        df = raw[raw.situation == "all"][["playerId", "name", "team", *GOALIE_COLS]].rename(columns=GOALIE_COLS)
        df = df.rename(columns={"playerId": "nhl_id", "name": "mp_name", "team": "mp_team"})
        df["mp_gsax"] = df["mp_xga"] - df["mp_ga"]
        df["mp_season"] = season
        return df

    def load(self, season: int) -> pd.DataFrame:
        return pd.concat([self.skaters(season), self.goalies(season).assign(position="G")], ignore_index=True)
