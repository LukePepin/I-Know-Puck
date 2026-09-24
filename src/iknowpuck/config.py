"""Environment, ESPN constants, and league settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"
RUNS_DIR = ROOT / "runs"

load_dotenv(ROOT / ".env")

# --- ESPN stat ids (fhl), decoded against known player seasons ---------------------------------
STAT_NAMES: dict[int, str] = {
    # goalie
    0: "GS", 1: "W", 2: "L", 3: "SA", 4: "GA", 6: "SV", 7: "SO", 8: "G_TOI", 9: "OTL", 10: "GAA", 11: "SV%",
    # skater
    13: "G", 14: "A", 15: "+/-", 16: "PTS", 17: "PIM", 18: "PPG", 19: "PPA", 20: "SHG", 21: "SHA", 22: "GWG",
    23: "FOW", 24: "FOL", 26: "TOI", 27: "ATOI", 28: "HAT", 29: "SOG", 30: "GP", 31: "HIT", 32: "BLK", 33: "DEF",
    35: "STG", 36: "STA", 37: "STP", 38: "PPP", 39: "SHP",
}
STAT_IDS: dict[str, int] = {v: k for k, v in STAT_NAMES.items()}

# Ratio categories are aggregated from components, never averaged: stat -> (numerator, denominator, scale)
RATIO_STATS: dict[int, tuple[int, int, float]] = {
    10: (4, 8, 3600.0),  # GAA = GA * 3600 / TOI_seconds
    11: (6, 3, 1.0),  # SV% = SV / SA
}
GOALIE_STATS = {0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11}

# --- lineup slots ------------------------------------------------------------------------------
SLOT_NAMES = {0: "C", 1: "LW", 2: "RW", 3: "F", 4: "D", 5: "G", 6: "UTIL", 7: "BN", 8: "IR"}
POSITION_NAMES = {1: "C", 2: "LW", 3: "RW", 4: "D", 5: "G"}  # ESPN defaultPositionId
BENCH_SLOT, IR_SLOT, GOALIE_SLOT = 7, 8, 5


@dataclass
class Category:
    stat_id: int
    reverse: bool = False  # lower is better (GAA)
    points: float = 0.0  # fantasy points per unit (points leagues)

    @property
    def name(self) -> str:
        return STAT_NAMES.get(self.stat_id, str(self.stat_id))

    @property
    def is_goalie(self) -> bool:
        return self.stat_id in GOALIE_STATS


@dataclass
class LeagueSettings:
    n_teams: int = 12
    categories: list[Category] = field(
        default_factory=lambda: [Category(STAT_IDS[s]) for s in ("G", "A", "+/-", "PIM", "PPP", "SOG", "HIT", "BLK", "W", "SO")]
        + [Category(STAT_IDS["GAA"], reverse=True), Category(STAT_IDS["SV%"])]
    )
    lineup: dict[int, int] = field(default_factory=lambda: {0: 2, 1: 2, 2: 2, 4: 4, 6: 1, 5: 2, 7: 5})
    matchup_weeks: int = 24
    draft_type: str = "SNAKE"
    scoring_type: str = "H2H_CATEGORY"  # or H2H_POINTS, H2H_MOST_CATEGORIES, ROTO, TOTAL_POINTS
    pick_order: list[int] = field(default_factory=list)  # round-1 team ids
    my_team_id: int | None = None
    team_names: dict[int, str] = field(default_factory=dict)

    @property
    def rounds(self) -> int:
        return sum(c for s, c in self.lineup.items() if s != IR_SLOT)

    @property
    def starting_slots(self) -> dict[int, int]:
        return {s: c for s, c in self.lineup.items() if s not in (BENCH_SLOT, IR_SLOT) and c > 0}

    @property
    def is_points(self) -> bool:
        return "POINTS" in self.scoring_type

    @property
    def scoring_categories(self) -> list[Category]:
        """Categories that actually score: nonzero weights in points leagues, all in category leagues."""
        return [c for c in self.categories if c.points != 0] if self.is_points else self.categories

    @property
    def category_names(self) -> list[str]:
        return [c.name for c in self.categories]


@dataclass
class Credentials:
    league_id: int | None
    espn_s2: str | None
    swid: str | None
    season: int

    @property
    def cookies(self) -> dict | None:
        if self.espn_s2 and self.swid:
            return {"espn_s2": self.espn_s2, "SWID": self.swid}
        return None

    @property
    def has_league(self) -> bool:
        return self.league_id is not None


def load_credentials() -> Credentials:
    lid = os.getenv("ESPN_LEAGUE_ID", "").strip()
    return Credentials(
        league_id=int(lid) if lid.isdigit() else None,
        espn_s2=os.getenv("ESPN_S2", "").strip() or None,
        swid=os.getenv("ESPN_SWID", "").strip() or None,
        season=int(os.getenv("ESPN_SEASON", "2027") or 2027),
    )
