"""ESPN Fantasy Hockey (fhl) v3 API client.

Public player data (projections, actuals, ADP) needs no auth. League data (settings,
teams, historical drafts, live draft) needs espn_s2 + SWID cookies for private leagues.
Raw JSON is cached to disk; parsing into tidy frames happens here too.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import requests

from puckcore.cache import DiskCache

from ..config import CACHE_DIR, POSITION_NAMES, STAT_NAMES, Category, Credentials, LeagueSettings

BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fhl"
HEADERS = {"User-Agent": "Mozilla/5.0 (i-know-puck research tool)", "Accept": "application/json"}
DAY = 24 * 3600
ADP_CEILING = 229.0  # ESPN's "undrafted" ADP sentinel region


class EspnError(RuntimeError):
    pass


class EspnClient:
    def __init__(self, creds: Credentials, cache: DiskCache | None = None, timeout: float = 30):
        self.creds = creds
        self.cache = cache or DiskCache(CACHE_DIR)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # --- raw requests -------------------------------------------------------------------------
    def _get(self, url: str, params: Any = None, fantasy_filter: dict | None = None, auth: bool = False) -> dict:
        headers = {"X-Fantasy-Filter": json.dumps(fantasy_filter)} if fantasy_filter else {}
        cookies = self.creds.cookies if auth else None
        r = self.session.get(url, params=params, headers=headers, cookies=cookies, timeout=self.timeout)
        if r.status_code == 401:
            raise EspnError("ESPN returned 401: private league needs valid ESPN_S2 and ESPN_SWID in .env")
        if r.status_code == 404:
            raise EspnError(f"ESPN returned 404 for {url}")
        r.raise_for_status()
        return r.json()

    def _league_raw(self, season: int, views: list[str]) -> dict:
        if not self.creds.has_league:
            raise EspnError("ESPN_LEAGUE_ID is not set in .env")
        lid = self.creds.league_id
        params = [("view", v) for v in views]
        try:
            return self._get(f"{BASE}/seasons/{season}/segments/0/leagues/{lid}", params=params, auth=True)
        except (EspnError, requests.HTTPError):
            data = self._get(f"{BASE}/leagueHistory/{lid}", params=params + [("seasonId", season)], auth=True)
            if isinstance(data, list):
                if not data:
                    raise EspnError(f"No league history for season {season}")
                data = data[0]
            return data

    def league_raw(self, season: int, views: list[str], max_age_s: float | None = None) -> dict:
        # Past seasons are immutable, so cache forever; the current season must refresh.
        if max_age_s is None and season >= self.creds.season:
            max_age_s = 300
        key = {"league": self.creds.league_id, "season": season, "views": sorted(views)}
        return self.cache.json("espn_league", key, lambda: self._league_raw(season, views), max_age_s=max_age_s)

    def players_raw(self, season: int, limit: int = 1200, max_age_s: float | None = None) -> list[dict]:
        if max_age_s is None:
            max_age_s = DAY if season >= self.creds.season else None
        flt = {
            "players": {
                "limit": limit,
                "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
                "filterStatsForTopScoringPeriodIds": {"value": 1, "additionalValue": [f"00{season}", f"10{season}"]},
            }
        }
        url = f"{BASE}/seasons/{season}/segments/0/leaguedefaults/1"

        def fetch() -> list[dict]:
            return self._get(url, params={"view": "kona_player_info"}, fantasy_filter=flt)["players"]

        return self.cache.json("espn_players", {"season": season, "limit": limit}, fetch, max_age_s=max_age_s)

    # --- tidy frames --------------------------------------------------------------------------
    def players(self, season: int, limit: int = 1200) -> pd.DataFrame:
        return parse_players(self.players_raw(season, limit), season)

    def settings(self, season: int) -> LeagueSettings:
        raw = self.league_raw(season, ["mSettings", "mTeam"])
        ls = parse_settings(raw)
        swid = self.creds.swid
        ls.my_team_id = next((t["id"] for t in raw.get("teams", []) if swid and swid in (t.get("owners") or [])), None)
        return ls

    def teams(self, season: int) -> pd.DataFrame:
        return parse_teams(self.league_raw(season, ["mTeam"]), season)

    def draft(self, season: int, live: bool = False) -> pd.DataFrame:
        raw = self.league_raw(season, ["mDraftDetail", "mTeam"], max_age_s=5 if live else None)
        return parse_draft(raw, season)

    def draft_order(self, season: int) -> list[int]:
        raw = self.league_raw(season, ["mSettings"])
        return list(raw.get("settings", {}).get("draftSettings", {}).get("pickOrder", []))

    def managers(self, seasons: list[int]) -> pd.DataFrame:
        """One row per manager (ESPN member id) across seasons: name, which team slot each season.
        Managers are identified by account id, never by team id, because team slots change hands."""
        teams = pd.concat([self.teams(s) for s in seasons], ignore_index=True)
        latest = max(seasons)
        rows = []
        for oid, g in teams.groupby("owner_id"):
            g = g.sort_values("season")
            rows.append(
                {
                    "owner_id": oid,
                    "manager": g["owner_name"].iloc[-1],
                    "seasons": g["season"].tolist(),
                    "team_by_season": dict(zip(g["season"].tolist(), g["team_id"].tolist())),
                    "current_team_id": int(g.loc[g.season == latest, "team_id"].iloc[0]) if (g.season == latest).any() else None,
                    "active": bool((g.season == latest).any()),
                }
            )
        return pd.DataFrame(rows)

    def seasons_available(self) -> list[int]:
        raw = self.league_raw(self.creds.season, ["mStatus"])
        prev = raw.get("status", {}).get("previousSeasons", [])
        return sorted(set(prev) | {self.creds.season})


def parse_players(raw: list[dict], season: int) -> pd.DataFrame:
    rows = []
    for p in raw:
        pl = p.get("player", p)
        own = pl.get("ownership") or {}
        rank = (pl.get("draftRanksByRankType") or {}).get("STANDARD", {})
        row: dict[str, Any] = {
            "season": season,
            "player_id": pl["id"],
            "name": pl.get("fullName"),
            "pos_id": pl.get("defaultPositionId"),
            "pos": POSITION_NAMES.get(pl.get("defaultPositionId"), "?"),
            "eligible_slots": list(pl.get("eligibleSlots", [])),
            "pro_team_id": pl.get("proTeamId"),
            "adp": own.get("averageDraftPosition"),
            "espn_rank": rank.get("rank"),
            "pct_owned": own.get("percentOwned"),
            "injury_status": pl.get("injuryStatus"),
        }
        for s in pl.get("stats", []):
            if s.get("seasonId") != season or s.get("statSplitTypeId") != 0:
                continue
            prefix = {0: "act", 1: "proj"}.get(s.get("statSourceId"))
            if prefix is None:
                continue
            for k, v in (s.get("stats") or {}).items():
                if int(k) in STAT_NAMES:
                    row[f"{prefix}_{int(k)}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in ("adp", "espn_rank", "pct_owned", *[c for c in df.columns if c.startswith(("act_", "proj_"))]):
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    # ESPN parks undrafted players at ~230 ADP (0 in some seasons) and wiped 2026's ADP entirely.
    # Keep genuine ADP; otherwise fall back to ESPN's preseason rank (Spearman ~0.97 with ADP).
    df["adp_raw"] = df["adp"]
    df.loc[(df["adp"] <= 0) | (df["adp"] >= ADP_CEILING), "adp"] = np.nan
    df["adp"] = df["adp"].fillna(df["espn_rank"])
    # The API returns players sorted by *current* % owned; for past seasons that is hindsight.
    # Shuffle deterministically so no downstream tie-break can leak it.
    df = df.sample(frac=1.0, random_state=season).reset_index(drop=True)
    df.loc[df["espn_rank"] <= 0, "espn_rank"] = np.nan
    return df


def parse_settings(raw: dict) -> LeagueSettings:
    st = raw.get("settings", {})
    scoring = st.get("scoringSettings", {})
    cats = [
        Category(int(i["statId"]), bool(i.get("isReverseItem", False)), float(i.get("points", 0.0) or 0.0))
        for i in scoring.get("scoringItems", [])
    ]
    slots = {int(k): int(v) for k, v in st.get("rosterSettings", {}).get("lineupSlotCounts", {}).items() if int(v) > 0}
    sched = st.get("scheduleSettings", {})
    ls = LeagueSettings(
        n_teams=len(raw.get("teams", [])) or 12,
        draft_type=st.get("draftSettings", {}).get("type", "SNAKE"),
        scoring_type=scoring.get("scoringType", "H2H_CATEGORY"),
        pick_order=list(st.get("draftSettings", {}).get("pickOrder", [])),
        matchup_weeks=int(sched.get("matchupPeriodCount", 24) or 24),
        team_names={t["id"]: _team_name(t) for t in raw.get("teams", [])},
    )
    if cats:
        ls.categories = cats
    if slots:
        ls.lineup = slots
    return ls


def _team_name(t: dict) -> str:
    return t.get("name") or f"{t.get('location', '')} {t.get('nickname', '')}".strip() or f"Team {t.get('id')}"


def parse_teams(raw: dict, season: int) -> pd.DataFrame:
    members = {m["id"]: m for m in raw.get("members", [])}
    rows = []
    for t in raw.get("teams", []):
        owners = t.get("owners") or [t.get("primaryOwner")]
        owner = owners[0] if owners else None
        m = members.get(owner, {})
        rows.append(
            {
                "season": season,
                "team_id": t["id"],
                "team_name": _team_name(t),
                "owner_id": owner,
                "owner_name": manager_name(m, owner),
            }
        )
    return pd.DataFrame(rows)


def manager_name(member: dict, fallback: str | None = None) -> str:
    """Prefer the real name; ESPN display names are often auto-generated 'ESPNfan123...'."""
    full = f"{(member.get('firstName') or '').strip()} {(member.get('lastName') or '').strip()}".strip()
    if full:
        return " ".join(w[:1].upper() + w[1:] for w in full.split())
    return member.get("displayName") or str(fallback)


def parse_draft(raw: dict, season: int) -> pd.DataFrame:
    picks = raw.get("draftDetail", {}).get("picks", [])
    teams = parse_teams(raw, season).set_index("team_id") if raw.get("teams") else None
    rows = []
    for p in picks:
        if not p.get("playerId") or p["playerId"] < 0:
            continue
        tid = p.get("teamId")
        rows.append(
            {
                "season": season,
                "overall": p.get("overallPickNumber"),
                "round": p.get("roundId"),
                "round_pick": p.get("roundPickNumber"),
                "team_id": tid,
                "player_id": p["playerId"],
                "keeper": bool(p.get("keeper", False)),
                "auto": p.get("autoDraftTypeId", 0) != 0,
                "owner_id": teams.loc[tid, "owner_id"] if teams is not None and tid in teams.index else None,
            }
        )
    return pd.DataFrame(rows).sort_values("overall").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["season", "overall", "round", "round_pick", "team_id", "player_id", "keeper", "auto", "owner_id"]
    )
