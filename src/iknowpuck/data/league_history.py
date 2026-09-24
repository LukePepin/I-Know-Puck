"""League history from ESPN: NHL schedules, per-game logs, and every transaction.

Everything is fetched per season, cached to disk, and returned as tidy frames:
  schedules    : one row per (NHL team, scoring period) game, with the date
  gamelogs     : one row per (player, game) with the player's NHL team and stats that game
  transactions : one row per transaction; items: one row per player moved in it
Scoring periods are ESPN's day index within a season.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from ..config import STAT_NAMES
from .espn import BASE, EspnClient

MAX_SCORING_PERIOD = 200
IR_SLOT_ID = 8


def schedules(client: EspnClient, season: int) -> pd.DataFrame:
    def fetch():
        d = client._get(f"{BASE}/seasons/{season}", params={"view": "proTeamSchedules_wl"})
        rows = []
        for t in d.get("settings", {}).get("proTeams", []):
            for sp, games in (t.get("proGamesByScoringPeriod") or {}).items():
                for g in games:
                    rows.append({"season": season, "pro_team_id": t["id"], "scoring_period": int(sp), "date_ms": g.get("date"), "game_id": g.get("id")})
        return rows

    rows = client.cache.json("espn_schedule", {"season": season}, fetch)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date_ms"], unit="ms").dt.normalize()
    return df


def gamelogs(client: EspnClient, season: int, player_ids: list[int], chunk: int = 80) -> pd.DataFrame:
    """Per-game stat lines for the given players (league endpoint, needs league cookies)."""
    ids = sorted({int(p) for p in player_ids})
    lid = client.creds.league_id

    def fetch_chunk(block: list[int]) -> list[dict]:
        flt = {"players": {"filterIds": {"value": block}, "filterStatsForTopScoringPeriodIds": {"value": MAX_SCORING_PERIOD, "additionalValue": [f"00{season}"]}}}

        def fetch():
            d = client._get(f"{BASE}/seasons/{season}/segments/0/leagues/{lid}", params={"view": "kona_playercard"}, fantasy_filter=flt, auth=True)
            out = []
            for p in d.get("players", []):
                pl = p.get("player", p)
                for s in pl.get("stats", []):
                    if s.get("statSplitTypeId") != 5 or s.get("seasonId") != season or s.get("statSourceId") != 0:
                        continue
                    row = {"player_id": pl["id"], "scoring_period": s.get("scoringPeriodId"), "pro_team_id": s.get("proTeamId"), "game_id": s.get("externalId")}
                    row.update({f"g_{int(k)}": v for k, v in (s.get("stats") or {}).items() if int(k) in STAT_NAMES})
                    out.append(row)
            return out

        return client.cache.json("espn_gamelogs", {"season": season, "ids": block}, fetch)

    blocks = [ids[i : i + chunk] for i in range(0, len(ids), chunk)]
    with ThreadPoolExecutor(6) as ex:
        parts = list(ex.map(fetch_chunk, blocks))
    df = pd.DataFrame([r for part in parts for r in part])
    if len(df):
        df.insert(0, "season", season)
        # ESPN returns a row for every team game; only rows with the games-played flag were played
        df = df[df.get("g_30", pd.Series(np.nan, index=df.index)).fillna(0) >= 1].reset_index(drop=True)
    return df


def transactions(client: EspnClient, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    lid = client.creds.league_id

    def fetch_sp(sp: int) -> list[dict]:
        def fetch():
            d = client._get(f"{BASE}/seasons/{season}/segments/0/leagues/{lid}", params=[("view", "mTransactions2"), ("scoringPeriodId", sp)], auth=True)
            return d.get("transactions", [])

        return client.cache.json("espn_transactions", {"league": lid, "season": season, "sp": sp}, fetch)

    with ThreadPoolExecutor(8) as ex:
        parts = list(ex.map(fetch_sp, range(0, MAX_SCORING_PERIOD + 1)))
    seen: set[str] = set()
    tx_rows, item_rows = [], []
    for part in parts:
        for t in part:
            if t["id"] in seen:
                continue
            seen.add(t["id"])
            tx_rows.append({
                "season": season, "tx_id": t["id"], "scoring_period": t.get("scoringPeriodId"), "type": t.get("type"),
                "status": t.get("status"), "team_id": t.get("teamId"), "member_id": t.get("memberId"),
                "related_id": t.get("relatedTransactionId"), "proposed_ms": t.get("proposedDate"), "processed_ms": t.get("processDate"),
            })
            for it in t.get("items", []) or []:
                item_rows.append({
                    "season": season, "tx_id": t["id"], "player_id": it.get("playerId"), "item_type": it.get("type"),
                    "from_team": it.get("fromTeamId"), "to_team": it.get("toTeamId"),
                    "from_slot": it.get("fromLineupSlotId"), "to_slot": it.get("toLineupSlotId"),
                })
    tx = pd.DataFrame(tx_rows)
    items = pd.DataFrame(item_rows)
    if len(tx):
        tx["when"] = pd.to_datetime(tx["processed_ms"].fillna(tx["proposed_ms"]), unit="ms")
    return tx, items


def executed_trades(tx: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Trades that actually went through: a proposal upheld/accepted and not vetoed or cancelled.
    One row per player moved, with the scoring period the trade took effect."""
    if tx.empty:
        return pd.DataFrame(columns=["season", "tx_id", "player_id", "from_team", "to_team", "scoring_period", "when"])
    props = tx[tx.type == "TRADE_PROPOSAL"]
    rel = tx[tx.related_id.notna()]
    upheld = rel[(rel.type.isin(["TRADE_UPHOLD", "TRADE_ACCEPT"])) & (rel.status == "EXECUTED")].groupby("related_id")
    vetoed = set(rel.loc[(rel.type == "TRADE_VETO") & (rel.status == "EXECUTED"), "related_id"])
    declined = set(rel.loc[rel.type == "TRADE_DECLINE", "related_id"])
    ok = props[props.tx_id.isin(upheld.groups.keys()) | (props.status == "EXECUTED")]
    ok = ok[~ok.tx_id.isin(vetoed | declined) & (ok.status != "CANCELED")]
    eff = upheld.agg(eff_sp=("scoring_period", "max"), eff_when=("when", "max")).reset_index().rename(columns={"related_id": "tx_id"})
    out = items[items.tx_id.isin(ok.tx_id) & (items.item_type == "TRADE")].copy()
    out = out.merge(ok[["tx_id", "scoring_period", "when"]], on="tx_id", how="left").merge(eff, on="tx_id", how="left")
    out["scoring_period"] = out["eff_sp"].fillna(out["scoring_period"])
    out["when"] = out["eff_when"].where(out["eff_when"].notna(), out["when"])
    return out.drop(columns=["eff_sp", "eff_when"])


def roster_tenures(drafts: pd.DataFrame, tx: pd.DataFrame, items: pd.DataFrame, season: int) -> pd.DataFrame:
    """Rebuild who was on which fantasy team, and when.

    Returns one row per stint: team_id, player_id, start_sp (inclusive), end_sp (exclusive),
    how the player was acquired (draft / free agent / waiver / trade).
    """
    events = []  # (sp, order, kind, team, player, how)
    for r in drafts[drafts.season == season].itertuples():
        events.append((0, 0, "add", int(r.team_id), int(r.player_id), "draft"))
    if not tx.empty:
        adds = items.merge(tx[["tx_id", "type", "status", "scoring_period"]], on="tx_id")
        adds = adds[adds.status.eq("EXECUTED") & adds.type.isin(["FREEAGENT", "WAIVER", "TRADE_ACCEPT", "TRADE_UPHOLD"])]
        for r in adds.itertuples():
            sp = int(r.scoring_period or 0)
            if r.item_type == "ADD":
                events.append((sp, 2, "add", int(r.to_team), int(r.player_id), "waiver" if r.type == "WAIVER" else "free agent"))
            elif r.item_type == "DROP":
                events.append((sp, 1, "drop", int(r.from_team), int(r.player_id), ""))
        for r in executed_trades(tx, items).itertuples():
            sp = int(r.scoring_period or 0)
            events.append((sp, 1, "drop", int(r.from_team), int(r.player_id), ""))
            events.append((sp, 2, "add", int(r.to_team), int(r.player_id), "trade"))
    events.sort(key=lambda e: (e[0], e[1]))
    open_: dict[tuple[int, int], tuple[int, str]] = {}
    stints = []
    for sp, _, kind, team, pid, how in events:
        key = (team, pid)
        if kind == "add":
            # a player can only be on one team: close any other open stint first
            for (t2, p2), (s2, h2) in list(open_.items()):
                if p2 == pid and t2 != team:
                    stints.append((t2, pid, s2, sp, h2))
                    del open_[(t2, p2)]
            open_.setdefault(key, (sp, how))
        elif key in open_:
            s0, h0 = open_.pop(key)
            stints.append((team, pid, s0, sp, h0))
    for (team, pid), (s0, h0) in open_.items():
        stints.append((team, pid, s0, MAX_SCORING_PERIOD + 50, h0))
    out = pd.DataFrame(stints, columns=["team_id", "player_id", "start_sp", "end_sp", "how"])
    out.insert(0, "season", season)
    return out


def lineup_activity(tx: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Manager-made lineup changes and IR placements (excludes ESPN's automatic nightly moves)."""
    if tx.empty:
        return pd.DataFrame()
    li = items.merge(tx[["tx_id", "type", "status", "scoring_period", "team_id", "member_id", "when"]], on="tx_id")
    li = li[(li.item_type == "LINEUP") & li.type.isin(["ROSTER", "FUTURE_ROSTER"]) & (li.member_id.astype(str).str.startswith("{"))]
    li["to_ir"] = li["to_slot"] == IR_SLOT_ID
    return li
