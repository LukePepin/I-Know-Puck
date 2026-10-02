"""Read picks from a copy of ESPN's draft-room log, for when ESPN's API does not publish live picks.

ESPN's log shows each pick as two lines:
    Nathan MacKinnon / COL C
    R1, P1 - Team Name
Join/leave messages and chat lines are ignored. The drafting team comes from the round and pick number
(the snake order), so renamed fantasy teams cannot break it.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

import pandas as pd

ESPN_TO_ABBR = {"TB": "TBL", "LA": "LAK", "SJ": "SJS", "NJ": "NJD", "UTAH": "UTA", "VGS": "VGK"}
PLAYER = re.compile(r"^\W*(?P<name>[^/]+?)\s*/\s*(?P<team>[A-Z]{2,4})\b")
SLOT = re.compile(r"R(?P<round>\d+)\s*,\s*P(?P<pick>\d+)\s*-\s*(?P<team_name>.+?)\s*$")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", s.lower())).strip()


def parse_log(text: str, frame: pd.DataFrame, order: list[int], n_teams: int, abbrev: dict) -> tuple[list[tuple[int, int]], list[str]]:
    """(picks [(team_id, player_id)] in pick order, unmatched players).

    Picks run from 1 up to the last pick found; a player that can't be matched keeps his slot with a
    placeholder id (-overall pick) so the pick count and the team on the clock stay right."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    by_name: dict[str, list[tuple[int, str | None]]] = {}
    for r in frame[["player_id", "name", "pro_team_id"]].itertuples():
        by_name.setdefault(_norm(str(r.name)), []).append((int(r.player_id), abbrev.get(r.pro_team_id)))
    names = list(by_name)
    found: dict[int, tuple[int, int]] = {}
    unmatched: list[str] = []
    for i in range(1, len(lines)):
        m = SLOT.search(lines[i])
        pm = PLAYER.match(lines[i - 1]) if m else None
        if not pm:
            continue
        overall = (int(m["round"]) - 1) * n_teams + int(m["pick"])
        if not 1 <= overall <= len(order):
            continue
        nm, tm = _norm(pm["name"]), ESPN_TO_ABBR.get(pm["team"], pm["team"])
        cands = by_name.get(nm)
        if not cands:
            close = difflib.get_close_matches(nm, names, n=1, cutoff=0.85)
            cands = by_name.get(close[0]) if close else None
        if not cands:
            unmatched.append(f"R{m['round']} P{m['pick']} {pm['name'].strip()}")
            found[overall] = (order[overall - 1], -overall)
            continue
        found[overall] = (order[overall - 1], next((p for p, a in cands if a == tm), cands[0][0]))
    board, board_unmatched = _parse_board(lines, by_name, names, order)
    for k, v in board.items():
        found.setdefault(k, v)
    unmatched += [u for u in board_unmatched if int(u.split()[1]) not in found or found[int(u.split()[1])][1] < 0]
    if not found:
        return [], unmatched
    last = max(found)
    missing = [k for k in range(1, last + 1) if k not in found]
    for k in missing:
        unmatched.append(f"pick {k} not in the pasted log")
        found[k] = (order[k - 1], -k)
    return [found[k] for k in range(1, last + 1)], unmatched


def _parse_board(lines: list[str], by_name: dict, names: list[str], order: list[int]) -> tuple[dict[int, tuple[int, int]], list[str]]:
    """ESPN's draft board copied as one field per line: pick number, player, [status], NHL team, position,
    fantasy team, points... A pick is an integer line followed by a line that names a player."""
    found: dict[int, tuple[int, int]] = {}
    unmatched: list[str] = []
    for i in range(len(lines) - 1):
        if not lines[i].isdigit():
            continue
        k, nxt = int(lines[i]), lines[i + 1]
        if not 1 <= k <= len(order) or nxt.replace(".", "").isdigit() or k in found:
            continue
        nm = _norm(nxt)
        cands = by_name.get(nm)
        if not cands:
            close = difflib.get_close_matches(nm, names, n=1, cutoff=0.88)
            cands = by_name.get(close[0]) if close else None
        if not cands:
            if len(nm.split()) >= 2 and not nm.startswith("round"):
                unmatched.append(f"pick {k} {nxt}")
            continue
        teams = {ESPN_TO_ABBR.get(x, x) for x in lines[i + 2 : i + 4]}
        found[k] = (order[k - 1], next((p for p, a in cands if a in teams), cands[0][0]))
    return found, unmatched
