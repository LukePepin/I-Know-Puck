import numpy as np
import pandas as pd

from iknowpuck.data.league_history import executed_trades, roster_tenures
from iknowpuck.history import absences
from iknowpuck.injuries import apply_injuries, injury_history


def test_absences_finds_mid_season_spell_and_ignores_goalies():
    sched = pd.DataFrame({"season": 2025, "pro_team_id": 1, "scoring_period": np.arange(1, 21), "date": pd.date_range("2024-10-01", periods=20)})
    played = [s for s in range(1, 21) if not 8 <= s <= 12]  # misses 5 straight team games
    gl = pd.DataFrame({"season": 2025, "player_id": 10, "scoring_period": played, "pro_team_id": 1})
    gl_g = pd.DataFrame({"season": 2025, "player_id": 20, "scoring_period": [1, 5, 9], "pro_team_id": 1})
    sp = absences(pd.concat([gl, gl_g]), sched, pd.Series({10: "C", 20: "G"}))
    assert len(sp) == 1 and sp.iloc[0].games == 5 and sp.iloc[0]["where"] == "mid-season" and sp.iloc[0].player_id == 10


def _tx():
    tx = pd.DataFrame([
        {"season": 2025, "tx_id": "p1", "scoring_period": 10, "type": "TRADE_PROPOSAL", "status": "PENDING", "team_id": 1, "member_id": "{a}", "related_id": None, "when": pd.Timestamp("2024-11-01")},
        {"season": 2025, "tx_id": "u1", "scoring_period": 12, "type": "TRADE_UPHOLD", "status": "EXECUTED", "team_id": 1, "member_id": "{a}", "related_id": "p1", "when": pd.Timestamp("2024-11-03")},
        {"season": 2025, "tx_id": "p2", "scoring_period": 20, "type": "TRADE_PROPOSAL", "status": "PENDING", "team_id": 2, "member_id": "{b}", "related_id": None, "when": pd.Timestamp("2024-11-10")},
        {"season": 2025, "tx_id": "v2", "scoring_period": 21, "type": "TRADE_VETO", "status": "EXECUTED", "team_id": 1, "member_id": "{a}", "related_id": "p2", "when": pd.Timestamp("2024-11-11")},
        {"season": 2025, "tx_id": "f1", "scoring_period": 30, "type": "FREEAGENT", "status": "EXECUTED", "team_id": 2, "member_id": "{b}", "related_id": None, "when": pd.Timestamp("2024-11-20")},
    ])
    items = pd.DataFrame([
        {"season": 2025, "tx_id": "p1", "player_id": 100, "item_type": "TRADE", "from_team": 1, "to_team": 2, "from_slot": 0, "to_slot": -1},
        {"season": 2025, "tx_id": "p1", "player_id": 200, "item_type": "TRADE", "from_team": 2, "to_team": 1, "from_slot": 0, "to_slot": -1},
        {"season": 2025, "tx_id": "p2", "player_id": 300, "item_type": "TRADE", "from_team": 2, "to_team": 1, "from_slot": 0, "to_slot": -1},
        {"season": 2025, "tx_id": "f1", "player_id": 400, "item_type": "ADD", "from_team": 0, "to_team": 2, "from_slot": -1, "to_slot": 7},
        {"season": 2025, "tx_id": "f1", "player_id": 100, "item_type": "DROP", "from_team": 2, "to_team": 0, "from_slot": 7, "to_slot": -1},
    ])
    return tx, items


def test_executed_trades_skips_vetoed_and_uses_uphold_date():
    tx, items = _tx()
    tr = executed_trades(tx, items)
    assert set(tr.tx_id) == {"p1"} and set(tr.scoring_period) == {12}


def test_roster_tenures_follow_draft_trade_and_pickup():
    tx, items = _tx()
    drafts = pd.DataFrame({"season": 2025, "team_id": [1, 2, 2], "player_id": [100, 200, 300]})
    st = roster_tenures(drafts, tx, items, 2025)
    p100 = st[st.player_id == 100].sort_values("start_sp")
    assert list(zip(p100.team_id, p100.start_sp, p100.end_sp, p100.how)) == [(1, 0, 12, "draft"), (2, 12, 30, "trade")]
    assert st[(st.player_id == 400)].how.iloc[0] == "free agent"
    assert st[(st.player_id == 300)].team_id.tolist() == [2]  # vetoed trade never moved him


def test_apply_injuries_scales_projection_and_respects_override():
    f = pd.DataFrame({"player_id": [1, 2, 3], "injury_status": ["OUT", "ACTIVE", "DAY_TO_DAY"], "p_30": [80.0, 80.0, 80.0], "p_13": [40.0, 40.0, 40.0]})
    out = apply_injuries(f, [13], overrides={3: 40})
    assert out.loc[0, "p_13"] == 40 * (80 - 15) / 80 and out.loc[1, "p_13"] == 40 and out.loc[2, "p_30"] == 40


def test_injury_history_flags_high_risk():
    avail = pd.DataFrame({"season": [2024, 2025, 2024, 2025], "player_id": [1, 1, 2, 2], "gp": [60, 55, 82, 80], "missed": [22, 27, 0, 2]})
    h = injury_history(avail, [2024, 2025]).set_index("player_id")
    assert h.loc[1, "injury_risk"] == "high" and h.loc[2, "injury_risk"] == "low"
