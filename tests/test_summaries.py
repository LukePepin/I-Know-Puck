import pandas as pd

from iknowpuck.summaries import league_takeaways, manager_reports, ordinal, position_timing, verdict


def test_ordinal():
    assert [ordinal(i) for i in (1, 2, 3, 4, 11, 12, 13, 21, 22)] == ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd"]


def test_position_timing_counts_kth_goalie():
    rows = [{"season": 2025, "overall": i + 1, "round": i // 4 + 1, "pos": "G" if i % 2 else "C"} for i in range(40)]
    t = position_timing(pd.DataFrame(rows), n_teams=4)
    assert t["G2"] == 1 and t["G4"] == 2  # 2nd goalie at pick 4 (round 1), 4th at pick 8 (round 2)


def _strategy():
    rows = []
    for s in (2024, 2025):
        for i, oid in enumerate("abcdef"):
            rows.append({"season": s, "owner_id": oid, "final_rank": i + 1, "win_pct": 0.7 - 0.08 * i,
                         "value_added_all": 200 - 60 * i, "first_goalie_round": 2 + i, "d_share_r1_6": 0.1 * i,
                         "reach_early": 0.05 * i, "goalies_r1_3": 1})
    return pd.DataFrame(rows)


def test_manager_reports_and_takeaways():
    ms = _strategy()
    reps = manager_reports(ms, {"a": "Alice"}, active=set("abcde"))
    assert reps[0].name == "Alice" and reps[0].headline == "A consistent contender"
    assert "Drafts well" in reps[0].bullets[0] and "early" in reps[0].bullets[1]
    assert reps[-1].active is False  # former managers sort last
    tk = league_takeaways(ms, top_n=2)
    assert tk["top_value"] > tk["rest_value"] and tk["top_goalie_round"] < tk["rest_goalie_round"]


def test_verdict():
    assert verdict({"significant": True, "ci_high": 1}) == "yes"
    assert verdict({"significant": False, "ci_high": -0.1}) == "no"
    assert verdict({"significant": False, "ci_high": 0.1}) == "tie"
