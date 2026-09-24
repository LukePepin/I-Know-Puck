"""Plain-language summaries of the league's history, generated from the data.

Everything here turns numbers into short sentences for the app. Labels ("early", "late",
"above average") are relative to the league's own spread, so they adapt to any league.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def ordinal(n: int) -> str:
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def position_timing(drafts: pd.DataFrame, n_teams: int) -> dict[str, float]:
    """Average round (across seasons) by which the k-th goalie / defenseman was drafted in this league."""
    out: dict[str, list[int]] = {}
    if "pos" not in drafts:
        return {}
    for _, g in drafts.groupby("season"):
        for pos, ks in (("G", (n_teams // 2, n_teams, 2 * n_teams)), ("D", (n_teams, 2 * n_teams, 4 * n_teams))):
            sub = g[g.pos == pos].sort_values("overall")
            for k in ks:
                if len(sub) >= k:
                    out.setdefault(f"{pos}{k}", []).append(int(sub.iloc[k - 1]["round"]))
    return {k: float(np.mean(v)) for k, v in out.items()}


@dataclass
class ManagerReport:
    owner_id: str
    name: str
    active: bool
    seasons: int
    avg_finish: float
    best_finish: int
    finishes: str
    win_pct: float
    headline: str
    bullets: list[str]
    sunday_tip: str


def _tercile(value: float, series: pd.Series) -> int:
    """-1 low third, 0 middle, +1 top third of the league's manager averages."""
    lo, hi = series.quantile([1 / 3, 2 / 3])
    return -1 if value < lo else (1 if value > hi else 0)


def manager_reports(strategy: pd.DataFrame, names: dict[str, str], active: set[str]) -> list[ManagerReport]:
    if strategy.empty:
        return []
    avg = strategy.groupby("owner_id").agg(
        seasons=("season", "nunique"), avg_finish=("final_rank", "mean"), best_finish=("final_rank", "min"),
        win_pct=("win_pct", "mean"), value=("value_added_all", "mean"), goalie_round=("first_goalie_round", "mean"),
        d_share=("d_share_r1_6", "mean"), reach=("reach_early", "mean"),
    )
    reports = []
    for oid, r in avg.iterrows():
        hist = strategy[strategy.owner_id == oid].sort_values("season")
        finishes = ", ".join(f"{ordinal(f)} in {s}" for s, f in zip(hist.season, hist.final_rank))
        bullets = []
        tips = []

        v = _tercile(r.value, avg.value)
        bullets.append({1: f"Drafts well: the players they pick score about {r.value:.0f} more points per season than their draft spots usually produce.",
                        0: f"Drafts about as well as the league average ({r.value:+.0f} points per season compared with their draft spots).",
                        -1: f"Drafts below average: their picks score about {abs(r.value):.0f} points per season less than their draft spots usually produce."}[v])

        g = _tercile(r.goalie_round, avg.goalie_round)
        bullets.append({-1: f"Grabs goalies early (first goalie around round {r.goalie_round:.0f}).",
                        0: f"Takes a first goalie at a normal time (around round {r.goalie_round:.0f}).",
                        1: f"Waits on goalies (first goalie around round {r.goalie_round:.0f})."}[g])
        if g == -1:
            tips.append("expect them to take a goalie in the first couple of rounds")

        d = _tercile(r.d_share, avg.d_share)
        if d == 1 and r.d_share >= 0.3:
            bullets.append(f"Likes defensemen early: about {r.d_share:.0%} of their first six picks are defensemen.")
            tips.append("they may take a top defenseman you want")
        elif d == 1:
            bullets.append(f"Takes slightly more defensemen early than most ({r.d_share:.0%} of their first six picks).")
        elif d == -1:
            bullets.append("Almost always takes forwards in the first six rounds.")

        rc = _tercile(r.reach, avg.reach)
        if rc == 1:
            bullets.append("Often picks players earlier than ESPN ranks them (a 'reacher').")
            tips.append("they can surprise you by taking someone early")
        elif rc == -1:
            bullets.append("Sticks closely to ESPN's rankings, so their picks are easy to predict.")
            tips.append("they will likely take the highest-ranked player left")

        avgf = r.avg_finish
        if avgf <= 3.5:
            headline = "A consistent contender"
        elif r.best_finish == 1:
            headline = "A past champion with up-and-down results"
        elif avgf <= 7:
            headline = "A solid middle-of-the-pack team"
        else:
            headline = "Has struggled in recent seasons"
        tip = ("On Sunday: " + "; ".join(tips) + ".") if tips else "On Sunday: no strong habits; they mostly follow the rankings."
        reports.append(ManagerReport(
            owner_id=str(oid), name=names.get(str(oid), str(oid)), active=str(oid) in active, seasons=int(r.seasons),
            avg_finish=float(avgf), best_finish=int(r.best_finish), finishes=finishes, win_pct=float(r.win_pct),
            headline=headline, bullets=bullets, sunday_tip=tip,
        ))
    return sorted(reports, key=lambda m: (not m.active, m.avg_finish))


def league_takeaways(strategy: pd.DataFrame, top_n: int = 4) -> dict[str, float]:
    """Compare managers who finished in the top ``top_n`` with everyone else."""
    if strategy.empty:
        return {}
    top = strategy[strategy.final_rank <= top_n]
    rest = strategy[strategy.final_rank > top_n]
    return {
        "top_value": float(top.value_added_all.mean()), "rest_value": float(rest.value_added_all.mean()),
        "top_goalie_round": float(top.first_goalie_round.mean()), "rest_goalie_round": float(rest.first_goalie_round.mean()),
        "top_d_share": float(top.d_share_r1_6.mean()), "rest_d_share": float(rest.d_share_r1_6.mean()),
        "top_reach": float(top.reach_early.mean()), "rest_reach": float(rest.reach_early.mean()),
        "top_win": float(top.win_pct.mean()), "rest_win": float(rest.win_pct.mean()),
    }


PLAIN_QUESTIONS = {
    "H1": "Are our player projections more accurate than ESPN's?",
    "H1b": "Does adding advanced stats (MoneyPuck) make our projections better?",
    "H2": "Does the app's way of drafting beat simply following ESPN's rankings?",
    "H2a": "Does chasing 'bargains' that only our model likes beat the rankings?",
    "H3": "Can we predict each manager's picks better than one league-wide model?",
    "H4": "Does grouping similar managers make those predictions better?",
}
PLAIN_MEANING = {
    ("H1", "yes"): "Our projections are used for every player.",
    ("H1b", "yes"): "A small improvement, so the advanced stats stay in.",
    ("H2", "tie"): "The app drafts about as well as the rankings (within about 2 percentage points of weekly win chance), and adds roster fit and 'will he last' odds.",
    ("H2", "yes"): "The app's drafting style wins more weeks than the rankings.",
    ("H2a", "no"): "Chasing bargains loses, so the app does not do it.",
    ("H3", "tie"): "Individual manager habits are shown for scouting only; the simulator uses the league-wide model.",
    ("H4", "tie"): "The manager groups are shown for interest only.",
    ("H4", "no"): "Grouping managers made pick predictions slightly worse, so the groups are used for scouting insight only.",
    ("H3", "no"): "Per-manager models predicted picks worse, so the simulator uses the league-wide model.",
    ("H1b", "tie"): "No clear gain, so the advanced stats could be dropped without losing accuracy.",
}


def verdict(test: dict) -> str:
    if test.get("significant"):
        return "yes"
    if test.get("ci_high", 0) < 0:
        return "no"
    return "tie"


def plain_answer(v: str) -> str:
    return {"yes": "Yes", "no": "No, it does worse", "tie": "No clear difference"}[v]
