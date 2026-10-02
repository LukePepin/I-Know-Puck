"""Export a public, credential-free snapshot for the showcase app (showcase/).

Run locally (needs .env and the data cache):   PYTHONPATH=src .venv/bin/python scripts/export_showcase.py

What goes out: public NHL player data (names, positions, projections, ADP), the league's draft picks and
rosters with every other manager anonymized ("Team 1".."Team 12" for 2027 by draft slot, "Manager A".. for
history), model results and test results. What never goes out: ESPN cookies, the league id, ESPN member ids,
fantasy team names, other managers' names.
"""

from __future__ import annotations

import json
import string
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iknowpuck import diagnostics as D  # noqa: E402
from iknowpuck import team_effects as T  # noqa: E402
from iknowpuck.config import RUNS_DIR, SLOT_NAMES, load_credentials  # noqa: E402
from iknowpuck.data.dataset import build_panel  # noqa: E402
from iknowpuck.data.espn import EspnClient  # noqa: E402
from iknowpuck.draft import crn_experiment  # noqa: E402
from iknowpuck.history import NHL_ABBREV  # noqa: E402
from iknowpuck.injuries import DEFAULT_GAMES_MISSED, load_overrides  # noqa: E402
from iknowpuck.pipeline import FIRST_PANEL_SEASON, build  # noqa: E402
from iknowpuck.spectral import eigengap_null  # noqa: E402
from iknowpuck.summaries import PLAIN_MEANING, PLAIN_QUESTIONS, plain_answer, verdict  # noqa: E402
from iknowpuck.valuation import COVAR_INFLATION  # noqa: E402

OUT = ROOT / "showcase" / "data"
AUTHOR = "Luke (author)"


def write(name: str, obj) -> None:
    path = OUT / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False, float_format="%.4g")
    else:
        path.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    print(f"  {name}: {len(obj) if hasattr(obj, '__len__') else ''}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    creds = load_credentials()
    client = EspnClient(creds)
    b = build(creds.season)
    S, H = b.settings, b.history
    me = S.my_team_id
    order = list(S.pick_order)
    pool = b.injured_pool(load_overrides(), dict(DEFAULT_GAMES_MISSED))
    ctx = b.context(order, me, pool=pool)
    val, f = ctx.val, pool.frame
    idx_of = {int(p): i for i, p in enumerate(f.player_id)}
    panel = build_panel(list(range(FIRST_PANEL_SEASON, b.season + 1)), client)
    cur = panel[panel.season == b.season].drop_duplicates("player_id").set_index("player_id")

    def team_label(t: int) -> str:
        return AUTHOR if t == me else f"Team {order.index(t) + 1}"

    def player_row(pid: int) -> dict:
        if pid in idx_of:
            r = f.loc[idx_of[pid]]
            status = r.get("injury_status")
            return {"player": r["name"], "pos": r.pos, "nhl": NHL_ABBREV.get(r.pro_team_id, ""), "adp": r.adp, "proj_pts": r.fpts,
                    "espn_proj_pts": r.get("fpts_espn"), "status": status if isinstance(status, str) and status else "ACTIVE"}
        r = cur.loc[pid] if pid in cur.index else None
        return {"player": r["name"] if r is not None else f"player {pid}", "pos": r.pos if r is not None else "", "nhl": NHL_ABBREV.get(r.pro_team_id, "") if r is not None else "",
                "adp": r.adp if r is not None else np.nan, "proj_pts": np.nan, "espn_proj_pts": np.nan, "status": "ACTIVE"}

    print("2027 draft and rosters")
    d = client.draft(b.season)
    picks = pd.DataFrame([{"overall": int(r.overall), "round": int(r["round"]), "round_pick": int(r.round_pick), "slot": order.index(int(r.team_id)) + 1,
                           "team": team_label(int(r.team_id)), **player_row(int(r.player_id))} for _, r in d.sort_values("overall").iterrows()])
    picks["value_vs_adp"] = picks.adp - picks.overall  # > 0: taken later than the market expected (a bargain)
    write("draft_2027.csv", picks)

    rosters = client.rosters(b.season) or {int(t): d[d.team_id == t].player_id.astype(int).tolist() for t in d.team_id.unique()}
    slot_lab = []
    cnt: dict[int, int] = {}
    for s_ in val.slot_list:
        cnt[s_] = cnt.get(s_, 0) + 1
        slot_lab.append(f"{SLOT_NAMES.get(s_, s_)}{cnt[s_]}")
    roster_rows, team_idx = [], {}
    for t, ids in rosters.items():
        r_idx = [idx_of[p] for p in ids if p in idx_of]
        team_idx[t] = r_idx
        cost, rows_, cols_, starter = val.assignment(r_idx)
        assigned = {int(i): int(j) for i, j in zip(rows_, cols_) if cost[i, j] < 1e6}
        u = val.usage(r_idx)
        for k, j in enumerate(r_idx):
            slot = slot_lab[assigned[k]] if k in assigned else "none"
            roster_rows.append({"team": team_label(t), **player_row(int(f.loc[j, "player_id"])), "lineup": "starter" if u[k] == 1 else "bench",
                                "slot": slot if u[k] == 1 else "BN", "weekly_pts": float(u[k] * val.pts[j] / 26.0)})
        for p in ids:
            if p not in idx_of:
                roster_rows.append({"team": team_label(t), **player_row(p), "lineup": "bench", "slot": "BN", "weekly_pts": 0.0})
    write("rosters_2027.csv", pd.DataFrame(roster_rows))

    dist = {t: val.team_dist(r) for t, r in team_idx.items()}
    teams = list(team_idx)
    h2h = pd.DataFrame([[float(val.win_probs(dist[a], dist[c])[0]) if a != c else np.nan for c in teams] for a in teams],
                       index=[team_label(t) for t in teams], columns=[team_label(t) for t in teams])
    team_rows = []
    rr = pd.DataFrame(roster_rows)
    for t in teams:
        lab = team_label(t)
        mine = rr[rr.team == lab]
        pk = picks[(picks.team == lab) & (picks["round"] <= 15) & (picks.adp < 200)]  # ESPN ADP caps near 229, so late rounds are noise
        stack = mine[mine.nhl != ""].nhl.value_counts()
        team_rows.append({
            "team": lab, "slot": order.index(t) + 1, "weekly_pts": float(dist[t][0][0]), "weekly_sd": float(np.sqrt(dist[t][1][0])),
            "win_chance": float(np.nanmean(h2h.loc[lab].to_numpy(float))), "starters_weekly": float(mine[mine.lineup == "starter"].weekly_pts.sum()),
            "bench_weekly": float(mine[mine.lineup == "bench"].weekly_pts.sum()), "injured": int(mine.status.fillna("ACTIVE").ne("ACTIVE").sum()),
            "top_stack": f"{stack.index[0]} x{int(stack.iloc[0])}" if len(stack) else "", "goalies": int((mine.pos == "G").sum()),
            "draft_value_vs_adp": float(pk.value_vs_adp.clip(-60, 60).sum()),
            "best_value": (lambda x: f"{x.player} (R{int(x['round'])}, {x.value_vs_adp:+.0f})")(pk.loc[pk.value_vs_adp.idxmax()]) if pk.value_vs_adp.notna().any() else "",
            "biggest_reach": (lambda x: f"{x.player} (R{int(x['round'])}, {x.value_vs_adp:+.0f})")(pk.loc[pk.value_vs_adp.idxmin()]) if pk.value_vs_adp.notna().any() else "",
        })
    tt = pd.DataFrame(team_rows).sort_values("weekly_pts", ascending=False)
    tt["rank"] = np.arange(1, len(tt) + 1)
    write("teams_2027.csv", tt)
    write("h2h_2027.csv", h2h.reset_index().rename(columns={"index": "team"}))

    print("league history")
    owners: list[str] = []
    for o in list(b.drafts.sort_values(["season", "overall"]).owner_id.astype(str)) + list(b.strategy.owner_id.astype(str)):
        if o not in owners:
            owners.append(o)
    my_owner = str(b.manager_of_team.get(me))
    letters = iter(string.ascii_uppercase)
    anon = {o: (AUTHOR if o == my_owner else f"Manager {next(letters)}") for o in owners}
    hd = b.drafts.copy()
    pts = H.gamelogs.groupby(["season", "player_id"]).fp.agg(["sum", "size"]).rename(columns={"sum": "actual_pts", "size": "games"}).reset_index()
    hd = hd.merge(pts, on=["season", "player_id"], how="left").fillna({"actual_pts": 0, "games": 0})
    hist = pd.DataFrame({"season": hd.season, "overall": hd.overall, "round": hd["round"], "manager": hd.owner_id.astype(str).map(anon),
                         "player": hd.player_id.map(H.player_names), "pos": hd.player_id.map(H.player_pos).fillna(hd.get("pos")),
                         "nhl": [NHL_ABBREV.get(H.player_team.get((s, p)), "") for s, p in zip(hd.season, hd.player_id)],
                         "adp": hd.adp, "actual_pts": hd.actual_pts, "games": hd.games})
    write("history_picks.csv", hist)
    ms = b.strategy.copy()
    keep = [c for c in ["season", "win_pct", "final_rank", "value_added_all", "draft_skill", "injury_luck", "first_goalie_round", "d_share_r1_6",
                        "reach_early", "pickups", "top_count"] if c in ms]
    ms_out = ms[keep].assign(manager=ms.owner_id.astype(str).map(anon))
    write("history_managers.csv", ms_out)
    cor = b.strategy_cor
    write("habits.csv", cor[cor.outcome == "Win %"][["strategy", "rho", "ci_low", "ci_high"]].sort_values("rho"))
    if b.spectral is not None:
        tb = b.spectral.table().reset_index()
        tb["group"] = tb.cluster.map(lambda c: b.cluster_names.get(int(c), {}).get("name", f"Group {c + 1}"))
        write("styles.csv", pd.DataFrame({"manager": tb.manager.astype(str).map(anon), "x": tb.x, "y": tb.y, "group": tb.group}))
        groups = [{"name": v["name"], "traits": v["traits"]} for v in b.cluster_names.values()]
    else:
        groups = []

    print("model results")
    runs = sorted(RUNS_DIR.glob("*_ikp/results.json"))
    tests = []
    if runs:
        res = json.loads(runs[-1].read_text())["results"]
        for r in res:
            v = verdict(r["test"])
            tests.append({"id": r["id"], "question": PLAIN_QUESTIONS.get(r["id"], r["hypothesis"]), "answer": plain_answer(v), "verdict": v,
                          "meaning": PLAIN_MEANING.get((r["id"], v), ""), "metric": r["metric"], "n": r["test"]["n"], "diff": r["test"]["mean_diff"],
                          "lo": r["test"]["ci_low"], "hi": r["test"]["ci_high"], "p_holm": r["test"]["p_adjusted"], "dz": r["test"]["cohens_dz"]})
    write("tests.json", tests)
    pe = b.proj_eval
    acc = pe.assign(err_espn=(pe.espn - pe.actual).abs(), err_ours=(pe.market_adj - pe.actual).abs()).groupby("season")[["err_espn", "err_ours"]].mean().reset_index()
    write("projection_accuracy.csv", acc)

    m, per = D.matchups(client, H.seasons)
    reg = m[~m.playoff].copy()
    nz = D.normality(D.standardised_scores(reg))
    wp = D.win_prob_backtest(m)
    wps = D.scores(wp.p_model, wp.result)
    wk = D.weekly_player_stats(H.gamelogs, per, S)
    disp = D.dispersion_table(wk, S, H.player_pos, n_boot=300)
    _, cov = D.covariance_inflation(wk, S, H.player_pos, n_boot=300)
    cal, cal_e = D.calibration_line(pe.market_adj, pe.actual), D.calibration_line(pe.espn, pe.actual)
    bu = D.bench_usage(m, per, H.gamelogs, H.stints, H.player_pos, S.lineup)
    picks_loso, _, _ = D.pick_model_loso(b.drafts, panel, S.lineup)
    p_styles = eigengap_null(b.spectral.features, n_perm=1000)[2] if b.spectral is not None else float("nan")
    rel = D.reliability(wp.p_model, wp.result, 8)
    write("weekly_scores.csv", pd.DataFrame({"z": D.standardised_scores(reg).round(4)}))
    write("win_calibration.csv", rel)
    assumptions = [
        {"assumption": "A team's weekly score follows a bell curve", "result": f"{nz['n']} real weekly scores: skew {nz['skew']:+.2f}, Shapiro-Wilk p = {nz['p_shapiro']:.2f}",
         "verdict": "Holds" if nz["p_shapiro"] > 0.05 else "Does not hold"},
        {"assumption": "The win-chance formula gives honest odds", "result": f"{len(wp)} past matchups: Brier {wps['brier']:.3f} vs 0.250 for a coin flip; winner called {wps['accuracy']:.0%} of the time",
         "verdict": "Holds" if wps["brier"] < 0.25 else "Does not hold"},
        {"assumption": "Only the 20 starters really score (weekly lineup lock)", "result": f"real score = {bu['starters']:.2f} x starters + {bu['bench']:.2f} x bench over {bu['team_weeks']} team-weeks; the model now counts the bench at 0.25",
         "verdict": "Fixed"},
        {"assumption": "Our projections are right on average", "result": f"slope {cal['slope']:.2f} (95% CI {cal['slope_lo']:.2f} to {cal['slope_hi']:.2f}); ESPN's own: {cal_e['slope']:.2f}",
         "verdict": "Holds" if cal["slope_lo"] <= 1 <= cal["slope_hi"] else "Does not hold"},
        {"assumption": "Players' stats swing like the model says", "result": f"{int((disp.estimated / disp.assumed).between(0.75, 1.25).sum())} of {len(disp)} stats within 25%; variance inflation {cov['estimated']:.2f} measured vs {COVAR_INFLATION} assumed",
         "verdict": "Partly holds"},
        {"assumption": "Opponents pick the way the model says", "result": f"actual pick in the model's top 5: {(picks_loso.model_rank <= 5).mean():.0%} (ESPN order alone: {(picks_loso.adp_rank <= 5).mean():.0%})",
         "verdict": "Roughly holds"},
        {"assumption": "Managers fall into distinct drafting styles", "result": f"permutation test p = {p_styles:.2f}", "verdict": "Holds" if p_styles < 0.05 else "Not supported"},
    ]
    write("assumptions.json", assumptions)

    print("optimization and theory")
    cands, crn, ind = crn_experiment(ctx, [], 5, 30, 0)
    write("crn.json", {"candidates": [f"{f.loc[c, 'name']} ({f.loc[c, 'pos']})" for c in cands], "crn": crn.round(5).tolist(), "independent": ind.round(5).tolist()})
    sc = []
    for pos, n_s in {"C": S.lineup.get(0, 0), "LW": S.lineup.get(1, 0), "RW": S.lineup.get(2, 0), "D": S.lineup.get(4, 0), "G": S.lineup.get(5, 0)}.items():
        sub = f[f.pos == pos].assign(v=ctx.value[f.pos == pos]).nlargest(110, "v").reset_index(drop=True)
        sc.append(pd.DataFrame({"pos": pos, "rank": sub.index + 1, "player": sub.name, "proj_pts": sub.v, "replacement_rank": S.n_teams * n_s}))
    write("scarcity.csv", pd.concat(sc, ignore_index=True))
    rates = T.projected_rates(panel, S, H.seasons)
    stn = T.stints(H.gamelogs, rates)
    dd = T.add_team_surprise(stn, 20)
    dd = dd[dd.grp != "G"].reset_index(drop=True)
    eff = T.team_effect(dd, 2000, 2000)
    w = T.halves(H.gamelogs, H.schedules, rates)
    pz = T.persistence(w, 2000)
    write("theory.json", {"slope": eff["slope"], "lo": eff["lo"], "hi": eff["hi"], "p": eff["p"], "n": eff["n"], "teams": eff["teams"], "icc": T.icc(dd),
                          "quintiles": T.by_quantile(dd).to_dict("records"), "carry_team_alone": pz["team_alone"], "carry_team": pz["team"], "carry_p": pz["p"]})

    write("meta.json", {
        "season": b.season, "snapshot": time.strftime("%Y-%m-%d"), "teams": S.n_teams, "rounds": S.rounds, "history_seasons": H.seasons,
        "scoring": [{"stat": c.name, "points": c.points} for c in S.scoring_categories],
        "lineup": {SLOT_NAMES.get(k, str(k)): v for k, v in S.lineup.items() if v}, "lineup_lock": "weekly" if S.weekly_lineups else "daily",
        "bench_usage": val.bench_factor, "bench_fit": bu, "author_slot": order.index(me) + 1, "groups": groups,
        "sources": ["ESPN Fantasy (public player projections, ADP and stats; league drafts and results)", "MoneyPuck (advanced stats)"],
    })
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
