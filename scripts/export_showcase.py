"""Export a public, credential-free snapshot for the showcase app (showcase/).

Run locally (needs .env and the data cache):   PYTHONPATH=src .venv/bin/python scripts/export_showcase.py
Then check before pushing:                     PYTHONPATH=src .venv/bin/python scripts/check_public.py

What goes out: public NHL player data (names, positions, projections, ADP), the league's picks, rosters,
schedule and results with managers shown by FIRST NAME only, and model and test results. What never goes
out: ESPN cookies, the league id, ESPN member ids, fantasy team names, managers' surnames.
"""

from __future__ import annotations

import json
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
from iknowpuck.opponents import group_of  # noqa: E402
from iknowpuck.pipeline import FIRST_PANEL_SEASON, build  # noqa: E402
from iknowpuck.spectral import eigengap_null  # noqa: E402
from iknowpuck.summaries import PLAIN_MEANING, PLAIN_QUESTIONS, plain_answer, verdict  # noqa: E402
from iknowpuck.valuation import COVAR_INFLATION, WEEKS_IN_SEASON  # noqa: E402

OUT = ROOT / "showcase" / "data"
GROUPS = {0: "forwards", 1: "defense", 2: "goalies"}


def write(name: str, obj) -> None:
    path = OUT / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False, float_format="%.4g")
    else:
        path.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    print(f"  {name}: {len(obj) if hasattr(obj, '__len__') else ''}")


def first_name_labels(managers: pd.DataFrame) -> dict[str, str]:
    """owner id -> first name (first name + last initial when two managers share a first name)."""
    full = dict(zip(managers.owner_id.astype(str), managers.manager.astype(str)))
    first = {o: n.split()[0] for o, n in full.items()}
    counts = pd.Series(list(first.values())).value_counts()
    return {o: (f if counts[f] == 1 else f"{f} {full[o].split()[-1][0]}.") for o, f in first.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*"):
        old.unlink()
    creds = load_credentials()
    client = EspnClient(creds)
    b = build(creds.season)
    S, H = b.settings, b.history
    order = list(S.pick_order)
    pool = b.injured_pool(load_overrides(), dict(DEFAULT_GAMES_MISSED))
    ctx = b.context(order, order[0], pool=pool)
    val, f = ctx.val, pool.frame
    idx_of = {int(p): i for i, p in enumerate(f.player_id)}
    panel = build_panel(list(range(FIRST_PANEL_SEASON, b.season + 1)), client)
    cur = panel[panel.season == b.season].drop_duplicates("player_id").set_index("player_id")
    names = first_name_labels(b.managers)

    def team_label(t: int) -> str:
        return names.get(str(b.manager_of_team.get(t)), f"Team {order.index(t) + 1}")

    def player_row(pid: int) -> dict:
        if pid in idx_of:
            r = f.loc[idx_of[pid]]
            status = r.get("injury_status")
            return {"player": r["name"], "pos": r.pos, "nhl": NHL_ABBREV.get(r.pro_team_id, ""), "adp": r.adp, "proj_pts": r.fpts,
                    "status": status if isinstance(status, str) and status else "ACTIVE"}
        r = cur.loc[pid] if pid in cur.index else None
        return {"player": r["name"] if r is not None else f"player {pid}", "pos": r.pos if r is not None else "",
                "nhl": NHL_ABBREV.get(r.pro_team_id, "") if r is not None else "", "adp": r.adp if r is not None else np.nan,
                "proj_pts": np.nan, "status": "ACTIVE"}

    # --- present: draft, rosters, rankings, this week ---------------------------------------------
    print("2027 season so far")
    d = client.draft(b.season)
    picks = pd.DataFrame([{"overall": int(r.overall), "round": int(r["round"]), "round_pick": int(r.round_pick), "slot": order.index(int(r.team_id)) + 1,
                           "manager": team_label(int(r.team_id)), **player_row(int(r.player_id))} for _, r in d.sort_values("overall").iterrows()])
    picks["value_vs_adp"] = picks.adp - picks.overall  # > 0: taken later than the market expected
    write("draft_2027.csv", picks)

    rosters = client.rosters(b.season) or {int(t): d[d.team_id == t].player_id.astype(int).tolist() for t in d.team_id.unique()}
    drafted = {int(t): set(g.player_id.astype(int)) for t, g in d.groupby("team_id")}
    slot_lab, cnt = [], {}
    for s_ in val.slot_list:
        cnt[s_] = cnt.get(s_, 0) + 1
        slot_lab.append(f"{SLOT_NAMES.get(s_, s_)}{cnt[s_]}")
    roster_rows, team_idx = [], {}
    for t, ids in rosters.items():
        r_idx = [idx_of[p] for p in ids if p in idx_of]
        team_idx[t] = r_idx
        cost, rows_, cols_, _ = val.assignment(r_idx)
        assigned = {int(i): int(j) for i, j in zip(rows_, cols_) if cost[i, j] < 1e6}
        u = val.usage(r_idx)
        for k, j in enumerate(r_idx):
            pid = int(f.loc[j, "player_id"])
            roster_rows.append({"manager": team_label(t), **player_row(pid), "lineup": "starter" if u[k] == 1 else "bench",
                                "slot": slot_lab[assigned[k]] if (k in assigned and u[k] == 1) else "BN", "weekly_pts": float(u[k] * val.pts[j] / WEEKS_IN_SEASON),
                                "acquired": "draft" if pid in drafted.get(t, set()) else "waivers/trade"})
        for p in ids:
            if p not in idx_of:
                roster_rows.append({"manager": team_label(t), **player_row(p), "lineup": "bench", "slot": "BN", "weekly_pts": 0.0,
                                    "acquired": "draft" if p in drafted.get(t, set()) else "waivers/trade"})
    rr = pd.DataFrame(roster_rows)
    write("rosters_2027.csv", rr)

    teams = list(team_idx)
    dist = {t: val.team_dist(r) for t, r in team_idx.items()}
    draft_dist = {t: val.team_dist([idx_of[p] for p in drafted.get(t, set()) if p in idx_of]) for t in teams}
    h2h = pd.DataFrame([[float(val.win_probs(dist[a], dist[c])[0]) if a != c else np.nan for c in teams] for a in teams],
                       index=[team_label(t) for t in teams], columns=[team_label(t) for t in teams])
    reg = None
    m_hist, per = D.matchups(client, H.seasons)
    reg = m_hist[~m_hist.playoff]
    obs_sd = float(reg.groupby(["season", "team_id"]).points.std().median())
    model_sd = float(np.median([np.sqrt(dist[t][1][0]) for t in teams]))
    sd_extra = float(np.sqrt(max(obs_sd**2 - model_sd**2, 0.0)))  # injuries, roster moves and schedule quirks the roster model leaves out
    pe = b.proj_eval
    per_player_week = float((pe.market_adj - pe.actual).std() / WEEKS_IN_SEASON)
    n_start = sum(c for s_, c in S.lineup.items() if s_ not in (7, 8))
    team_unc = round(float(per_player_week * np.sqrt(n_start)) * 2) / 2
    starters_all = rr[rr.lineup == "starter"].assign(grp=lambda x: x.pos.map(lambda p: GROUPS[group_of(p)]))
    grp_rank = starters_all.groupby(["manager", "grp"]).weekly_pts.sum().unstack().rank(ascending=False)
    team_rows = []
    for t in teams:
        lab = team_label(t)
        mine = rr[rr.manager == lab]
        pk = picks[(picks.manager == lab) & (picks["round"] <= 15) & (picks.adp < 200)]  # ESPN ADP caps near 229, so late rounds are noise
        stack = mine[mine.nhl != ""].nhl.value_counts()
        g = grp_rank.loc[lab] if lab in grp_rank.index else pd.Series(dtype=float)
        team_rows.append({
            "manager": lab, "slot": order.index(t) + 1, "weekly_pts": float(dist[t][0][0]), "weekly_sd": float(np.sqrt(dist[t][1][0])),
            "sd_sim": float(np.sqrt(dist[t][1][0] + sd_extra**2)), "draft_weekly_pts": float(draft_dist[t][0][0]),
            "win_chance": float(np.nanmean(h2h.loc[lab].to_numpy(float))), "injured": int(mine.status.fillna("ACTIVE").ne("ACTIVE").sum()),
            "top_stack": f"{stack.index[0]} x{int(stack.iloc[0])}" if len(stack) else "", "goalies": int((mine.pos == "G").sum()),
            "moves": int((mine.acquired != "draft").sum()), "strongest": g.idxmin() if len(g) else "", "weakest": g.idxmax() if len(g) else "",
            "best_value": (lambda x: f"{x.player} (R{int(x['round'])}, {x.value_vs_adp:+.0f})")(pk.loc[pk.value_vs_adp.idxmax()]) if pk.value_vs_adp.notna().any() else "",
            "biggest_reach": (lambda x: f"{x.player} (R{int(x['round'])}, {x.value_vs_adp:+.0f})")(pk.loc[pk.value_vs_adp.idxmin()]) if pk.value_vs_adp.notna().any() else "",
        })
    tt = pd.DataFrame(team_rows)
    tt["change_since_draft"] = tt.weekly_pts - tt.draft_weekly_pts
    tt = tt.sort_values("weekly_pts", ascending=False)
    tt["rank"] = np.arange(1, len(tt) + 1)
    write("teams_2027.csv", tt)
    write("h2h_2027.csv", h2h.reset_index().rename(columns={"index": "manager"}))

    raw = client.league_raw(b.season, ["mMatchupScore", "mMatchup", "mSettings", "mStatus"], max_age_s=0)
    ss = raw.get("settings", {}).get("scheduleSettings", {})
    status = raw.get("status", {})
    week_now = int(status.get("currentMatchupPeriod", 1) or 1)
    sched = []
    for mm in raw.get("schedule", []):
        if not mm.get("home") or not mm.get("away") or mm.get("playoffTierType", "NONE") != "NONE":
            continue
        wk = int(mm["matchupPeriodId"])
        h, a = mm["home"], mm["away"]
        done = mm.get("winner") not in (None, "UNDECIDED")
        live = wk == week_now and not done
        sched.append({"week": wk, "home": team_label(int(h["teamId"])), "away": team_label(int(a["teamId"])),
                      "home_pts": float(h.get("totalPointsLive", h.get("totalPoints", 0)) or 0) if (done or live) else np.nan,
                      "away_pts": float(a.get("totalPointsLive", a.get("totalPoints", 0)) or 0) if (done or live) else np.nan,
                      "status": "final" if done else ("in progress" if live else "upcoming")})
    write("schedule_2027.csv", pd.DataFrame(sched))

    # --- history --------------------------------------------------------------------------------
    print("2024-26 history")
    owner_label = lambda o: names.get(str(o), "former manager")  # noqa: E731
    hd = b.drafts.copy()
    pts = H.gamelogs.groupby(["season", "player_id"]).fp.agg(["sum", "size"]).rename(columns={"sum": "actual_pts", "size": "games"}).reset_index()
    hd = hd.merge(pts, on=["season", "player_id"], how="left").fillna({"actual_pts": 0, "games": 0})
    write("history_picks.csv", pd.DataFrame({
        "season": hd.season, "overall": hd.overall, "round": hd["round"], "manager": hd.owner_id.map(owner_label),
        "player": hd.player_id.map(H.player_names), "pos": hd.player_id.map(H.player_pos).fillna(hd.get("pos")),
        "nhl": [NHL_ABBREV.get(H.player_team.get((s, p)), "") for s, p in zip(hd.season, hd.player_id)],
        "adp": hd.adp, "actual_pts": hd.actual_pts, "games": hd.games}))
    ms = b.strategy
    write("history_standings.csv", pd.DataFrame({
        "season": ms.season, "manager": ms.owner_id.map(owner_label), "final_rank": ms.final_rank, "win_pct": ms.win_pct, "points_for": ms.points_for,
        "draft_value": ms.value_added_all, "draft_skill": ms.get("draft_skill"), "injury_luck": ms.get("injury_luck"), "pickups": ms.get("pickups")}))
    cor = b.strategy_cor
    write("habits.csv", cor[cor.outcome == "Win %"][["strategy", "rho", "ci_low", "ci_high"]].sort_values("rho"))
    groups = []
    if b.spectral is not None:
        tb = b.spectral.table().reset_index()
        tb["group"] = tb.cluster.map(lambda c: b.cluster_names.get(int(c), {}).get("name", f"Group {c + 1}"))
        write("styles.csv", pd.DataFrame({"manager": tb.manager.map(owner_label), "x": tb.x, "y": tb.y, "group": tb.group}))
        names, W = tb.manager.map(owner_label).tolist(), b.spectral.affinity
        i, j = np.triu_indices(len(names), 1)
        write("style_links.csv", pd.DataFrame({"a": [names[k] for k in i], "b": [names[k] for k in j], "w": W[i, j].round(4)}))
        groups = [{"name": v["name"], "traits": v["traits"]} for v in b.cluster_names.values()]
    rates = T.projected_rates(panel, S, H.seasons)
    stn = T.stints(H.gamelogs, rates)
    dd = T.add_team_surprise(stn, 20)
    dd = dd[dd.grp != "G"].reset_index(drop=True)
    eff = T.team_effect(dd, 2000, 2000)
    pz = T.persistence(T.halves(H.gamelogs, H.schedules, rates), 2000)
    write("theory.json", {"slope": eff["slope"], "lo": eff["lo"], "hi": eff["hi"], "p": eff["p"], "n": eff["n"], "teams": eff["teams"], "icc": T.icc(dd),
                          "quintiles": T.by_quantile(dd).to_dict("records"), "carry_team_alone": pz["team_alone"], "carry_team": pz["team"], "carry_p": pz["p"]})

    # --- models and checks ------------------------------------------------------------------------
    print("models and checks")
    runs = sorted(RUNS_DIR.glob("*_ikp/results.json"))
    tests = []
    if runs:
        for r in json.loads(runs[-1].read_text())["results"]:
            v = verdict(r["test"])
            tests.append({"id": r["id"], "question": PLAIN_QUESTIONS.get(r["id"], r["hypothesis"]), "answer": plain_answer(v), "verdict": v,
                          "meaning": PLAIN_MEANING.get((r["id"], v), ""), "metric": r["metric"], "n": r["test"]["n"], "diff": r["test"]["mean_diff"],
                          "lo": r["test"]["ci_low"], "hi": r["test"]["ci_high"], "p_holm": r["test"]["p_adjusted"]})
    write("tests.json", tests)
    write("projection_accuracy.csv", pe.assign(err_espn=(pe.espn - pe.actual).abs(), err_ours=(pe.market_adj - pe.actual).abs())
          .groupby("season")[["err_espn", "err_ours"]].mean().reset_index())
    nz = D.normality(D.standardised_scores(reg))
    wp = D.win_prob_backtest(m_hist)
    wps = D.scores(wp.p_model, wp.result)
    wk = D.weekly_player_stats(H.gamelogs, per, S)
    disp = D.dispersion_table(wk, S, H.player_pos, n_boot=300)
    _, cov = D.covariance_inflation(wk, S, H.player_pos, n_boot=300)
    cal, cal_e = D.calibration_line(pe.market_adj, pe.actual), D.calibration_line(pe.espn, pe.actual)
    bu = D.bench_usage(m_hist, per, H.gamelogs, H.stints, H.player_pos, S.lineup)
    picks_loso, _, _ = D.pick_model_loso(b.drafts, panel, S.lineup)
    p_styles = eigengap_null(b.spectral.features, n_perm=1000)[2] if b.spectral is not None else float("nan")
    write("weekly_scores.csv", pd.DataFrame({"z": D.standardised_scores(reg).round(4)}))
    write("win_calibration.csv", D.reliability(wp.p_model, wp.result, 8))
    write("assumptions.json", [
        {"assumption": "A team's weekly score follows a bell curve", "result": f"{nz['n']} real weekly scores: skew {nz['skew']:+.2f}, Shapiro-Wilk p = {nz['p_shapiro']:.2f}",
         "verdict": "Holds" if nz["p_shapiro"] > 0.05 else "Does not hold"},
        {"assumption": "The win-chance formula gives honest odds", "result": f"{len(wp)} past matchups: Brier score {wps['brier']:.3f} vs 0.250 for a coin flip; winner called {wps['accuracy']:.0%} of the time",
         "verdict": "Holds" if wps["brier"] < 0.25 else "Does not hold"},
        {"assumption": "Only the starting lineup really scores", "result": f"real score = {bu['starters']:.2f} x starters + {bu['bench']:.2f} x bench across {bu['team_weeks']} team-weeks; bench now counts {val.bench_factor:.2f}",
         "verdict": "Fixed"},
        {"assumption": "Season projections are right on average", "result": f"calibration slope {cal['slope']:.2f} (95% range {cal['slope_lo']:.2f} to {cal['slope_hi']:.2f}); ESPN's own: {cal_e['slope']:.2f}",
         "verdict": "Holds" if cal["slope_lo"] <= 1 <= cal["slope_hi"] else "Does not hold"},
        {"assumption": "Players' stats swing the way the model says", "result": f"{int((disp.estimated / disp.assumed).between(0.75, 1.25).sum())} of {len(disp)} stats within 25%; variance inflation {cov['estimated']:.2f} measured vs {COVAR_INFLATION} assumed",
         "verdict": "Partly holds"},
        {"assumption": "Managers pick the way the pick model says", "result": f"actual pick in the model's top 5: {(picks_loso.model_rank <= 5).mean():.0%} (ESPN order alone: {(picks_loso.adp_rank <= 5).mean():.0%})",
         "verdict": "Roughly holds"},
        {"assumption": "Managers fall into distinct drafting styles", "result": f"permutation test p = {p_styles:.2f}", "verdict": "Holds" if p_styles < 0.05 else "Not supported"},
    ])
    cands, crn, ind = crn_experiment(ctx, [], 5, 30, 0)
    write("crn.json", {"candidates": [f"{f.loc[c, 'name']} ({f.loc[c, 'pos']})" for c in cands], "crn": crn.round(5).tolist(), "independent": ind.round(5).tolist()})

    write("meta.json", {
        "season": b.season, "snapshot": time.strftime("%Y-%m-%d %H:%M"), "teams": S.n_teams, "rounds": S.rounds, "history_seasons": H.seasons,
        "scoring": [{"stat": c.name, "points": c.points} for c in S.scoring_categories],
        "lineup": {SLOT_NAMES.get(k, str(k)): v for k, v in S.lineup.items() if v}, "starters": n_start, "lineup_lock": "weekly" if S.weekly_lineups else "daily",
        "bench_usage": val.bench_factor, "bench_fit": bu, "groups": groups, "regular_weeks": int(ss.get("matchupPeriodCount", 24) or 24),
        "playoff_teams": int(ss.get("playoffTeamCount", 8) or 8), "current_week": week_now, "observed_weekly_sd": obs_sd, "model_weekly_sd": model_sd,
        "sd_extra": sd_extra, "per_player_weekly_miss": per_player_week, "team_uncertainty": team_unc,
        "spectral": {"k": int(b.spectral.k), "sigma": round(b.spectral.sigma, 3), "p": round(p_styles, 3), "n_features": int(b.spectral.features.drop(columns=["n_picks"], errors="ignore").shape[1]),
                     "eigenvalues": b.spectral.eigenvalues[:6].round(4).tolist()} if b.spectral is not None else None,
        "sources": ["ESPN Fantasy (player projections, ADP and stats; league drafts, rosters, schedule and results)", "MoneyPuck (advanced stats)"],
    })
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
