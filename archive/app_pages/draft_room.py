"""Draft room: live board, suggested pick, roster, pick log."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from draft_log import parse_log
from ui import NAVY, RUST, app_state, fig_show, note

from iknowpuck.config import SLOT_NAMES
from iknowpuck.data.espn import EspnClient, EspnError
from iknowpuck.draft import recommend
from iknowpuck.history import NHL_ABBREV

A = app_state()
b, ctx, pool, names, order, my_team = A["b"], A["ctx"], A["pool"], A["names"], A["order"], A["my_team"]
frame = pool.frame
pid_to_idx = {int(p): i for i, p in enumerate(frame.player_id)}

st.markdown("## Draft room")
live = st.toggle("Live sync with ESPN draft", value=False, key="live_sync",
                 help="Reads the ESPN draft every 5 seconds. ESPN's API may not publish picks until the draft ends; if nothing appears, paste the draft log below.")
LOG_FILE = Path(__file__).resolve().parents[2] / "data" / "draft_log.txt"  # a saved copy of ESPN's board or log, loaded whenever it changes
if LOG_FILE.exists() and st.session_state.get("log_file_mtime") != LOG_FILE.stat().st_mtime:
    st.session_state.log_file_mtime = LOG_FILE.stat().st_mtime
    parsed, unmatched = parse_log(LOG_FILE.read_text(encoding="utf-8"), frame, ctx.order, len(order), NHL_ABBREV)
    if parsed and len(parsed) >= len(st.session_state.picks):
        st.session_state.picks = parsed
        st.session_state.practice = False
        st.session_state.log_msg = ("ok", f"Loaded {len(parsed)} picks from data/draft_log.txt." + (f" Not matched: {', '.join(unmatched)}." if unmatched else ""))
with st.expander("Paste the ESPN draft log (use this if live sync shows no picks)", expanded=True, icon=":material/content_paste:"):
    st.caption("In ESPN's draft room, select the draft chat/pick list (Cmd+A inside it works), copy, and paste it here. Paste the whole log each time: "
               "it replaces the picks, so it also fixes any mistakes. Join/leave messages are ignored.")
    with st.form("paste_log", clear_on_submit=True, border=False):
        log_text = st.text_area("ESPN draft log", height=110, label_visibility="collapsed", placeholder="Nathan MacKinnon / COL C\nR1, P1 - Team Name\n...")
        if st.form_submit_button("Load picks from log", icon=":material/download:", type="primary"):
            parsed, unmatched = parse_log(log_text or "", frame, ctx.order, len(order), NHL_ABBREV)
            if parsed:
                st.session_state.picks = parsed
                st.session_state.practice = False
                st.session_state.log_msg = ("ok", f"Loaded {len(parsed)} picks." + (f" Not matched (kept as placeholders): {', '.join(unmatched)}." if unmatched else ""))
            else:
                st.session_state.log_msg = ("err", "No picks found. Copy the lines that look like 'Player / TEAM POS' and 'R1, P1 - Team name'.")
            st.rerun()
    msg = st.session_state.get("log_msg")
    if msg:
        (st.success if msg[0] == "ok" else st.error)(msg[1])
if live and st.session_state.get("practice"):  # practice picks must never mix with the real draft
    st.session_state.picks = []
    st.session_state.practice = False
PRESETS = {"Quick": (8, 15), "Standard": (12, 30), "Deep": (16, 60)}  # (candidates, simulations each): about 1 s, 2 s, 5 s
nhl_teams = sorted({int(t) for t in frame.pro_team_id.dropna() if int(t) in NHL_ABBREV}, key=lambda t: NHL_ABBREV[t])
with st.container(horizontal=True, vertical_alignment="bottom"):
    depth = st.segmented_control("Search depth", list(PRESETS), default="Standard", key="depth",
                                 help="Quick takes about 1 second, Standard about 2, Deep about 5. ESPN gives you 45 seconds a pick.") or "Standard"
    stack_teams = st.multiselect("NHL teams you'd like to stack (optional)", nhl_teams, format_func=NHL_ABBREV.get, key="stack_teams", max_selections=3,
                                 help="Their players get a star. When one is tied with the best pick, he becomes the suggestion; otherwise you see what he would cost.")
n_cand, n_roll = PRESETS[depth]
if st.session_state.get("practice"):
    st.warning("Practice picks are loaded. Press **Reset picks** (or turn on live sync) before the real draft.", icon=":material/science:")


def picks_as_idx() -> list[tuple[int, int]]:
    return [(t, pid_to_idx[p]) for t, p in st.session_state.picks if p in pid_to_idx]


def sync_espn():
    try:
        d = EspnClient(A["creds"]).draft(A["season"], live=True)
    except EspnError as e:
        st.error(str(e))
        return
    if len(d):
        st.session_state.picks = [(int(r.team_id), int(r.player_id)) for r in d.itertuples()]


def stack_note(team_state, j) -> str:
    team = frame.loc[j, "pro_team_id"]
    same = sum(1 for k in team_state.roster if frame.loc[k, "pro_team_id"] == team)
    return f"{same + 1} {NHL_ABBREV.get(team, '')}" if same >= 2 else ""


@st.fragment(run_every=5 if live else None)
def draft_room():
    if live:
        sync_espn()
    picks = picks_as_idx()
    n_done = len(st.session_state.picks)
    total = len(ctx.order)
    if n_done >= total:
        st.success("Draft complete. Open In-season moves for waiver-wire help.")
    on_clock = ctx.order[min(n_done, total - 1)]
    rnd = n_done // len(order) + 1
    until = next((d for d, t in enumerate(ctx.order[n_done:]) if t == my_team), None)
    with st.container(horizontal=True):
        st.metric("Overall pick", f"{n_done + 1} of {total}", border=True)
        st.metric("Round", rnd, border=True)
        st.metric("On the clock", names.get(on_clock, on_clock), border=True)
        st.metric("Picks until mine", "Now" if until == 0 else (until if until is not None else "-"), border=True)

    taken, teams = ctx.initial_state(picks)
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Recommendation" + (": you are on the clock" if on_clock == my_team else f" for {names.get(on_clock)}"))
        if st.button("Run simulation", type="primary", icon=":material/play_arrow:") or (live and on_clock == my_team and st.session_state.get("rec", (None,))[0] != n_done):
            extra = []
            if stack_teams and on_clock == my_team:  # also evaluate stack-team players the market expects to go within two rounds
                av = ctx.adp_order[~taken[ctx.adp_order]][: 2 * len(order)]
                fav = av[np.isin(frame.loc[av, "pro_team_id"].fillna(-1).astype(int).to_numpy(), stack_teams)]
                extra = fav[ctx.starter_fit(teams[on_clock], fav) > 0][:4].tolist() if len(fav) else []
            with st.spinner(f"Comparing {n_cand} candidates across {n_roll} simulated drafts each..."):
                rec = recommend(ctx, picks, n_candidates=int(n_cand), n_rollouts=int(n_roll), extra=extra)
            st.session_state.rec = (n_done, rec)
        if "rec" in st.session_state and st.session_state.rec[0] == n_done:
            full = st.session_state.rec[1].copy()
            full["fav"] = [on_clock == my_team and int(frame.loc[j, "pro_team_id"] if pd.notna(frame.loc[j, "pro_team_id"]) else -1) in stack_teams for j in full.pool_idx]
            rec = pd.concat([full.head(12), full.iloc[12:][full.iloc[12:].fav]]).copy()
            rec["status"] = frame.loc[rec.pool_idx, "injury_status"].fillna("ACTIVE").str.replace("_", " ").str.lower().replace("active", "").to_numpy()
            rec["risk"] = frame.loc[rec.pool_idx, "injury_risk"].fillna("").to_numpy() if "injury_risk" in frame else ""
            rec["stack"] = [stack_note(teams[on_clock], j) for j in rec.pool_idx]
            tied = rec[rec.gap_to_best >= -2 * rec.gap_se.fillna(0)]
            pick_row = tied.sort_values(["p_avail_next_pick", "win_prob"], ascending=[True, False]).iloc[0]
            why = ("It gives the best chance of winning weeks." if len(tied) == 1 else
                   f"{len(tied)} players are tied for best; this one is the least likely to still be there at your next pick ({pick_row.p_avail_next_pick:.0%}).")
            fav_tied = tied[tied.fav]
            if len(fav_tied):
                pick_row = fav_tied.sort_values("win_prob", ascending=False).iloc[0]
                why = f"He is statistically tied for the best pick and fits your {NHL_ABBREV.get(int(frame.loc[pick_row.pool_idx, 'pro_team_id']), '')} stack."
            notes = []
            if pick_row.status:
                notes.append(f"Note: currently {pick_row.status}; his value already assumes the missed games.")
            if pick_row.stack and not pick_row.fav:
                notes.append(f"This would give you {pick_row.stack} players, which makes your weekly score swing more with one NHL team.")
            st.success(f"**Suggested pick: {pick_row['name']} ({pick_row.pos}).** {why} " + " ".join(notes))
            fav_rest = rec[rec.fav & (rec.pool_idx != pick_row.pool_idx)].sort_values("win_prob", ascending=False)
            if len(fav_rest):
                fr = fav_rest.iloc[0]
                cost = pick_row.win_prob - fr.win_prob
                st.info(f"**Stack option: {fr['name']} ({fr.pos}, {NHL_ABBREV.get(int(frame.loc[fr.pool_idx, 'pro_team_id']), '')})** "
                        + (f"costs about {100 * cost:.1f} points of weekly win chance ({fr.win_prob:.1%} vs {pick_row.win_prob:.1%})." if cost > 0.0005 else "is just as good."),
                        icon=":material/star:")
            elif stack_teams and on_clock == my_team:
                st.caption("No player from your stack teams is a sensible pick here; the market expects them later.")
            rec["name"] = np.where(rec.fav, "★ " + rec["name"], rec["name"])
            fig = go.Figure(go.Bar(
                x=rec.win_prob, y=rec.name + " (" + rec.pos + ")", orientation="h",
                error_x=dict(type="data", array=1.96 * rec.gap_se.fillna(0), color="#444"),
                marker_color=[NAVY if i == 0 else ("#7F93AD" if abs(g) <= 2 * s else "#C8CED8") for i, (g, s) in enumerate(zip(rec.gap_to_best, rec.gap_se.fillna(0)))],
                hovertemplate="%{y}<br>chance of winning a week %{x:.3f}<extra></extra>",
            ))
            lo = float(rec.win_prob.min()) - 0.01
            fig.update_layout(title="Chance of winning a week if you draft each player now", xaxis=dict(range=[lo, float(rec.win_prob.max()) + 0.01], title="chance of winning a week"), yaxis=dict(autorange="reversed"))
            fig_show(fig, 60 + 32 * len(rec))
            st.dataframe(
                rec[["name", "pos", "proj_value", "adp", "market_rank", "win_prob", "gap_to_best", "gap_se", "p_avail_next_pick", "status", "risk", "stack"]],
                hide_index=True,
                column_config={
                    "proj_value": st.column_config.NumberColumn("proj pts", format="%.0f"), "adp": st.column_config.NumberColumn("ADP", format="%.0f"),
                    "market_rank": "market rank", "win_prob": st.column_config.NumberColumn("win chance", format="%.3f"),
                    "gap_to_best": st.column_config.NumberColumn("gap to best", format="%+.3f"), "gap_se": st.column_config.NumberColumn("gap SE", format="%.3f"),
                    "p_avail_next_pick": st.column_config.ProgressColumn("still there next pick", format="percent", min_value=0, max_value=1),
                    "status": "injury", "risk": "injury history", "stack": "same NHL team",
                },
            )
            note("A longer bar means a better chance of winning weeks. The dark bar is the top estimate; mid-blue bars are statistically tied with it. "
                 "Among tied players, take the one least likely to still be there at your next pick.")
        st.markdown("#### Record a pick")
        taken_ids = {p for _, p in st.session_state.picks}
        avail = frame[~frame.player_id.isin(taken_ids)].sort_values("fpts", ascending=False)
        with st.container(horizontal=True, vertical_alignment="bottom"):
            choice = st.selectbox("Player", avail.player_id.tolist(), key="rec_player",
                                  format_func=lambda p: f"{frame.loc[pid_to_idx[p], 'name']} ({frame.loc[pid_to_idx[p], 'pos']}, {frame.loc[pid_to_idx[p], 'fpts']:.0f} pts)")
            if st.session_state.get("rec_team_pick_no") != n_done and on_clock in names:  # new pick: follow whoever is on the clock
                st.session_state["rec_team"] = on_clock
                st.session_state["rec_team_pick_no"] = n_done
            team_pick = st.selectbox("Drafted by", list(names), format_func=lambda t: names[t], key="rec_team")
            if st.button("Add", icon=":material/add:"):
                st.session_state.picks.append((team_pick, int(choice)))
                st.rerun()
        with st.container(horizontal=True):
            if st.button("Undo last pick", icon=":material/undo:") and st.session_state.picks:
                st.session_state.picks.pop()
                st.rerun()
            if st.button("Sync once from ESPN", icon=":material/sync:"):
                sync_espn()
                st.rerun()
        with st.container(horizontal=True):
            if st.button("Practice: other teams pick until my turn", icon=":material/science:", disabled=live,
                         help="Rehearse the draft: the other managers pick the way the model expects. Turn live sync on for the real draft."):
                rng = np.random.default_rng(len(st.session_state.picks) + 17)
                taken_p, teams_p = ctx.initial_state(picks_as_idx())
                for p_no in range(len(st.session_state.picks), len(ctx.order)):
                    tid = ctx.order[p_no]
                    if tid == my_team:  # stop when you are on the clock; record your own pick with Add
                        break
                    j = ctx.opponent_pick(tid, teams_p[tid], taken_p, p_no, rng)
                    taken_p[j] = True
                    if not ctx.place(teams_p[tid], j):
                        teams_p[tid].roster.append(j)
                    st.session_state.picks.append((tid, int(frame.loc[j, "player_id"])))
                st.session_state.practice = True
                st.rerun()
            if st.button("Reset picks", icon=":material/restart_alt:", disabled=not st.session_state.picks):
                st.session_state.picks = []
                st.session_state.practice = False
                st.rerun()

    with right:
        mine = teams[my_team]
        st.markdown("#### My roster")
        rows = [{"player": frame.loc[j, "name"], "pos": frame.loc[j, "pos"], "NHL": NHL_ABBREV.get(frame.loc[j, "pro_team_id"], ""), "proj pts": round(ctx.value[j])} for j in mine.roster]
        st.dataframe(pd.DataFrame(rows, columns=["player", "pos", "NHL", "proj pts"]), hide_index=True)
        st.caption("Open slots: " + ", ".join(f"{SLOT_NAMES[s]} {c}" for s, c in mine.open.items() if c > 0))
        wk = pd.DataFrame([{"manager": names.get(t, t), "weekly pts": ctx.val.weekly_points(teams[t].roster)} for t in order]).sort_values("weekly pts")
        fig = go.Figure(go.Bar(x=wk["weekly pts"], y=wk.manager, orientation="h", marker_color=[RUST if m == names[my_team] else NAVY for m in wk.manager],
                               hovertemplate="%{y}<br>%{x:.1f} expected weekly points<extra></extra>"))
        fig.update_layout(title="Projected weekly points of rosters so far", xaxis_title="expected weekly fantasy points")
        fig_show(fig, 420)

    with st.expander("Pick log"):
        log = pd.DataFrame([{"pick": i + 1, "manager": names.get(t, t), "player": frame.loc[pid_to_idx[p], "name"] if p in pid_to_idx else p}
                            for i, (t, p) in enumerate(st.session_state.picks)])
        st.dataframe(log, hide_index=True)


draft_room()
