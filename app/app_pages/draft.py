"""Draft room: live board, suggested pick, roster, pick log."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
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
live = st.toggle("Live sync with ESPN draft", value=False, key="live_sync", help="Reads the ESPN draft every 5 seconds; the suggestion runs by itself when you are on the clock")
with st.expander("Simulation settings (optional)"):
    with st.container(horizontal=True):
        n_cand = st.number_input("Candidates to compare", 4, 20, 10, key="n_cand")
        n_roll = st.number_input("Simulations per candidate", 5, 100, 25, key="n_roll", help="More simulations give tighter estimates but take longer (about 7 seconds at the defaults)")


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
            with st.spinner(f"Comparing {n_cand} candidates across {n_roll} simulated drafts each..."):
                rec = recommend(ctx, picks, n_candidates=int(n_cand), n_rollouts=int(n_roll))
            st.session_state.rec = (n_done, rec)
        if "rec" in st.session_state and st.session_state.rec[0] == n_done:
            rec = st.session_state.rec[1].head(12).copy()
            rec["status"] = frame.loc[rec.pool_idx, "injury_status"].fillna("ACTIVE").str.replace("_", " ").str.lower().replace("active", "").to_numpy()
            rec["risk"] = frame.loc[rec.pool_idx, "injury_risk"].fillna("").to_numpy() if "injury_risk" in frame else ""
            rec["stack"] = [stack_note(teams[on_clock], j) for j in rec.pool_idx]
            tied = rec[rec.gap_to_best >= -2 * rec.gap_se.fillna(0)]
            pick_row = tied.sort_values(["p_avail_next_pick", "win_prob"], ascending=[True, False]).iloc[0]
            why = ("It gives the best chance of winning weeks." if len(tied) == 1 else
                   f"{len(tied)} players are tied for best; this one is the least likely to still be there at your next pick ({pick_row.p_avail_next_pick:.0%}).")
            extra = []
            if pick_row.status:
                extra.append(f"Note: currently {pick_row.status}; his value already assumes the missed games.")
            if pick_row.stack:
                extra.append(f"This would give you {pick_row.stack} players, which makes your weekly score swing more with one NHL team.")
            st.success(f"**Suggested pick: {pick_row['name']} ({pick_row.pos}).** {why} " + " ".join(extra))
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
            team_pick = st.selectbox("Drafted by", list(names), index=list(names).index(on_clock) if on_clock in names else 0, format_func=lambda t: names[t], key="rec_team")
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
