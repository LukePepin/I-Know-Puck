"""Pre-draft plan: injuries to check, simulated plan from your slot, availability, player board."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import NAVY, PALETTE, app_state, fig_show, note

from iknowpuck.draft import availability, predraft_plan
from iknowpuck.injuries import DEFAULT_GAMES_MISSED, save_overrides

A = app_state()
b, ctx, pool, names, my_team = A["b"], A["ctx"], A["pool"], A["names"], A["my_team"]
f = pool.frame

st.markdown("## Pre-draft plan")
st.caption("Do these three things before the draft: check injuries, simulate your plan, and see who is likely to be there at your first pick.")

# --- 1. injuries -------------------------------------------------------------------------------------
st.markdown("### 1. Check injured and suspended players")
st.markdown(
    "ESPN marks some players as out, on injured reserve, day-to-day or suspended, but does not say for how long. "
    "The app assumes a typical number of missed games and lowers each player's projection to match. **If you know better, change the number below and press Save.**"
)
with st.expander("Default games missed by status"):
    cols = st.columns(len(DEFAULT_GAMES_MISSED))
    new_defaults = {}
    for c, (status, dflt) in zip(cols, DEFAULT_GAMES_MISSED.items()):
        new_defaults[status] = c.number_input(status.replace("_", " ").title(), 0, 82, int(st.session_state.injury_defaults.get(status, dflt)), key=f"def_{status}")
    if new_defaults != st.session_state.injury_defaults:
        st.session_state.injury_defaults = new_defaults
        st.rerun()

base = b.pool.frame
flagged = base[(base.injury_status.fillna("ACTIVE") != "ACTIVE") | base.player_id.isin(st.session_state.injury_overrides.keys())]
flagged = flagged.nsmallest(60, "adp").copy()
flagged["games missed"] = flagged.player_id.map(lambda p: st.session_state.injury_overrides.get(int(p))).fillna(
    flagged.injury_status.map(lambda s: st.session_state.injury_defaults.get(str(s), 0))).astype(int)
flagged["status"] = flagged.injury_status.fillna("ACTIVE").str.replace("_", " ").str.title()
edited = st.data_editor(
    flagged[["player_id", "name", "pos", "adp", "status", "p_30", "games missed"]],
    hide_index=True, key="inj_editor", disabled=["player_id", "name", "pos", "adp", "status", "p_30"],
    column_config={"player_id": None, "adp": st.column_config.NumberColumn("ADP", format="%.0f"),
                   "p_30": st.column_config.NumberColumn("projected games", format="%.0f"),
                   "games missed": st.column_config.NumberColumn("games missed (edit)", min_value=0, max_value=82, step=1)},
)
with st.container(horizontal=True):
    if st.button("Save injury changes", type="primary", icon=":material/save:"):
        ov = dict(st.session_state.injury_overrides)
        for _, r in edited.iterrows():
            pid, games = int(r["player_id"]), int(r["games missed"])
            default = st.session_state.injury_defaults.get(str(base.loc[base.player_id == pid, "injury_status"].iloc[0]), 0)
            if games != default:
                ov[pid] = games
            else:
                ov.pop(pid, None)
        st.session_state.injury_overrides = ov
        save_overrides(ov)
        st.rerun()
    if st.button("Reset to defaults", icon=":material/restart_alt:"):
        st.session_state.injury_overrides = {}
        save_overrides({})
        st.rerun()
with st.expander("Add a player who is not on the list (for example, fresh injury news)"):
    c1, c2, c3 = st.columns([3, 1, 1])
    extra = c1.selectbox("Player", base.sort_values("adp").player_id.tolist(), format_func=lambda p: base.loc[base.player_id == p, "name"].iloc[0], key="inj_add_player")
    gm = c2.number_input("Games missed", 0, 82, 10, key="inj_add_games")
    c3.markdown("&nbsp;")
    if c3.button("Add", key="inj_add_btn"):
        st.session_state.injury_overrides = {**st.session_state.injury_overrides, int(extra): int(gm)}
        save_overrides(st.session_state.injury_overrides)
        st.rerun()
changed = f[f.injury_factor < 1].nsmallest(8, "adp") if "injury_factor" in f else f.iloc[0:0]
if len(changed):
    st.caption("Biggest effects right now: " + "; ".join(f"{r.name} {r.fpts * r.injury_factor:.0f} pts (was {r.fpts:.0f})" for r in changed.itertuples()))

# --- 2. plan -----------------------------------------------------------------------------------------
st.markdown("### 2. Simulate your draft")
st.markdown("Full drafts are played out from your slot: the other managers pick the way your league usually does, and you pick with the app's method.")
n_sims = st.slider("Simulated drafts", 20, 400, 100, key="plan_sims")
if st.button("Simulate plan", type="primary", icon=":material/play_arrow:"):
    with st.spinner("Simulating full drafts..."):
        st.session_state.plan = predraft_plan(ctx, n_sims=n_sims)
if "plan" in st.session_state:
    targets, summary = st.session_state.plan
    with st.container(horizontal=True):
        st.metric("Chance of winning a week", f"{summary.win_prob.mean():.1%}", f"±{1.96 * summary.win_prob.std() / np.sqrt(len(summary)):.1%} (95% range of the average)", delta_color="off", border=True)
        st.metric("Expected weekly points", f"{summary.weekly_points.mean():.1f}", border=True)
    c1, c2 = st.columns(2)
    with c1:
        pos_mix = targets.groupby(["round", "pos"])["share"].sum().unstack(fill_value=0)
        fig = go.Figure()
        for i, p in enumerate([c for c in ["C", "LW", "RW", "D", "G"] if c in pos_mix]):
            fig.add_bar(x=pos_mix.index, y=pos_mix[p], name=p, marker_color=PALETTE[i])
        fig.update_layout(barmode="stack", title="Which positions you usually take each round", xaxis_title="your round", yaxis_title="share of simulations")
        fig_show(fig, 340)
    with c2:
        fig = go.Figure(go.Histogram(x=summary.win_prob, nbinsx=25, marker_color=NAVY))
        fig.update_layout(title="How good your team ends up across simulations", xaxis=dict(title="chance of winning a week", tickformat=".0%"), yaxis_title="drafts")
        fig_show(fig, 340)
    st.markdown("**Who you usually get, round by round** (the share is how often that player was your pick)")
    st.dataframe(targets, hide_index=True, column_config={"adp": st.column_config.NumberColumn("ADP", format="%.0f"),
                                                         "proj_value": st.column_config.NumberColumn("projected pts", format="%.0f"),
                                                         "share": st.column_config.ProgressColumn("how often", format="percent", min_value=0, max_value=1)})

# --- 3. availability ---------------------------------------------------------------------------------
st.markdown("### 3. Who will still be there at your first pick?")
if st.button("Compute availability", icon=":material/query_stats:"):
    st.session_state.avail = availability(ctx, [], n_rollouts=80)
if "avail" in st.session_state:
    df = f.assign(p_avail=st.session_state.avail)[["name", "pos", "adp", "fpts", "p_avail"]].sort_values("adp").head(40)
    fig = go.Figure(go.Bar(x=df.name, y=df.p_avail, marker_color=NAVY, hovertemplate="%{x}<br>%{y:.0%} chance still available<extra></extra>"))
    fig.update_layout(title="Chance each player is still available at your first pick (top 40 by ADP)", yaxis=dict(range=[0, 1], tickformat=".0%"), xaxis_tickangle=-60)
    fig_show(fig, 380)

# --- 4. board ----------------------------------------------------------------------------------------
st.markdown("### Player board")
with st.container(horizontal=True):
    pos_f = st.pills("Positions", ["C", "LW", "RW", "D", "G"], selection_mode="multi", default=["C", "LW", "RW", "D", "G"], key="board_pos")
    only_healthy = st.toggle("Hide injured and suspended", value=False, key="board_healthy")
board = f[f.pos.isin(pos_f or [])].copy()
if only_healthy:
    board = board[board.injury_status.fillna("ACTIVE") == "ACTIVE"]
board["final pts"] = board["fpts"] * board.get("injury_factor", 1.0)
top = board.nsmallest(150, "adp")
fig = go.Figure()
for i, p in enumerate(["C", "LW", "RW", "D", "G"]):
    d = top[top.pos == p]
    fig.add_scatter(x=d.adp, y=d["final pts"], mode="markers", name=p, text=d.name, marker=dict(size=8, color=PALETTE[i]),
                    hovertemplate="%{text}<br>ADP %{x:.0f}<br>%{y:.0f} projected pts<extra></extra>")
fig.update_layout(title="Projected points vs ESPN ranking (top 150 by ADP; high and right = value later in the draft)", xaxis_title="ADP", yaxis_title="projected fantasy points")
fig_show(fig, 420)
cols = ["name", "pos", "adp", "final pts", "fpts_espn", "fpts_own", "p_30", "injury_status", "games_missed_assumed", "injury_risk"]
cols = [c for c in cols if c in board]
st.dataframe(board.sort_values("final pts", ascending=False)[cols], hide_index=True, height=480,
             column_config={"adp": st.column_config.NumberColumn("ADP", format="%.0f"), "final pts": st.column_config.NumberColumn(format="%.0f"),
                            "fpts_espn": st.column_config.NumberColumn("ESPN pts", format="%.0f"), "fpts_own": st.column_config.NumberColumn("our model pts", format="%.0f"),
                            "p_30": st.column_config.NumberColumn("games", format="%.0f"), "games_missed_assumed": st.column_config.NumberColumn("games missed", format="%.0f")})
note("**final pts** already includes the mix with the ESPN ranking and any expected missed games. **injury risk** comes from how many games the player missed in each of the last three seasons.")
