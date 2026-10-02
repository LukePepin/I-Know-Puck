"""The 2027 draft: how every roster projects, the draft board, steals and reaches, and notes on each roster."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import AUTHOR, DIV, FAINT, GROUP, INK, ME, MUTED, csv, fig_show, group_of, js, note, rgba

meta = js("meta.json")
teams = csv("teams_2027.csv")
picks = csv("draft_2027.csv")
rosters = csv("rosters_2027.csv")
h2h = csv("h2h_2027.csv").set_index("team")
me = teams[teams.team == AUTHOR].iloc[0]
n_start = sum(v for k, v in meta["lineup"].items() if k not in ("BN", "IR"))

st.markdown("## The 2027 draft")
st.caption(f"{meta['teams']} teams, {meta['rounds']} rounds, {len(picks)} picks, drafted on 27 September 2026. "
           f"I picked {meta['author_slot']}{'th' if meta['author_slot'] > 3 else ['st', 'nd', 'rd'][meta['author_slot'] - 1]}. Other teams are labelled by their draft slot.")

with st.container(horizontal=True):
    st.metric("My projected rank", f"#{int(me['rank'])} of {len(teams)}", border=True)
    st.metric("Expected weekly points", f"{me.weekly_pts:.0f}", border=True, help=f"The {n_start} starters count fully; bench players count a quarter (weekly lineup lock).")
    st.metric("Chance to win a typical week", f"{me.win_chance:.0%}", border=True)
    st.metric("Biggest same-team stack", me.top_stack or "-", border=True, help="Most players from one NHL team, for fun to watch.")

# --- how every roster projects -------------------------------------------------------------------------
t = teams.sort_values("weekly_pts")
avg = float(teams.weekly_pts.mean())
fig = go.Figure()
for _, r_ in t.iterrows():
    fig.add_scatter(x=[avg, r_.weekly_pts], y=[r_.team] * 2, mode="lines", line=dict(color=FAINT, width=2), hoverinfo="skip", showlegend=False)
fig.add_scatter(x=t.weekly_pts, y=t.team, mode="markers", showlegend=False,
                marker=dict(size=14, color=[ME if x == AUTHOR else MUTED for x in t.team], line=dict(color="white", width=2)),
                customdata=np.stack([t.win_chance, t.weekly_sd, t.top_stack.fillna(""), t.best_value.fillna("")], axis=1),
                hovertemplate="%{y}<br>%{x:.1f} expected points a week (swing +/- %{customdata[1]:.0f})<br>wins a typical week %{customdata[0]:.0%}"
                              "<br>stack: %{customdata[2]}<br>best value pick: %{customdata[3]}<extra></extra>")
fig.add_vline(x=avg, line_color=INK, line_width=1, annotation_text=f"league average {avg:.0f}", annotation_position="top")
fig.update_layout(title="Expected weekly points of each roster after the draft", xaxis_title="expected fantasy points per week")
fig_show(fig, 80 + 28 * len(t))
note(f"**How to read this:** each dot is a team's expected points in a typical week; lines show the distance from the league average. Only the **{n_start} players in the starting lineup** "
     "(4 centres, 4 left wings, 4 right wings, 6 defensemen, 2 goalies) score. This league locks lineups once a week, so a bench player counts only about "
     "a quarter as much, mainly when he covers an injury or has more games that week. Projections are for the full season and will change with injuries and trades.")

c1, c2 = st.columns([3, 2])
with c1:
    order = teams.sort_values("slot").team.tolist()
    m = h2h.loc[order, order].to_numpy(float)
    dev = float(np.nanmax(np.abs(m - 0.5)))
    fig = go.Figure(go.Heatmap(z=m, x=order, y=order, colorscale=DIV, zmid=0.5, zmin=0.5 - dev, zmax=0.5 + dev, xgap=1, ygap=1,
                               colorbar=dict(title="win", thickness=10, tickformat=".0%"),
                               hovertemplate="%{y} beats %{x} in %{z:.0%} of weeks<extra></extra>"))
    fig.update_layout(title="Who would win a typical week? (row team vs column team)", yaxis=dict(autorange="reversed"), xaxis=dict(tickangle=-45))
    fig_show(fig, 460)
with c2:
    st.markdown("**The win-chance formula**")
    st.markdown(
        "Each team's weekly score is treated as a bell curve with a mean (the dot above) and a spread. The chance that one team beats another is the chance "
        "that its curve lands higher, which depends on the **gap between the means** compared with **how much both scores swing**. Blue cells favour the "
        "row team; rust cells favour the column team. Even the best roster wins only about "
        f"{np.nanmax(m):.0%} of weeks against the weakest, because weekly scores swing a lot."
    )

# --- the draft board ----------------------------------------------------------------------------------
st.markdown("### The draft board")
pk = picks.copy()
pk["grp"] = pk.pos.map(group_of)
slot_names = pk.drop_duplicates("slot").sort_values("slot").set_index("slot").team
fig = go.Figure()
for g, (lab, colr) in GROUP.items():
    x = pk[pk.grp == g]
    fig.add_scatter(x=x.slot, y=x["round"], mode="markers", name=lab, marker=dict(symbol="square", size=15, color=colr, line=dict(color="white", width=1)),
                    customdata=np.stack([x.player, x.pos, x.nhl.fillna(""), x.overall, x.adp.round(0), x.team], axis=1),
                    hovertemplate="Pick %{customdata[3]}: %{customdata[0]} (%{customdata[1]}, %{customdata[2]})<br>%{customdata[5]} · ADP %{customdata[4]}<extra></extra>")
fig.add_vrect(x0=meta["author_slot"] - 0.5, x1=meta["author_slot"] + 0.5, fillcolor=rgba(ME, 0.12), line_width=0)
fig.update_layout(title="Every pick, coloured by position (hover a square; my column is shaded)",
                  xaxis=dict(title="", tickmode="array", tickvals=list(slot_names.index), ticktext=list(slot_names), tickangle=-45, side="top"),
                  yaxis=dict(title="round", range=[26.6, 0.4], tick0=1, dtick=5), legend=dict(orientation="h", y=-0.05))
fig_show(fig, 700)
share = pk[pk["round"] <= 6].grp.value_counts(normalize=True)
st.caption(f"In rounds 1-6 the league took {share.get('F', 0):.0%} forwards, {share.get('D', 0):.0%} defensemen and {share.get('G', 0):.0%} goalies.")

st.markdown("### Steals and reaches")
st.markdown("ADP is where a player usually goes in ESPN drafts. A pick **above the line** came later than ADP (a bargain); **below the line** is a reach. "
            "Only rounds 1-15 are shown: past about pick 200 ESPN's ADP stops being informative.")
c1, c2 = st.columns([3, 2])
with c1:
    q = pk.dropna(subset=["adp"])
    q = q[(q["round"] <= 15) & (q.adp < 200)]
    mine = q.team == AUTHOR
    fig = go.Figure()
    fig.add_scatter(x=q[~mine].overall, y=q[~mine].adp, mode="markers", name="other teams", marker=dict(size=7, color=rgba(MUTED, 0.55)),
                    customdata=np.stack([q[~mine].player, q[~mine].team], axis=1), hovertemplate="%{customdata[0]} (%{customdata[1]})<br>pick %{x}, ADP %{y:.0f}<extra></extra>")
    fig.add_scatter(x=q[mine].overall, y=q[mine].adp, mode="markers", name="my picks", marker=dict(size=11, color=ME, line=dict(color="white", width=1.5)),
                    customdata=q[mine].player, hovertemplate="%{customdata}<br>pick %{x}, ADP %{y:.0f}<extra></extra>")
    lim = float(q.overall.max())
    fig.add_scatter(x=[1, lim], y=[1, lim], mode="lines", line=dict(color=INK, width=1), name="picked exactly at ADP", hoverinfo="skip")
    fig.update_layout(title="Pick number vs ADP", xaxis_title="pick number", yaxis=dict(title="ADP (market rank)", range=[0, lim * 1.15]), legend=dict(orientation="h", y=-0.2))
    fig_show(fig, 420)
with c2:
    st.markdown("**Biggest bargains**")
    st.dataframe(q.nlargest(6, "value_vs_adp")[["overall", "player", "team", "value_vs_adp"]].rename(columns={"overall": "pick", "value_vs_adp": "vs ADP"}),
                 hide_index=True, column_config={"vs ADP": st.column_config.NumberColumn(format="%+.0f", help="How many picks later than his ADP he went")})
    st.markdown("**Biggest reaches**")
    st.dataframe(q.nsmallest(6, "value_vs_adp")[["overall", "player", "team", "value_vs_adp"]].rename(columns={"overall": "pick", "value_vs_adp": "vs ADP"}),
                 hide_index=True, column_config={"vs ADP": st.column_config.NumberColumn(format="%+.0f", help="How many picks later than his ADP he went")})

# --- rosters -------------------------------------------------------------------------------------------
st.markdown("### Notes on the rosters")
team_pick = st.selectbox("Team", teams.sort_values("rank").team.tolist(), index=int(np.flatnonzero(teams.sort_values("rank").team.to_numpy() == AUTHOR)[0]), key="roster_team")
r = rosters[rosters.team == team_pick].copy()
row = teams[teams.team == team_pick].iloc[0]
r["grp"] = r.pos.map(group_of)
starters = r[r.lineup == "starter"]
by_grp = starters.groupby("grp").weekly_pts.sum()
league_grp = rosters[rosters.lineup == "starter"].assign(grp=lambda x: x.pos.map(group_of)).groupby(["team", "grp"]).weekly_pts.sum().unstack().rank(ascending=False)
bullets = [f"Projected **#{int(row['rank'])}** of {len(teams)}: {row.weekly_pts:.0f} points in a typical week, winning about {row.win_chance:.0%} of weeks."]
strong = league_grp.loc[team_pick].idxmin() if team_pick in league_grp.index else None
weak = league_grp.loc[team_pick].idxmax() if team_pick in league_grp.index else None
if strong:
    bullets.append(f"Strongest group: **{GROUP[strong][0].lower()}** (#{int(league_grp.loc[team_pick, strong])} in the league). "
                   f"Weakest: **{GROUP[weak][0].lower()}** (#{int(league_grp.loc[team_pick, weak])}).")
if isinstance(row.top_stack, str) and row.top_stack:
    bullets.append(f"Biggest same-team stack: **{row.top_stack}**, so the weekly score will swing with that NHL team.")
inj = r[r.status.fillna("ACTIVE").ne("ACTIVE")]
if len(inj):
    bullets.append("Starting the season hurt or suspended: " + ", ".join(f"{p} ({s.replace('_', ' ').lower()})" for p, s in zip(inj.player, inj.status)) + ".")
if isinstance(row.best_value, str) and row.best_value:
    bullets.append(f"Best value pick: {row.best_value} picks after his ADP. Biggest reach: {row.biggest_reach}.")
note("\n".join(f"- {x}" for x in bullets))
c1, c2 = st.columns([3, 2])
with c1:
    show = r.sort_values(["lineup", "weekly_pts"], ascending=[False, False])
    st.dataframe(show[["slot", "player", "pos", "nhl", "proj_pts", "weekly_pts", "status"]], hide_index=True, height=420,
                 column_config={"slot": "spot", "proj_pts": st.column_config.NumberColumn("season pts", format="%.0f", help="Projected fantasy points for the season"),
                                "weekly_pts": st.column_config.NumberColumn("per week", format="%.1f", help="Points this player adds to a typical week (bench counts a quarter)"),
                                "status": "status"})
with c2:
    s_ = starters.groupby("grp").weekly_pts.sum().reindex(["F", "D", "G"]).fillna(0)
    b_ = r[r.lineup == "bench"].weekly_pts.sum()
    fig = go.Figure(go.Bar(x=[GROUP[g][0] for g in s_.index] + ["Bench"], y=list(s_.values) + [b_],
                           marker_color=[GROUP[g][1] for g in s_.index] + [FAINT], hovertemplate="%{x}: %{y:.1f} points a week<extra></extra>"))
    fig.update_layout(title="Where the weekly points come from", yaxis_title="expected points per week")
    fig_show(fig, 360)
