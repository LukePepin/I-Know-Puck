"""League history 2024-26: what separates winners, drafting styles, and every past pick."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import AUTHOR, CAT, FAINT, GROUP, INK, ME, MUTED, csv, fig_show, group_of, js, note, rgba

meta = js("meta.json")
habits = csv("habits.csv")
styles = csv("styles.csv")
hist = csv("history_picks.csv")
mgr = csv("history_managers.csv")
seasons = sorted(hist.season.unique())

st.markdown("## League history")
st.caption(f"The league's {len(seasons)} past drafts ({seasons[0]}-{seasons[-1]}), {len(hist):,} picks. Other managers appear as Manager A, B, C...")

# --- what wins ---------------------------------------------------------------------------------------
st.markdown("### What separates the winners?")
h = habits.sort_values("rho")
colr = [CAT[0] if lo > 0 else (CAT[1] if hi < 0 else FAINT) for lo, hi in zip(h.ci_low, h.ci_high)]
fig = go.Figure()
for (_, r), c in zip(h.iterrows(), colr):
    fig.add_scatter(x=[r.ci_low, r.ci_high], y=[r.strategy] * 2, mode="lines", line=dict(color=c if c != FAINT else MUTED, width=3), hoverinfo="skip", showlegend=False)
fig.add_scatter(x=h.rho, y=h.strategy, mode="markers", marker=dict(size=11, color=[c if c != FAINT else MUTED for c in colr], line=dict(color="white", width=2)),
                showlegend=False, customdata=np.stack([h.ci_low, h.ci_high], axis=1),
                hovertemplate="%{y}<br>correlation with win rate %{x:+.2f} (95%% range %{customdata[0]:+.2f} to %{customdata[1]:+.2f})<extra></extra>")
fig.add_vline(x=0, line_color=INK, line_width=1)
fig.update_layout(title="How each habit relates to a manager's weekly win rate (blue = clearly helps, rust = clearly hurts, grey = unclear)",
                  xaxis=dict(title="rank correlation with win rate (-1 to +1)", range=[-1, 1]))
fig_show(fig, 80 + 30 * len(h))
note("**What stands out:** drafting well matters most. The managers whose drafted players scored the most for their draft slots won the most weeks. "
     "Taking lots of defensemen early went with losing. Most other habits, including stacking one NHL team and trading, showed no clear link. "
     f"These come from {len(mgr)} manager-seasons, so they are clues rather than proof.")

# --- styles -----------------------------------------------------------------------------------------
st.markdown("### Drafting styles")
c1, c2 = st.columns([3, 2])
groups = sorted(styles.group.unique())
gcol = {g: CAT[i % len(CAT)] for i, g in enumerate(groups)}
with c1:
    fig = go.Figure()
    for g in groups:
        x = styles[styles.group == g]
        fig.add_scatter(x=x.x, y=x.y, mode="markers", name=g, marker=dict(size=16, color=gcol[g], line=dict(color="white", width=2)),
                        customdata=x.manager, hovertemplate="%{customdata}<extra>" + g + "</extra>")
    a = styles[styles.manager == AUTHOR]
    if len(a):
        fig.add_scatter(x=a.x, y=a.y, mode="markers+text", text=["me"], textposition="top center", marker=dict(size=24, color="rgba(0,0,0,0)", line=dict(color=INK, width=2)),
                        showlegend=False, hoverinfo="skip")
    fig.update_layout(title="Managers placed so that similar drafters sit close together (hover for names)", xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=-0.05))
    fig_show(fig, 420)
with c2:
    st.markdown("Each manager is described by ten habits: how far they reach ahead of the rankings, when they take a goalie, how many defensemen they take early, "
                "how loyal they are to one NHL team, and so on. A **similarity graph** links managers with similar habits, and the eigenvectors of its "
                "Laplacian matrix give the map on the left (graph spectral clustering).")
    for g in meta.get("groups", []):
        st.markdown(f"- **{g['name']}**: {', '.join(g['traits'])}")
    st.caption("Honest caveat: a permutation test could not rule out that these groups are chance (p about 0.1), so treat them as tendencies, not types.")

# --- past drafts -------------------------------------------------------------------------------------
st.markdown("### Every past pick")
season = st.segmented_control("Season", seasons, default=seasons[-1], key="h_season") or seasons[-1]
d = hist[hist.season == season].copy()
d["grp"] = d.pos.fillna("C").map(group_of)
d["kind"] = np.where(d.grp == "G", "G", "S")  # goalies score fewer points in this league, so compare like with like
fits = {k: np.polyfit(np.log(x.overall), x.actual_pts, 1) for k, x in d.groupby("kind") if len(x) >= 10}
d["expected"] = [np.polyval(fits.get(k, fits["S"]), np.log(o)) for k, o in zip(d.kind, d.overall)]
d["vs_slot"] = d.actual_pts - d.expected
coef = fits["S"]
c1, c2 = st.columns([3, 2])
with c1:
    fig = go.Figure()
    for g, (lab, colr_) in GROUP.items():
        x = d[d.grp == g]
        fig.add_scatter(x=x.overall, y=x.actual_pts, mode="markers", name=lab, marker=dict(size=8, color=rgba(colr_, 0.7),
                        line=dict(color=[INK if m_ == AUTHOR else "white" for m_ in x.manager], width=[2 if m_ == AUTHOR else 0.5 for m_ in x.manager])),
                        customdata=np.stack([x.player, x.manager, x["round"], x.games], axis=1),
                        hovertemplate="%{customdata[0]} (round %{customdata[2]}, %{customdata[1]})<br>%{y:.0f} points in %{customdata[3]} games<extra></extra>")
    xs = np.arange(1, d.overall.max() + 1)
    fig.add_scatter(x=xs, y=np.polyval(coef, np.log(xs)), mode="lines", line=dict(color=INK, width=2), name="typical skater for the slot", hoverinfo="skip")
    fig.update_layout(title=f"What each {season} pick actually scored (black ring = my picks)", xaxis_title="pick number", yaxis_title="fantasy points that season",
                      legend=dict(orientation="h", y=-0.2))
    fig_show(fig, 440)
with c2:
    st.markdown("**Best picks** (points above what that slot usually gives)")
    st.dataframe(d.nlargest(6, "vs_slot")[["overall", "player", "manager", "vs_slot"]].rename(columns={"overall": "pick", "vs_slot": "vs slot"}),
                 hide_index=True, column_config={"vs slot": st.column_config.NumberColumn(format="%+.0f", help="Points above what that slot usually gives a skater (or a goalie)")})
    st.markdown("**Biggest misses**")
    st.dataframe(d.nsmallest(6, "vs_slot")[["overall", "player", "manager", "vs_slot"]].rename(columns={"overall": "pick", "vs_slot": "vs slot"}),
                 hide_index=True, column_config={"vs slot": st.column_config.NumberColumn(format="%+.0f", help="Points above what that slot usually gives a skater (or a goalie)")})
    st.caption("Big misses are usually injuries: points only count for games played.")

mine = mgr[mgr.manager == AUTHOR].sort_values("season")
if len(mine):
    st.markdown("### My seasons")
    st.dataframe(pd.DataFrame({"season": mine.season.astype(str), "finish": mine.final_rank.map(lambda x: f"{int(x)}" if pd.notna(x) else ""),
                               "weekly win rate": mine.win_pct.map(lambda x: f"{x:.0%}"), "draft skill": mine.draft_skill.map(lambda x: f"{x:+.0f}"),
                               "injury luck": mine.injury_luck.map(lambda x: f"{x:+.0f}")}), hide_index=True)
    st.caption("Draft skill: points my drafted players scored above what those slots usually give, if injuries had been average. Injury luck: points not lost to injuries vs an average team.")
