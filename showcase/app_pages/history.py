"""History: the league's 2024-26 seasons. Champions, what separates winners, drafting styles, every past pick,
and one theory tested."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import CAT, FAINT, GROUP, INK, MUTED, SEQ, csv, definitions, fig_show, group_of, header, how_built, js, label_positions, math, note, rgba

meta = js("meta.json")
stand = csv("history_standings.csv")
habits = csv("habits.csv")
styles = csv("styles.csv")
hist = csv("history_picks.csv")
theory = js("theory.json")
seasons = sorted(hist.season.unique())

header("History", f"The league's {len(seasons)} past seasons ({seasons[0]}-{seasons[-1]}): {meta['teams']} teams, {len(hist):,} draft picks")

# --- champions and standings -------------------------------------------------------------------------
st.markdown("### Champions and final standings")
cols = st.columns(len(seasons))
for c, s in zip(cols, seasons):
    row = stand[(stand.season == s) & (stand.final_rank == 1)]
    c.metric(f"{s} champion", row.manager.iloc[0] if len(row) else "-", border=True,
             help=f"Regular-season weekly win rate {row.win_pct.iloc[0]:.0%}" if len(row) else None)
piv = stand.pivot_table(index="manager", columns="season", values="final_rank")
piv = piv.loc[piv.mean(axis=1).sort_values().index]
fig = go.Figure(go.Heatmap(z=-piv.to_numpy(float), x=[str(s) for s in piv.columns], y=piv.index, colorscale=SEQ, showscale=False, xgap=3, ygap=3,
                           text=piv.map(lambda v: "" if pd.isna(v) else f"{int(v)}").to_numpy(), texttemplate="%{text}", textfont=dict(size=13),
                           customdata=piv.to_numpy(float), hovertemplate="%{y}, %{x}: finished %{customdata:.0f}<extra></extra>"))
fig.update_layout(title="Final finish each season (1 = champion; darker = better; blank = not in the league)", yaxis=dict(autorange="reversed"), xaxis=dict(side="top", type="category"))
fig_show(fig, 60 + 26 * len(piv))

# --- what wins ---------------------------------------------------------------------------------------
st.markdown("### What separates the winners?")
h = habits.sort_values("rho")
colr = [CAT[0] if lo > 0 else (CAT[1] if hi < 0 else MUTED) for lo, hi in zip(h.ci_low, h.ci_high)]
fig = go.Figure()
for (_, r), c in zip(h.iterrows(), colr):
    fig.add_scatter(x=[r.ci_low, r.ci_high], y=[r.strategy] * 2, mode="lines", line=dict(color=c, width=3), hoverinfo="skip", showlegend=False)
fig.add_scatter(x=h.rho, y=h.strategy, mode="markers", marker=dict(size=11, color=colr, line=dict(color="white", width=2)), showlegend=False,
                customdata=np.stack([h.ci_low, h.ci_high], axis=1),
                hovertemplate="%{y}<br>correlation with win rate %{x:+.2f} (95%% range %{customdata[0]:+.2f} to %{customdata[1]:+.2f})<extra></extra>")
fig.add_vline(x=0, line_color=INK, line_width=1)
fig.update_layout(title="How each habit relates to a manager's weekly win rate (blue = clearly helps, rust = clearly hurts, grey = unclear)",
                  xaxis=dict(title="rank correlation with weekly win rate (-1 to +1)", range=[-1, 1]))
fig_show(fig, 80 + 30 * len(h))
note("**What stands out:** drafting well matters most. Managers whose drafted players scored the most for their draft slots won the most weeks. "
     "Taking many defensemen in the first six rounds went with losing. Most other habits, including stacking one NHL team and trading, show no clear link. "
     f"These come from {len(stand)} manager-seasons, so they are clues rather than proof.")

# --- styles ------------------------------------------------------------------------------------------
st.markdown("### Drafting styles")
c1, c2 = st.columns([3, 2])
groups = sorted(styles.group.unique())
gcol = {g: CAT[i % len(CAT)] for i, g in enumerate(groups)}
with c1:
    tpos = label_positions(styles.x, styles.y, styles.manager.tolist())
    fig = go.Figure()
    for g in groups:
        idx = np.flatnonzero(styles.group.to_numpy() == g)
        x = styles.iloc[idx]
        fig.add_scatter(x=x.x, y=x.y, mode="markers+text", name=g, text=x.manager, textposition=[tpos[i] for i in idx], textfont=dict(color=INK, size=12),
                        marker=dict(size=16, color=gcol[g], line=dict(color="white", width=2)), hovertemplate="%{text}<extra>" + g + "</extra>")
    pad_x, pad_y = 0.15 * np.ptp(styles.x), 0.12 * np.ptp(styles.y)
    fig.update_layout(title="Managers placed so that similar drafters sit close together", legend=dict(orientation="h", y=-0.05),
                      xaxis=dict(visible=False, range=[styles.x.min() - pad_x, styles.x.max() + pad_x]),
                      yaxis=dict(visible=False, range=[styles.y.min() - pad_y, styles.y.max() + pad_y]))
    fig_show(fig, 440)
with c2:
    st.markdown("Each manager is described by ten draft habits, such as how far they reach ahead of the rankings, when they take a goalie, and how "
                "loyal they are to one NHL team. A **similarity graph** links managers with similar habits; the graph's eigenvectors place them on the map, "
                "and clustering the map finds the groups (**graph spectral clustering**).")
    for g in meta.get("groups", []):
        st.markdown(f"- **{g['name']}**: {', '.join(g['traits'])}")
    st.caption("Honest caveat: a permutation test could not rule out that these groups are chance (p about 0.1). Treat them as tendencies, not types.")

# --- past picks --------------------------------------------------------------------------------------
st.markdown("### Every past pick")
season = st.segmented_control("Season", seasons, default=seasons[-1], key="h_season") or seasons[-1]
d = hist[hist.season == season].copy()
d["grp"] = d.pos.fillna("C").map(group_of)
d["kind"] = np.where(d.grp == "G", "G", "S")  # goalies score fewer points in this league, so compare like with like
fits = {k: np.polyfit(np.log(x.overall), x.actual_pts, 1) for k, x in d.groupby("kind") if len(x) >= 10}
d["vs_slot"] = d.actual_pts - [np.polyval(fits.get(k, fits["S"]), np.log(o)) for k, o in zip(d.kind, d.overall)]
c1, c2 = st.columns([3, 2])
with c1:
    fig = go.Figure()
    for g, (lab, colr_) in GROUP.items():
        x = d[d.grp == g]
        fig.add_scatter(x=x.overall, y=x.actual_pts, mode="markers", name=lab, marker=dict(size=8, color=rgba(colr_, 0.7), line=dict(color="white", width=0.5)),
                        customdata=np.stack([x.player, x.manager, x["round"], x.games], axis=1),
                        hovertemplate="%{customdata[0]} (round %{customdata[2]}, %{customdata[1]})<br>%{y:.0f} points in %{customdata[3]} games<extra></extra>")
    xs = np.arange(1, d.overall.max() + 1)
    fig.add_scatter(x=xs, y=np.polyval(fits["S"], np.log(xs)), mode="lines", line=dict(color=INK, width=2), name="typical skater for the slot", hoverinfo="skip")
    fig.update_layout(title=f"What each {season} pick actually scored", xaxis_title="pick number", yaxis_title="fantasy points that season",
                      legend=dict(orientation="h", y=-0.2))
    fig_show(fig, 440)
with c2:
    for title, fn in (("Best picks", "nlargest"), ("Biggest misses", "nsmallest")):
        st.markdown(f"**{title}** (points vs what the slot usually gives)")
        st.dataframe(getattr(d, fn)(6, "vs_slot")[["overall", "player", "manager", "vs_slot"]].rename(columns={"overall": "pick", "vs_slot": "vs slot"}),
                     hide_index=True, column_config={"vs slot": st.column_config.NumberColumn(format="%+.0f")})
    st.caption("Big misses are usually injuries: points only count for games played.")

# --- a theory -----------------------------------------------------------------------------------------
st.markdown("### Myth check: does a hot NHL team lift all its players?")
q = theory["quintiles"]
c1, c2 = st.columns([3, 2])
with c1:
    labels = ["coldest fifth", "2nd", "middle", "4th", "hottest fifth"][: len(q)]
    beat = [x["beat"] for x in q]
    fig = go.Figure(go.Bar(x=labels, y=beat, marker_color=[CAT[0] if b_ >= np.mean(beat) else FAINT for b_ in beat],
                           error_y=dict(type="data", symmetric=False, array=[x["hi"] - x["beat"] for x in q], arrayminus=[x["beat"] - x["lo"] for x in q], color=MUTED),
                           hovertemplate="%{x} of teams: %{y:.0%} of players beat their projection<extra></extra>"))
    fig.update_layout(title="Share of players who beat their projection, by how their NHL teammates did", yaxis=dict(title="beat projection", tickformat=".0%", range=[0, 1]),
                      xaxis_title="NHL teams grouped by how the player's teammates did vs projection")
    fig_show(fig, 340)
with c2:
    note(f"**Partly true.** Each +10% for a player's teammates lifted him about **{10 * theory['slope']:+.1f}%** (95% range {10 * theory['lo']:+.1f}% to "
         f"{10 * theory['hi']:+.1f}%, p = {theory['p']:.3f}). But the team explains only about {max(theory['icc'], 0):.0%} of why a player beats his projection, "
         "half the players on the hottest teams still fell short, and a hot first half said **nothing** about a player's second half once his own first "
         f"half was known ({10 * theory['carry_team']:+.1f}% per +10%, p = {theory['carry_p']:.2f}).")

# --- systems overview ----------------------------------------------------------------------------------
how_built(
    [("ESPN league data", "drafts, weekly results"), ("ESPN game logs", "every player, every game"), ("Match & clean", "managers by ESPN account"),
     ("Measure", "draft value, habits"), ("Test", "correlations, permutations"), ("Charts", "this page")],
    ["**Collect.** The league's drafts, weekly matchup results and transactions for 2024-26 come from ESPN's fantasy API; each player's game-by-game stats "
     "come from ESPN game logs. Everything is cached so the analysis is repeatable.",
     "**Clean.** Managers are matched by ESPN account, not by team slot, because slots changed hands. Real names are reduced to first names before publishing.",
     "**Measure.** For every pick: points scored that season against what that pick slot usually produces. For every manager-season: ten draft habits.",
     "**Test.** Habits are compared with win rate by rank correlation, with bootstrap 95% ranges. Drafting styles use graph spectral clustering, checked by "
     "a permutation test. The NHL-team theory compares each player with his teammates only and shuffles whole teams to see what luck looks like."],
)
math((r"\text{draft value} = \text{points scored} - (a + b \ln(\text{pick number}))", "A pick's value is how far it beat the typical points for its slot; a and b are fit on every pick that season."),
     (r"\rho = \text{correlation of the ranks of two measures, from } -1 \text{ to } +1", "Spearman rank correlation: +1 means the habit and win rate always rise together."))
definitions([
    ("ADP", "Average draft position: where a player usually goes in ESPN drafts. It is the crowd's ranking."),
    ("Draft value", "Fantasy points a pick scored above (or below) what that pick slot usually gives."),
    ("Win rate", "Share of weekly head-to-head matchups a team won in the regular season."),
    ("Correlation", "How strongly two measures move together, from -1 to +1. Zero means no link."),
    ("95% range", "The range the true value most likely falls in. If it crosses zero, the link could be luck."),
    ("Spectral clustering", "Grouping by the eigenvectors of a similarity graph; similar items end up close together."),
    ("Permutation test", "Shuffle the data many times to see how often luck alone gives a result this strong."),
    ("Projection", "A preseason estimate of how many fantasy points a player will score."),
])
