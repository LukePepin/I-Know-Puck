"""History: the league's 2024-26 seasons. Champions, what separates winners, drafting styles, every past pick,
and one theory tested."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import (BG, CAT, FAINT, GROUP, INK, MUTED, csv, definitions, esc, fig_show, group_of, header, how_built, js, label_positions, math, note,
                rgba, table)

meta = js("meta.json")
stand = csv("history_standings.csv")
habits = csv("habits.csv")
styles = csv("styles.csv")
links = csv("style_links.csv")
hist = csv("history_picks.csv")
theory = js("theory.json")
seasons = sorted(hist.season.unique())
n_teams = int(meta["teams"])

header("History", f"Three seasons of the league, {seasons[0]} to {seasons[-1]}: who won, what the winning managers did differently, how each manager "
       f"drafts, and all {len(hist):,} picks.")

# --- champions and standings -------------------------------------------------------------------------
st.markdown("### Champions and final standings")
with st.container(horizontal=True):
    for s in seasons:
        row = stand[(stand.season == s) & (stand.final_rank == 1)]
        st.metric(f"{s} champion", row.manager.iloc[0] if len(row) else "-", border=True,
                  help=f"Regular-season weekly win rate {row.win_pct.iloc[0]:.0%}" if len(row) else None)
piv = stand.pivot_table(index="manager", columns="season", values="final_rank")
piv = piv.loc[piv.mean(axis=1).sort_values().index]
head = "<th>Manager</th>" + "".join(f'<th class="n">{s}</th>' for s in piv.columns) + '<th class="n">Avg</th>'
body = []
for m, r in piv.iterrows():
    cells = []
    for s in piv.columns:
        v = r[s]
        if pd.isna(v):
            cells.append(f'<td class="cell" style="color:{FAINT}">-</td>')
        elif v == 1:
            cells.append(f'<td class="cell" style="background:{CAT[2]};color:{BG}">1</td>')
        else:
            a = 0.08 + 0.5 * (n_teams - v) / (n_teams - 1)
            cells.append(f'<td class="cell" style="background:{rgba(CAT[0], a)}">{int(v)}</td>')
    body.append(f'<tr><td>{esc(m)}</td>{"".join(cells)}<td class="n">{r.mean():.1f}</td></tr>')
st.markdown("**Final finish each season**")
st.html(f'<div class="kp-wrap"><table class="kp"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')
st.caption("1 is the champion (gold). Brighter blue is a better finish. A dash means the manager wasn't in the league that season.")

# --- what wins ---------------------------------------------------------------------------------------
st.markdown("### What separates the winners?")
SHORT = {"Defense share of rounds 1-6": "Defensemen in rounds 1-6", "Follows market order (Spearman)": "Drafts in ADP order",
         "First goalie round (higher = waited)": "Waits on goalies", "Reach vs market, rounds 1-6 (>0 = early)": "Reaches early, rounds 1-6",
         "Most players from one NHL team (stacking)": "Stacks one NHL team", "Injury luck (points not lost to injuries vs league)": "Injury luck",
         "Draft value added, rounds 1-6 (actual pts vs slot)": "Draft value, rounds 1-6", "Lineup changes (rank within season)": "Lineup changes",
         "Share of points from pickups": "Points from pickups", "Waiver activity (rank within season)": "Waiver activity",
         "Draft skill (value added, injuries evened out)": "Draft skill, injuries evened out", "Draft value added, all rounds": "Draft value, all rounds",
         "Actual pts of best 16 drafted players": "Points from the best 16 picks"}
h = habits.sort_values("rho").reset_index(drop=True)
colr = [CAT[0] if lo > 0 else (CAT[1] if hi < 0 else MUTED) for lo, hi in zip(h.ci_low, h.ci_high)]
fig = go.Figure()
for i, (r, c) in enumerate(zip(h.itertuples(), colr)):
    fig.add_scatter(x=[r.ci_low, r.ci_high], y=[i, i], mode="lines", line=dict(color=c, width=3), hoverinfo="skip", showlegend=False)
    fig.add_annotation(x=-1, y=i, text=esc(SHORT.get(r.strategy, r.strategy)), xanchor="left", yanchor="bottom", yshift=5, showarrow=False,
                       font=dict(size=12, color=INK))
fig.add_scatter(x=h.rho, y=list(range(len(h))), mode="markers", marker=dict(size=11, color=colr, line=dict(color=BG, width=2)), showlegend=False,
                customdata=np.stack([h.strategy, h.ci_low, h.ci_high], axis=1),
                hovertemplate="%{customdata[0]}<br>correlation %{x:+.2f} (95%% range %{customdata[1]:+.2f} to %{customdata[2]:+.2f})<extra></extra>")
fig.add_vline(x=0, line_color=MUTED, line_width=1)
fig.update_layout(title="How each habit relates to a manager's weekly win rate", yaxis=dict(visible=False, range=[-0.6, len(h) - 0.1]),
                  xaxis=dict(title="rank correlation with win rate", range=[-1, 1], dtick=0.5))
fig_show(fig, 60 + 40 * len(h))
st.caption("Each line is a 95% range. Blue: the habit goes with winning. Rust: it goes with losing. Grey: the range crosses zero.")
note("**What stands out:** drafting well matters most. Managers whose picks scored more than their draft slot usually gives won the most weeks. "
     "Taking a lot of defensemen in the first six rounds went with losing. Stacking one NHL team and trading show no link either way. "
     f"This is {len(stand)} manager-seasons, so only strong links clear zero.")

# --- styles: graph spectral analysis ---------------------------------------------------------------------
st.markdown("### Drafting styles: the managers' similarity graph")
sp = meta.get("spectral") or {}
groups = sorted(styles.group.unique())
gcol = {g: CAT[i % len(CAT)] for i, g in enumerate(groups)}
c1, c2 = st.columns([3, 2], gap="large")
with c2:
    st.markdown(f"I used **graph spectral analysis** in three steps:\n"
                f"1. Describe each manager by {sp.get('n_features', 10)} draft habits, such as how far they reach ahead of the rankings, "
                "when they take a goalie, and how loyal they are to one NHL team.\n"
                "2. Link every pair of managers. The link is strong (near 1) when their habits match and weak (near 0) when they differ.\n"
                "3. The graph's **eigenvectors** place the managers on the map, and the **eigengap** sets the number of groups "
                f"(here {sp.get('k', len(groups))}).")
    thr = st.slider("Show links stronger than", 0.3, 0.9, 0.7, 0.05, key="h_thr",
                    help="Raise it and only the most alike pairs stay linked; the graph breaks apart where the links are weakest.")
    for g in meta.get("groups", []):
        st.markdown(f"<span style='color:{gcol.get(g['name'], INK)}'>●</span> **{g['name']}**: {', '.join(g['traits'])}", unsafe_allow_html=True)
with c1:
    pos = styles.set_index("manager")[["x", "y"]]
    near = {m: ", ".join(f"{r.other} ({r.w:.2f})" for r in grp.nlargest(3, "w").itertuples())
            for m, grp in pd.concat([links.rename(columns={"a": "m", "b": "other"}), links.rename(columns={"b": "m", "a": "other"})]).groupby("m")}
    shown = links[links.w >= thr]
    fig = go.Figure()
    for r in shown.itertuples():
        s = (r.w - thr) / max(1 - thr, 1e-9)
        fig.add_scatter(x=[pos.x[r.a], pos.x[r.b]], y=[pos.y[r.a], pos.y[r.b]], mode="lines", hoverinfo="skip", showlegend=False,
                        line=dict(width=0.8 + 5 * s, color=rgba(MUTED, 0.2 + 0.5 * s)))
    tpos = label_positions(styles.x, styles.y, styles.manager.tolist(), width_px=400, height_px=380)
    for g in groups:
        idx = np.flatnonzero(styles.group.to_numpy() == g)
        x = styles.iloc[idx]
        fig.add_scatter(x=x.x, y=x.y, mode="markers+text", name=g, text=x.manager, textposition=[tpos[i] for i in idx], textfont=dict(color=INK, size=12),
                        marker=dict(size=18, color=gcol[g], line=dict(color=BG, width=2)), customdata=[near.get(m, "") for m in x.manager],
                        hovertemplate="<b>%{text}</b> (" + g + ")<br>drafts most like: %{customdata}<extra></extra>")
    pad_x, pad_y = 0.15 * np.ptp(styles.x), 0.12 * np.ptp(styles.y)
    fig.update_layout(title=f"Managers who draft alike, linked ({len(shown)} of {len(links)} links shown)",
                      xaxis=dict(visible=False, range=[styles.x.min() - pad_x, styles.x.max() + pad_x]),
                      yaxis=dict(visible=False, range=[styles.y.min() - pad_y, styles.y.max() + pad_y]))
    fig_show(fig, 460)
    st.caption("Thicker line = more alike. Tap a manager to see who drafts most like them.")
p_sp = sp.get("p", 0.1)
note(f"**How solid are the groups?** I shuffled the habits 1,000 times and checked how often random managers split this cleanly: "
     f"{p_sp:.0%} of the time (p = {p_sp:.2f}). That isn't statistically significant. The links themselves are direct measurements.")

# --- past picks --------------------------------------------------------------------------------------
st.markdown("### Every past pick")
season = st.segmented_control("Season", seasons, default=seasons[-1], key="h_season") or seasons[-1]
d = hist[hist.season == season].copy()
d["grp"] = d.pos.fillna("C").map(group_of)
d["kind"] = np.where(d.grp == "G", "G", "S")  # goalies score fewer points in this league, so compare like with like
fits = {k: np.polyfit(np.log(x.overall), x.actual_pts, 1) for k, x in d.groupby("kind") if len(x) >= 10}
d["vs_slot"] = d.actual_pts - [np.polyval(fits.get(k, fits["S"]), np.log(o)) for k, o in zip(d.kind, d.overall)]
fig = go.Figure()
for g, (lab, colr_) in GROUP.items():
    x = d[d.grp == g]
    fig.add_scatter(x=x.overall, y=x.actual_pts, mode="markers", name=lab, marker=dict(size=8, color=rgba(colr_, 0.75), line=dict(color=BG, width=0.5)),
                    customdata=np.stack([x.player, x.manager, x["round"], x.games], axis=1),
                    hovertemplate="%{customdata[0]} (round %{customdata[2]}, %{customdata[1]})<br>%{y:.0f} points in %{customdata[3]} games<extra></extra>")
xs = np.arange(1, d.overall.max() + 1)
fig.add_scatter(x=xs, y=np.polyval(fits["S"], np.log(xs)), mode="lines", line=dict(color=INK, width=2), name="typical skater", hoverinfo="skip")
fig.update_layout(title=f"What each {season} pick scored", xaxis_title="pick number", yaxis_title="fantasy points that season")
fig_show(fig, 420)
c1, c2 = st.columns(2, gap="large")
for col, title, fn in ((c1, "Best picks", "nlargest"), (c2, "Biggest misses", "nsmallest")):
    with col:
        st.markdown(f"**{title}** (points vs what the slot usually gives)")
        table(getattr(d, fn)(6, "vs_slot"), [("overall", "Pick", "{:.0f}"), ("player", "Player", None, "manager"), ("vs_slot", "vs slot", "{:+.0f}")])
st.caption("Most of the big misses are injuries: points only count for games played.")

# --- a theory -----------------------------------------------------------------------------------------
st.markdown("### Does a hot NHL team lift all its players?")
q = theory["quintiles"]
c1, c2 = st.columns([3, 2], gap="large")
with c1:
    labels = ["coldest", "2nd", "middle", "4th", "hottest"][: len(q)]
    beat = [x["beat"] for x in q]
    fig = go.Figure(go.Bar(x=labels, y=beat, marker_color=[CAT[0] if b_ >= np.mean(beat) else FAINT for b_ in beat],
                           text=[f"{b_:.0%}" for b_ in beat], textposition="outside", cliponaxis=False,
                           error_y=dict(type="data", symmetric=False, array=[x["hi"] - x["beat"] for x in q], arrayminus=[x["beat"] - x["lo"] for x in q],
                                        color=MUTED, thickness=1),
                           hovertemplate="%{x} fifth of teams: %{y:.0%} of players beat their projection<extra></extra>"))
    fig.update_layout(title="Players who beat their projection, by how their teammates did", yaxis=dict(title="beat projection", tickformat=".0%", range=[0, 1]),
                      xaxis_title="teams, by how teammates did")
    fig_show(fig, 340)
with c2:
    note(f"**Partly true.** Each +10% for a player's teammates lifted him about **{10 * theory['slope']:+.1f}%** (95% range {10 * theory['lo']:+.1f}% to "
         f"{10 * theory['hi']:+.1f}%, p = {theory['p']:.3f}). But the team explains only about {max(theory['icc'], 0):.0%} of why a player beats his projection, "
         "half the players on the hottest teams still fell short, and a hot first half told me **nothing** about a player's second half once I knew his own "
         f"first half ({10 * theory['carry_team']:+.1f}% per +10%, p = {theory['carry_p']:.2f}).")

# --- systems overview ----------------------------------------------------------------------------------
how_built(
    [("ESPN league data", "drafts, weekly results"), ("ESPN game logs", "every player, every game"), ("Match and clean", "managers by ESPN account"),
     ("Measure", "draft value, habits"), ("Similarity graph", "links, eigenvectors"), ("Test", "correlations, permutations")],
    ["**Collect.** The league's drafts, weekly results and transactions for 2024-26 come from ESPN's fantasy API, and each player's game-by-game stats "
     "from ESPN game logs. I cache everything so the analysis is repeatable.",
     "**Clean.** I match managers by ESPN account, not by team slot, because slots changed hands. Names are cut to first names before anything is published.",
     "**Measure.** For every pick: points scored that season against what that pick slot usually produces. For every manager-season: ten draft habits.",
     "**Graph.** I standardise each manager's habits and give every pair a link strength from a Gaussian kernel. The eigenvectors of the normalised graph "
     "Laplacian give the map, its eigengap the number of groups, and k-means on the eigenvectors the groups.",
     "**Test.** Habits are compared with win rate by rank correlation, with bootstrap 95% ranges. The drafting styles get a permutation test. The NHL-team "
     "theory compares each player with his own teammates and shuffles whole teams to see what luck looks like."],
)
math((r"\text{draft value} = \text{points scored} - (a + b \ln(\text{pick number}))", "A pick's value is how far it beat the typical points for its slot; a and b are fit on every pick that season."),
     (r"w_{ij} = \exp\!\left(-\frac{\lVert z_i - z_j \rVert^2}{2\sigma^2}\right), \qquad L = I - D^{-1/2} W D^{-1/2}",
      "Link strength between managers i and j (z = their standardised habits, sigma = the typical distance). L is the normalised graph Laplacian; "
      "D holds each manager's total link strength. Its smallest eigenvectors place the managers on the map."),
     (r"\rho = \text{correlation of the ranks of two measures, from } -1 \text{ to } +1", "Spearman rank correlation: +1 means the habit and win rate always rise together."))
definitions([
    ("ADP", "Average draft position: where a player usually goes in ESPN drafts. It's the crowd's ranking."),
    ("Draft value", "Fantasy points a pick scored above (or below) what that pick slot usually gives."),
    ("Win rate", "Share of weekly head-to-head matchups a team won in the regular season."),
    ("Correlation", "How strongly two measures move together, from -1 to +1. Zero means no link."),
    ("95% range", "The range the true value most likely falls in. If it crosses zero, the link could be luck."),
    ("Similarity graph", "Dots (managers) joined by lines whose strength says how alike two managers draft."),
    ("Spectral clustering", "Grouping by the eigenvectors of a similarity graph; similar items end up close together."),
    ("Eigengap", "A jump in the graph's eigenvalues. A jump after the k-th one suggests the graph splits into k groups."),
    ("Permutation test", "Shuffle the data many times to see how often luck alone gives a result this strong."),
    ("Projection", "A preseason estimate of how many fantasy points a player will score."),
])
