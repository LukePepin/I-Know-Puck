"""Present: the 2026-27 season so far. This week's matchups, power rankings, who beats whom, the draft and every roster."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import ACCENT, DIV, FAINT, GROUP, INK, csv, definitions, fig_show, group_of, header, how_built, js, math, note, rgba

meta = js("meta.json")
teams = csv("teams_2027.csv")
picks = csv("draft_2027.csv")
rosters = csv("rosters_2027.csv")
h2h = csv("h2h_2027.csv").set_index("manager")
sched = csv("schedule_2027.csv")
week = int(meta["current_week"])

header("Present", f"The {meta['season'] - 1}-{str(meta['season'])[2:]} season: the draft, every roster, and week {week}")

# --- this week ---------------------------------------------------------------------------------------
wk = sched[sched.week == week]
state = wk.status.iloc[0] if len(wk) else "upcoming"
st.markdown(f"### Week {week} ({state})")
if len(wk):
    rows = []
    for r in wk.itertuples():
        p_home = float(h2h.loc[r.home, r.away])
        rows.append({"matchup": f"{r.home} vs {r.away}", "score": f"{r.home_pts:.1f} - {r.away_pts:.1f}" if pd.notna(r.home_pts) else "-",
                     "leader": (r.home if r.home_pts > r.away_pts else r.away) if pd.notna(r.home_pts) and r.home_pts != r.away_pts else "-",
                     "model favourite": r.home if p_home >= 0.5 else r.away, "favourite's chance": 100 * max(p_home, 1 - p_home)})
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 column_config={"favourite's chance": st.column_config.ProgressColumn("model's pre-week chance", min_value=50, max_value=100, format="%.0f%%")})
    st.caption("Scores are a snapshot taken partway through the week. The model's chance is computed before any games, from each roster's projected "
               "weekly points and how much those points swing. Most matchups are close to a coin flip, which is normal in head-to-head fantasy.")

# --- power rankings -----------------------------------------------------------------------------------
st.markdown("### Power rankings")
t = teams.sort_values("rank")
c1, c2 = st.columns([2, 3])
with c1:
    st.dataframe(t[["rank", "manager", "weekly_pts", "win_chance", "change_since_draft", "moves"]].assign(win_chance=lambda x: 100 * x.win_chance), hide_index=True, height=460,
                 column_config={"rank": "#", "weekly_pts": st.column_config.NumberColumn("weekly pts", format="%.1f", help="Expected fantasy points in a typical week"),
                                "win_chance": st.column_config.NumberColumn("wins a week", format="%.0f%%", help="Chance to beat a typical opponent"),
                                "change_since_draft": st.column_config.NumberColumn("since draft", format="%+.1f", help="Change in weekly points from roster moves since the draft"),
                                "moves": st.column_config.NumberColumn("adds", help="Players on the roster who were not drafted by this team")})
with c2:
    avg = float(t.weekly_pts.mean())
    tt = t.sort_values("weekly_pts")
    fig = go.Figure()
    for _, r_ in tt.iterrows():
        fig.add_scatter(x=[avg, r_.weekly_pts], y=[r_.manager] * 2, mode="lines", line=dict(color=FAINT, width=2), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=tt.weekly_pts, y=tt.manager, mode="markers", showlegend=False, marker=dict(size=13, color=ACCENT, line=dict(color="white", width=2)),
                    customdata=np.stack([tt.win_chance, tt.sd_sim, tt.strongest.fillna(""), tt.top_stack.fillna("")], axis=1),
                    hovertemplate="%{y}: %{x:.1f} points a week (swing +/- %{customdata[1]:.0f})<br>wins a typical week %{customdata[0]:.0%}"
                                  "<br>strongest group: %{customdata[2]} · stack: %{customdata[3]}<extra></extra>")
    fig.add_vline(x=avg, line_color=INK, line_width=1, annotation_text=f"league average {avg:.0f}", annotation_position="top")
    fig.update_layout(title="Expected weekly points of each roster", xaxis_title="expected fantasy points per week")
    fig_show(fig, 440)
spread = float(t.weekly_pts.max() - t.weekly_pts.min())
note(f"**The league is tight.** The best and worst rosters are only **{spread:.0f} points a week** apart, while a single team's score swings by about "
     f"±{meta['observed_weekly_sd']:.0f} points from one week to the next. Even the top roster wins only about {t.win_chance.max():.0%} of its weeks against an "
     "average opponent. Rankings will move as injuries, waiver pickups and trades change the rosters.")

c1, c2 = st.columns([3, 2])
with c1:
    order = t.manager.tolist()
    m = h2h.loc[order, order].to_numpy(float)
    dev = float(np.nanmax(np.abs(m - 0.5)))
    fig = go.Figure(go.Heatmap(z=m, x=order, y=order, colorscale=DIV, zmid=0.5, zmin=0.5 - dev, zmax=0.5 + dev, xgap=1, ygap=1,
                               colorbar=dict(title="win", thickness=10, tickformat=".0%"), hovertemplate="%{y} beats %{x} in %{z:.0%} of weeks<extra></extra>"))
    fig.update_layout(title="Who beats whom in a typical week (row team vs column team, ranked order)", yaxis=dict(autorange="reversed"), xaxis=dict(tickangle=-45))
    fig_show(fig, 460)
with c2:
    st.markdown("**Reading the grid.** Each cell is the chance that the row team outscores the column team in a typical week. Blue favours the row; rust "
                "favours the column. The grid comes from one formula (see *The math* below): the gap between two teams' expected points, divided by how "
                "much their scores swing.")

# --- the draft ---------------------------------------------------------------------------------------
st.markdown("### The draft")
pk = picks.copy()
pk["grp"] = pk.pos.map(group_of)
slot_names = pk.drop_duplicates("slot").sort_values("slot").set_index("slot").manager
fig = go.Figure()
for g, (lab, colr) in GROUP.items():
    x = pk[pk.grp == g]
    fig.add_scatter(x=x.slot, y=x["round"], mode="markers", name=lab, marker=dict(symbol="square", size=15, color=colr, line=dict(color="white", width=1)),
                    customdata=np.stack([x.player, x.pos, x.nhl.fillna(""), x.overall, x.adp.round(0), x.manager], axis=1),
                    hovertemplate="Pick %{customdata[3]}: %{customdata[0]} (%{customdata[1]}, %{customdata[2]})<br>%{customdata[5]} · ADP %{customdata[4]}<extra></extra>")
fig.update_layout(title=f"All {len(pk)} picks by draft slot and round, coloured by position (hover a square)",
                  xaxis=dict(title="", tickmode="array", tickvals=list(slot_names.index), ticktext=list(slot_names), side="top"),
                  yaxis=dict(title="round", range=[meta["rounds"] + 0.6, 0.4], tick0=1, dtick=5), legend=dict(orientation="h", y=-0.05))
fig_show(fig, 680)
share = pk[pk["round"] <= 6].grp.value_counts(normalize=True)
st.caption(f"In rounds 1-6 the league took {share.get('F', 0):.0%} forwards, {share.get('D', 0):.0%} defensemen and {share.get('G', 0):.0%} goalies.")

q = pk.dropna(subset=["adp"])
q = q[(q["round"] <= 15) & (q.adp < 200)]
c1, c2 = st.columns([3, 2])
with c1:
    fig = go.Figure()
    fig.add_scatter(x=q.overall, y=q.adp, mode="markers", marker=dict(size=7, color=rgba(ACCENT, 0.55)), showlegend=False,
                    customdata=np.stack([q.player, q.manager], axis=1), hovertemplate="%{customdata[0]} (%{customdata[1]})<br>pick %{x}, ADP %{y:.0f}<extra></extra>")
    lim = float(q.overall.max())
    fig.add_scatter(x=[1, lim], y=[1, lim], mode="lines", line=dict(color=INK, width=1), name="picked exactly at ADP", hoverinfo="skip", showlegend=False)
    fig.update_layout(title="Pick number vs ADP, rounds 1-15 (above the line = a bargain, below = a reach)", xaxis_title="pick number",
                      yaxis=dict(title="ADP", range=[0, lim * 1.15]))
    fig_show(fig, 400)
with c2:
    for title, fn in (("Biggest bargains", "nlargest"), ("Biggest reaches", "nsmallest")):
        st.markdown(f"**{title}**")
        st.dataframe(getattr(q, fn)(5, "value_vs_adp")[["overall", "player", "manager", "value_vs_adp"]].rename(columns={"overall": "pick", "value_vs_adp": "vs ADP"}),
                     hide_index=True, column_config={"vs ADP": st.column_config.NumberColumn(format="%+.0f", help="How many picks after his ADP he went")})
st.caption("Past about pick 200 ESPN's ADP stops being informative, so later rounds are left out.")

# --- every roster ------------------------------------------------------------------------------------
st.markdown("### Every roster")
mgr = st.selectbox("Manager", t.manager.tolist(), key="p_mgr")
r = rosters[rosters.manager == mgr].copy()
row = t[t.manager == mgr].iloc[0]
bullets = [f"Ranked **#{int(row['rank'])}**: {row.weekly_pts:.0f} expected points a week, beating a typical opponent {row.win_chance:.0%} of the time."]
if isinstance(row.strongest, str) and row.strongest:
    bullets.append(f"Strongest group: **{row.strongest}**. Weakest: **{row.weakest}**.")
if isinstance(row.top_stack, str) and row.top_stack:
    bullets.append(f"Most players from one NHL team: **{row.top_stack}**.")
inj = r[r.status.fillna("ACTIVE").ne("ACTIVE")]
if len(inj):
    bullets.append("Injured or suspended: " + ", ".join(f"{p} ({s.replace('_', ' ').lower()})" for p, s in zip(inj.player, inj.status)) + ".")
if int(row.moves):
    bullets.append(f"{int(row.moves)} player(s) added since the draft: " + ", ".join(r[r.acquired != "draft"].player) + ".")
note("\n".join(f"- {x}" for x in bullets))
c1, c2 = st.columns([3, 2])
with c1:
    show = r.sort_values(["lineup", "weekly_pts"], ascending=[False, False])
    st.dataframe(show[["slot", "player", "pos", "nhl", "proj_pts", "weekly_pts", "status", "acquired"]], hide_index=True, height=420,
                 column_config={"slot": "spot", "proj_pts": st.column_config.NumberColumn("season pts", format="%.0f", help="Projected fantasy points for the season"),
                                "weekly_pts": st.column_config.NumberColumn("per week", format="%.1f", help="Points this player adds to a typical week (bench counts a quarter)")})
with c2:
    starters = r[r.lineup == "starter"].assign(grp=lambda x: x.pos.map(group_of))
    s_ = starters.groupby("grp").weekly_pts.sum().reindex(["F", "D", "G"]).fillna(0)
    fig = go.Figure(go.Bar(x=[GROUP[g][0] for g in s_.index] + ["Bench"], y=list(s_.values) + [r[r.lineup == "bench"].weekly_pts.sum()],
                           marker_color=[GROUP[g][1] for g in s_.index] + [FAINT], hovertemplate="%{x}: %{y:.1f} points a week<extra></extra>"))
    fig.update_layout(title="Where this roster's weekly points come from", yaxis_title="expected points per week")
    fig_show(fig, 360)

# --- systems overview ----------------------------------------------------------------------------------
how_built(
    [("ESPN league", "draft, rosters, live scores"), ("Projections", "each player's season"), ("Injuries", "games expected to miss"),
     ("Best lineup", "20 starters, Hungarian"), ("Weekly points", "mean and swing"), ("Win chance", "rankings, grid")],
    ["**Collect.** The draft results, current rosters, schedule and live scores come from ESPN's fantasy API at the time of the snapshot.",
     "**Project.** Each player's season is projected from his last three seasons (recent seasons count more), pulled toward his position's average, "
     "blended with ESPN's projection and with where the market drafts him. Injured or suspended players lose the games they are expected to miss.",
     f"**Fill the lineup.** Only {meta['starters']} starters score each week. The Hungarian algorithm assigns players to the starting spots to maximise points. "
     f"This league locks lineups weekly, so bench players count {meta['bench_usage']:.2f} of their points (measured from real weekly scores).",
     "**Score the roster.** Expected weekly points and their week-to-week swing give each team a bell curve; comparing two curves gives the chance one beats the other."],
)
math((r"\mu = \frac{1}{26}\sum_i u_i \, p_i", "Expected weekly points: each player's projected season points p, times his usage u (1 for a starter, 0.25 for bench), over 26 weeks."),
     (r"P(A \text{ beats } B) = \Phi\!\left(\frac{\mu_A - \mu_B}{\sqrt{\sigma_A^2 + \sigma_B^2}}\right)",
      "Φ is the bell-curve (standard Normal) probability; σ is how much each team's weekly score swings."))
definitions([
    ("Starting lineup", f"The {meta['starters']} players whose points count each week: " + ", ".join(f"{v} {k}" for k, v in meta["lineup"].items() if k not in ("BN", "IR")) + "."),
    ("Bench", "Roster spots whose points don't count unless the player is moved into the lineup before his first game of the week."),
    ("Weekly lineup lock", "Each player's spot locks at his first game of the week, so the bench can't fill nightly gaps."),
    ("Expected weekly points", "The average points a roster's starting lineup should score in a typical week."),
    ("Win chance", "The chance a roster outscores an opponent in one week, from the formula in The math."),
    ("ADP", "Average draft position: where a player usually goes in ESPN drafts."),
    ("Bargain / reach", "A player taken well after (bargain) or well before (reach) his ADP."),
    ("Injury status", "DTD = day to day, O = out, IR = injured reserve, SSPD = suspended."),
])
