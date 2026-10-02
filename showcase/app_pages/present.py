"""Present: the 2026-27 season so far. This week's games, power rankings, the draft and a page for every team."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import (ACCENT, BG, CAT, FAINT, GROUP, INK, MUTED, csv, definitions, esc, fig_show, group_of, header, how_built, js, math, note, pick_one,
                snapshot_date, table)

meta = js("meta.json")
teams = csv("teams_2027.csv")
picks = csv("draft_2027.csv")
rosters = csv("rosters_2027.csv")
h2h = csv("h2h_2027.csv").set_index("manager")
sched = csv("schedule_2027.csv")
week = int(meta["current_week"])
t = teams.sort_values("rank")

header("Present", f"I built I Know Puck to draft for my {meta['teams']}-team ESPN league and kept it running into the season. This tab is the league "
       f"as of {snapshot_date()}: this week's games, my power rankings, how the draft went, and a page for every team.")

# --- this week ---------------------------------------------------------------------------------------
wk = sched[sched.week == week]
state = wk.status.iloc[0] if len(wk) else "upcoming"
st.markdown(f"### Week {week}")
st.caption(f"{state.capitalize()} · scores as of {snapshot_date()}")
cards = []
for r in wk.itertuples():
    p_home = float(h2h.loc[r.home, r.away])
    fav, p_fav = (r.home, p_home) if p_home >= 0.5 else (r.away, 1 - p_home)
    played = pd.notna(r.home_pts)
    lead = (r.home if r.home_pts > r.away_pts else r.away) if played and r.home_pts != r.away_pts else None
    rows = "".join(f'<div class="row{" lead" if m == lead else ""}"><span>{esc(m)}</span><b>{pts:.1f}</b></div>' if played else
                   f'<div class="row"><span>{esc(m)}</span><b>-</b></div>' for m, pts in ((r.home, r.home_pts), (r.away, r.away_pts)))
    cards.append(f'<div class="kp-game">{rows}<div class="foot">Model before the week: {esc(fav)}, {p_fav:.0%}</div></div>')
st.html('<div class="kp-board">' + "".join(cards) + "</div>")
st.caption("The model makes its pick before any games, from each roster's projected points and how much they swing. Most games start near 50-50.")

# --- power rankings -----------------------------------------------------------------------------------
st.markdown("### Power rankings")
st.caption("Ranked by expected points a week from each team's best lineup. Win % is the chance to beat an average team in one week.")
c1, c2 = st.columns(2, gap="large")
with c1:
    table(t, [("rank", "#", "{:.0f}"), ("manager", "Team", None), ("weekly_pts", "Pts/wk", "{:.1f}"),
              ("win_chance", "Win %", lambda v: f"{v:.0%}"), ("change_since_draft", "Since draft", "{:+.1f}")])
with c2:
    avg = float(t.weekly_pts.mean())
    tt = t.sort_values("weekly_pts")
    fig = go.Figure()
    for _, r_ in tt.iterrows():
        fig.add_scatter(x=[avg, r_.weekly_pts], y=[r_.manager] * 2, mode="lines", line=dict(color=FAINT, width=2), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=tt.weekly_pts, y=tt.manager, mode="markers", showlegend=False, marker=dict(size=14, color=ACCENT, line=dict(color=BG, width=2)),
                    customdata=np.stack([tt.win_chance, tt.sd_sim, tt.strongest.fillna(""), tt.top_stack.fillna("")], axis=1),
                    hovertemplate="%{y}: %{x:.1f} points a week (swing ±%{customdata[1]:.0f})<br>beats an average team %{customdata[0]:.0%} of weeks"
                                  "<br>strongest: %{customdata[2]} · stack: %{customdata[3]}<extra></extra>")
    fig.add_vline(x=avg, line_color=MUTED, line_width=1, line_dash="dot", annotation_text=f"average {avg:.0f}", annotation_position="top",
                  annotation_font_color=MUTED)
    fig.update_layout(title="Expected points a week", xaxis_title="points per week")
    fig_show(fig, 430)
spread = float(t.weekly_pts.max() - t.weekly_pts.min())
note(f"**The league is tight.** The best and worst rosters are {spread:.0f} points a week apart, but one team's score swings about "
     f"±{meta['observed_weekly_sd']:.0f} points from week to week. Against an average opponent, the top roster wins {t.win_chance.max():.0%} of its weeks.")

# --- the draft ---------------------------------------------------------------------------------------
st.markdown("### The draft")
pk = picks.copy()
pk["grp"] = pk.pos.map(group_of)
st.caption(f"{len(pk)} picks over {meta['rounds']} rounds in snake order.")
c1, c2 = st.columns(2, gap="large")
with c1:
    mix = pk.groupby(["round", "grp"]).size().unstack(fill_value=0).reindex(columns=["F", "D", "G"], fill_value=0)
    fig = go.Figure()
    for g in ("F", "D", "G"):
        lab, colr = GROUP[g]
        fig.add_bar(y=mix.index, x=mix[g], orientation="h", name=lab, marker=dict(color=colr, line=dict(color=BG, width=1)),
                    hovertemplate=f"Round %{{y}}: %{{x}} {lab.lower()}<extra></extra>")
    fig.update_layout(barmode="stack", title="Positions taken in each round", bargap=0.15, legend_traceorder="normal",
                      xaxis=dict(title=f"picks (of {meta['teams']})", dtick=3), yaxis=dict(title="round", autorange="reversed", dtick=5))
    fig_show(fig, 520)
    share = pk[pk["round"] <= 6].grp.value_counts(normalize=True)
    st.caption(f"In rounds 1-6 the league took {share.get('F', 0):.0%} forwards, {share.get('D', 0):.0%} defensemen and {share.get('G', 0):.0%} goalies.")
q = pk.dropna(subset=["adp"])
q = q[(q["round"] <= 15) & (q.adp < 200)]
with c2:
    fig = go.Figure()
    lim = float(q.overall.max())
    fig.add_scatter(x=[1, lim], y=[1, lim], mode="lines", line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip", showlegend=False)
    for lab, mask, colr in (("bargain", q.value_vs_adp > 0, CAT[0]), ("reach", q.value_vs_adp <= 0, CAT[1])):
        x = q[mask]
        fig.add_scatter(x=x.overall, y=x.adp, mode="markers", name=lab, marker=dict(size=9, color=colr, opacity=0.8, line=dict(color=BG, width=1)),
                        customdata=np.stack([x.player, x.manager], axis=1), hovertemplate="%{customdata[0]} (%{customdata[1]})<br>pick %{x}, ADP %{y:.0f}<extra></extra>")
    fig.update_layout(title="Pick number vs ADP, rounds 1-15", xaxis_title="pick number", yaxis=dict(title="ADP", range=[0, lim * 1.15]))
    fig_show(fig, 420)
    st.caption("ADP is where a player usually goes in ESPN drafts. Above the line he went later than his ADP (a bargain); below it, earlier "
               "(a reach). I leave out picks after ADP 200, where it stops meaning much.")
c1, c2 = st.columns(2, gap="large")
for col, title, fn in ((c1, "Biggest bargains", "nlargest"), (c2, "Biggest reaches", "nsmallest")):
    with col:
        st.markdown(f"**{title}**")
        table(getattr(q, fn)(5, "value_vs_adp"), [("overall", "Pick", "{:.0f}"), ("player", "Player", None, "manager"), ("value_vs_adp", "vs ADP", "{:+.0f}")])

# --- team pages --------------------------------------------------------------------------------------
st.markdown("### Team pages")
mgr = pick_one("Team", t.manager.tolist(), key="p_mgr")
row = t[t.manager == mgr].iloc[0]
r = rosters[rosters.manager == mgr].copy()
st.markdown(f"#### #{int(row['rank'])} {mgr}")
st.caption(f"{row.weekly_pts:.1f} expected points a week · beats an average team {row.win_chance:.0%} of the time · {row.change_since_draft:+.1f} since the draft")
bullets = []
if isinstance(row.strongest, str) and row.strongest:
    bullets.append(f"Strongest group: **{row.strongest}**. Weakest: **{row.weakest}**.")
if isinstance(row.top_stack, str) and row.top_stack:
    bullets.append(f"Most players from one NHL team: **{row.top_stack}**.")
inj = r[r.status.fillna("ACTIVE").ne("ACTIVE")]
if len(inj):
    bullets.append("Injured or suspended: " + ", ".join(f"{p} ({s.replace('_', ' ').lower()})" for p, s in zip(inj.player, inj.status)) + ".")
if int(row.moves):
    bullets.append("Added since the draft: " + ", ".join(r[r.acquired != "draft"].player) + ".")
if bullets:
    note("\n".join(f"- {x}" for x in bullets))

SLOT_ORDER = {"C": 0, "LW": 1, "RW": 2, "D": 3, "G": 4, "BN": 5}
STATUS = {"INJURY_RESERVE": "IR", "OUT": "out", "SUSPENSION": "suspended", "DAY_TO_DAY": "day to day"}
r["spot"] = r.slot.astype(str).str.rstrip("0123456789")
r["order"] = r.spot.map(SLOT_ORDER).fillna(9)
r["sub"] = [" · ".join(x for x in (p, n if isinstance(n, str) else "", STATUS.get(s, ""), "added" if a != "draft" else "") if x)
            for p, n, s, a in zip(r.pos, r.nhl, r.status, r.acquired)]
c1, c2 = st.columns([3, 2], gap="large")
with c1:
    table(r.sort_values(["order", "weekly_pts"], ascending=[True, False]),
          [("spot", "Spot", None), ("player", "Player", None, "sub"), ("weekly_pts", "Pts/wk", "{:.1f}"), ("proj_pts", "Season", "{:.0f}")])
    st.caption("Pts/wk is what a player adds to a typical week. Bench players count a quarter, because lineups lock for the week.")
with c2:
    starters = r[r.lineup == "starter"].assign(grp=lambda x: x.pos.map(group_of))
    s_ = starters.groupby("grp").weekly_pts.sum().reindex(["F", "D", "G"]).fillna(0)
    vals = list(s_.values) + [r[r.lineup == "bench"].weekly_pts.sum()]
    fig = go.Figure(go.Bar(x=[GROUP[g][0] for g in s_.index] + ["Bench"], y=vals, marker_color=[GROUP[g][1] for g in s_.index] + [FAINT],
                           text=[f"{v:.0f}" for v in vals], textposition="outside", cliponaxis=False, hovertemplate="%{x}: %{y:.1f} points a week<extra></extra>"))
    fig.update_layout(title=f"Where {mgr}'s points come from", yaxis=dict(title="points per week", range=[0, max(vals) * 1.2]))
    fig_show(fig, 300)
    opp = h2h.loc[mgr].drop(mgr).astype(float).sort_values()
    dev = max(0.12, float(np.abs(opp - 0.5).max()) + 0.04)
    fig = go.Figure(go.Bar(y=opp.index, x=opp.values - 0.5, base=0.5, orientation="h", marker_color=[CAT[0] if v >= 0.5 else CAT[1] for v in opp],
                           text=[f"{v:.0%}" for v in opp], textposition="outside", cliponaxis=False,
                           hovertemplate=f"{mgr} beats %{{y}} in %{{customdata:.0%}} of weeks<extra></extra>", customdata=opp.values))
    fig.add_vline(x=0.5, line_color=INK, line_width=1)
    fig.update_layout(title=f"{mgr}'s chance to win a week against each team", xaxis=dict(range=[0.5 - dev, 0.5 + dev], tickformat=".0%"))
    fig_show(fig, 380)
mine = picks[picks.manager == mgr].assign(sub=lambda x: x.pos + " · " + x.nhl.fillna(""))
with st.expander(f"{mgr}'s draft: all {len(mine)} picks"):
    table(mine, [("round", "Rd", "{:.0f}"), ("overall", "Pick", "{:.0f}"), ("player", "Player", None, "sub"), ("value_vs_adp", "vs ADP", "{:+.0f}")])

# --- systems overview ----------------------------------------------------------------------------------
how_built(
    [("ESPN league", "draft, rosters, live scores"), ("Projections", "each player's season"), ("Injuries", "games expected to miss"),
     ("Best lineup", f"{meta['starters']} starters, Hungarian"), ("Weekly points", "average and swing"), ("Win chance", "rankings and odds")],
    ["**Collect.** I pull the draft, current rosters, schedule and live scores from ESPN's fantasy API when I take the snapshot.",
     "**Project.** I project each player's season from his last three seasons (recent ones count more), pull it toward his position's average, and blend "
     "it with ESPN's projection and with where the market drafts him. Injured and suspended players lose the games they're expected to miss.",
     f"**Fill the lineup.** Only {meta['starters']} starters score each week. The Hungarian algorithm puts players in the starting spots that maximise points. "
     f"The league locks lineups weekly, so a bench player counts for {meta['bench_usage']:.2f} of his points. I measured that from real weekly scores.",
     "**Score the roster.** Expected weekly points and their week-to-week swing give each team a bell curve. Comparing two curves gives the chance one team "
     "beats the other."],
)
math((r"\mu = \frac{1}{26}\sum_i u_i \, p_i", "Expected weekly points: each player's projected season points p, times his usage u (1 for a starter, 0.25 for bench), over 26 weeks."),
     (r"P(A \text{ beats } B) = \Phi\!\left(\frac{\mu_A - \mu_B}{\sqrt{\sigma_A^2 + \sigma_B^2}}\right)",
      "Φ is the bell-curve (standard Normal) probability; σ is how much each team's weekly score swings."))
definitions([
    ("Starting lineup", f"The {meta['starters']} players whose points count each week: " + ", ".join(f"{v} {k}" for k, v in meta["lineup"].items() if k not in ("BN", "IR")) + "."),
    ("Bench", "Roster spots whose points don't count unless the player is moved into the lineup before his first game of the week."),
    ("Weekly lineup lock", "Each player's spot locks at his first game of the week, so the bench can't fill nightly gaps."),
    ("Expected weekly points", "The average points a roster's best lineup should score in a typical week."),
    ("Win chance", "The chance a roster outscores an opponent in one week, from the formula in The math."),
    ("ADP", "Average draft position: where a player usually goes in ESPN drafts."),
    ("Bargain / reach", "A player taken well after (bargain) or well before (reach) his ADP."),
    ("Injury status", "Day to day, out, IR (injured reserve) or suspended, as ESPN reports it."),
])
