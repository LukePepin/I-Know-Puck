"""How the model picks: lineup -> weekly points -> win chance -> simulated drafts -> scarcity."""

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from ui import AUTHOR, CAT, FAINT, GROUP, INK, ME, MUTED, csv, fig_show, group_of, js, note, rgba

meta = js("meta.json")
teams = csv("teams_2027.csv")
rosters = csv("rosters_2027.csv")
crn = js("crn.json")
sc = csv("scarcity.csv")
lineup = {k: v for k, v in meta["lineup"].items() if k not in ("IR",)}
n_start = sum(v for k, v in lineup.items() if k != "BN")

st.markdown("## How the model picks a player")
st.caption("Four ideas, each small enough to explain on one screen.")

# --- 1. lineup --------------------------------------------------------------------------------------
st.markdown("### 1. Only the starting lineup scores")
c1, c2 = st.columns([2, 3])
with c1:
    st.markdown(
        f"Every team has {sum(lineup.values())} roster spots, but only **{n_start} starters** score: "
        + ", ".join(f"{v} {k}" for k, v in lineup.items() if k != "BN") + f", plus {lineup.get('BN', 0)} bench spots. "
        "The model fills the starting spots with the best combination of players (a centre who can also play wing goes wherever he helps most). "
        "This is a classic **assignment problem**, solved exactly by the Hungarian algorithm."
    )
    bu = meta.get("bench_fit")
    note("**Checking the rules paid off.** This league locks each lineup once a week, so a bench player can't be swapped in on a night off. "
         + (f"Across {bu['team_weeks']} real team-weeks, a team's score was about **{bu['starters']:.2f} x its likely starters' points + "
            f"{bu['bench']:.2f} x its bench points**. " if bu else "")
         + "The model had assumed daily lineups (bench worth about a third); it now counts the bench at a quarter.")
with c2:
    r = rosters[rosters.team == AUTHOR].copy()
    r = r.sort_values(["lineup", "weekly_pts"], ascending=[False, True])
    fig = go.Figure(go.Bar(x=r.weekly_pts, y=r.player + " (" + r.slot + ")", orientation="h",
                           marker_color=[GROUP[group_of(p)][1] if lu == "starter" else FAINT for p, lu in zip(r.pos, r.lineup)],
                           hovertemplate="%{y}: %{x:.1f} points a week<extra></extra>"))
    fig.update_layout(title="My roster: what each player adds to a typical week (grey = bench)", xaxis_title="expected points per week")
    fig_show(fig, 80 + 17 * len(r))

# --- 2. win chance ----------------------------------------------------------------------------------
st.markdown("### 2. From weekly points to a chance of winning")
sd = float(teams.weekly_sd.mean())
c1, c2 = st.columns([1, 2])
with c1:
    me_pts = st.slider("My expected weekly points", 80, 140, int(round(teams[teams.team == AUTHOR].weekly_pts.iloc[0])), key="m_me")
    opp_pts = st.slider("Opponent's expected weekly points", 80, 140, int(round(teams.weekly_pts.median())), key="m_opp")
    p = 0.5 * (1 + math.erf((me_pts - opp_pts) / (sd * math.sqrt(2) * math.sqrt(2))))
    st.metric("My chance to win the week", f"{p:.0%}", border=True)
    st.caption(f"Each team's weekly score swings by about ±{sd:.0f} points (one standard deviation), so even a 10-point favourite loses often.")
with c2:
    xs = np.linspace(min(me_pts, opp_pts) - 4 * sd, max(me_pts, opp_pts) + 4 * sd, 300)
    pdf = lambda mu: np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))  # noqa: E731
    fig = go.Figure()
    fig.add_scatter(x=xs, y=pdf(me_pts), name="my team", line=dict(color=ME, width=2), fill="tozeroy", fillcolor=rgba(ME, 0.15))
    fig.add_scatter(x=xs, y=pdf(opp_pts), name="opponent", line=dict(color=CAT[0], width=2), fill="tozeroy", fillcolor=rgba(CAT[0], 0.12))
    fig.update_layout(title="Two teams' possible weekly scores", xaxis_title="fantasy points in a week", yaxis=dict(title="how likely", showticklabels=False),
                      legend=dict(orientation="h", y=-0.2))
    fig_show(fig, 330)
st.markdown("The model scores a finished draft by averaging this chance over all 11 opponents. That single number is what every suggestion tries to raise.")

# --- 3. simulation ----------------------------------------------------------------------------------
st.markdown("### 3. Play the draft forward many times")
st.markdown(
    "You can't know who the others will take, so for each candidate the model **pretends you took him**, lets the other managers pick the way they "
    "usually do (mostly by ESPN rank, with some randomness), finishes the draft, and scores the result. Repeating this many times gives an average. "
    "One trick matters a lot: every candidate is tested against **the same set of simulated futures** (common random numbers), so luck cancels out "
    "when candidates are compared."
)
c1, c2 = st.columns(2)
names = crn["candidates"]
A, B = np.array(crn["crn"]), np.array(crn["independent"])
R = A.shape[1]
k = np.arange(1, R + 1)
with c1:
    fig = go.Figure()
    for i, n in enumerate(names):
        fig.add_scatter(x=k, y=np.cumsum(A[i]) / k, mode="lines", name=n, line=dict(color=CAT[i % len(CAT)], width=2),
                        hovertemplate=f"{n}<br>after %{{x}} drafts: %{{y:.1%}}<extra></extra>")
    fig.update_layout(title="Estimated win chance for five first-pick candidates, as simulations add up", xaxis_title="simulated drafts",
                      yaxis=dict(title="win a typical week", tickformat=".0%"), legend=dict(orientation="h", y=-0.3))
    fig_show(fig, 400)
with c2:
    best = int(A.mean(1).argmax())
    oth = [i for i in range(len(names)) if i != best]
    se_c = [np.std(A[i] - A[best], ddof=1) / math.sqrt(R) for i in oth]
    se_i = [math.sqrt(np.var(B[i], ddof=1) / R + np.var(B[best], ddof=1) / R) for i in oth]
    fig = go.Figure()
    fig.add_bar(y=[names[i] for i in oth], x=se_i, orientation="h", name="separate futures", marker_color=FAINT)
    fig.add_bar(y=[names[i] for i in oth], x=se_c, orientation="h", name="same futures", marker_color=CAT[0])
    fig.update_layout(barmode="group", title=f"Uncertainty in each candidate's gap to the best ({names[best]})", xaxis=dict(title="standard error of the gap", tickformat=".1%"),
                      legend=dict(orientation="h", y=-0.3))
    fig_show(fig, 400)
ratio = math.sqrt(np.mean(np.square(se_i))) / max(math.sqrt(np.mean(np.square(se_c))), 1e-9)
note(f"**Result:** sharing the simulated futures cut the uncertainty of each comparison about **{ratio:.0f}x**, which means roughly {ratio ** 2:.0f}x fewer "
     "simulations for the same precision. That is why a suggestion takes about two seconds instead of a minute.")

# --- 4. scarcity -------------------------------------------------------------------------------------
st.markdown("### 4. Value depends on who is left")
st.markdown("If 48 centres will start across the league, the 49th-best centre is **replacement level**: you can find one on waivers. A position whose "
            "values fall off quickly before that line is worth drafting early.")
colors = {"C": CAT[0], "LW": CAT[3], "RW": CAT[4], "D": CAT[1], "G": CAT[2]}
fig = go.Figure()
for pos, g in sc.groupby("pos", sort=False):
    fig.add_scatter(x=g["rank"], y=g.proj_pts, mode="lines", name=pos, line=dict(color=colors.get(pos, MUTED), width=2), customdata=g.player,
                    hovertemplate=f"{pos} #%{{x}}: %{{customdata}}, %{{y:.0f}} pts<extra></extra>")
    rep = int(g.replacement_rank.iloc[0])
    if rep <= g["rank"].max():
        fig.add_scatter(x=[rep], y=[g.loc[g["rank"] == rep, "proj_pts"].iloc[0]], mode="markers", marker=dict(size=10, color=colors.get(pos, MUTED),
                        line=dict(color="white", width=2)), showlegend=False, hovertemplate=f"{pos} replacement level (#{rep})<extra></extra>")
fig.update_layout(title="Projected season points by rank within each position (dots = replacement level)", xaxis_title="rank within position",
                  yaxis_title="projected season points", legend=dict(orientation="h", y=-0.2))
fig_show(fig, 400)
st.caption("Goalies sit lower because this league's scoring gives them fewer points per game, and only 24 start, so the replacement line comes early.")
