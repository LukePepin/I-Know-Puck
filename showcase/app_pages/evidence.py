"""Does it work? Pre-registered tests, projection accuracy, assumption checks, and one theory tested."""

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from ui import CAT, FAINT, INK, ME, MUTED, badge, csv, fig_show, js, note, rgba

tests = js("tests.json")
acc = csv("projection_accuracy.csv")
assumptions = js("assumptions.json")
theory = js("theory.json")
z = csv("weekly_scores.csv").z.dropna()
cal = csv("win_calibration.csv")

st.markdown("## Does it work?")
st.markdown("Every idea was written down as a yes-or-no question **before** testing it, then checked on seasons the model never saw. "
            "Ideas that failed were removed from the app, and the failures are listed here too.")

# --- tests ------------------------------------------------------------------------------------------
UNITS = {"abs fantasy-pt error": ("fantasy points of error saved per player", 1.0), "pick log-likelihood": ("log-likelihood per pick", 1.0),
         "P(win weekly matchup), actual stats": ("percentage points of weekly win chance", 100.0),
         "abs points-per-game error, second half": ("points per game of error saved", 1.0)}
st.markdown("### The tests")
cols = st.columns(2)
for i, t in enumerate(tests):
    with cols[i % 2]:
        with st.container(border=True):
            badge(t["answer"])
            st.markdown(f"**{t['question']}**")
            if t["meaning"]:
                st.markdown(t["meaning"])
            unit, scale = UNITS.get(t["metric"], ("", 1.0))
            st.caption(f"Test {t['id']} on {t['n']:,} cases: {scale * t['diff']:+.2f} {unit} (95% range {scale * t['lo']:+.2f} to {scale * t['hi']:+.2f}; "
                       "positive = the new idea did better)" + ((", p < 0.001" if t["p_holm"] < 0.001 else f", p = {t['p_holm']:.3f}") + f" after correcting for {len(tests)} tests." if t["verdict"] == "yes" else "."))
with st.expander("How to read a test"):
    st.markdown("Each test compares the old way and the new way on the **same** players, drafts or picks, and asks whether the average difference could be luck. "
                "The 95% range is where the true difference most likely sits: if the whole range is on the good side of zero, the idea helps. "
                "Because several tests were run, each p-value is corrected (Holm's method) so that one lucky result can't sneak through.")

# --- projections -------------------------------------------------------------------------------------
st.markdown("### Are the player projections better than ESPN's?")
c1, c2 = st.columns([3, 2])
with c1:
    fig = go.Figure()
    fig.add_bar(x=acc.season.astype(str), y=acc.err_espn, name="ESPN's projection", marker_color=FAINT)
    fig.add_bar(x=acc.season.astype(str), y=acc.err_ours, name="Our projection", marker_color=CAT[0])
    fig.update_layout(barmode="group", title="Average miss per player, by season (smaller is better)", yaxis_title="fantasy points off",
                      legend=dict(orientation="h", y=-0.2))
    fig_show(fig, 340)
with c2:
    st.markdown(
        f"Across 2024-26 our projection missed by **{acc.err_ours.mean():.1f}** points per player on average, ESPN's by **{acc.err_espn.mean():.1f}**. "
        "Those seasons were held out: the model was built only from the seasons before each one."
    )
    note("**Why it's better:** it blends a player's own history with ESPN's view and with where the market drafts him, in proportions learned from past seasons.")

# --- assumptions -------------------------------------------------------------------------------------
st.markdown("### Are the model's assumptions true?")
st.markdown("A model is only as good as the simplifications inside it. Each one was checked against the league's real results.")
cols = st.columns(2)
for i, a in enumerate(assumptions):
    with cols[i % 2]:
        with st.container(border=True):
            badge(a["verdict"])
            st.markdown(f"**{a['assumption']}**  \n{a['result']}")
c1, c2 = st.columns(2)
with c1:
    xs = np.linspace(-3.5, 3.5, 200)
    fig = go.Figure(go.Histogram(x=z, histnorm="probability density", xbins=dict(size=0.25), marker_color=rgba(CAT[0], 0.55),
                                 marker_line=dict(color="white", width=1), name="real weeks", hoverinfo="skip"))
    fig.add_scatter(x=xs, y=np.exp(-xs ** 2 / 2) / math.sqrt(2 * math.pi), mode="lines", line=dict(color=INK, width=2), name="bell curve")
    fig.update_layout(title="Real weekly team scores vs a bell curve", xaxis_title="how far from the team's own average (standard deviations)",
                      yaxis=dict(title="share of weeks", showticklabels=False), bargap=0.02, legend=dict(orientation="h", y=-0.25))
    fig_show(fig, 340)
with c2:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, width=1), name="perfect", hoverinfo="skip")
    fig.add_scatter(x=cal.predicted, y=cal.observed, mode="markers+lines", name="real matchups", line=dict(color=CAT[0], width=2),
                    marker=dict(size=10, color=CAT[0], line=dict(color="white", width=2)),
                    error_y=dict(type="data", symmetric=False, array=cal.hi - cal.observed, arrayminus=cal.observed - cal.lo, color=rgba(CAT[0], 0.5)),
                    customdata=cal.n, hovertemplate="predicted %{x:.0%}, actually won %{y:.0%} (%{customdata} matchups)<extra></extra>")
    fig.update_layout(title="When the formula said 70%, did teams win 70%?", xaxis=dict(title="predicted chance of winning", tickformat=".0%", range=[0, 1]),
                      yaxis=dict(title="share actually won", tickformat=".0%", range=[0, 1]), legend=dict(orientation="h", y=-0.25))
    fig_show(fig, 340)
st.caption("Left: 756 real weekly scores follow the bell curve closely. Right: past matchups predicted from each team's other weeks; dots near the line mean honest odds.")

# --- a theory ----------------------------------------------------------------------------------------
st.markdown("### Testing a theory: does a hot NHL team lift all its players?")
st.markdown("A theory from the league: when an NHL team beats its preseason expectations, all of its players benefit. The fair test compares each player "
            "with his **teammates** (leaving him out), and shuffles whole teams to see what luck alone looks like.")
q = theory["quintiles"]
c1, c2 = st.columns([3, 2])
with c1:
    labels = ["coldest fifth", "2nd", "middle", "4th", "hottest fifth"][: len(q)]
    beat = [x["beat"] for x in q]
    fig = go.Figure(go.Bar(x=labels, y=beat, marker_color=[CAT[0] if b_ >= np.mean(beat) else FAINT for b_ in beat],
                           error_y=dict(type="data", symmetric=False, array=[x["hi"] - x["beat"] for x in q], arrayminus=[x["beat"] - x["lo"] for x in q], color=MUTED),
                           hovertemplate="%{x} of teams: %{y:.0%} of players beat their projection<extra></extra>"))
    fig.update_layout(title="Share of players who beat their projection, by how their team did", yaxis=dict(title="beat projection", tickformat=".0%", range=[0, 1]),
                      xaxis_title="NHL teams grouped by how their other players did")
    fig_show(fig, 340)
with c2:
    note(f"**Partly true.** Each +10% for a player's teammates lifted him about **{10 * theory['slope']:+.1f}%** (95% range {10 * theory['lo']:+.1f}% to "
         f"{10 * theory['hi']:+.1f}%, p = {theory['p']:.3f}). But the team explains only about {max(theory['icc'], 0):.0%} of why a player beats his projection, "
         f"half the players on the hottest teams still fell short, and a hot first half told us **nothing** about a player's second half once his own "
         f"first half was known ({10 * theory['carry_team']:+.1f}% per +10%, p = {theory['carry_p']:.2f}). Judge a player by his own season, not his team's.")
