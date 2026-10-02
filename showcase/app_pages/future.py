"""Future: simulate the rest of the season, explain how the models work, and the whole-system overview."""

import math as pymath

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import (ACCENT, BG, CAT, FAINT, INK, MUTED, badge, csv, definitions, fig_show, header, how_built, js, math, note, rgba, system_map, table)

meta = js("meta.json")
teams = csv("teams_2027.csv").sort_values("rank").reset_index(drop=True)
sched = csv("schedule_2027.csv")
tests = js("tests.json")
assumptions = js("assumptions.json")
acc = csv("projection_accuracy.csv")
z = csv("weekly_scores.csv").z.dropna()
cal = csv("win_calibration.csv")
crn = js("crn.json")
season = f"{meta['season'] - 1}-{str(meta['season'])[2:]}"

header("Future", f"I simulate the rest of the {season} season 10,000 times on the real schedule. Below that: how each model behind the numbers works, "
       "how I tested them, and how the whole system fits together.")


@st.cache_data(max_entries=16)
def simulate(mu: tuple, sd: tuple, games: tuple, n_weeks: int, n_playoff: int, uncertainty: float, n: int = 10_000, seed: int = 7):
    """Monte Carlo seasons: each team gets a season-long strength (projection + uncertainty), each week's score is
    strength + Normal noise, standings by wins (points break ties), then a seeded single-week playoff bracket."""
    rng = np.random.default_rng(seed)
    mu, sd = np.asarray(mu), np.asarray(sd)
    k = len(mu)
    strength = mu + rng.normal(0, uncertainty, (n, k))
    scores = strength[:, None, :] + rng.normal(0, 1, (n, n_weeks, k)) * sd
    wins = np.zeros((n, k))
    for w, h, a in games:
        if w <= n_weeks:
            win_h = scores[:, w - 1, h] > scores[:, w - 1, a]
            wins[:, h] += win_h
            wins[:, a] += ~win_h
    key = wins + scores.sum(axis=1) / 1e6
    order = np.argsort(-key, axis=1)  # seed 1 first
    finish = np.empty_like(order)
    np.put_along_axis(finish, order, np.arange(k)[None, :].repeat(n, 0), axis=1)

    def game(a_, b_):
        sa = np.take_along_axis(strength, a_[:, None], 1)[:, 0] + rng.normal(0, 1, n) * sd[a_]
        sb = np.take_along_axis(strength, b_[:, None], 1)[:, 0] + rng.normal(0, 1, n) * sd[b_]
        return np.where(sa > sb, a_, b_)

    seeds = order[:, :n_playoff]
    if n_playoff == 8:
        q = [game(seeds[:, 0], seeds[:, 7]), game(seeds[:, 3], seeds[:, 4]), game(seeds[:, 1], seeds[:, 6]), game(seeds[:, 2], seeds[:, 5])]
        champ = game(game(q[0], q[1]), game(q[2], q[3]))
    else:
        champ = seeds[:, 0]
    return {"wins": wins, "finish": finish, "playoff": (finish < n_playoff).mean(0), "top_seed": (finish == 0).mean(0),
            "title": np.bincount(champ, minlength=k) / n, "n": n}


# --- season simulation ---------------------------------------------------------------------------------
st.markdown("### Simulating the season")
idx = {m: i for i, m in enumerate(teams.manager)}
games = tuple((int(r.week), idx[r.home], idx[r.away]) for r in sched.itertuples() if r.home in idx and r.away in idx)
n_po = int(meta["playoff_teams"])
c1, c2 = st.columns([1, 2], gap="large")
with c1:
    unc = st.slider("How unsure am I about each team's true strength? (points per week)", 0.0, 12.0, float(meta["team_uncertainty"]), 0.5, format="%.1f",
                    key="f_unc", help="0 = the projections are exactly right. The default comes from how much player projections usually miss.")
    st.caption(f"Default ±{meta['team_uncertainty']:.1f}: a player's season projection usually misses by about {meta['per_player_weekly_miss']:.1f} points "
               f"a week, and a lineup has {meta['starters']} starters (√{meta['starters']} × {meta['per_player_weekly_miss']:.1f} ≈ {meta['team_uncertainty']:.1f}).")
sim = simulate(tuple(teams.weekly_pts), tuple(teams.sd_sim), games, int(meta["regular_weeks"]), n_po, unc)
fin = sim["finish"]
res = pd.DataFrame({"manager": teams.manager, "proj": teams.weekly_pts, "wins": sim["wins"].mean(0), "lo": np.quantile(sim["wins"], 0.1, axis=0),
                    "hi": np.quantile(sim["wins"], 0.9, axis=0), "playoff": sim["playoff"], "top_seed": sim["top_seed"], "title": sim["title"],
                    "f1": (fin == 0).mean(0), "f2": ((fin >= 1) & (fin <= 3)).mean(0), "f5": ((fin >= 4) & (fin < n_po)).mean(0), "miss": (fin >= n_po).mean(0),
                    "avg_finish": fin.mean(0) + 1}).sort_values("title", ascending=False)
with c2:
    fav = res.iloc[0]
    with st.container(horizontal=True):
        st.metric("Title favourite", fav.manager, delta=f"{fav.title:.0%} chance", delta_color="off", delta_arrow="off", border=True)
        st.metric("Seasons simulated", f"{sim['n']:,}", border=True)
        st.metric("Teams with 10%+ title odds", int((res.title >= 0.10).sum()), border=True)
    st.markdown(f"Each simulated season plays all {meta['regular_weeks']} weeks of the real schedule, ranks teams by wins, and runs the "
                f"{n_po}-team bracket (1 v 8, 4 v 5, 2 v 7, 3 v 6, one week per round). More uncertainty pulls every team toward the middle.")
c1, c2 = st.columns(2, gap="large")
with c1:
    r = res.sort_values("playoff")
    fig = go.Figure()
    fig.add_bar(y=r.manager, x=r.playoff, orientation="h", name="makes playoffs", marker_color=rgba(ACCENT, 0.3), hovertemplate="%{y}: playoffs %{x:.0%}<extra></extra>")
    fig.add_bar(y=r.manager, x=r.title, orientation="h", name="wins the title", marker_color=ACCENT, text=[f"{v:.0%}" for v in r.title],
                textposition="outside", cliponaxis=False, hovertemplate="%{y}: title %{x:.0%}<extra></extra>")
    fig.update_layout(barmode="overlay", title="Chance to make the playoffs and win the title", xaxis=dict(tickformat=".0%", range=[0, 1]))
    fig_show(fig, 440)
with c2:
    r = res.sort_values("avg_finish", ascending=False)
    fig = go.Figure()
    for col, lab, colr in (("f1", "1st", CAT[2]), ("f2", "2nd-4th", ACCENT), ("f5", f"5th-{n_po}th", rgba(ACCENT, 0.45)), ("miss", "misses playoffs", FAINT)):
        fig.add_bar(y=r.manager, x=r[col], orientation="h", name=lab, marker=dict(color=colr, line=dict(color=BG, width=1)),
                    hovertemplate=f"%{{y}} finishes {lab}: %{{x:.0%}}<extra></extra>")
    fig.update_layout(barmode="stack", title="Where each team finishes the regular season", legend_traceorder="normal", xaxis=dict(tickformat=".0%", range=[0, 1]))
    fig_show(fig, 440)
res["range"] = [f"range {lo:.0f}-{hi:.0f}" for lo, hi in zip(res.lo, res.hi)]
table(res, [("manager", "Team", None), ("wins", "Wins", "{:.1f}", "range"), ("playoff", "Playoffs", ("bar", 1.0, "{:.0%}")),
            ("title", "Title", ("bar", max(0.3, float(res.title.max())), "{:.0%}"))])
st.caption(f"Wins are out of {meta['regular_weeks']} weeks; the range covers the middle 80% of simulated seasons.")
one_in = 1 / max(float(fav.title), 1e-9)
note(f"**Reading the odds.** Rosters are close and weekly scores swing a lot, so even the favourite wins the title in about 1 season in {one_in:.0f}. "
     "The simulation holds today's rosters fixed for the whole season; injuries, pickups and trades will move these numbers.")
how_built(
    [("Rosters", f"today's {meta['teams']} teams"), ("Weekly points", "average and swing"), ("ESPN schedule", f"{meta['regular_weeks']} weeks"),
     ("10,000 seasons", "random weekly scores"), ("Standings", "wins, then points"), ("Playoffs", f"{n_po}-team bracket")],
    [f"**Inputs.** Each team's expected weekly points from the Present tab, and its week-to-week swing. The roster model alone says ±{meta['model_weekly_sd']:.0f} "
     f"points; real teams in 2024-26 swung by ±{meta['observed_weekly_sd']:.0f}, because injuries, roster moves and schedules add noise. I add the missing "
     f"±{meta['sd_extra']:.0f} so simulated weeks are as unpredictable as real ones.",
     "**One season.** Each team draws a season-long strength (its projection plus a random error set by the slider), then a fresh random score every week. "
     "The higher score wins each real matchup on the schedule.",
     "**Repeat.** 10,000 seasons give a distribution of wins, finishing places, playoff spots and champions. Each share is accurate to about ±1 percentage point.",
     "**Runs live.** The simulation reruns on the server each time the slider moves, from the published rosters and schedule."],
)
math((r"\text{strength}_t \sim \mathcal{N}(\mu_t, u^2), \qquad \text{score}_{t,w} \sim \mathcal{N}(\text{strength}_t, \sigma_t^2)", "u is the slider; μ and σ are each team's expected weekly points and swing."),
     (r"\sigma_t^2 = \sigma_{\text{model},t}^2 + \sigma_{\text{extra}}^2", f"σ_extra = {meta['sd_extra']:.1f}, chosen so simulated weeks swing like the league's real weeks."))

# --- how the models work -------------------------------------------------------------------------------
st.markdown("### How the models work")
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("**1. Projecting players.** I take a player's last three seasons, weighted 5/4/3 toward the most recent, pull them toward his position's "
                "average, correct them with advanced stats (expected goals, ice time), blend in ESPN's projection, and shrink the result toward where the "
                "market drafts him.")
    fig = go.Figure()
    fig.add_bar(x=acc.season.astype(str), y=acc.err_espn, name="ESPN's projection", marker_color=FAINT, text=[f"{v:.0f}" for v in acc.err_espn], textposition="outside", cliponaxis=False)
    fig.add_bar(x=acc.season.astype(str), y=acc.err_ours, name="my model", marker_color=ACCENT, text=[f"{v:.0f}" for v in acc.err_ours], textposition="outside", cliponaxis=False)
    fig.update_layout(barmode="group", title="Average miss per player, on seasons the model never saw",
                      yaxis=dict(title="fantasy points off", range=[0, float(acc.err_espn.max()) * 1.25]))
    fig_show(fig, 320)
    st.caption("Smaller is better.")
with c2:
    st.markdown(f"**2. Turning a roster into a chance of winning.** I treat the best lineup's weekly points as a bell curve. I checked that on {len(z)} real "
                "weekly scores (left) and on past matchups (right): when the formula said 70%, teams won about 70% of the time.")
    xs = np.linspace(-3.5, 3.5, 200)
    cc1, cc2 = st.columns(2)
    with cc1:
        fig = go.Figure(go.Histogram(x=z, histnorm="probability density", xbins=dict(size=0.25), marker_color=rgba(ACCENT, 0.55), marker_line=dict(color=BG, width=1),
                                     hoverinfo="skip"))
        fig.add_scatter(x=xs, y=np.exp(-xs**2 / 2) / pymath.sqrt(2 * pymath.pi), mode="lines", line=dict(color=INK, width=2), hoverinfo="skip")
        fig.update_layout(title="Real weekly scores vs a bell curve", showlegend=False, xaxis_title="standard deviations", yaxis=dict(showticklabels=False, title=""))
        fig_show(fig, 260)
    with cc2:
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=cal.predicted, y=cal.observed, mode="markers+lines", line=dict(color=ACCENT, width=2), marker=dict(size=9, color=ACCENT), showlegend=False,
                        hovertemplate="predicted %{x:.0%}, won %{y:.0%}<extra></extra>")
        fig.update_layout(title="Predicted vs actual wins", xaxis=dict(tickformat=".0%", range=[0, 1], title="predicted"), yaxis=dict(tickformat=".0%", range=[0, 1], title="actual"))
        fig_show(fig, 260)
names_ = crn["candidates"]
A, B = np.array(crn["crn"]), np.array(crn["independent"])
R = A.shape[1]
best = int(A.mean(1).argmax())
oth = [i for i in range(len(names_)) if i != best]
se_c = [np.std(A[i] - A[best], ddof=1) / pymath.sqrt(R) for i in oth]
se_i = [pymath.sqrt(np.var(B[i], ddof=1) / R + np.var(B[best], ddof=1) / R) for i in oth]
ratio = pymath.sqrt(np.mean(np.square(se_i))) / max(pymath.sqrt(np.mean(np.square(se_c))), 1e-9)
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("**3. Simulating the draft.** For each candidate pick, the model pretends I took him, lets the other managers pick the way they usually do, "
                "finishes the draft and scores the result, many times over. Every candidate faces the *same* simulated futures, so luck cancels out of the comparison.")
    kk = np.arange(1, R + 1)
    fig = go.Figure()
    for i, n_ in enumerate(names_):
        fig.add_scatter(x=kk, y=np.cumsum(A[i]) / kk, mode="lines", name=n_, line=dict(color=CAT[i % len(CAT)], width=2), hovertemplate=f"{n_}: %{{y:.1%}}<extra></extra>")
    fig.update_layout(title="Win-chance estimates for five first-pick candidates as simulations add up", xaxis_title="simulated drafts",
                      yaxis=dict(title="wins a typical week", tickformat=".0%"), hovermode="x unified")
    fig_show(fig, 380)
with c2:
    st.markdown(f"**Why share the futures?** Two candidates compared on different random drafts differ partly by luck. Giving both the *same* random drafts "
                f"(**common random numbers**) cancels that luck. Here it cut the uncertainty of each comparison about **{ratio:.0f}x**, so a draft suggestion "
                "took about two seconds instead of a minute.")
    note("I used the draft simulation for live pick suggestions on draft night. The season simulation above reuses the same weekly-points model.")

st.markdown("**4. Testing every idea.** I wrote each idea down as a yes-or-no question before testing it, then checked it on seasons the model never saw. "
            "Ideas that failed came out of the model. The failures are listed too.")
h1 = next((t for t in tests if t["id"] == "H1"), None)
MEANING = {"H1": f"The blended projections miss by about {h1['diff']:.1f} fewer fantasy points per player than ESPN's." if h1 else "",
           "H1b": "A small gain: advanced stats (expected goals, ice time) make the projections slightly better.",
           "H2": "Drafting with the model did about as well as following ESPN's rankings, so the rankings stay the backbone.",
           "H2a": "Always taking the player the model likes most compared with his ranking picks the model's own mistakes (the winner's curse).",
           "H3": "One league-wide pick model predicts managers' picks as well as a separate model for each manager.",
           "H4": "Grouping managers by drafting style made pick predictions slightly worse.",
           "H5": "A hot NHL team's first half adds nothing to a player's own first half when predicting his second half."}
cols = st.columns(2)
for i, t in enumerate(tests):
    with cols[i >= (len(tests) + 1) // 2]:  # fill down, so a phone's stacked columns keep the order
        with st.container(border=True):
            badge(t["answer"])
            st.markdown(f"**{t['question']}**  \n{MEANING.get(t['id'], t['meaning'])}")
with st.expander("Assumption checks: are the model's simplifications true?"):
    cols = st.columns(2)
    for i, a in enumerate(assumptions):
        with cols[i >= (len(assumptions) + 1) // 2]:
            with st.container(border=True):
                badge(a["verdict"])
                st.markdown(f"**{a['assumption']}**  \n{a['result']}")
math((r"\text{rate} = \frac{\sum_i w_i x_i + R\,\bar{x}_{\text{pos}}}{\sum_i w_i\,\text{GP}_i + R}, \quad w = (5, 4, 3)",
      "Projection per game: weighted past production pulled toward the position average with strength R."),
     (r"\text{final} = a + b\cdot\text{projection} + c\cdot\ln(\text{ADP})", "Market adjustment: fit on past seasons, it shrinks the projection toward where the crowd drafts the player."))

# --- whole-system overview -----------------------------------------------------------------------------
st.markdown("### Systems overview")
st.caption("Data flows left to right (top to bottom on a phone). Verification checks the models; only the cleaned export reaches this site.")
system_map([
    ("1 · Data", [("ESPN fantasy API", "players, drafts, results"), ("MoneyPuck", "advanced stats"), ("Local cache", "repeatable runs")]),
    ("2 · Models", [("Player projections", "three seasons + market"), ("Lineup + win chance", "Hungarian, bell curve"), ("How managers pick", "choice model")]),
    ("3 · Simulations", [("Draft simulation", "live pick suggestions"), ("Season simulation", "this tab")]),
    ("4 · Verification", [("Unit tests", "synthetic data"), ("Pre-registered tests", "7 questions, unseen seasons"), ("Assumption checks", "against real results")]),
    ("5 · Outputs", [("Private research app", "uses the league login"), ("Clean export", "first names, privacy scan"), ("This website", "no login, no ids")]),
])
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown(
        "- **Data.** ESPN's fantasy API (ten seasons of player stats, projections and draft positions; three seasons of league drafts and results) and MoneyPuck's "
        "advanced stats. Everything is cached so results can be reproduced.\n"
        "- **Models.** Player projections, lineup assignment and win chance, and a choice model of how managers pick.\n"
        "- **Simulations.** The draft simulation gave me live pick suggestions; the season simulation powers this tab."
    )
with c2:
    st.markdown(
        "- **Verification.** Unit tests on synthetic data, seven pre-registered paired tests on unseen seasons (with a Holm correction for running several), "
        "and assumption checks against the league's real results.\n"
        "- **Privacy.** The league login stays on my computer. This site is built from an exported snapshot with managers shown by first name only, "
        "and every published file is scanned for private data first.\n"
        "- **Tools.** Python, pandas, NumPy, SciPy, scikit-learn, Streamlit, Plotly."
    )
definitions([
    ("Monte Carlo simulation", "Repeating a random experiment many times and counting the outcomes."),
    ("Normal distribution", "The bell curve. Described by its mean (centre) and standard deviation (spread)."),
    ("Standard deviation (±)", "A typical distance from the average; about two thirds of weeks fall within ±1 of it."),
    ("Playoff odds", f"The share of simulated seasons in which a team finishes in the top {n_po}."),
    ("Common random numbers", "Giving every option the same random scenarios so comparisons between them are fair and precise."),
    ("Calibration", "Whether predicted chances match how often things really happen (70% predictions come true about 70% of the time)."),
    ("Pre-registered test", "A yes-or-no question written down before looking at the results, so the answer can't be tuned."),
    ("Holm correction", "A stricter p-value when several tests run at once, so one lucky result can't sneak through."),
])
