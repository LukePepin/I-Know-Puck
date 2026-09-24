"""Walkthrough: plain-language guide, game plan, how it works, evidence, glossary."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.special import softmax
from scipy.stats import norm
from ui import GREY, NAVY, RUST, SAGE, app_state, fig_show, load_runs, note

from iknowpuck.summaries import PLAIN_MEANING, PLAIN_QUESTIONS, plain_answer, position_timing, verdict

A = app_state()
b, S, names, order, my_team, ctx = A["b"], A["S"], A["names"], A["order"], A["my_team"], A["ctx"]
pool = A["pool"].frame  # player table (injury-adjusted)

me_name = names.get(my_team, "you")
slot = order.index(my_team) + 1 if my_team in order else None
timing = position_timing(b.drafts, S.n_teams)
n = S.n_teams
g_half, g_one, g_two = timing.get(f"G{n // 2}"), timing.get(f"G{n}"), timing.get(f"G{2 * n}")
pe = b.proj_eval

st.markdown("## Your draft, explained simply")
st.markdown(
    f"This app helps **{me_name}** make the best pick every time it is your turn. "
    "It looks at years of hockey stats and at how the people in your league have drafted before, "
    "then plays out the rest of the draft many times to see which pick gives you the best chance "
    "of winning each week."
)

# --- the short version -------------------------------------------------------------------------
st.markdown("### The short version")
short = [
    "**Follow the rankings, but pick smart inside them.** The app suggests players close to where ESPN ranks them. Taking players much earlier than their ranking lost in our draft tests and never helped anyone in your league.",
    "**Take forwards early.** In your league, teams that took lots of defensemen in the first six rounds won less often.",
]
if g_one:
    short.append(f"**Don't wait too long for a goalie.** In your league, about half the starting goalies are gone by round {g_half:.0f}, and one goalie per team is gone by round {g_one:.0f}.")
short += [
    "**Good drafting really matters here.** Managers whose picks beat their draft spots finished near the top.",
    "**When the app shows a tie, take the player least likely to come back to you.**",
]
for s_ in short:
    st.markdown(f"- {s_}")

# --- action checklist --------------------------------------------------------------------------
st.markdown("### Your game plan, step by step")
st.caption("The pages in the left-hand menu follow this order.")
c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.markdown("#### Before the draft")
        st.markdown(
            f"1. **Check the left-hand panel.** \"My team\" should say **{me_name}**" + (f" and your slot should be **{slot}**." if slot else ".") + " "
            "If ESPN changed the pick order, press **Rebuild data and models**.\n"
            "2. **Read League history.** See who grabs goalies early, who reaches, and which NHL teams people love.\n"
            "3. **Open Pre-draft plan and check the injury list.** Change the games-missed number for anyone you have news about.\n"
            "4. **Press \"Simulate plan\".** Write down the two or three names it shows for each of your first rounds."
        )
with c2:
    with st.container(border=True):
        st.markdown("#### During the draft")
        st.markdown(
            "5. **Open Draft room and turn on \"Live sync with ESPN draft\".** Picks appear by themselves.\n"
            "6. **When it says \"You are on the clock\",** read the green **Suggested pick** box and draft that player.\n"
            "7. **Tied bars** (same blue) are equally good picks.\n"
            "8. **If live sync stops,** add picks yourself with **Record a pick**. **Undo last pick** fixes mistakes."
        )
with c3:
    with st.container(border=True):
        st.markdown("#### After the draft")
        st.markdown(
            "9. **Open In-season moves once a week.** It lists the free agents who would raise your weekly points most, and who to drop.\n"
            "10. **Stay active on the waiver wire.** In your league, managers who made more pickups won more, even after accounting for how well they drafted.\n"
            "11. **Re-check injuries.** Lower the games-missed numbers as players return."
        )
st.markdown("#### Rules of thumb from your league's history")
rules = ["Rounds 1 to 6: mostly forwards. Take a defenseman only if he is clearly the best player left."]
if g_one:
    second = f"{round(g_two) - 2}" if g_two else "14"
    rules.append(f"First goalie: by about round {max(2, round(g_one) - 1)}. Second goalie: by about round {second}.")
rules += [
    "Never reach far ahead of a player's ESPN ranking just because our projection likes him.",
    "Injured stars: the app lowers their value by the games they are expected to miss. If you know better, change the number on the Pre-draft plan page.",
    "Late rounds: take the best projected player who fits an empty spot on your roster.",
]
for r_ in rules:
    st.markdown(f"- {r_}")

# --- how it works ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("### How it works, in seven simple steps")

st.markdown("#### Step 1. What wins in your league?")
c1, c2 = st.columns([3, 2])
with c1:
    w = pd.DataFrame([(c.name, c.points) for c in S.scoring_categories], columns=["stat", "points"]).sort_values("points")
    fig = go.Figure(go.Bar(x=w.points, y=w.stat, orientation="h", marker_color=[RUST if p < 0 else NAVY for p in w.points]))
    fig.update_layout(title="Points you get for each stat", xaxis_title="fantasy points", yaxis_title="")
    fig_show(fig, 340)
with c2:
    st.markdown(
        "Every week you play one other team. **Whoever scores more fantasy points wins that week.** "
        "A goal is worth 2 points, an assist 1, a goalie win 4, and every goal a goalie allows costs 2 points (the red bar). "
        f"There are {n} teams and {S.rounds} rounds, and the pick order reverses every round."
    )
    note("<b>What the app aims for:</b> the team with the best chance of winning each week, not just the most total points.")

st.markdown("#### Step 2. Guessing how good each player will be")
if len(pe):
    srcs = [("espn", "ESPN's guess"), ("market_adj", "Our guess")]
    mae = pe.groupby("season").apply(lambda g: pd.Series({lab: (g[c] - g.actual).abs().mean() for c, lab in srcs})).reset_index()
    miss_ours = (pe.market_adj - pe.actual).abs().mean()
    miss_espn = (pe.espn - pe.actual).abs().mean()
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        for i, (_, lab) in enumerate(srcs):
            fig.add_bar(x=mae.season.astype(str), y=mae[lab], name=lab, marker_color=[GREY, NAVY][i])
        fig.update_layout(barmode="group", title="How far off each guess was (smaller is better)", xaxis_title="season", yaxis_title="average miss (fantasy points)")
        fig_show(fig, 330)
    with c2:
        st.markdown(
            "Before a season starts, nobody knows exactly how many points a player will score. ESPN makes a guess. "
            "We make our own from each player's last three seasons plus advanced stats. "
            f"Then we checked both against what really happened. **Our guess missed by about {miss_ours:.0f} points per player; "
            f"ESPN's missed by about {miss_espn:.0f}.**"
        )
        note("<b>Why it matters:</b> better guesses mean better picks, especially when two players look alike.")
    with st.expander("Technical details"):
        st.markdown(
            "Own model (Marcel+): recency-weighted per-game rates (5/4/3 by games played) shrunk toward the position mean with a "
            "per-stat constant fit on past seasons; a ridge regression on MoneyPuck features (xG minus goals, power-play TOI, TOI, "
            "shot attempts) corrects the residual; games played is a linear model on the previous two seasons. The blend weights "
            "own vs ESPN per statistic by least squares on out-of-sample predictions, then the market adjustment in step 3 is applied. "
            "The chart uses held-out seasons only."
        )

st.markdown("#### Step 3. Don't trust our guesses too much")
st.markdown(
    "Here is a trap: when our guess says a player is great but ESPN ranks him low, **we are usually the ones who are wrong.** "
    "ESPN's rankings reflect things our numbers can't see, like injuries or a player losing his spot on the power play. "
    "So the app **mixes our guess with the ranking** before it recommends anyone."
)
if b.market is not None and b.market.coef_:
    with st.expander("Technical details (the winner's curse)"):
        cf = pd.DataFrame([{"group": g, "weight on projection": c[1], "weight on log ADP": c[2], "R squared": b.market.r2_[g]} for g, c in b.market.coef_.items()])
        st.dataframe(cf.round(3), hide_index=True)
        st.markdown(
            "Choosing the maximum of many noisy estimates selects estimates that are too high (the winner's curse). We regress actual "
            "fantasy points on our projection and log(ADP) separately for forwards, defense and goalies, using past seasons, and use the "
            "fitted value. A projection weight below 1 means shrinkage; goalies are shrunk most because their projections are least reliable."
        )

st.markdown("#### Step 4. Turning a team into a chance of winning")
_, _, demo = ctx.rollout(*ctx.initial_state([]), 0, None, np.random.default_rng(3))
mu_sd = {t: ctx.val.team_dist(demo[t].roster) for t in ctx.teams}
ranked = sorted(ctx.teams, key=lambda t: mu_sd[t][0][0])
(m1, v1), (m2, v2) = mu_sd[ranked[-1]], mu_sd[ranked[len(ranked) // 2]]
sd_max = np.sqrt(max(v1[0], v2[0]))
xs = np.linspace(min(m1[0], m2[0]) - 4 * sd_max, max(m1[0], m2[0]) + 4 * sd_max, 300)
p_demo = norm.cdf((m1[0] - m2[0]) / np.sqrt(v1[0] + v2[0]))
c1, c2 = st.columns([3, 2])
with c1:
    fig = go.Figure()
    fig.add_scatter(x=xs, y=norm.pdf(xs, m1[0], np.sqrt(v1[0])), name="A strong team", line=dict(color=NAVY), fill="tozeroy", fillcolor="rgba(31,58,95,0.15)")
    fig.add_scatter(x=xs, y=norm.pdf(xs, m2[0], np.sqrt(v2[0])), name="An average team", line=dict(color=RUST), fill="tozeroy", fillcolor="rgba(165,69,43,0.12)")
    fig.update_layout(title="How many points each team might score in a week", xaxis_title="weekly fantasy points", yaxis_title="how likely")
    fig_show(fig, 320)
with c2:
    st.markdown(
        "A team does not score the same number of points every week. Some weeks are lucky and some are not. "
        "The curves show the range of weekly scores for a strong team and an average team. "
        f"Because the curves overlap, **the strong team wins about {p_demo:.0%} of the time, not every time.** "
        "The app judges every possible pick by how much it raises this chance."
    )
with st.expander("Technical details"):
    st.markdown(
        "Players are slotted by the Hungarian algorithm (starters count fully, bench about one third because daily lineups let them fill idle days). "
        "Weekly team points are modelled as Normal: the mean is expected weekly points and the variance comes from over-dispersed Poisson variation in each scoring stat, "
        "inflated for correlation between stats. P(win) = Phi((mu_me - mu_opp) / sqrt(var_me + var_opp)), averaged over the other rosters."
    )

st.markdown("#### Step 5. Guessing what everyone else will pick")
opp = b.sim_opp or b.opp_model
top = ctx.adp_order[:25]
probs = softmax(opp.utilities(None, 1, ctx.neg_log_adp[top], ctx.grp[top], np.ones(len(top))))
c1, c2 = st.columns([3, 2])
with c1:
    fig = go.Figure(go.Bar(x=[pool.loc[j, "name"] for j in top], y=probs, marker_color=NAVY))
    fig.update_layout(title="Chance each player is the very first pick (top 25 by ESPN ranking)", yaxis_title="chance", xaxis_tickangle=-60)
    fig_show(fig, 360)
with c2:
    st.markdown(
        "We studied every pick your league made in its last three drafts. **People in your league mostly follow ESPN's rankings, but not always.** "
        "So the app can't know exactly who will be gone before your next turn, but it can give good odds. "
        "That is where the \"available next pick\" percentage in the Draft room comes from."
    )
with st.expander("Technical details"):
    st.markdown(
        f"Conditional logit fitted by maximum likelihood on every historical pick: utility = b_adp * (-log ADP) + b_need * positional need + position-by-round effects. "
        f"Fitted market sensitivity b_adp = {opp.global_[0]:.1f}. Per-manager versions did not predict held-out drafts better (test H3), so the simulator uses the league-wide model."
    )

st.markdown("#### Step 6. Playing out the draft many times")
st.markdown(
    "When it is your turn, the app takes each good candidate and **pretends you picked him.** Then it plays out the rest of the draft "
    "many times: the other managers pick the way they usually do, and you keep picking sensibly. At the end it checks how often your finished "
    "team would win a week. **The candidate with the highest winning chance becomes the suggestion.** It works like a weather forecast: "
    "many possible futures, summed up as one clear answer."
)
with st.expander("Technical details"):
    st.markdown(
        "Monte Carlo rollouts with common random numbers (every candidate sees the same simulated futures, so the comparison between them is precise). "
        "Inside each rollout your later picks use the market-anchored policy: the highest projected value among the next two roster-fitting players by ADP. "
        "The Draft room reports each candidate's gap to the best with its paired standard error; gaps within about two standard errors count as ties."
    )

st.markdown("#### Step 7. Grouping the managers")
if b.cluster_names:
    groups = "; ".join(f"**{v['name']}** ({', '.join(v['traits'][:2])})" for v in b.cluster_names.values())
    st.markdown(
        "We drew a map of the managers where **people who draft in similar ways sit close together**, then let the computer find the groups. "
        f"Your league has {len(b.cluster_names)} styles: {groups}. Each manager's group is on the League history page."
    )
if b.spectral is not None:
    with st.expander("Technical details (graph spectral analysis)"):
        ev = b.spectral.eigenvalues[: min(8, len(b.spectral.eigenvalues))]
        fig = go.Figure(go.Scatter(x=list(range(1, len(ev) + 1)), y=ev, mode="lines+markers", line=dict(color=NAVY)))
        fig.add_vline(x=b.spectral.k + 0.5, line_dash="dash", line_color=RUST)
        fig.update_layout(title=f"Laplacian eigenvalues: largest gap after k = {b.spectral.k}", xaxis_title="index", yaxis_title="eigenvalue")
        fig_show(fig, 300)
        st.markdown(
            "Managers are nodes of a similarity graph (Gaussian kernel on standardised draft-behaviour features). The normalised Laplacian's smallest "
            "eigenvectors reveal clusters; k is chosen at the largest eigengap and k-means runs on the row-normalised eigenvectors. Pooling by cluster "
            "did not improve pick prediction (test H4), so the groups are descriptive."
        )

# --- scorecard ---------------------------------------------------------------------------------
st.markdown("---")
st.markdown("### Did we check that it actually works?")
st.markdown(
    "Yes. Each idea was written down as a question **before** testing it, then checked on past seasons the models had never seen. "
    "The whole set of tests was run twice with different random numbers to make sure the answers hold up."
)
runs2 = load_runs(2)
if not runs2:
    st.info("No test results yet. Run: python -m iknowpuck.experiments")
else:
    rows = []
    for r in runs2[0]["results"]:
        vs = [verdict(next(x["test"] for x in run_["results"] if x["id"] == r["id"])) for run_ in runs2 if any(x["id"] == r["id"] for x in run_["results"])]
        v = vs[0]
        rows.append({
            "Question": PLAIN_QUESTIONS.get(r["id"], r["hypothesis"]),
            "Answer": plain_answer(v),
            "Same answer in every run": f"{sum(x == v for x in vs)} of {len(vs)}",
            "What it means for you": PLAIN_MEANING.get((r["id"], v), ""),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    note("<b>Bottom line:</b> our player guesses are clearly better than ESPN's. When it comes to drafting, the app does about as well as following the rankings, and on top of that it checks roster fit and tells you who is likely to still be there. Ideas that failed the tests were removed.")
    with st.expander("Technical details (effect sizes and p-values)"):
        res = pd.DataFrame([{**x["test"], "id": x["id"], "label": f"{x['id']}: {x['treatment']} vs {x['baseline']}"} for x in runs2[0]["results"]])
        res["sd"] = res["mean_diff"] / res["cohens_dz"].replace(0, np.nan)
        res["dz_lo"], res["dz_hi"] = res["ci_low"] / res["sd"], res["ci_high"] / res["sd"]
        fig = go.Figure()
        for _, r in res.iterrows():
            col = SAGE if r["significant"] else (RUST if r["ci_high"] < 0 else GREY)
            fig.add_scatter(x=[r["dz_lo"], r["dz_hi"]], y=[r["label"]] * 2, mode="lines", line=dict(color=col, width=3), showlegend=False)
            fig.add_scatter(x=[r["cohens_dz"]], y=[r["label"]], mode="markers", marker=dict(color=col, size=11), showlegend=False)
        fig.add_vline(x=0, line_color="#444")
        fig.update_layout(title="Cohen's d_z with 95% bootstrap intervals (right of zero favours the new idea)", xaxis_title="standardised paired effect")
        fig_show(fig, 60 + 70 * len(res))
        for x in runs2[0]["results"]:
            t = x["test"]
            st.markdown(f"- **{x['id']}**: mean paired difference {t['mean_diff']:+.3f}, 95% CI [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}], n = {t['n']}, Holm-adjusted p = {t['p_adjusted']:.3f}.")



st.markdown("---")
st.markdown("### Glossary")
st.markdown("Plain meanings first. The technical term is in brackets for anyone who wants to look it up.")
terms = [
    ("ESPN ranking / ADP", "Where a player usually gets picked in ESPN drafts. ADP stands for average draft position. It is the crowd's opinion of each player."),
    ("Market rank", "Among players still available, who the crowd expects to go next. Rank 1 is the player most people would take next."),
    ("Projection", "Our best guess of how many fantasy points a player will score this season."),
    ("P(win a week)", "The chance your team beats an opponent in a weekly matchup. The app tries to make this as high as possible."),
    ("Available next pick", "The chance a player will still be there when it is your turn again."),
    ("Tie", "Two picks are tied when the difference between them is smaller than the app's measurement error. Either one is fine."),
    ("Reaching", "Taking a player much earlier than the crowd would. It has not paid off in your league."),
    ("Draft points above expected", "How many more (or fewer) points a manager's picks scored than players taken at the same draft spots usually score."),
    ("Simulation", "Playing out the rest of the draft on the computer many times to see what usually happens. [Monte Carlo rollout]"),
    ("Same simulated futures", "Every candidate is tested against the same set of simulated futures, so the comparison is fair. [common random numbers]"),
    ("Winner's curse", "If you always pick the player our model likes most compared with the crowd, you tend to pick the players our model got wrong."),
    ("Mixing with the ranking", "Blending our projection with the crowd's ranking to avoid the winner's curse. [market adjustment: regression on projection and log ADP]"),
    ("Pick model", "A formula that gives each available player a chance of being picked next, based on ranking and team needs. [conditional logit]"),
    ("Manager map", "A picture where managers who draft alike sit close together. [graph spectral clustering: Laplacian eigenvectors, eigengap]"),
    ("Tested on unseen seasons", "Checking a model only on seasons it was not built from. This is the only fair test. [out-of-sample, leave-one-season-out]"),
    ("Confidence interval", "The range the true answer most likely falls in. If the whole range is above zero, the idea clearly helps. [95% bootstrap CI]"),
    ("Statistically significant", "Very unlikely to be luck, even after allowing for running several tests at once. [permutation test with Holm correction]"),
    ("Effect size", "How big a difference is on a common scale: 0.2 is small, 0.5 medium, 0.8 large. [Cohen's d_z]"),
    ("Correlation", "How strongly two things move together, from -1 to +1. [Spearman rank correlation]"),
    ("VONA", "Value over next available: how much better a player is than the best similar player likely to be left at your next pick."),
    ("Draft skill", "How many more points a manager's picks scored than those draft spots usually produce, if every team had average injury luck."),
    ("Injury luck", "Points a manager did NOT lose to injuries compared with an average team. Positive means luckier than average."),
    ("Absence", "Three or more games in a row that a player's NHL team played without him (usually an injury)."),
    ("Stacking", "Drafting several players from the same NHL team. It makes your weekly score swing with that one team."),
    ("Style group", "A group of managers who draft in similar ways, found by the manager map. The name describes most members, not every one."),
    ("Pickup percentile", "How a manager's number of waiver pickups ranks against the rest of the league in the same season (100% = most active)."),
]
for t, d in terms:
    st.markdown(f"**{t}.** {d}")

