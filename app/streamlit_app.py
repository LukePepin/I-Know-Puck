"""I Know Puck: draft decision support.   Run:  streamlit run app/streamlit_app.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from scipy.special import softmax
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iknowpuck.config import RUNS_DIR, SLOT_NAMES, STAT_NAMES, load_credentials  # noqa: E402
from iknowpuck.data.espn import EspnClient, EspnError  # noqa: E402
from iknowpuck.draft import availability, predraft_plan, recommend  # noqa: E402
from iknowpuck.pipeline import build  # noqa: E402
from iknowpuck.summaries import PLAIN_MEANING, PLAIN_QUESTIONS, league_takeaways, manager_reports, ordinal, plain_answer, position_timing, verdict  # noqa: E402
from iknowpuck.valuation import WEEKS_IN_SEASON  # noqa: E402

# --- look and feel -----------------------------------------------------------------------------------
NAVY, RUST, SAGE, SAND, GREY = "#1F3A5F", "#A5452B", "#5E7D5B", "#C9A35B", "#8A8A8A"
PALETTE = [NAVY, RUST, SAGE, SAND, "#6B5B8C", GREY]
pio.templates["academic"] = go.layout.Template(
    layout=dict(
        font=dict(family="Georgia, 'Times New Roman', serif", size=13, color="#1B1B1B"),
        colorway=PALETTE,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, linecolor="#444", ticks="outside", zeroline=False),
        yaxis=dict(gridcolor="#E6E4DE", linecolor="#444", ticks="outside", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=20, t=50, b=50),
    )
)
pio.templates.default = "academic"

st.set_page_config(page_title="I Know Puck", layout="wide")
st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-family: Georgia, 'Times New Roman', serif; }
      h1, h2, h3 { font-weight: 600; letter-spacing: 0.2px; }
      .note { border-left: 3px solid #1F3A5F; background: #F4F3EF; padding: 0.6rem 0.9rem; margin: 0.4rem 0 1rem 0; font-size: 0.95rem; }
      .caption { color: #555; font-size: 0.88rem; }
      div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def note(text: str):
    st.markdown(f"<div class='note'>{text}</div>", unsafe_allow_html=True)


def fig_show(fig, height: int = 380):
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True)


# --- data ----------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Building projections and opponent models (first run takes about 2 minutes)...")
def get_bundle(season: int, refresh_token: int):
    return build(season, refresh=refresh_token > 0)


def load_runs(n: int = 2) -> list[dict]:
    """Most recent experiment-suite runs, newest first (used to show replication)."""
    paths = sorted(RUNS_DIR.glob("*_ikp/results.json"), reverse=True) if RUNS_DIR.exists() else []
    return [json.loads(p.read_text()) for p in paths[:n]]


creds = load_credentials()
st.session_state.setdefault("refresh", 0)
st.session_state.setdefault("picks", [])  # list[(team_id, player_id)]

with st.sidebar:
    st.markdown("### I Know Puck")
    st.markdown("<span class='caption'>Draft decision support for an ESPN H2H fantasy hockey league</span>", unsafe_allow_html=True)
    season = st.number_input("ESPN season (year the season ends)", value=creds.season, step=1)
    if st.button("Rebuild data and models", help="Re-download ESPN and MoneyPuck data and refit every model"):
        st.session_state.refresh += 1
        get_bundle.clear()

b = get_bundle(int(season), st.session_state.refresh)
S = b.settings
names = b.team_labels() or {t: f"Team {t}" for t in range(1, S.n_teams + 1)}

with st.sidebar:
    order_default = S.pick_order or list(names)
    order_txt = st.text_input("Round-1 pick order (team ids)", ",".join(map(str, order_default)))
    order = [int(x) for x in order_txt.split(",") if x.strip().isdigit()]
    my_team = st.selectbox("My team", list(names), index=list(names).index(S.my_team_id) if S.my_team_id in names else 0, format_func=lambda t: names[t])
    st.markdown(f"<span class='caption'>Draft slot {order.index(my_team) + 1 if my_team in order else '?'} of {len(order)} &middot; {S.rounds} rounds &middot; {S.scoring_type.replace('_', ' ').title()}</span>", unsafe_allow_html=True)
    st.divider()
    n_cand = st.slider("Candidates evaluated", 4, 20, 10)
    n_roll = st.slider("Simulations per candidate", 5, 100, 25, help="More simulations give tighter estimates but take longer")
    live = st.toggle("Live sync with ESPN draft", value=False, help="Polls the ESPN draft every 5 seconds")
    for n in b.notes:
        st.warning(n)

ctx = b.context(order, my_team)
pool = b.pool.frame
pid_to_idx = {int(p): i for i, p in enumerate(pool.player_id)}


def picks_as_idx() -> list[tuple[int, int]]:
    return [(t, pid_to_idx[p]) for t, p in st.session_state.picks if p in pid_to_idx]


def sync_espn():
    try:
        d = EspnClient(creds).draft(int(season), live=True)
    except EspnError as e:
        st.sidebar.error(str(e))
        return
    if len(d):
        st.session_state.picks = [(int(r.team_id), int(r.player_id)) for r in d.itertuples()]


tabs = st.tabs(["Walkthrough", "Draft room", "Pre-draft plan", "Players", "League and strategy", "Research", "Glossary"])
tab_walk, tab_room, tab_plan, tab_players, tab_league, tab_research, tab_gloss = tabs


# ======================================================================================================
# WALKTHROUGH (plain language first, technical details in expanders)
# ======================================================================================================
with tab_walk:
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
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Before the draft")
        st.markdown(
            f"1. **Check the left-hand panel.** \"My team\" should say **{me_name}**" + (f" and your draft slot should be **{slot}**." if slot else ".") + "\n"
            "   If ESPN changed the pick order, press **Rebuild data and models**.\n"
            "2. **Open the Pre-draft plan tab and press \"Simulate plan\".** Write down the two or three names it shows for each of your first few rounds.\n"
            "3. **Press \"Compute availability\"** on the same tab. It shows how likely each top player is to still be there at your first pick.\n"
            "4. **Read the League and strategy tab.** Note which managers grab goalies early; they will take goalies before you expect."
        )
    with c2:
        st.markdown("#### During the draft")
        st.markdown(
            "5. **Turn on \"Live sync with ESPN draft\"** in the left-hand panel. Picks will appear by themselves.\n"
            "6. **When the Draft room says \"You are on the clock\",** the app runs by itself. Read the green **Suggested pick** box and draft that player.\n"
            "7. **Want a second opinion?** Look at the bar chart. Bars in the same blue are tied, so any of them is a good pick.\n"
            "8. **If live sync stops working,** add each pick yourself with **Record a pick** (choose the player and the manager, then press Add).\n"
            "9. **Made a mistake?** Press **Undo last pick**."
        )
    st.markdown("#### Rules of thumb from your league's history")
    rules = ["Rounds 1 to 6: mostly forwards. Take a defenseman only if he is clearly the best player left."]
    if g_one:
        second = f"{round(g_two) - 2}" if g_two else "14"
        rules.append(f"First goalie: by about round {max(2, round(g_one) - 1)}. Second goalie: by about round {second}.")
    rules += ["Never reach far ahead of a player's ESPN ranking just because our projection likes him.",
              "Late rounds: take the best projected player who fits an empty spot on your roster."]
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
            st.dataframe(cf.round(3), hide_index=True, use_container_width=True)
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
    st.markdown(
        "We also drew a map of the managers where **people who draft in similar ways sit close together.** It shows two main styles in your league: "
        "managers who grab goalies early, and managers who follow the rankings and wait on goalies. The map is in the Detailed data section of the League and strategy tab."
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
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
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


# ======================================================================================================
# DRAFT ROOM
# ======================================================================================================
@st.fragment(run_every=5 if live else None)
def draft_room():
    if live:
        sync_espn()
    picks = picks_as_idx()
    n_done = len(st.session_state.picks)
    total = len(ctx.order)
    if n_done >= total:
        st.success("Draft complete.")
    on_clock = ctx.order[min(n_done, total - 1)]
    rnd = n_done // len(order) + 1
    until = next((d for d, t in enumerate(ctx.order[n_done:]) if t == my_team), None)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall pick", f"{n_done + 1} of {total}")
    c2.metric("Round", rnd)
    c3.metric("On the clock", names.get(on_clock, on_clock))
    c4.metric("Picks until mine", "Now" if until == 0 else (until if until is not None else "-"))

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Recommendation" + (": you are on the clock" if on_clock == my_team else f" for {names.get(on_clock)}"))
        if st.button("Run simulation", type="primary") or (live and on_clock == my_team):
            with st.spinner(f"Simulating {n_cand} candidates x {n_roll} drafts..."):
                rec = recommend(ctx, picks, n_candidates=n_cand, n_rollouts=n_roll)
            st.session_state.rec = (n_done, rec)
        if "rec" in st.session_state and st.session_state.rec[0] == n_done:
            rec = st.session_state.rec[1].head(12)
            tied = rec[rec.gap_to_best >= -2 * rec.gap_se.fillna(0)]
            pick_row = tied.sort_values(["p_avail_next_pick", "win_prob"], ascending=[True, False]).iloc[0]
            why = ("It gives the best chance of winning weeks." if len(tied) == 1 else
                   f"{len(tied)} players are tied for best; this one is the least likely to still be there at your next pick ({pick_row.p_avail_next_pick:.0%}).")
            st.success(f"Suggested pick: {pick_row['name']} ({pick_row.pos}). {why}")
            fig = go.Figure(go.Bar(
                x=rec.win_prob, y=rec.name + " (" + rec.pos + ")", orientation="h",
                error_x=dict(type="data", array=1.96 * rec.gap_se.fillna(0), color="#444"),
                marker_color=[NAVY if i == 0 else ("#7F93AD" if abs(g) <= 2 * s else "#C8CED8") for i, (g, s) in enumerate(zip(rec.gap_to_best, rec.gap_se.fillna(0)))],
            ))
            lo = float(rec.win_prob.min()) - 0.01
            fig.update_layout(title="Simulated P(win a week) if you draft each player now", xaxis=dict(range=[lo, float(rec.win_prob.max()) + 0.01], title="P(win a week)"), yaxis=dict(autorange="reversed"))
            fig_show(fig, 60 + 32 * len(rec))
            show = rec[["name", "pos", "proj_value", "adp", "market_rank", "win_prob", "gap_to_best", "gap_se", "p_best", "p_avail_next_pick"]].rename(columns={
                "proj_value": "proj pts", "market_rank": "market rank", "win_prob": "P(win week)", "gap_to_best": "gap to best",
                "gap_se": "gap SE", "p_best": "P(best)", "p_avail_next_pick": "available next pick"})
            st.dataframe(show.style.format({"proj pts": "{:.0f}", "adp": "{:.1f}", "P(win week)": "{:.3f}", "gap to best": "{:+.3f}", "gap SE": "{:.3f}", "P(best)": "{:.0%}", "available next pick": "{:.0%}"}), hide_index=True, use_container_width=True)
            note("How to read this: a longer bar means a better chance of winning weeks. The dark bar is the top estimate; mid-blue bars are tied with it. Among tied players, take the one least likely to be available at your next pick (last column).")
        st.markdown("#### Record a pick")
        taken_ids = {p for _, p in st.session_state.picks}
        avail = pool[~pool.player_id.isin(taken_ids)].sort_values("fpts", ascending=False)
        pc1, pc2, pc3 = st.columns([3, 2, 1])
        choice = pc1.selectbox("Player", avail.player_id.tolist(), format_func=lambda p: f"{pool.loc[pid_to_idx[p], 'name']} ({pool.loc[pid_to_idx[p], 'pos']}, {pool.loc[pid_to_idx[p], 'fpts']:.0f} pts)")
        team_pick = pc2.selectbox("Drafted by", list(names), index=list(names).index(on_clock) if on_clock in names else 0, format_func=lambda t: names[t])
        pc3.markdown("&nbsp;")
        if pc3.button("Add"):
            st.session_state.picks.append((team_pick, int(choice)))
            st.rerun()
        u1, u2 = st.columns(2)
        if u1.button("Undo last pick") and st.session_state.picks:
            st.session_state.picks.pop()
            st.rerun()
        if u2.button("Sync once from ESPN"):
            sync_espn()
            st.rerun()

    with right:
        taken, teams = ctx.initial_state(picks)
        mine = teams[my_team]
        st.markdown("#### My roster")
        rows = [{"player": pool.loc[j, "name"], "pos": pool.loc[j, "pos"], "proj pts": round(ctx.value[j])} for j in mine.roster]
        st.dataframe(pd.DataFrame(rows, columns=["player", "pos", "proj pts"]), hide_index=True, use_container_width=True)
        st.markdown("<span class='caption'>Open slots: " + ", ".join(f"{SLOT_NAMES[s]} {c}" for s, c in mine.open.items() if c > 0) + "</span>", unsafe_allow_html=True)
        wk = pd.DataFrame([{"manager": names.get(t, t), "picks": len(teams[t].roster), "weekly pts": ctx.val.weekly_points(teams[t].roster)} for t in order]).sort_values("weekly pts")
        fig = go.Figure(go.Bar(x=wk["weekly pts"], y=wk.manager, orientation="h", marker_color=[RUST if m == names[my_team] else NAVY for m in wk.manager]))
        fig.update_layout(title="Projected weekly points of rosters so far", xaxis_title="expected weekly fantasy points")
        fig_show(fig, 420)

    with st.expander("Pick log"):
        log = pd.DataFrame([{"pick": i + 1, "manager": names.get(t, t), "player": pool.loc[pid_to_idx[p], "name"] if p in pid_to_idx else p} for i, (t, p) in enumerate(st.session_state.picks)])
        st.dataframe(log, hide_index=True, use_container_width=True)


with tab_room:
    draft_room()


# ======================================================================================================
# PRE-DRAFT PLAN
# ======================================================================================================
with tab_plan:
    st.markdown("## Pre-draft plan")
    st.markdown("Full drafts are simulated from your slot: opponents follow the fitted pick model and you follow the market-anchored policy. The result is a realistic picture of who tends to be available to you, round by round.")
    n_sims = st.slider("Simulated drafts", 20, 400, 100)
    if st.button("Simulate plan"):
        with st.spinner("Simulating full drafts..."):
            st.session_state.plan = predraft_plan(ctx, n_sims=n_sims)
    if "plan" in st.session_state:
        targets, summary = st.session_state.plan
        c1, c2 = st.columns(2)
        c1.metric("Mean P(win a week)", f"{summary.win_prob.mean():.3f}", f"95% CI +/- {1.96 * summary.win_prob.std() / np.sqrt(len(summary)):.3f}", delta_color="off")
        c2.metric("Mean weekly points", f"{summary.weekly_points.mean():.1f}")
        fig = go.Figure(go.Histogram(x=summary.win_prob, nbinsx=25, marker_color=NAVY))
        fig.update_layout(title="Distribution of final P(win a week) across simulated drafts", xaxis_title="P(win a week)", yaxis_title="drafts")
        fig_show(fig, 300)
        pos_mix = targets.groupby(["round", "pos"])["share"].sum().unstack(fill_value=0)
        fig = go.Figure()
        for i, p in enumerate([c for c in ["C", "LW", "RW", "D", "G"] if c in pos_mix]):
            fig.add_bar(x=pos_mix.index, y=pos_mix[p], name=p, marker_color=PALETTE[i % len(PALETTE)])
        fig.update_layout(barmode="stack", title="Position of the most common targets by round", xaxis_title="your round", yaxis_title="share of simulations (top 3 targets)")
        fig_show(fig, 320)
        st.dataframe(targets.style.format({"adp": "{:.1f}", "proj_value": "{:.0f}", "share": "{:.0%}"}), hide_index=True, use_container_width=True)
    st.markdown("#### Chance each player is still available at your first pick")
    if st.button("Compute availability"):
        a = availability(ctx, picks_as_idx(), n_rollouts=80)
        df = pool.assign(p_avail=a)[["name", "pos", "adp", "fpts", "p_avail"]].sort_values("adp").head(40)
        fig = go.Figure(go.Bar(x=df.name, y=df.p_avail, marker_color=NAVY))
        fig.update_layout(title="P(available at my next pick), top 40 by ADP", yaxis=dict(range=[0, 1], title="probability"), xaxis_tickangle=-60)
        fig_show(fig, 380)


# ======================================================================================================
# PLAYERS
# ======================================================================================================
with tab_players:
    st.markdown("## Player board")
    pos_f = st.multiselect("Positions", ["C", "LW", "RW", "D", "G"], default=["C", "LW", "RW", "D", "G"])
    hide_taken = st.checkbox("Hide drafted players", value=True)
    board = pool[pool.pos.isin(pos_f)].copy()
    if hide_taken:
        board = board[~board.player_id.isin({p for _, p in st.session_state.picks})]
    board["rank_gap"] = board["fpts"].rank(ascending=False) - board["adp"].rank()
    top = board.nsmallest(150, "adp")
    fig = go.Figure()
    for i, p in enumerate(["C", "LW", "RW", "D", "G"]):
        d = top[top.pos == p]
        fig.add_scatter(x=d.adp, y=d.fpts, mode="markers", name=p, text=d.name, marker=dict(size=7, color=PALETTE[i % len(PALETTE)]), hovertemplate="%{text}<br>ADP %{x:.1f}<br>%{y:.0f} pts")
    fig.update_layout(title="Market-adjusted projection vs ADP (top 150 by ADP)", xaxis_title="ADP (earlier is left)", yaxis_title="projected fantasy points")
    fig_show(fig, 420)
    note("Points high and to the right are value later in the draft. The projection already includes the market adjustment, so remaining gaps are modest by design.")
    cols = ["name", "pos", "adp", "fpts", "fpts_model", "fpts_espn", "fpts_own", "p_30", "rank_gap", "injury_status"]
    st.dataframe(
        board.sort_values("fpts", ascending=False)[cols].rename(columns={"fpts": "final pts", "fpts_model": "blend pts", "fpts_espn": "ESPN pts", "fpts_own": "own pts", "p_30": "GP", "rank_gap": "rank gap vs ADP"}).round(1),
        hide_index=True, use_container_width=True, height=520,
    )


# ======================================================================================================
# LEAGUE AND STRATEGY
# ======================================================================================================
with tab_league:
    mgr_names = b.manager_names
    active = set(b.managers.loc[b.managers.active, "owner_id"].astype(str)) if len(b.managers) else set()
    my_owner = b.manager_of_team.get(my_team)

    def who(m):
        n_ = mgr_names.get(m, m)
        return n_ if m in active else f"{n_} (former)"

    st.markdown("## Your league: who wins, and how")
    tk = league_takeaways(b.strategy)
    if tk:
        st.markdown("### The big picture")
        st.markdown("We looked at every team from the last three seasons and compared the **top four finishers** with **everyone else**.")
        comp = pd.DataFrame({
            "": ["Weekly win rate", "Draft points above expected, per season", "Round of first goalie", "Share of defensemen in rounds 1 to 6"],
            "Top four finishers": [f"{tk['top_win']:.0%}", f"{tk['top_value']:+.0f}", f"{tk['top_goalie_round']:.1f}", f"{tk['top_d_share']:.0%}"],
            "Everyone else": [f"{tk['rest_win']:.0%}", f"{tk['rest_value']:+.0f}", f"{tk['rest_goalie_round']:.1f}", f"{tk['rest_d_share']:.0%}"],
        })
        st.dataframe(comp, hide_index=True, use_container_width=True)
        st.markdown(
            f"- **The top teams drafted better.** Their picks scored about {tk['top_value'] - tk['rest_value']:.0f} more points per season than the other teams' picks, compared with what those draft spots normally produce.\n"
            f"- **The top teams took their first goalie a little earlier:** around round {tk['top_goalie_round']:.0f}, compared with round {tk['rest_goalie_round']:.0f}.\n"
            "- **Reaching did not help anyone.** Winners and losers took players ahead of their ranking about equally often.\n"
            "- **Loading up on defensemen in rounds 1 to 6 went with fewer wins** across the league."
        )
        note("These patterns come from 36 team-seasons. Treat them as strong hints, not guarantees.")

    reports = manager_reports(b.strategy, mgr_names, active)
    mine_r = next((r for r in reports if r.owner_id == str(my_owner)), None)
    if mine_r:
        st.markdown(f"### Your review: {mine_r.name}")
        st.markdown(f"**Finishes:** {mine_r.finishes}. **Average win rate:** {mine_r.win_pct:.0%}.")
        for x in mine_r.bullets:
            st.markdown(f"- {x}")
        mine_rows = b.strategy[b.strategy.owner_id == my_owner].sort_values("season")
        st.dataframe(pd.DataFrame({
            "Season": mine_rows.season.astype(str), "Finish": mine_rows.final_rank.map(ordinal),
            "Weekly win rate": mine_rows.win_pct.map(lambda x: f"{x:.0%}"),
            "Draft points above expected": mine_rows.value_added_all.map(lambda x: f"{x:+.0f}"),
            "First goalie round": mine_rows.first_goalie_round.astype(int),
        }), hide_index=True, use_container_width=True)
        league_median_value = b.strategy.groupby("owner_id").value_added_all.mean().median()
        if mine_rows.value_added_all.mean() >= league_median_value and mine_r.avg_finish > 6.5:
            note("<b>The key insight:</b> your drafts have been better than most of the league's, but your finishes have been lower than your drafts suggest. "
                 "That points to what happens after the draft (weekly lineups, pickups, injuries and luck), not the draft itself. "
                 "The draft tool keeps your edge; staying active on the waiver wire and setting daily lineups is where the extra wins are.")
        advice = []
        if tk and mine_rows.first_goalie_round.mean() > tk["top_goalie_round"] + 0.5:
            advice.append(f"You took your first goalie later (round {mine_rows.first_goalie_round.mean():.0f} on average) than the top teams (round {tk['top_goalie_round']:.0f}). Aim a little earlier.")
        if tk and mine_rows.d_share_r1_6.mean() > tk["top_d_share"] + 0.03:
            advice.append("You took more defensemen early than the top teams did. Lean toward forwards in rounds 1 to 6.")
        if tk and mine_rows.value_added_all.mean() < tk["top_value"]:
            advice.append("Your picks have not beaten their draft spots by as much as the top teams' picks have. That is exactly the gap this app is built to close: better guesses and no reaching.")
        if advice:
            st.markdown("**What to change this year:**")
            for a in advice:
                st.markdown(f"- {a}")

    st.markdown("### Scouting reports")
    st.markdown("One card for each manager in this year's league, best average finish first.")
    act = [r for r in reports if r.active and r.owner_id != str(my_owner)]
    cols = st.columns(2)
    for i, r in enumerate(act):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{r.name}**: {r.headline.lower()}")
                st.markdown(f"<span class='caption'>Finishes: {r.finishes} &middot; win rate {r.win_pct:.0%}</span>", unsafe_allow_html=True)
                for x in r.bullets:
                    st.markdown(f"- {x}")
                st.markdown(f"*{r.sunday_tip}*")
    former = [r for r in reports if not r.active]
    if former:
        with st.expander("Former managers (their drafts still help the app learn the league's habits)"):
            for r in former:
                st.markdown(f"**{r.name}**: finishes {r.finishes}. " + " ".join(r.bullets))

    with st.expander("Detailed data and charts"):
        if len(b.managers):
            md = b.managers.copy()
            md["team slot by season"] = md["team_by_season"].map(lambda d: ", ".join(f"{s}: #{t}" for s, t in sorted(d.items())))
            md["picks modelled"] = md["owner_id"].map(b.drafts.groupby("owner_id").size()).fillna(0).astype(int)
            md["status"] = np.where(md.active, "active", "left league")
            st.dataframe(md[["manager", "status", "team slot by season", "picks modelled"]].sort_values(["status", "manager"]), hide_index=True, use_container_width=True)
            st.markdown("<span class='caption'>Managers are tracked by ESPN account, not team slot, so history follows the person when slots change hands.</span>", unsafe_allow_html=True)
        if len(b.strategy):
            ms = b.strategy.copy()
            ms["manager"] = ms["owner_id"].map(who)
            fig = go.Figure()
            fig.add_scatter(x=ms.value_added_all, y=ms.win_pct, mode="markers", text=ms.manager + " " + ms.season.astype(str),
                            marker=dict(size=9, color=[RUST if o == my_owner else NAVY for o in ms.owner_id]), hovertemplate="%{text}<br>draft points above expected %{x:.0f}<br>win rate %{y:.2f}")
            fig.update_layout(title="Better drafts, more wins (each dot is one team in one season; yours are rust)", xaxis_title="draft points above what those draft spots usually produce", yaxis_title="win rate")
            fig_show(fig, 400)
            cor = b.strategy_cor
            wp = cor[cor.outcome == "Win %"].sort_values("rho")
            fig = go.Figure()
            for _, r in wp.iterrows():
                col = SAGE if r.ci_low > 0 else (RUST if r.ci_high < 0 else GREY)
                fig.add_scatter(x=[r.ci_low, r.ci_high], y=[r.strategy] * 2, mode="lines", line=dict(color=col, width=3), showlegend=False)
                fig.add_scatter(x=[r.rho], y=[r.strategy], mode="markers", marker=dict(color=col, size=10), showlegend=False)
            fig.add_vline(x=0, line_color="#444")
            fig.update_layout(title="Draft habits vs win rate (Spearman correlation, 95% bootstrap interval)", xaxis_title="rank correlation")
            fig_show(fig, 400)
            st.dataframe(ms[["season", "manager", "final_rank", "win_pct", "points_for", "first_goalie_round", "goalies_r1_3", "d_share_r1_6", "reach_early", "value_added_all"]].sort_values(["season", "final_rank"]).round(2), hide_index=True, use_container_width=True)
            st.dataframe(cor.round(3), hide_index=True, use_container_width=True)
        if b.spectral is not None:
            tbl = b.spectral.table().reset_index()
            tbl["who"] = tbl["manager"].map(who)
            fig = go.Figure()
            for i, cl in enumerate(sorted(tbl.cluster.unique())):
                d = tbl[tbl.cluster == cl]
                fig.add_scatter(x=d.x, y=d.y, mode="markers+text", text=d.who, textposition="top center", name=f"group {cl + 1}", marker=dict(size=11, color=PALETTE[i]))
            fig.update_layout(title="Map of managers: similar drafters sit close together", xaxis_title="Laplacian eigenvector 2 (Fiedler)", yaxis_title="eigenvector 3")
            fig_show(fig, 480)
            desc = b.opp_model.describe()
            desc["who"] = desc["manager"].map(lambda m: who(m) if m != "(league)" else "League average")
            st.dataframe(desc.drop(columns="manager").set_index("who").round(2), use_container_width=True)
            st.markdown("<span class='caption'>adp_sensitivity: how strictly a manager follows ADP. Position biases are relative to forwards in each round phase; positive means that position is taken earlier than the league does.</span>", unsafe_allow_html=True)


# ======================================================================================================
# RESEARCH
# ======================================================================================================
with tab_research:
    st.markdown("## Research record")
    runs = sorted(RUNS_DIR.glob("*/results.json"), reverse=True) if RUNS_DIR.exists() else []
    if not runs:
        st.info("No experiment runs yet. Run: python -m iknowpuck.experiments")
    for r in runs[:5]:
        payload = json.loads(r.read_text())
        with st.expander(f"Run {r.parent.name} (git {payload['meta']['git_commit']})", expanded=r == runs[0]):
            st.markdown((r.parent / "report.md").read_text())
    st.markdown("### Projection blend weights (share on our model, per statistic)")
    bw = pd.DataFrame({"stat": [STAT_NAMES.get(k, k) for k in b.blend_weights], "weight on own model": list(b.blend_weights.values())})
    fig = go.Figure(go.Bar(x=bw.stat, y=bw["weight on own model"], marker_color=NAVY))
    fig.add_hline(y=0.5, line_dash="dash", line_color=GREY)
    fig.update_layout(title="Blend weights estimated out of sample", yaxis=dict(range=[0, 1], title="weight on own model"))
    fig_show(fig, 320)
    st.markdown("Full methodology: `docs/METHODOLOGY.md` in the repository.")


# ======================================================================================================
# GLOSSARY
# ======================================================================================================
with tab_gloss:
    st.markdown("## Glossary")
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
    ]
    for t, d in terms:
        st.markdown(f"**{t}.** {d}")
