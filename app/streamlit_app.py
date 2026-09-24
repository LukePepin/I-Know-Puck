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


def latest_run() -> dict | None:
    runs = sorted(RUNS_DIR.glob("*/results.json"), reverse=True) if RUNS_DIR.exists() else []
    return json.loads(runs[0].read_text()) if runs else None


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
run = latest_run()


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
# WALKTHROUGH
# ======================================================================================================
with tab_walk:
    st.markdown("## How the recommendation is built")
    st.markdown(
        "This walkthrough follows the pipeline end to end: what data goes in, how each model works, how it was "
        "tested, and what the evidence says you should do on draft day. Every chart below is computed from "
        "your league's data."
    )

    # 1. league -------------------------------------------------------------------------------------------
    st.markdown("### 1. The decision problem")
    c1, c2 = st.columns([3, 2])
    with c1:
        w = pd.DataFrame([(c.name, c.points) for c in S.scoring_categories], columns=["stat", "points"]).sort_values("points")
        fig = go.Figure(go.Bar(x=w.points, y=w.stat, orientation="h", marker_color=[RUST if p < 0 else NAVY for p in w.points]))
        fig.update_layout(title="Fantasy points per unit of each statistic", xaxis_title="points", yaxis_title="")
        fig_show(fig, 360)
    with c2:
        slots = pd.DataFrame([(SLOT_NAMES.get(s, s), n) for s, n in S.lineup.items()], columns=["slot", "count"])
        st.dataframe(slots, hide_index=True, use_container_width=True)
        note(
            f"{S.n_teams} teams draft {S.rounds} rounds in snake order. Each week you play one opponent and the team with "
            "more fantasy points wins. The quantity we maximise is therefore <b>P(win a weekly matchup)</b>, not total "
            "points: a roster's value depends on how its weekly score compares with the rosters the other managers "
            "actually build."
        )

    # 2. projections -------------------------------------------------------------------------------------
    st.markdown("### 2. Projecting player seasons")
    st.markdown(
        "Three projections are combined. **ESPN** publishes a preseason projection. **Our own model** "
        "(Marcel+) weights each player's last three seasons 5/4/3, shrinks per-game rates toward the position "
        "average, and corrects with MoneyPuck features (expected goals, power-play time, ice time). The "
        "**blend** is a per-statistic weighted average of the two, with weights estimated on seasons the model "
        "never saw."
    )
    pe = b.proj_eval
    if len(pe):
        srcs = [("espn", "ESPN"), ("own", "Own model"), ("blend", "Blend"), ("market_adj", "Blend + market")]
        mae = pe.groupby("season").apply(lambda g: pd.Series({lab: (g[c] - g.actual).abs().mean() for c, lab in srcs})).reset_index()
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for i, (_, lab) in enumerate(srcs):
                fig.add_bar(x=mae.season.astype(str), y=mae[lab], name=lab, marker_color=PALETTE[i])
            fig.update_layout(barmode="group", title="Out-of-sample error by season", xaxis_title="season", yaxis_title="mean absolute error (fantasy pts)")
            fig_show(fig)
        with c2:
            lim = float(max(pe.blend.max(), pe.actual.max()))
            fig = go.Figure()
            fig.add_scatter(x=pe.espn, y=pe.actual, mode="markers", name="ESPN", marker=dict(color=GREY, size=5, opacity=0.45))
            fig.add_scatter(x=pe.market_adj, y=pe.actual, mode="markers", name="Blend + market", marker=dict(color=NAVY, size=5, opacity=0.55))
            fig.add_scatter(x=[0, lim], y=[0, lim], mode="lines", name="perfect", line=dict(color=RUST, dash="dash"))
            fig.update_layout(title="Projected vs actual fantasy points", xaxis_title="projected", yaxis_title="actual")
            fig_show(fig)
        tot = {lab: (pe[c] - pe.actual).abs().mean() for c, lab in srcs}
        note(
            "How to read this: lower bars are better. Pooled over the held-out seasons the mean absolute error is "
            + ", ".join(f"{k} {v:.1f}" for k, v in tot.items())
            + " fantasy points per player. ESPN's projections are systematically optimistic (points sit below the dashed "
            "line), which is why the blend leans toward our model for most statistics."
        )

    # 3. market ------------------------------------------------------------------------------------------
    st.markdown("### 3. Respecting the market: the winner's curse")
    st.markdown(
        "A draft policy that simply takes the player our model likes most relative to ADP systematically picks "
        "players whose projections happen to be too high. Across hundreds of noisy projections, the largest "
        "disagreements with the market are disproportionately errors. We therefore regress actual points on "
        "both our projection and log(ADP), separately for forwards, defense and goalies, and use the fitted value."
    )
    if b.market is not None and b.market.coef_:
        c1, c2 = st.columns([2, 3])
        with c1:
            cf = pd.DataFrame([{"group": g, "weight on projection": c[1], "weight on log ADP": c[2], "R squared": b.market.r2_[g]} for g, c in b.market.coef_.items()])
            st.dataframe(cf.round(3), hide_index=True, use_container_width=True)
        with c2:
            adp_grid = np.linspace(1, 230, 100)
            fig = go.Figure()
            for i, (g, c) in enumerate(b.market.coef_.items()):
                med = float(pool.loc[pool.pos.map(lambda p: "G" if p == "G" else ("D" if p == "D" else "F")) == g, "fpts_model"].median())
                fig.add_scatter(x=adp_grid, y=c[0] + c[1] * med + c[2] * np.log(adp_grid), name=f"{g} (median projection)", line=dict(color=PALETTE[i]))
            fig.update_layout(title="Market-adjusted expectation vs ADP, holding projection fixed", xaxis_title="ADP", yaxis_title="expected actual pts")
            fig_show(fig, 330)
        note(
            "How to read this: a weight on projection below 1 means projections are shrunk; a negative weight on log ADP "
            "means that, for the same projection, a player the market drafts later tends to score less. Goalie projections "
            "carry the least information, so goalie values are shrunk the most."
        )

    # 4. valuation ---------------------------------------------------------------------------------------
    st.markdown("### 4. From a roster to P(win a week)")
    st.markdown(
        "Each roster is slotted optimally (Hungarian assignment: starters count fully, bench players about a third "
        "because daily lineups let them fill idle days). A team's weekly score is modelled as Normal with mean equal "
        "to its expected weekly points and variance from Poisson-type variation in each scoring statistic. The "
        "probability of beating an opponent is the area where your distribution exceeds theirs."
    )
    _, _, demo = ctx.rollout(*ctx.initial_state([]), 0, None, np.random.default_rng(3))
    mu_sd = {t: ctx.val.team_dist(demo[t].roster) for t in ctx.teams}
    ranked = sorted(ctx.teams, key=lambda t: mu_sd[t][0][0])
    top_t, mid_t = ranked[-1], ranked[len(ranked) // 2]
    (m1, v1), (m2, v2) = mu_sd[top_t], mu_sd[mid_t]
    xs = np.linspace(min(m1[0], m2[0]) - 4 * np.sqrt(max(v1[0], v2[0])), max(m1[0], m2[0]) + 4 * np.sqrt(max(v1[0], v2[0])), 300)
    fig = go.Figure()
    fig.add_scatter(x=xs, y=norm.pdf(xs, m1[0], np.sqrt(v1[0])), name=f"Strongest simulated roster (mean {m1[0]:.0f})", line=dict(color=NAVY), fill="tozeroy", fillcolor="rgba(31,58,95,0.15)")
    fig.add_scatter(x=xs, y=norm.pdf(xs, m2[0], np.sqrt(v2[0])), name=f"Median simulated roster (mean {m2[0]:.0f})", line=dict(color=RUST), fill="tozeroy", fillcolor="rgba(165,69,43,0.12)")
    p_demo = norm.cdf((m1[0] - m2[0]) / np.sqrt(v1[0] + v2[0]))
    fig.update_layout(title=f"Weekly score distributions from one simulated draft: P(strongest beats median) = {p_demo:.2f}", xaxis_title="weekly fantasy points", yaxis_title="density")
    fig_show(fig, 340)
    note("How to read this: the wider the overlap, the closer the matchup is to a coin flip. A few points of weekly mean can move P(win) by several percentage points because weekly variance is large.")

    # 5. opponents ---------------------------------------------------------------------------------------
    st.markdown("### 5. Predicting what the other managers will do")
    st.markdown(
        "Opponent picks are modelled with a conditional logit: the chance a manager takes player j is proportional to "
        "exp(utility), where utility rises with market rank (negative log ADP), with positional need, and with a "
        "position-by-round tendency. It is estimated by maximum likelihood on every pick in the league's past drafts."
    )
    opp = b.sim_opp or b.opp_model
    top = ctx.adp_order[:40]
    for_round = st.select_slider("Illustrate at round", options=[1, 3, 6, 10, 15], value=1)
    probs = softmax(opp.utilities(None, for_round, ctx.neg_log_adp[top], ctx.grp[top], np.ones(len(top))))
    fig = go.Figure(go.Bar(x=[f"{pool.loc[j, 'name']}" for j in top], y=probs, marker_color=[RUST if ctx.grp[j] == 2 else (SAGE if ctx.grp[j] == 1 else NAVY) for j in top]))
    fig.update_layout(title=f"Probability each of the top 40 (by ADP) is the next pick, round {for_round} (blue F, green D, rust G)", yaxis_title="pick probability", xaxis_tickangle=-60)
    fig_show(fig, 380)
    note(f"The fitted market sensitivity is {opp.global_[0]:.1f}: managers in this league follow ADP closely, but not perfectly. That residual randomness is why a player's chance of lasting to your next pick is a probability, not a certainty.")

    # 6. spectral ----------------------------------------------------------------------------------------
    st.markdown("### 6. Graph spectral analysis of manager behaviour")
    if b.spectral is not None:
        c1, c2 = st.columns(2)
        with c1:
            ev = b.spectral.eigenvalues[: min(8, len(b.spectral.eigenvalues))]
            fig = go.Figure(go.Scatter(x=list(range(1, len(ev) + 1)), y=ev, mode="lines+markers", line=dict(color=NAVY)))
            fig.add_vline(x=b.spectral.k + 0.5, line_dash="dash", line_color=RUST)
            fig.update_layout(title=f"Laplacian eigenvalues: largest gap after k = {b.spectral.k}", xaxis_title="index", yaxis_title="eigenvalue")
            fig_show(fig, 330)
        with c2:
            note(
                "Managers are nodes; edges are weighted by how similar their drafting is (reach vs ADP, when they take goalies "
                "and defense, loyalty to one NHL team). The normalised graph Laplacian's small eigenvalues reveal clusters; the "
                "number of clusters is chosen where the eigenvalue sequence jumps (the eigengap). The map of managers is on the "
                "<b>League and strategy</b> tab. Tested in H4: pooling managers by cluster did <b>not</b> improve pick prediction "
                "over simpler models, so it is used descriptively rather than in the simulator."
            )

    # 7. simulation --------------------------------------------------------------------------------------
    st.markdown("### 7. The recommendation: Monte Carlo rollouts")
    note(
        "For each candidate you could draft now, the rest of the draft is simulated many times: opponents pick from the "
        "logit model and your later picks follow the market-anchored policy (best projected value among the next two "
        "players by ADP). Each finished league is scored with P(win a week). All candidates share the same random draws "
        "(common random numbers), so their difference is measured precisely even with few simulations. The Draft room "
        "reports the gap to the best candidate with its standard error; a gap smaller than about two standard errors is "
        "a statistical tie, and then you should take the player less likely to be available at your next pick."
    )

    # 8. evidence ----------------------------------------------------------------------------------------
    st.markdown("### 8. Does it work? Pre-registered hypothesis tests")
    if run is None:
        st.info("No experiment results found. Run: python -m iknowpuck.experiments")
    else:
        res = pd.DataFrame([{**r["test"], "id": r["id"], "label": f"{r['id']}: {r['treatment']} vs {r['baseline']}", "hyp": r["hypothesis"]} for r in run["results"]])
        res["sd"] = res["mean_diff"] / res["cohens_dz"].replace(0, np.nan)
        res["dz_lo"], res["dz_hi"] = res["ci_low"] / res["sd"], res["ci_high"] / res["sd"]
        fig = go.Figure()
        for _, r in res.iterrows():
            col = SAGE if r["significant"] else (RUST if r["mean_diff"] < 0 else GREY)
            fig.add_scatter(x=[r["dz_lo"], r["dz_hi"]], y=[r["label"]] * 2, mode="lines", line=dict(color=col, width=3), showlegend=False)
            fig.add_scatter(x=[r["cohens_dz"]], y=[r["label"]], mode="markers", marker=dict(color=col, size=11), showlegend=False)
        fig.add_vline(x=0, line_color="#444")
        fig.update_layout(title="Effect sizes (Cohen's d_z) with 95% bootstrap intervals; right of zero favours the treatment", xaxis_title="standardised paired effect", yaxis_title="")
        fig_show(fig, 60 + 70 * len(res))
        for _, r in res.iterrows():
            verdict = "supported" if r["significant"] else ("contradicted: the treatment was worse" if r["ci_high"] < 0 else "not supported")
            st.markdown(f"- **{r['id']}** ({r['hyp']}): {verdict}. Mean paired difference {r['mean_diff']:+.3f}, 95% CI [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}], n = {r['n']}, Holm-adjusted p = {r['p_adjusted']:.3f}.")
        note("Green: significant after Holm correction for testing several hypotheses at once. Rust: the interval lies entirely below zero, so the idea was rejected and removed from the draft engine. Grey: inconclusive.")

    # 9. aims --------------------------------------------------------------------------------------------
    st.markdown("### 9. What to aim for on draft day")
    cor = b.strategy_cor
    if len(cor):
        wp = cor[cor.outcome == "Win %"].sort_values("rho")
        fig = go.Figure()
        for _, r in wp.iterrows():
            col = SAGE if r.ci_low > 0 else (RUST if r.ci_high < 0 else GREY)
            fig.add_scatter(x=[r.ci_low, r.ci_high], y=[r.strategy] * 2, mode="lines", line=dict(color=col, width=3), showlegend=False)
            fig.add_scatter(x=[r.rho], y=[r.strategy], mode="markers", marker=dict(color=col, size=10), showlegend=False)
        fig.add_vline(x=0, line_color="#444")
        fig.update_layout(title="Association of draft behaviour with regular-season win % (Spearman, 95% bootstrap CI)", xaxis_title="rank correlation", yaxis_title="")
        fig_show(fig, 420)
        note(
            "Evidence from this league's past three seasons (36 manager-seasons). Descriptive, not causal, and "
            "several comparisons are being made, so treat borderline results as hints.<br>"
            "<b>1. The draft matters.</b> The actual points of the players you draft correlate strongly with finishing "
            "position.<br>"
            "<b>2. Win on value, not on reaches.</b> Reaching ahead of ADP shows no association with results, and the "
            "backtest shows that chasing projection-vs-ADP gaps loses. Stay inside the market window.<br>"
            "<b>3. Forwards first.</b> A heavy defense share in rounds 1 to 6 is associated with lower win %.<br>"
            "<b>4. Goalies.</b> Managers who waited longest on goalies scored fewer points, but goalie projections are the "
            "least reliable. Secure a starting goalie by the middle rounds instead of in round 1."
        )


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
            note("Dark bar: best estimate. Mid-blue: statistically tied with the best (gap within two standard errors); among tied players prefer the one least likely to be available at your next pick. Market rank 1 is the next player the market expects to go.")
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
    st.markdown("## League managers and strategy")
    mgr_names = b.manager_names
    active = set(b.managers.loc[b.managers.active, "owner_id"].astype(str)) if len(b.managers) else set()

    def who(m):
        n = mgr_names.get(m, m)
        return n if m in active else f"{n} (former)"

    if len(b.managers):
        md = b.managers.copy()
        md["team slot by season"] = md["team_by_season"].map(lambda d: ", ".join(f"{s}: #{t}" for s, t in sorted(d.items())))
        md["picks modelled"] = md["owner_id"].map(b.drafts.groupby("owner_id").size()).fillna(0).astype(int)
        md["status"] = np.where(md.active, "active", "left league")
        st.dataframe(md[["manager", "status", "team slot by season", "picks modelled"]].sort_values(["status", "manager"]), hide_index=True, use_container_width=True)
        st.markdown("<span class='caption'>Managers are tracked by ESPN account, not team slot, so history follows the person when slots change hands.</span>", unsafe_allow_html=True)

    if len(b.strategy):
        st.markdown("### Results and draft behaviour by season")
        ms = b.strategy.copy()
        ms["manager"] = ms["owner_id"].map(who)
        fig = go.Figure()
        fig.add_scatter(x=ms.value_added_all, y=ms.win_pct, mode="markers", text=ms.manager + " " + ms.season.astype(str), marker=dict(size=9, color=[RUST if o == S.my_team_id or mgr_names.get(o) == names.get(my_team) else NAVY for o in ms.owner_id]), hovertemplate="%{text}<br>value added %{x:.0f}<br>win %% %{y:.2f}")
        fig.update_layout(title="Draft value added vs regular-season win % (each point is a manager-season)", xaxis_title="actual points of drafted players above what their pick slots usually yield", yaxis_title="win %")
        fig_show(fig, 400)
        st.dataframe(ms[["season", "manager", "final_rank", "win_pct", "points_for", "first_goalie_round", "goalies_r1_3", "d_share_r1_6", "reach_early", "value_added_all"]].sort_values(["season", "final_rank"]).round(2), hide_index=True, use_container_width=True)
        prof = ms.groupby("manager")[["win_pct", "value_added_all", "first_goalie_round", "d_share_r1_6", "reach_early"]].mean().sort_values("win_pct", ascending=False)
        st.markdown("### Manager profiles (averaged over seasons)")
        st.dataframe(prof.round(2), use_container_width=True)
        st.markdown("### Strategy associations, all outcomes")
        st.dataframe(b.strategy_cor.round(3), hide_index=True, use_container_width=True)

    if b.spectral is not None:
        st.markdown("### Manager archetypes (spectral clustering)")
        tbl = b.spectral.table().reset_index()
        tbl["who"] = tbl["manager"].map(who)
        fig = go.Figure()
        for i, cl in enumerate(sorted(tbl.cluster.unique())):
            d = tbl[tbl.cluster == cl]
            fig.add_scatter(x=d.x, y=d.y, mode="markers+text", text=d.who, textposition="top center", name=f"cluster {cl}", marker=dict(size=11, color=PALETTE[i]))
        fig.update_layout(title="Spectral embedding of managers", xaxis_title="Laplacian eigenvector 2 (Fiedler)", yaxis_title="eigenvector 3")
        fig_show(fig, 480)
        st.dataframe(tbl[["who", "cluster", "mean_reach", "reach_early", "d_share_early", "g_share_early", "first_goalie_round", "homer_index", "fiedler"]].round(2), hide_index=True, use_container_width=True)
        st.markdown("### Fitted pick tendencies (conditional logit, per manager)")
        desc = b.opp_model.describe()
        desc["who"] = desc["manager"].map(lambda m: who(m) if m != "(league)" else "League average")
        st.dataframe(desc.drop(columns="manager").set_index("who").round(2), use_container_width=True)
        st.markdown("<span class='caption'>adp_sensitivity: how strictly a manager follows ADP. Position biases are relative to forwards in each round phase; positive values mean the manager takes that position earlier than the league. These per-manager fits are descriptive; the simulator uses the league-wide model (see H3).</span>", unsafe_allow_html=True)


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
    terms = [
        ("ADP", "Average draft position across ESPN leagues. The market's consensus of when a player is drafted."),
        ("Market rank", "A player's position among still-available players when sorted by ADP; 1 is the next player the market expects to go."),
        ("H2H points", "Head-to-head points scoring: each week you face one opponent and the higher fantasy-point total wins."),
        ("P(win a week)", "Probability that a roster beats an opponent in a weekly matchup, averaged over the league's other rosters. The objective the tool maximises."),
        ("Marcel+", "Our projection model: recency-weighted (5/4/3) per-game rates shrunk toward the position mean, with a MoneyPuck-based correction and a games-played model."),
        ("Blend", "A per-statistic weighted average of our projection and ESPN's, with weights fit on seasons not used for training."),
        ("Market adjustment", "A regression of actual points on our projection and log(ADP), fit on past seasons; protects against the winner's curse."),
        ("Winner's curse", "When choosing the option with the highest noisy estimate, that estimate is biased upward; the largest projection-vs-market gaps are disproportionately errors."),
        ("VONA", "Value over next available: a player's value minus the best same-position value expected to remain at your next pick."),
        ("Market-anchored policy", "Choose the highest projected value among the next two roster-fitting players by ADP. Used for your future picks inside simulations."),
        ("Usage / slotting", "Players are assigned to lineup slots optimally (Hungarian algorithm). Starters count fully, bench players partially."),
        ("Conditional logit", "A discrete-choice model: the probability of choosing an option is proportional to exp(utility). Used to model opponent picks."),
        ("Monte Carlo rollout", "Simulating the remainder of the draft many times to estimate the expected outcome of a decision."),
        ("Common random numbers", "Using the same random draws for every candidate so that differences between candidates are not swamped by simulation noise."),
        ("Standard error (SE)", "The uncertainty of an average estimated from simulations or samples. A gap smaller than about 2 SE is not statistically distinguishable from zero."),
        ("Graph Laplacian", "L = I - D^(-1/2) W D^(-1/2) for a similarity graph with weights W; its smallest eigenvectors reveal cluster structure."),
        ("Eigengap", "The largest jump in the sorted Laplacian eigenvalues; used to choose the number of clusters."),
        ("Fiedler vector", "The eigenvector of the second-smallest Laplacian eigenvalue; orders nodes along the graph's main division."),
        ("Spearman correlation (rho)", "Correlation of ranks; robust to outliers and non-linear but monotone relationships."),
        ("Paired test", "Both arms are measured on the same units (same players, same simulated drafts), removing between-unit variation."),
        ("Permutation test", "Randomly flips the sign of paired differences to build the null distribution; makes no normality assumption."),
        ("Bootstrap CI", "Resamples units with replacement many times; the middle 95% of the resampled statistic forms the confidence interval."),
        ("Cohen's d_z", "Mean paired difference divided by the standard deviation of differences; a unit-free effect size (0.2 small, 0.5 medium, 0.8 large)."),
        ("Holm correction", "Adjusts p-values when testing several hypotheses so that the chance of any false positive stays at 5%."),
        ("Leave-one-season-out", "Fit on all seasons but one and evaluate on the held-out season, repeated for each season."),
        ("Out-of-sample", "Evaluated on data the model did not see when it was fit; the only honest measure of predictive accuracy."),
    ]
    for t, d in terms:
        st.markdown(f"**{t}.** {d}")
