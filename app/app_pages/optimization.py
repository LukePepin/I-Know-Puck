"""Optimization: what the app maximises and the operations-research methods that do it (normal-approximation
objective, Hungarian lineup assignment, Monte Carlo rollouts with common random numbers, scarcity)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm
from ui import CAT, DIV, FAINT, INK, MUTED, SEQ, app_state, fig_show, get_matchups, note, rgba

from iknowpuck.config import SLOT_NAMES
from iknowpuck.draft import crn_experiment, pick_distribution
from iknowpuck.valuation import WEEKS_IN_SEASON, Valuator

A = app_state()
b, S, ctx, names, order, my_team = A["b"], A["S"], A["ctx"], A["names"], A["order"], A["my_team"]
pool = A["pool"]
f = pool.frame
val = ctx.val
GROUP = {0: "Forward", 1: "Defense", 2: "Goalie"}

st.markdown("## Optimization")
st.caption("What the app maximises when it suggests a pick, and the operations-research methods it uses to do that fast enough for a live draft.")
c1, c2 = st.columns([3, 2])
with c1:
    st.markdown(
        "A snake draft is a **sequential decision problem under uncertainty**. At each of your picks you choose one player. You can't control what "
        "the other managers pick, and the payoff only arrives once all the rosters are filled. Solving it exactly would mean searching every possible "
        "future draft, which is far too many. So the app breaks the problem into pieces that can each be solved well:\n\n"
        "1. **The objective** turns a finished league into one number, P(win a week), with a Normal approximation (section 1).\n"
        "2. **Lineup assignment** decides who starts in which slot with the Hungarian algorithm, an exact method (section 2).\n"
        "3. **Monte Carlo search** compares candidate picks by playing the draft forward many times, with common random numbers "
        "to cut the noise (section 3).\n"
        "4. **Opponent simulation** samples each manager's pick from a choice model, which shows who will still be there (section 4).\n"
        "5. **Scarcity** measures each player against the replacement level at his position (section 5)."
    )
with c2:
    st.graphviz_chart("""
    digraph G {
      rankdir=TB; bgcolor="transparent"; nodesep=0.25; ranksep=0.3;
      node [shape=box, style="rounded,filled", fillcolor="#F4F3EF", color="#2A6BB0", fontname="Georgia", fontsize=12, width=2.6];
      edge [color="#555555", fontname="Georgia", fontsize=10];
      s [label="Draft state: who is gone, rosters"]; c [label="Candidates: next players by ADP + top VONA"];
      r [label="Monte Carlo rollouts\\nopponents sample the pick model\\nyou follow the market-anchored policy"];
      h [label="Lineup assignment (Hungarian)"]; o [label="Objective: P(win a week)"]; d [label="Decision: best mean;\\nties go to the player least likely to return"];
      s -> c -> r -> h -> o -> d; r -> r [label=" same seeds for\\n every candidate"];
    }
    """, width="stretch")

section = st.segmented_control("Section", ["The objective", "Lineup assignment", "Monte Carlo search", "Where players go", "Scarcity and value"],
                               default="The objective", key="op_section", label_visibility="collapsed")


@st.cache_resource(max_entries=4, show_spinner=False)
def demo_draft(_ctx, key: tuple, seed: int):
    _, _, teams = _ctx.rollout(*_ctx.initial_state([]), 0, None, np.random.default_rng([seed]))
    return teams


# ======================================================================================================
if section == "The objective" or section is None:
    st.markdown("### 1. The objective: P(win a week)")
    st.latex(r"P(\text{win}) = \Phi\!\left(\frac{\mu_{\text{you}} - \mu_{\text{opp}}}{\sqrt{\sigma^2_{\text{you}} + \sigma^2_{\text{opp}}}}\right), \qquad "
             r"\text{score} = \frac{1}{11}\sum_{\text{opponents}} P(\text{win})")
    st.markdown("Each roster's weekly score is treated as Normal with mean mu and spread sigma (the Assumption tests page checks this). "
                "That turns a simulation-sized question into one formula, so thousands of finished leagues can be scored per second.")
    teams = demo_draft(ctx, (tuple(order), my_team, id(ctx)), 3)
    dist = {t: val.team_dist(teams[t].roster) for t in ctx.teams}
    ranked = sorted(ctx.teams, key=lambda t: dist[t][0][0])
    (m1, v1), (m2, v2) = dist[ranked[-1]], dist[ranked[len(ranked) // 2]]
    c1, c2 = st.columns([1, 2])
    with c1:
        mu_a = st.slider("Your weekly mean", 40.0, 200.0, float(round(m1[0])), 1.0, format="%.0f", key="ob_mu_a")
        sd_a = st.slider("Your weekly SD", 2.0, 50.0, float(round(np.sqrt(v1[0]))), 0.5, format="%.1f", key="ob_sd_a")
        mu_b = st.slider("Opponent's weekly mean", 40.0, 200.0, float(round(m2[0])), 1.0, format="%.0f", key="ob_mu_b")
        sd_b = st.slider("Opponent's weekly SD", 2.0, 50.0, float(round(np.sqrt(v2[0]))), 0.5, format="%.1f", key="ob_sd_b")
        p = norm.cdf((mu_a - mu_b) / np.hypot(sd_a, sd_b))
        st.metric("P(you win the week)", f"{p:.1%}", border=True)
    with c2:
        lo_, hi_ = min(mu_a - 4 * sd_a, mu_b - 4 * sd_b), max(mu_a + 4 * sd_a, mu_b + 4 * sd_b)
        xs = np.linspace(lo_, hi_, 400)
        fig = go.Figure()
        fig.add_scatter(x=xs, y=norm.pdf(xs, mu_a, sd_a), name="you", line=dict(color=CAT[0], width=2), fill="tozeroy", fillcolor=rgba(CAT[0], 0.15))
        fig.add_scatter(x=xs, y=norm.pdf(xs, mu_b, sd_b), name="opponent", line=dict(color=CAT[1], width=2), fill="tozeroy", fillcolor=rgba(CAT[1], 0.12))
        fig.update_layout(title="Weekly score distributions (defaults: a strong and an average simulated roster)", xaxis_title="weekly fantasy points",
                          yaxis_title="density", legend=dict(orientation="h", y=-0.2))
        fig_show(fig, 380)

    c1, c2 = st.columns(2)
    with c1:
        dm = np.linspace(-40, 40, 161)
        sg = np.linspace(4, 60, 113)
        Pz = norm.cdf(dm[None, :] / sg[:, None])
        fig = go.Figure(go.Contour(z=Pz, x=dm, y=sg, colorscale=DIV, zmin=0, zmax=1, contours=dict(start=0.1, end=0.9, size=0.1, showlabels=True,
                                   labelfont=dict(size=10, color=INK)), line=dict(width=0.5, color="white"), colorbar=dict(title="P(win)", thickness=10, tickformat=".0%"),
                                   hovertemplate="mean gap %{x:+.0f}<br>combined SD %{y:.0f}<br>P(win) %{z:.1%}<extra></extra>"))
        fig.add_scatter(x=[mu_a - mu_b], y=[np.hypot(sd_a, sd_b)], mode="markers", marker=dict(size=13, color=INK, symbol="x"), name="your sliders",
                        hovertemplate="your sliders<extra></extra>")
        fig.update_layout(title="P(win) over mean gap and combined spread", xaxis_title="your mean minus opponent's mean", yaxis_title="combined SD",
                          showlegend=False)
        fig_show(fig, 380)
    with c2:
        sds = np.linspace(4, 50, 100)
        gapv = max(abs(mu_a - mu_b), 5.0)
        fig = go.Figure()
        fig.add_scatter(x=sds, y=norm.cdf(gapv / np.hypot(sds, sd_b)), name=f"favourite by {gapv:.0f}", line=dict(color=CAT[0], width=2))
        fig.add_scatter(x=sds, y=norm.cdf(-gapv / np.hypot(sds, sd_b)), name=f"underdog by {gapv:.0f}", line=dict(color=CAT[1], width=2))
        fig.add_hline(y=0.5, line_color=MUTED, line_width=1)
        fig.update_layout(title="Should you want a steady or a streaky roster?", xaxis_title="your weekly SD", yaxis=dict(title="P(win)", tickformat=".0%"),
                          legend=dict(orientation="h", y=-0.2))
        fig_show(fig, 380)
    note("**The optimisation insight:** when you are the **favourite**, more week-to-week variance *lowers* your chance of winning, so steady players are worth more. "
         "When you are the **underdog**, variance *raises* it, so boom-or-bust players (and stacking one NHL team) become rational. This is why the objective is "
         "P(win) and not just total points: two rosters with the same average can have different odds.")

    st.markdown("#### Risk and return of every player")
    top = f.assign(_a=np.where(f.adp.notna(), f.adp, 999)).nsmallest(300, "_a").index.to_numpy()
    wm, wsd = val.pts[top] / WEEKS_IN_SEASON, np.sqrt(val.pts_var[top] / WEEKS_IN_SEASON)
    grp = ctx.grp[top]
    fig = go.Figure()
    for gi, gl in GROUP.items():
        sel = grp == gi
        fig.add_scatter(x=wm[sel], y=wsd[sel], mode="markers", name=gl, marker=dict(size=9, color=CAT[gi], line=dict(color="white", width=1)),
                        customdata=np.stack([f.loc[top[sel], "name"], f.loc[top[sel], "pos"], f.loc[top[sel], "adp"].fillna(np.nan)], axis=1),
                        hovertemplate="%{customdata[0]} (%{customdata[1]}), ADP %{customdata[2]:.0f}<br>%{x:.1f} pts/week, SD %{y:.1f}<extra></extra>")
    fig.update_layout(title="Weekly mean vs weekly SD as a starter (top 300 by ADP)", xaxis_title="expected fantasy points per week", yaxis_title="weekly SD",
                      legend=dict(orientation="h", y=-0.2))
    fig_show(fig, 420)
    st.caption("Goalies sit high on this chart: a goalie's week swings with how many games he starts and wins, so each point costs more variance than a skater's.")

# ======================================================================================================
elif section == "Lineup assignment":
    st.markdown("### 2. Lineup assignment: the Hungarian algorithm")
    st.markdown("Before a roster can be scored, the app has to decide **who starts where**. A centre who is also eligible at left wing can fill either slot. "
                "This is the classic **assignment problem**: every slot takes one player and every player fills at most one slot, and the goal is to maximise "
                "the value of the starters. The Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) solves it exactly, in a fraction of a millisecond, "
                "for every roster in every simulation.")
    st.latex(r"\max_{x}\ \sum_{i,j} v_i\, s_j\, x_{ij}\quad \text{s.t.}\ \sum_j x_{ij} \le 1,\ \sum_i x_{ij} = 1,\ x_{ij} = 0 \text{ if } i \text{ is not eligible for } j")
    c1, c2, c3 = st.columns([2, 1, 1])
    team = c1.selectbox("Roster from a simulated draft", ctx.teams, index=ctx.teams.index(my_team) if my_team in ctx.teams else 0,
                        format_func=lambda t: names.get(t, f"Team {t}") + (" (you)" if t == my_team else ""), key="la_team")
    seed = int(c2.number_input("Simulated draft #", 1, 50, 1, key="la_seed"))
    bench_f = c3.slider("Bench usage", 0.0, 1.0, float(val.bench_factor), 0.05, key="la_bench",
                        help="Share of a bench player's points that count. Your league locks lineups weekly, so only the weekly lineup choice and injury cover use the bench (measured about 0.25).")
    teams = demo_draft(ctx, (tuple(order), my_team, id(ctx)), seed)
    roster = teams[team].roster
    cost, rows_, cols_, starter = val.assignment(roster)
    r = np.asarray(roster)
    slot_lab, cnt = [], {}
    for s_ in val.slot_list:
        cnt[s_] = cnt.get(s_, 0) + 1
        slot_lab.append(f"{SLOT_NAMES.get(s_, s_)}{cnt[s_]}")
    pts = val.pts[r]
    o = np.argsort(-pts)
    elig = cost < 1e6
    Vm = np.where(elig, pts[:, None], np.nan)[o]
    who_lab = [f"{f.loc[j, 'name']} ({f.loc[j, 'pos']})" for j in r[o]]
    assigned = {int(i): int(j) for i, j in zip(rows_, cols_) if cost[i, j] < 1e6}
    inv = {int(i): k for k, i in enumerate(o)}
    fig = go.Figure(go.Heatmap(z=Vm, x=slot_lab, y=who_lab, colorscale=SEQ, xgap=2, ygap=2, colorbar=dict(title="season pts", thickness=10),
                               hovertemplate="%{y}<br>slot %{x}: eligible, %{z:.0f} season points<extra></extra>"))
    fig.add_scatter(x=[slot_lab[assigned[i]] for i in assigned], y=[who_lab[inv[i]] for i in assigned], mode="markers",
                    marker=dict(symbol="circle-open", size=16, color=CAT[1], line=dict(width=3)), name="assigned", hoverinfo="skip")
    fig.add_vline(x=sum(starter) - 0.5, line_color=INK, line_width=1)
    fig.update_layout(title="Eligibility matrix (coloured = eligible) and the optimal assignment (rust circles)", yaxis=dict(autorange="reversed"),
                      xaxis=dict(side="top", tickangle=-60, tickfont=dict(size=11)), showlegend=False)
    fig_show(fig, 110 + 24 * len(r))
    st.caption("Columns left of the black line are starting slots; the rest are bench. Blank cells are slots the player is not eligible for.")
    v2 = Valuator(pool, bench_factor=bench_f, goalie_bench_factor=val.goalie_bench_factor)
    u = v2.usage(roster)
    mu, var = v2.team_dist(roster)
    tab = pd.DataFrame({"player": [f.loc[j, "name"] for j in r], "pos": [f.loc[j, "pos"] for j in r],
                        "slot": [slot_lab[assigned[i]] if i in assigned else "none" for i in range(len(r))],
                        "usage": u, "season points": pts, "counted weekly": u * pts / WEEKS_IN_SEASON}).sort_values(["usage", "season points"], ascending=False)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(tab, hide_index=True, height=360, column_config={"usage": st.column_config.NumberColumn(format="%.2f"),
                     "season points": st.column_config.NumberColumn(format="%.0f"), "counted weekly": st.column_config.NumberColumn(format="%.1f")})
    with c2:
        st.metric("Weekly mean", f"{mu[0]:.1f} pts", border=True)
        st.metric("Weekly SD", f"{np.sqrt(var[0]):.1f} pts", border=True)
        st.metric("Starters' share of the mean", f"{(u * pts)[u == 1].sum() / max((u * pts).sum(), 1e-9):.0%}", border=True)

    st.markdown("#### Which bench usage matches real scores?")
    m, _ = get_matchups(tuple(b.history.seasons)) if b.history is not None else (pd.DataFrame(), None)
    grid = np.round(np.arange(0, 1.01, 0.05), 2)
    mus = []
    for g_ in grid:
        vg = Valuator(pool, bench_factor=float(g_), goalie_bench_factor=val.goalie_bench_factor)
        mus.append(np.mean([vg.team_dist(teams[t].roster)[0][0] for t in ctx.teams]))
    fig = go.Figure()
    fig.add_scatter(x=grid, y=mus, mode="lines+markers", line=dict(color=CAT[0], width=2), marker=dict(size=7), name="model: average simulated team",
                    hovertemplate="bench usage %{x:.2f}: %{y:.1f} pts/week<extra></extra>")
    if len(m):
        real = m[~m.playoff].points.mean()
        fig.add_hline(y=real, line_color=CAT[1], line_width=2, annotation_text=f"real league average {real:.0f}", annotation_position="bottom right")
    fig.add_vline(x=bench_f, line_color=MUTED)
    fig.update_layout(title="Average weekly score of a simulated team vs the bench usage assumption", xaxis_title="bench usage", yaxis_title="points per week",
                      showlegend=False)
    fig_show(fig, 320)
    st.caption("Where the blue line crosses the rust line, the model's average matches the real weekly average in your league (2024-26). "
               "Treat it as a rough check: this season's player pool is not the same as past seasons'.")

# ======================================================================================================
elif section == "Monte Carlo search":
    st.markdown("### 3. Monte Carlo rollouts with common random numbers")
    st.markdown("To judge a candidate, the app **pretends you took him**, plays the rest of the draft forward many times (opponents sample from the pick "
                "model, and you keep drafting with the market-anchored policy), and scores every finished league. This is a one-step lookahead with a base "
                "policy, a standard approximate dynamic programming method. The trick is **common random numbers (CRN)**: every candidate faces the *same* "
                "simulated futures, so the luck of the draw cancels out when two candidates are compared.")
    frame_ids = {int(p): i for i, p in enumerate(f.player_id)}
    picks = [(t, frame_ids[p]) for t, p in st.session_state.get("picks", []) if p in frame_ids]
    with st.form("mc_form", border=True):
        c1, c2, c3 = st.columns(3)
        n_c = c1.slider("Candidates", 2, 5, 5, key="mc_c")
        n_r = c2.slider("Rollouts per candidate", 10, 80, 30, 5, key="mc_r")
        seed = int(c3.number_input("Random seed", 0, 999, 0, key="mc_seed"))
        st.caption("Starts from the first pick of the draft. Each run takes a few seconds.")
        go_ = st.form_submit_button("Run the comparison", icon=":material/play_arrow:", type="primary")
    key = (tuple(order), my_team, tuple(picks), n_c, n_r, seed)
    if go_:
        with st.spinner("Rolling out drafts twice: shared futures, then independent futures..."):
            st.session_state["mc_result"] = (key, crn_experiment(ctx, picks, n_c, n_r, seed))
    got = st.session_state.get("mc_result")
    if not got or got[0][:3] != key[:3]:
        st.info("Press **Run the comparison** to simulate.")
        st.stop()
    cands, crn, ind = got[1]
    R = crn.shape[1]
    on_clock = ctx.order[len(picks)] if len(picks) < len(ctx.order) else my_team
    st.caption(f"Scored for the team on the clock at pick {len(picks) + 1}: **{names.get(on_clock, on_clock)}**"
               + (" (you)." if on_clock == my_team else "."))
    lab = [f"{f.loc[c, 'name']} ({f.loc[c, 'pos']})" for c in cands]
    best = int(crn.mean(1).argmax())
    k_ = np.arange(1, R + 1)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for ci in range(len(cands)):
            run = np.cumsum(crn[ci]) / k_
            sd = np.array([crn[ci, :i].std(ddof=1) if i > 1 else 0 for i in k_]) / np.sqrt(k_)
            fig.add_scatter(x=np.r_[k_, k_[::-1]], y=np.r_[run + 2 * sd, (run - 2 * sd)[::-1]], fill="toself", fillcolor=rgba(CAT[ci], 0.12),
                            line=dict(width=0), hoverinfo="skip", showlegend=False)
            fig.add_scatter(x=k_, y=run, mode="lines", name=lab[ci], line=dict(color=CAT[ci], width=2),
                            hovertemplate=f"{lab[ci]}<br>after %{{x}} rollouts: %{{y:.3f}}<extra></extra>")
        fig.update_layout(title="Running estimate of P(win) per candidate (bands = +/- 2 SE)", xaxis_title="rollouts", yaxis=dict(title="P(win a week)", tickformat=".1%"),
                          legend=dict(orientation="h", y=-0.25))
        fig_show(fig, 420)
    with c2:
        oth = [i for i in range(len(cands)) if i != best]
        se_crn = [(crn[i] - crn[best]).std(ddof=1) / np.sqrt(R) for i in oth]
        se_ind = [np.sqrt(ind[i].var(ddof=1) / R + ind[best].var(ddof=1) / R) for i in oth]
        fig = go.Figure()
        for y_, a_, c_ in zip([lab[i] for i in oth], se_crn, se_ind):
            fig.add_scatter(x=[a_, c_], y=[y_, y_], mode="lines", line=dict(color=FAINT, width=3), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=se_crn, y=[lab[i] for i in oth], mode="markers", name="shared futures (CRN)", marker=dict(size=12, color=CAT[0], line=dict(color="white", width=2)),
                        hovertemplate="%{y}<br>CRN: SE %{x:.4f}<extra></extra>")
        fig.add_scatter(x=se_ind, y=[lab[i] for i in oth], mode="markers", name="independent futures", marker=dict(size=12, color=CAT[1], line=dict(color="white", width=2)),
                        hovertemplate="%{y}<br>independent: SE %{x:.4f}<extra></extra>")
        fig.update_layout(title=f"Standard error of each gap to the best ({lab[best]})", xaxis=dict(title="SE of the P(win) gap", rangemode="tozero"),
                          legend=dict(orientation="h", y=-0.25))
        fig_show(fig, 420)
    # root-mean-square SE across candidates: a candidate whose gap is nearly constant has CRN SE ~ 0, which would blow up a ratio of medians
    ratio = float(np.sqrt(np.mean(np.square(se_ind))) / max(np.sqrt(np.mean(np.square(se_crn))), 1e-9))
    with st.container(horizontal=True):
        st.metric("SE reduction from CRN", f"{ratio:.1f}x", border=True, help="Root-mean-square SE of the gaps, independent vs shared futures.")
        st.metric("Rollouts saved for equal precision", f"{ratio ** 2:.0f}x fewer", border=True, help="SE falls with the square root of the rollouts, so an x-fold SE cut saves x-squared rollouts.")
        st.metric("Best candidate", lab[best], border=True)
    tab = pd.DataFrame({"candidate": lab, "ADP": [f.loc[c, "adp"] for c in cands], "season points": [ctx.value[c] for c in cands],
                        "P(win), CRN": crn.mean(1), "gap to best": crn.mean(1) - crn.mean(1)[best],
                        "gap SE, CRN": [(crn[i] - crn[best]).std(ddof=1) / np.sqrt(R) for i in range(len(cands))],
                        "gap SE, independent": [np.sqrt(ind[i].var(ddof=1) / R + ind[best].var(ddof=1) / R) if i != best else 0.0 for i in range(len(cands))]})
    st.dataframe(tab, hide_index=True, column_config={"ADP": st.column_config.NumberColumn(format="%.0f"), "season points": st.column_config.NumberColumn(format="%.0f"),
                 **{c: st.column_config.NumberColumn(format="%.4f") for c in ["P(win), CRN", "gap to best", "gap SE, CRN", "gap SE, independent"]}})
    note("**Why this matters on draft day:** with independent futures, most gaps between candidates are smaller than the noise, so the app would need many more "
         "rollouts to tell them apart. With shared futures, the same number of rollouts separates them, which is how a recommendation fits in a few seconds.")

# ======================================================================================================
elif section == "Where players go":
    st.markdown("### 4. Where each player gets drafted")
    st.markdown("The same rollouts show how uncertain the draft is. Each row is a player (by ESPN ranking); each cell is the share of simulated drafts in which "
                "he went at that pick. Vertical lines mark your picks.")

    @st.cache_resource(max_entries=4, show_spinner="Simulating 200 full drafts...")
    def dist(_ctx, key: tuple):
        return pick_distribution(_ctx, 200)

    D_ = dist(ctx, (tuple(order), my_team, id(ctx)))
    my_slots = [p for p, t in enumerate(ctx.order) if t == my_team]
    c1, c2, c3 = st.columns([2, 2, 3])
    n_show = c1.slider("Players", 15, 80, 40, 5, key="wp_n")
    rounds = c2.slider("Rounds shown", 2, S.rounds, min(5, S.rounds), key="wp_r")
    posf = c3.pills("Positions", ["C", "LW", "RW", "D", "G"], selection_mode="multi", default=["C", "LW", "RW", "D", "G"], key="wp_pos")
    cand = [j for j in ctx.adp_order if f.loc[j, "pos"] in (posf or [])][:n_show]
    P = S.n_teams * rounds
    Hm = np.stack([np.bincount(D_[:, j][(D_[:, j] >= 0) & (D_[:, j] < P)], minlength=P) / len(D_) for j in cand])
    ylab = [f"{f.loc[j, 'name']} ({f.loc[j, 'pos']}, ADP {f.loc[j, 'adp']:.0f})" if np.isfinite(f.loc[j, "adp"]) else f.loc[j, "name"] for j in cand]
    zmax = float(np.quantile(Hm[Hm > 0], 0.95)) if (Hm > 0).any() else 1.0  # a few near-certain early picks would wash out the rest
    fig = go.Figure(go.Heatmap(z=np.where(Hm > 0, Hm, np.nan), x=np.arange(1, P + 1), y=ylab, colorscale=SEQ, zmin=0, zmax=zmax,
                               colorbar=dict(title="share", thickness=10, tickformat=".0%"),
                               hovertemplate="%{y}<br>taken at pick %{x} in %{z:.0%} of drafts<extra></extra>", xgap=0, ygap=1))
    for p in my_slots:
        if p < P:
            fig.add_vline(x=p + 1, line_color=CAT[1], line_width=1.5)
    fig.update_layout(title=f"Pick-number distribution from 200 simulated drafts (rust lines = your picks, {names.get(my_team, 'you')})",
                      yaxis=dict(autorange="reversed"), xaxis_title="overall pick")
    fig_show(fig, 110 + 16 * len(cand))

    st.markdown("#### Will he still be there?")
    med = {j: float(np.median(np.where(D_[:, j] >= 0, D_[:, j], P + 50))) for j in cand}
    near_me = sorted(sorted(cand, key=lambda j: abs(med[j] - my_slots[0]))[:3], key=lambda j: med[j]) if my_slots else cand[:3]
    st.caption("Starts with the three players most likely to go around your first pick.")
    pick3 = st.multiselect("Compare up to three players", cand, default=near_me, max_selections=3,
                           format_func=lambda j: f"{f.loc[j, 'name']} ({f.loc[j, 'pos']})", key="wp_sel")
    fig = go.Figure()
    xs = np.arange(1, P + 1)
    for i, j in enumerate(pick3):
        tk = D_[:, j]
        surv = np.array([np.mean((tk < 0) | (tk >= p - 1)) for p in xs])
        fig.add_scatter(x=xs, y=surv, mode="lines", name=f.loc[j, "name"], line=dict(color=CAT[i], width=2, shape="hv"),
                        hovertemplate=f"{f.loc[j, 'name']}<br>still available at pick %{{x}}: %{{y:.0%}}<extra></extra>")
    for k, p in enumerate(my_slots):
        if p < P:
            fig.add_vline(x=p + 1, line_color=MUTED, line_width=1, annotation_text=f"R{k + 1}" if k < 8 else None, annotation_position="top")
    fig.update_layout(title="Chance each player is still available at every pick (grey lines = your picks)", xaxis_title="overall pick",
                      yaxis=dict(title="still available", tickformat=".0%", range=[0, 1.02]), legend=dict(orientation="h", y=-0.2))
    fig_show(fig, 360)
    rows = []
    for j in pick3:
        tk = D_[:, j]
        rows.append({"player": f.loc[j, "name"], **{f"R{k + 1} (pick {p + 1})": float(np.mean((tk < 0) | (tk >= p))) for k, p in enumerate(my_slots[:6])}})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={c: st.column_config.ProgressColumn(c, min_value=0, max_value=1, format="percent")
                                                                         for c in rows[0] if c != "player"})

# ======================================================================================================
elif section == "Scarcity and value":
    st.markdown("### 5. Positional scarcity and value over replacement")
    st.markdown("A player's value depends on who else is left. If 48 centres will start across the league, the 49th-best centre is **replacement level**: anyone can "
                "find one on waivers. The steeper a position's curve falls before its replacement line, the more an early pick at that position is worth.")
    starters = {"C": S.lineup.get(0, 0), "LW": S.lineup.get(1, 0), "RW": S.lineup.get(2, 0), "D": S.lineup.get(4, 0), "G": S.lineup.get(5, 0)}
    pos_color = {"C": CAT[0], "LW": CAT[3], "RW": CAT[4], "D": CAT[1], "G": CAT[2]}  # D and G match the defense / goalie colours above
    fig = go.Figure()
    for pos, n_s in starters.items():
        sub = f[f.pos == pos].assign(v=ctx.value[f.pos == pos]).nlargest(110, "v").reset_index(drop=True)
        if not len(sub):
            continue
        fig.add_scatter(x=sub.index + 1, y=sub.v, mode="lines", name=pos, line=dict(color=pos_color[pos], width=2), customdata=sub.name,
                        hovertemplate=f"{pos} #%{{x}}: %{{customdata}}, %{{y:.0f}} pts<extra></extra>")
        rep = S.n_teams * n_s
        if rep <= len(sub):
            fig.add_scatter(x=[rep], y=[sub.v.iloc[rep - 1]], mode="markers",
                            marker=dict(size=10, color=pos_color[pos], line=dict(color="white", width=2)), showlegend=False,
                            hovertemplate=f"{pos} replacement level (#{rep}): %{{y:.0f}} pts<extra></extra>")
    reps: dict[int, list[str]] = {}
    for pos, n_s in starters.items():
        if n_s:
            reps.setdefault(S.n_teams * n_s, []).append(pos)
    for rep, ps in reps.items():  # one labelled line per rank, so positions sharing a rank don't stack labels
        fig.add_vline(x=rep, line_color=MUTED, line_width=1, line_dash="dot", annotation_text=f"{'/'.join(ps)} replacement (#{rep})",
                      annotation_position="top right", annotation_font=dict(color=INK, size=11))
    fig.update_layout(title="Projected season points by rank within position (dots = replacement level: league-wide starters)", xaxis_title="rank within position",
                      yaxis_title="projected fantasy points", legend=dict(orientation="h", y=-0.2))
    fig_show(fig, 420)

    st.markdown("#### Value over next available (VONA) at your first pick")
    first = next((p for p, t in enumerate(ctx.order) if t == my_team), 0)
    taken, teams = ctx.initial_state([])
    avail, sc = ctx.greedy_scores(teams[my_team], taken, first)
    o = np.argsort(-sc)[:15]
    fig = go.Figure(go.Bar(x=sc[o], y=[f"{f.loc[j, 'name']} ({f.loc[j, 'pos']})" for j in avail[o]], orientation="h",
                           marker_color=[CAT[int(ctx.grp[j])] for j in avail[o]], customdata=np.stack([ctx.value[avail[o]], f.loc[avail[o], "adp"].to_numpy()], axis=1),
                           hovertemplate="%{y}<br>VONA %{x:.0f} (projected %{customdata[0]:.0f}, ADP %{customdata[1]:.0f})<extra></extra>"))
    fig.update_layout(title="Top 15 by VONA: projected points minus the best same-group player likely left at your next pick", xaxis_title="VONA (season points)",
                      yaxis=dict(autorange="reversed"))
    fig_show(fig, 460)
    st.caption("Colours: blue = forward, rust = defense, violet = goalie. VONA alone lost to ADP drafting in the backtest (experiment H2a), because it trusts the "
               "projection too much; the app uses it only to add candidates, and the market-anchored policy and Monte Carlo search make the call.")
