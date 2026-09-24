"""Systems engineering view: architecture, traceability, verification, findings, trade study, risks,
a short lab-presentation outline, and what transfers to other projects."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import GREY, NAVY, RUST, SAGE, app_state, fig_show, load_runs, note

from iknowpuck.summaries import verdict

A = app_state()
b = A["b"]
runs = load_runs(2)
res = {r["id"]: r["test"] for r in runs[0]["results"]} if runs else {}


def status(hid: str, want: str = "yes") -> str:
    if hid not in res:
        return "not run"
    v = verdict(res[hid])
    reps = [verdict(next(x["test"] for x in run["results"] if x["id"] == hid)) for run in runs if any(x["id"] == hid for x in run["results"])]
    agree = f" ({sum(r == v for r in reps)}/{len(reps)} runs)"
    return ("Verified" if v == want else "Not met" if want == "yes" else "Rejected as designed") + agree


st.markdown("## Systems engineering view")
st.caption("A compact systems-engineering record of the project: what it must do, how it is built, how it was verified, what we learned, and what can be reused.")

# --- 1. context ----------------------------------------------------------------------------------------
st.markdown("### 1. Mission and stakeholders")
c1, c2 = st.columns(2)
with c1:
    note("**Mission:** give a fantasy manager a decision-support tool that recommends the draft pick (and later the waiver move) with the highest "
         "probability of winning weekly head-to-head matchups, and prove every modelling claim with pre-registered tests.")
with c2:
    st.dataframe(pd.DataFrame([
        ("Fantasy manager (primary user)", "Clear, fast recommendation during a live draft; plain-language explanations"),
        ("Analyst / researcher", "Reproducible experiments, honest statistics, reusable framework"),
        ("League (environment)", "ESPN rules, rosters, schedule, opponent behaviour"),
        ("Data providers", "ESPN Fantasy API, MoneyPuck (rate limits, format changes, data quality)"),
    ], columns=["Stakeholder", "Need"]), hide_index=True)

# --- 2. architecture ---------------------------------------------------------------------------------
st.markdown("### 2. System architecture")
st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#F4F3EF", color="#1F3A5F", fontname="Georgia", fontsize=11];
  edge [color="#555555", fontname="Georgia", fontsize=9];
  subgraph cluster_data { label="Data layer"; color="#8A8A8A"; fontname="Georgia";
    espn [label="ESPN Fantasy API\\nplayers, league, drafts,\\ntransactions, game logs"]; mp [label="MoneyPuck\\nadvanced stats"]; cache [label="Disk cache\\n(content-addressed)"]; }
  subgraph cluster_models { label="Models"; color="#8A8A8A"; fontname="Georgia";
    proj [label="Projections\\nMarcel+ / blend"]; mkt [label="Market adjustment\\n(winner's-curse guard)"]; inj [label="Injury model\\nstatus + history"];
    val [label="Valuation\\nslotting + P(win week)"]; opp [label="Opponent model\\nconditional logit"]; spec [label="Spectral clustering\\nof managers"]; hist [label="League history\\nstints, trades, absences"]; }
  subgraph cluster_decide { label="Decision engine"; color="#8A8A8A"; fontname="Georgia";
    mc [label="Monte Carlo rollouts\\ncommon random numbers"]; ws [label="Waiver optimiser\\nadd/drop search"]; }
  subgraph cluster_ui { label="Interface"; color="#8A8A8A"; fontname="Georgia"; ui [label="Streamlit app\\n6 pages"]; }
  subgraph cluster_vv { label="Verification & validation"; color="#8A8A8A"; fontname="Georgia"; tests [label="Unit tests"]; exps [label="Experiment suite\\nH1-H4, Holm"]; }
  espn -> cache; mp -> cache; cache -> proj; cache -> hist; cache -> opp; proj -> mkt -> inj -> val; val -> mc; opp -> mc; spec -> opp [style=dashed, label="tested H4"];
  hist -> spec; hist -> inj; val -> ws; mc -> ui; ws -> ui; hist -> ui; exps -> ui [style=dashed]; tests -> proj [style=dashed]; exps -> proj [style=dashed]; exps -> mc [style=dashed];
}
""", width="stretch")
st.caption("Generic parts live in the `puckcore` package (cache, statistics, experiment runner, interfaces); hockey-specific parts live in `iknowpuck`.")

# --- 3. requirements traceability ---------------------------------------------------------------------
st.markdown("### 3. Requirements traceability")
rtm = pd.DataFrame([
    ("R1", "Projections more accurate than the ESPN baseline", "projections.py (Marcel+ blend)", "H1: paired test on held-out seasons", status("H1")),
    ("R2", "Advanced stats justify their cost", "MoneyPuck residual model", "H1b: ablation", status("H1b")),
    ("R3", "Draft policy at least as good as market (ADP) drafting", "draft.py market-anchored policy", "H2: backtest on actual outcomes", "Met (no difference)" if res.get("H2") and verdict(res["H2"]) == "tie" else status("H2")),
    ("R4", "Reject policies that lose to the market", "VONA projection-greedy policy", "H2a: backtest", status("H2a", want="no")),
    ("R5", "Opponent model only as complex as the data supports", "opponents.py + spectral.py", "H3, H4: leave-one-season-out likelihood", "Simplest model kept"),
    ("R6", "Recommendation in under 15 s during a live draft", "MC rollouts with common random numbers", "Timing: about 7 s for 12 x 20 rollouts", "Verified"),
    ("R7", "Plain-language explanations for non-experts", "Walkthrough, notes, glossary", "Usability review", "Reviewed"),
    ("R8", "Current injuries reflected in values", "injuries.py + editable overrides", "Unit tests + manual check on flagged stars", "Verified"),
    ("R9", "Reproducible results", "Experiment runner logs git commit + seed", "Two independent runs, identical verdicts", "Verified" if len(runs) >= 2 else "Pending"),
    ("R10", "Secrets never leave the machine", ".env git-ignored, cookies only in HTTP headers", "Pre-commit checks of tracked files", "Verified"),
], columns=["ID", "Requirement", "Design element", "Verification method", "Status"])
st.dataframe(rtm, hide_index=True, column_config={"Status": st.column_config.TextColumn(width="medium")})

# --- 4. V-model ----------------------------------------------------------------------------------------
st.markdown("### 4. Verification and validation (V-model)")
c1, c2 = st.columns(2)
with c1:
    st.dataframe(pd.DataFrame([
        ("Stakeholder needs", "Validation", "Real draft on Sunday; season results vs league"),
        ("System requirements", "System verification", "Backtests on 2024-26 scored with actual stats (H2, H2a)"),
        ("Architecture", "Integration testing", "End-to-end pipeline build; every app page loads with live data"),
        ("Component design", "Component verification", "Out-of-sample model tests (H1, H1b, H3, H4)"),
        ("Implementation", "Unit testing", "pytest suite on synthetic data (stats, valuation, logit recovery, spectral, summaries)"),
    ], columns=["Left side (define)", "Right side (verify)", "How it is done here"]), hide_index=True)
with c2:
    fig = go.Figure()
    left = [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
    right = [(8, 4), (7, 3), (6, 2), (5, 1), (4, 0)]
    labels_l = ["Needs", "Requirements", "Architecture", "Components", "Code"]
    labels_r = ["Validation", "System tests", "Integration", "Component tests", "Unit tests"]
    fig.add_scatter(x=[p[0] for p in left], y=[p[1] for p in left], mode="lines+markers+text", text=labels_l, textposition="middle left", line=dict(color=NAVY, width=3), showlegend=False)
    fig.add_scatter(x=[p[0] for p in right], y=[p[1] for p in right], mode="lines+markers+text", text=labels_r, textposition="middle right", line=dict(color=SAGE, width=3), showlegend=False)
    for (xl, y), (xr, _) in zip(left[:-1], right[:-1]):
        fig.add_scatter(x=[xl, xr], y=[y, y], mode="lines", line=dict(color=GREY, dash="dot"), showlegend=False, hoverinfo="skip")
    fig.update_layout(title="V-model", xaxis=dict(visible=False, range=[-3, 11]), yaxis=dict(visible=False, range=[-0.5, 4.5]))
    fig_show(fig, 320)

# --- 5. critical findings -----------------------------------------------------------------------------
st.markdown("### 5. Critical findings")
h2a = res.get("H2a", {})
findings = [
    ("Verification caught data leakage", "ESPN parks undrafted players at an ADP of about 230; ties then fell back to end-of-season ownership order, "
     "which is hindsight. Fixing it dropped the ADP baseline's backtest score from 0.75 to 0.63, so every earlier comparison had been biased."),
    ("An optimiser exploits its own model errors", "Taking the player our projection likes most relative to ADP lost to plain ADP drafting "
     f"(H2a: {h2a.get('mean_diff', float('nan')):+.3f} win probability). This is the winner's curse; the fix was a market-anchored policy plus a regression that shrinks projections toward the market."),
    ("More model complexity did not pay", "Per-manager pick models and spectral pooling did not predict held-out drafts better (H3, H4). "
     "The simulator keeps the simpler league-wide model; the richer models are used only for scouting insight."),
    ("Requirements must be checked against source data", "The league was described as a categories league; ESPN's settings showed head-to-head points in every season. "
     "The objective function was changed before any modelling built on the wrong assumption."),
    ("Results replicate", "The full experiment suite was run twice with different random seeds; every verdict matched."),
    ("Data quality is a system risk", "ESPN returns a row for every team game (not only games played), wiped one season's ADP, and logged far fewer lineup moves in later seasons. "
     "Each needed a detection check and a documented fix."),
    ("In-season behaviour matters", "Beyond draft quality, managers who made more waiver pickups won more (borderline evidence). This motivated the In-season moves page."),
]
for t, d in findings:
    with st.container(border=True):
        st.markdown(f"**{t}.** {d}")

# --- 6. trade study ------------------------------------------------------------------------------------
st.markdown("### 6. Trade study: which draft policy?")
crit = {"Backtest win chance": 0.40, "Robustness to model error": 0.25, "Compute time": 0.15, "Explainability": 0.20}
opts = pd.DataFrame({
    "ADP only": [3, 4, 5, 5],
    "Projection-greedy (VONA)": [1, 1, 5, 4],
    "Market-anchored (chosen)": [3, 4, 5, 4],
    "Market-anchored + Monte Carlo lookahead": [4, 4, 3, 3],
}, index=list(crit))
weights = pd.Series(crit)
scores = (opts.T * weights).sum(axis=1).sort_values()
c1, c2 = st.columns([3, 2])
with c1:
    st.dataframe(opts.assign(weight=weights), column_config={"weight": st.column_config.NumberColumn(format="%.2f")})
    st.caption("Scores 1 (poor) to 5 (best). Backtest scores come from H2/H2a; the Monte Carlo lookahead score is an engineering judgement pending its own backtest (future H2b).")
with c2:
    fig = go.Figure(go.Bar(x=scores.values, y=scores.index, orientation="h", marker_color=[RUST if "greedy" in i else NAVY for i in scores.index]))
    fig.update_layout(title="Weighted score", xaxis_title="weighted score (max 5)")
    fig_show(fig, 280)

# --- 7. risks ------------------------------------------------------------------------------------------
st.markdown("### 7. Failure modes and effects (FMEA)")
fmea = pd.DataFrame([
    ("ESPN cookies expire during the draft", "No live sync", 7, 3, 2, "Manual pick entry; clear error message; re-copy cookies"),
    ("ESPN API format changes", "Pipeline fails", 8, 2, 3, "Cached raw JSON; parsing isolated in one module; history optional"),
    ("Pick order changes before the draft", "Wrong 'on the clock' team", 6, 4, 2, "Order read from ESPN each rebuild; editable in sidebar"),
    ("Stale injury information", "Over- or under-valued players", 6, 5, 4, "Editable games-missed table; status shown in recommendations"),
    ("Projection error on a player", "Poor pick", 5, 6, 6, "Market adjustment; recommendations stay inside the ADP window"),
    ("Monte Carlo noise", "Wrong candidate ranked first", 4, 4, 3, "Common random numbers; standard errors; ties broken by availability"),
    ("Data leakage in backtests", "Overstated performance", 8, 3, 7, "Out-of-sample protocol; leakage audit; shuffled source order"),
], columns=["Failure mode", "Effect", "Severity", "Occurrence", "Detection", "Mitigation"])
fmea["RPN"] = fmea.Severity * fmea.Occurrence * fmea.Detection
st.dataframe(fmea.sort_values("RPN", ascending=False), hide_index=True,
             column_config={"RPN": st.column_config.ProgressColumn("risk priority (S x O x D)", min_value=0, max_value=500, format="%d")})

# --- 8. lab presentation --------------------------------------------------------------------------------
st.markdown("### 8. Short lab presentation (about 7 minutes)")
slides = [
    ("1. Problem (45 s)", "Snake draft as a sequential decision under uncertainty; objective = P(win a weekly matchup).", "Walkthrough, step 1"),
    ("2. Architecture (60 s)", "Data layer, models, decision engine, interface; reusable core vs domain plugin.", "This page, section 2"),
    ("3. Method (75 s)", "Projections + market adjustment, valuation by optimal slotting, opponent logit, Monte Carlo with common random numbers.", "Walkthrough, steps 2-6"),
    ("4. Verification (75 s)", "Pre-registered hypotheses, paired tests, Holm correction, replication; show the scorecard.", "Walkthrough scorecard"),
    ("5. Critical findings (90 s)", "Leakage caught by verification; winner's curse; negative results kept honest.", "This page, section 5"),
    ("6. Live demo (60 s)", "Draft room: suggested pick with uncertainty and availability; manager map click-through.", "Draft room, League history"),
    ("7. Transfer (30 s)", "What moves to other projects (next section).", "This page, section 9"),
]
st.dataframe(pd.DataFrame(slides, columns=["Slide", "Message", "Show in the app"]), hide_index=True)

# --- 9. what transfers -----------------------------------------------------------------------------------
st.markdown("### 9. What we could move over to other projects")
st.dataframe(pd.DataFrame([
    ("Experiment runner (puckcore.experiment)", "Any model or policy comparison", "Drop-in", "Hypothesis, paired design, Holm correction, report with git commit and seed"),
    ("Paired statistics (puckcore.stats)", "A/B tests, simulation studies, ablations", "Drop-in", "Permutation test, bootstrap CI, effect size, power estimate"),
    ("Disk cache (puckcore.cache)", "Any API-backed analysis", "Drop-in", "Content-addressed JSON/parquet cache with expiry"),
    ("Monte Carlo rollouts with common random numbers", "Scheduling, inventory, logistics, game AI", "Adapt", "Candidate comparison under uncertainty with paired variance reduction"),
    ("Conditional-logit choice model", "Customer choice, route choice, competitor behaviour", "Adapt", "MLE with shrinkage toward group means"),
    ("Spectral clustering of agents", "Customer or user segmentation, network communities", "Adapt", "Laplacian eigenmaps, eigengap selection, rule-based naming"),
    ("Market-anchoring (winner's-curse guard)", "Any optimiser selecting from noisy estimates", "Concept", "Shrink estimates toward an external consensus before optimising"),
    ("Leakage audit checklist", "Every backtest", "Concept", "Source ordering, sentinel values, point-in-time data, out-of-sample protocol"),
    ("Other ESPN fantasy sports", "Football, basketball, baseball leagues", "Adapt", "Same client pattern; new stat map and roster slots"),
], columns=["Component", "Where it applies", "Effort", "What it gives you"]), hide_index=True)
note("**Takeaway for the lab:** the most valuable transferable asset is not the hockey model but the **verification discipline**: "
     "pre-registered paired tests, replication, and leakage audits caught two errors that would otherwise have shipped.")
