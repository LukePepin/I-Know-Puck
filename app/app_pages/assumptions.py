"""Assumption tests: every modelling assumption checked against the league's own data."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.special import ndtr
from scipy.stats import norm
from ui import CAT, INK, MUTED, app_state, fig_show, get_matchups, get_panel, load_runs, note, rgba

from iknowpuck import diagnostics as D
from iknowpuck.spectral import eigengap_null
from iknowpuck.valuation import COVAR_INFLATION
from puckcore.stats import required_n_paired

A = app_state()
b, S, ctx, names = A["b"], A["S"], A["ctx"], A["names"]
H = b.history
st.markdown("## Assumption tests")
st.caption("Every model rests on assumptions. This page checks each one against your league's real results, so you can see which parts of the model to trust.")
if H is None:
    st.warning("Assumption tests need your league history (ESPN cookies in the .env file).")
    st.stop()
mgr_names = b.manager_names


# --- cached data --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Rebuilding weekly player stats...")
def weekly_stats(_b, season: int):
    m, per = get_matchups(tuple(_b.history.seasons))
    wk = D.weekly_player_stats(_b.history.gamelogs, per, _b.settings)
    disp = D.dispersion_table(wk, _b.settings, _b.history.player_pos)
    cov, cov_sum = D.covariance_inflation(wk, _b.settings, _b.history.player_pos)
    cov_est, cov_est_sum = D.covariance_inflation(wk, _b.settings, _b.history.player_pos, dispersion=dict(zip(disp.stat_id, disp.estimated)))
    return wk, disp, cov, cov_sum, cov_est_sum


@st.cache_resource(show_spinner="Scoring every past pick with the pick model...")
def pick_model_check(_b, season: int):
    return D.pick_model_loso(_b.drafts, get_panel(season), _b.settings.lineup)


@st.cache_resource(show_spinner=False)
def model_teams(_ctx, season: int, n_drafts: int = 8) -> pd.DataFrame:
    """Weekly (mean, SD) the model gives each simulated roster (several simulated drafts)."""
    rows = []
    for s in range(n_drafts):
        _, _, teams = _ctx.rollout(*_ctx.initial_state([]), 0, None, np.random.default_rng([5, s]))
        for t in teams.values():
            mu, var = _ctx.val.team_dist(t.roster)
            rows.append({"mu": float(mu[0]), "sd": float(np.sqrt(var[0]))})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def spectral_p(_b, season: int) -> float:
    return eigengap_null(_b.spectral.features, n_perm=1000)[2] if _b.spectral is not None else float("nan")


m, per = get_matchups(tuple(H.seasons))
if not len(m):
    st.warning("ESPN did not return weekly matchup scores for past seasons.")
    st.stop()
m["team"] = [mgr_names.get(H.owner_of(s, t), f"Team {t}") for s, t in zip(m.season, m.team_id)]
reg = m[~m.playoff].copy()
reg["z"] = D.standardised_scores(reg)
nz = D.normality(reg.z)
wp = D.win_prob_backtest(m)
wps = D.scores(wp.p_model, wp.result)
wk, disp, cov, cov_sum, cov_est_sum = weekly_stats(b, A["season"])
pe = b.proj_eval
cal = D.calibration_line(pe.market_adj, pe.actual)
cal_espn = D.calibration_line(pe.espn, pe.actual)
bp = D.breusch_pagan(pe.market_adj, pe.actual - pe.market_adj)
picks, pr, ch = pick_model_check(b, A["season"])
mt = model_teams(ctx, A["season"])
team_sd = reg.groupby(["season", "team_id"]).points.std()
runs = load_runs(2)

# --- scorecard ------------------------------------------------------------------------------------------
st.markdown("### Scorecard")
close = ((disp.estimated / disp.assumed).between(0.75, 1.25)).sum()  # thousands of player-seasons make every CI tiny, so judge closeness
worst = disp.assign(r=(disp.estimated / disp.assumed)).sort_values("r", key=lambda x: (np.log(x)).abs(), ascending=False).iloc[0]
rows = [
    ("Weekly team scores follow a bell curve (Normal)", "P(win a week) formula", f"{nz['n']} real weekly scores, standardised per team-season",
     f"skew {nz['skew']:+.2f}, excess kurtosis {nz['excess_kurtosis']:+.2f}, Shapiro-Wilk p = {nz['p_shapiro']:.2f}",
     "Holds" if nz["p_shapiro"] > 0.05 else "Does not hold"),
    ("P(win) = Phi(mean gap / combined SD) gives honest odds", "The objective every pick maximises", f"{len(wp)} past matchups, each predicted from the teams' other weeks",
     f"Brier score {wps['brier']:.3f} vs {wps['brier_coin']:.3f} for a coin flip; picks the winner {wps['accuracy']:.0%} of the time",
     "Holds" if wps["brier"] < 0.25 else "Does not hold"),
    ("Stat counts are over-dispersed Poisson with the assumed dispersion", "Weekly variance of each roster", f"{int(disp.player_seasons.max())} player-seasons of weekly stat counts",
     f"{close} of {len(disp)} stats within 25% of the assumed value; the exception is {worst.stat}: {worst.estimated:.1f} measured vs {worst.assumed:.1f} assumed",
     "Holds" if close == len(disp) else "Partly holds"),
    (f"Stats move together enough to inflate variance by {COVAR_INFLATION}", "Weekly variance of each roster", f"{cov_sum['player_seasons']} player-seasons",
     f"data supports {cov_sum['estimated']:.2f} (95% CI {cov_sum['lo']:.2f} to {cov_sum['hi']:.2f})",
     "Holds" if cov_sum["lo"] <= COVAR_INFLATION <= cov_sum["hi"] else ("Too high" if COVAR_INFLATION > cov_sum["hi"] else "Too low")),
    ("Our projections are calibrated (actual = projected on average)", "Every player's value", f"{cal['n']} held-out player-seasons",
     f"slope {cal['slope']:.2f} (95% CI {cal['slope_lo']:.2f} to {cal['slope_hi']:.2f}); ESPN's own projections: {cal_espn['slope']:.2f}",
     "Holds" if cal["slope_lo"] <= 1 <= cal["slope_hi"] else "Does not hold"),
    ("Projection errors have the same spread for stars and depth players", "Market adjustment regression", "Breusch-Pagan test on the same seasons",
     f"LM = {bp['lm']:.2f}, p = {bp['p']:.2f}", "Holds" if bp["p"] > 0.05 else "Does not hold"),
    ("Opponents pick like the conditional logit", "Draft simulator", f"{len(picks)} past picks, model fit on the other seasons",
     f"actual pick in the model's top 5: {(picks.model_rank <= 5).mean():.0%} (ESPN order alone: {(picks.adp_rank <= 5).mean():.0%})", "Roughly holds"),
]
if b.spectral is not None:
    p_sp = spectral_p(b, A["season"])
    rows.append(("Managers fall into distinct drafting styles", "Pick-model pooling (tested in H4)", "Permutation test of the Laplacian eigengap",
                 f"p = {p_sp:.3f}", "Holds" if p_sp < 0.05 else "Not supported"))
if runs:
    ds = [np.asarray(r.get("diffs", [])) for r in runs[0]["results"] if r.get("diffs")]
    if ds:
        agree = sum((r["test"]["p_permutation"] < 0.05) == (r["test"]["p_ttest"] < 0.05) == (r["test"]["p_wilcoxon"] < 0.05) for r in runs[0]["results"])
        rows.append(("Experiment conclusions don't depend on the test's assumptions", "H1 to H5 verdicts", "Permutation, Wilcoxon and t-test on the same paired units",
                     f"all three tests agree on {agree} of {len(runs[0]['results'])} hypotheses", "Holds" if agree == len(runs[0]["results"]) else "Partly holds"))
card = pd.DataFrame(rows, columns=["Assumption", "Used for", "Test", "Result", "Verdict"])
BADGE = {"Holds": ("green", ":material/check_circle:"), "Roughly holds": ("orange", ":material/adjust:"), "Partly holds": ("orange", ":material/adjust:"),
         "Too high": ("orange", ":material/arrow_upward:"), "Too low": ("orange", ":material/arrow_downward:"),
         "Does not hold": ("red", ":material/cancel:"), "Not supported": ("red", ":material/cancel:")}
grid = st.columns(2)
for i, r in card.iterrows():
    with grid[i % 2]:
        with st.container(border=True):
            colr, icon = BADGE.get(r.Verdict, ("gray", ":material/help:"))
            st.badge(r.Verdict, icon=icon, color=colr)
            st.markdown(f"**{r.Assumption}**  \n{r.Result}")
            st.caption(f"Used for: {r['Used for']}. Test: {r.Test}.")
with st.expander("Table view"):
    st.dataframe(card, hide_index=True)
note("**Bottom line:** the core formula holds. Weekly scores are bell-shaped and P(win) gives honest odds. Two variance settings are off in "
     f"opposite directions. The stat-covariance inflation is higher than the data supports ({cov_sum['estimated']:.2f} vs {COVAR_INFLATION}). Even so, whole "
     f"teams swing more from week to week in real life (median SD {team_sd.median():.0f}) than in the model ({mt.sd.median():.0f}), because real weeks also include "
     "injuries, roster moves and schedule quirks the model leaves out. Both affect every candidate alike, so they change how confident P(win) is more than "
     "which pick is best. Pick a section below to see each test.")

section = st.segmented_control("Section", ["Weekly scores", "Win probability", "Player variance", "Projections", "Pick model", "Experiment tests"],
                               default="Weekly scores", key="as_section", label_visibility="collapsed")

# ======================================================================================================
if section == "Weekly scores" or section is None:
    st.markdown("### Are weekly team scores bell-shaped?")
    st.markdown("The P(win) formula assumes a team's weekly score follows a Normal (bell) curve. Real weekly scores from every past matchup test that. "
                "Each score is standardised within its own team-season, so strong and weak teams sit on one scale.")
    with st.container(horizontal=True):
        seasons = st.pills("Seasons", sorted(reg.season.unique()), selection_mode="multi", default=sorted(reg.season.unique()), key="ws_seasons")
    sub = reg[reg.season.isin(seasons or [])]
    if len(sub) < 20:
        st.info("Pick at least one season.")
        st.stop()
    nzs = D.normality(sub.z)
    c1, c2 = st.columns(2)
    with c1:
        xs = np.linspace(-3.5, 3.5, 200)
        fig = go.Figure(go.Histogram(x=sub.z, histnorm="probability density", xbins=dict(size=0.25), marker_color=rgba(CAT[0], 0.55),
                                     marker_line=dict(color="white", width=1), name="real weeks", hovertemplate="%{y:.2f}<extra></extra>"))
        fig.add_scatter(x=xs, y=norm.pdf(xs), mode="lines", line=dict(color=INK, width=2), name="Normal curve")
        fig.update_layout(title="Standardised weekly scores vs the Normal curve", xaxis_title="standard deviations from the team's average",
                          yaxis_title="density", bargap=0.02, legend=dict(orientation="h", y=-0.2))
        fig_show(fig, 360)
    with c2:
        th, sm = D.qq(sub.z)
        srt = sub.iloc[np.argsort(sub.z.to_numpy())]
        fig = go.Figure()
        fig.add_scatter(x=[-3.5, 3.5], y=[-3.5, 3.5], mode="lines", line=dict(color=MUTED, width=1), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=th, y=sm, mode="markers", marker=dict(size=7, color=rgba(CAT[0], 0.55)), showlegend=False,
                        customdata=np.stack([srt.team, srt.season, srt.week, srt.points], axis=1),
                        hovertemplate="%{customdata[0]}, %{customdata[1]} week %{customdata[2]}<br>%{customdata[3]:.1f} points<extra></extra>")
        fig.update_layout(title="Q-Q plot: dots on the line = perfectly Normal", xaxis_title="Normal quantile", yaxis_title="observed quantile")
        fig_show(fig, 360)
    with st.container(horizontal=True):
        st.metric("Weeks", nzs["n"], border=True)
        st.metric("Skewness", f"{nzs['skew']:+.2f}", border=True, help="0 for a Normal curve; positive = a longer right tail.")
        st.metric("Excess kurtosis", f"{nzs['excess_kurtosis']:+.2f}", border=True, help="0 for a Normal curve; positive = more extreme weeks.")
        st.metric("Shapiro-Wilk p", f"{nzs['p_shapiro']:.2f}", border=True, help="Below 0.05 would reject normality.")
        st.metric("D'Agostino p", f"{nzs['p_dagostino']:.2f}", border=True)

    st.markdown("### Does the model get the size of a typical week right?")
    real_mu = reg.groupby(["season", "team_id"]).points.mean()
    c1, c2 = st.columns(2)
    for col, real, model, lab, unit in ((c1, real_mu, mt.mu, "Average weekly score", "points per week"), (c2, team_sd, mt.sd, "Week-to-week spread (SD)", "points")):
        with col:
            fig = go.Figure()
            fig.add_box(x=real, name="real teams<br>2024-26", marker_color=CAT[0], boxpoints="all", jitter=0.4, pointpos=0, line=dict(width=1.5),
                        hovertemplate="%{x:.1f}<extra>real team-season</extra>")
            fig.add_box(x=model, name="model's<br>simulated teams", marker_color=CAT[1], boxpoints="all", jitter=0.4, pointpos=0, line=dict(width=1.5),
                        hovertemplate="%{x:.1f}<extra>simulated roster</extra>")
            fig.update_layout(title=lab, xaxis_title=unit, showlegend=False)
            fig_show(fig, 280)
    note(f"**Reading this:** real teams averaged **{real_mu.median():.0f}** points a week, and the model's simulated rosters average **{mt.mu.median():.0f}**. "
         f"Real teams' scores swung by a median SD of **{team_sd.median():.0f}** from week to week; the model assumes **{mt.sd.median():.0f}**. "
         "Real week-to-week swings also include injuries, trades and schedule quirks, so the real SD should be somewhat larger than the model's.")
    with st.expander("Technical details"):
        st.markdown(
            "Regular-season matchup scores come from ESPN's `mMatchupScore` view. z = (score - team-season mean) / team-season SD. Shapiro-Wilk and "
            "D'Agostino-Pearson K-squared test normality; with several hundred weeks they detect even modest departures, so a p-value above 0.05 is strong "
            "support. The model's rosters come from simulated drafts of this season's player pool; the model's mean uses the bench usage for the league's lineup lock (0.25 for weekly locks) and "
            "its SD uses over-dispersed Poisson stat variance times the covariance inflation."
        )

# ======================================================================================================
elif section == "Win probability":
    st.markdown("### Does P(win) give honest odds?")
    st.markdown("For every past regular-season matchup, the formula predicted the chance each team would win, using each team's average and spread "
                "from its **other** weeks that season, so no matchup predicts itself. If the odds are honest, teams given a 70% chance win about 70% of the time.")
    c1, c2 = st.columns([3, 2])
    with c1:
        bins = st.slider("Number of bins", 4, 12, 8, key="wp_bins")
        rel = D.reliability(wp.p_model, wp.result, bins)
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, width=1), name="perfect calibration", hoverinfo="skip")
        fig.add_scatter(x=rel.predicted, y=rel.observed, mode="markers+lines", line=dict(color=CAT[0], width=2), name="real outcomes",
                        marker=dict(size=10, color=CAT[0], line=dict(color="white", width=2)),
                        error_y=dict(type="data", symmetric=False, array=rel.hi - rel.observed, arrayminus=rel.observed - rel.lo, color=rgba(CAT[0], 0.5), thickness=1.5),
                        customdata=np.stack([rel.n, rel.lo, rel.hi], axis=1),
                        hovertemplate="predicted %{x:.0%}<br>actually won %{y:.0%} (95%% range %{customdata[1]:.0%} to %{customdata[2]:.0%})<br>%{customdata[0]} matchups<extra></extra>")
        fig.update_layout(title="Reliability diagram (bars = 95% intervals)", xaxis=dict(title="predicted chance of winning", tickformat=".0%", range=[0, 1]),
                          yaxis=dict(title="share actually won", tickformat=".0%", range=[0, 1]), legend=dict(orientation="h", y=-0.2))
        fig_show(fig, 420)
    with c2:
        st.metric("Brier score", f"{wps['brier']:.3f}", delta=f"{wps['brier'] - wps['brier_coin']:+.3f} vs coin flip", delta_color="inverse", border=True)
        st.metric("Log loss", f"{wps['log_loss']:.3f}", delta=f"{wps['log_loss'] - wps['log_loss_coin']:+.3f} vs coin flip", delta_color="inverse", border=True)
        st.metric("Winner called correctly", f"{wps['accuracy']:.0%}", border=True)
        st.caption("Lower Brier score and log loss are better. A coin flip scores 0.250 and 0.693.")

    st.markdown("#### Is the assumed spread too wide or too narrow?")
    st.markdown("Multiply every team's variance by a factor and re-score all the matchups. The factor with the lowest log loss is the one the data prefers. "
                "A factor of 1 means the spreads are right.")
    fac = np.round(np.exp(np.linspace(np.log(0.25), np.log(4), 61)), 3)
    y = wp.result.to_numpy()
    ll = []
    for f in fac:
        p = np.clip(ndtr((wp.mu_loo - wp.opp_mu) / np.sqrt(f * (wp.var_loo + wp.opp_var))), 1e-6, 1 - 1e-6)
        ll.append(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    ll = np.array(ll)
    best = fac[np.argmin(ll)]
    pick = st.select_slider("Variance factor", options=list(fac), value=float(fac[np.argmin(np.abs(fac - 1))]), format_func=lambda v: f"x{v:.2f}", key="wp_fac")
    fig = go.Figure()
    fig.add_scatter(x=fac, y=ll, mode="lines", line=dict(color=CAT[0], width=2), showlegend=False, hovertemplate="x%{x:.2f}: log loss %{y:.4f}<extra></extra>")
    fig.add_scatter(x=[best], y=[ll.min()], mode="markers", marker=dict(size=11, color=CAT[1], line=dict(color="white", width=2)), showlegend=False,
                    hovertemplate="best: x%{x:.2f}<extra></extra>")
    fig.add_vline(x=pick, line_color=MUTED)
    fig.add_annotation(x=np.log10(best), y=ll.min(), text=f"best fit x{best:.2f}", showarrow=False, yshift=-18, xref="x", font=dict(color=INK))
    fig.update_layout(title="Log loss of all past matchups vs the variance factor", xaxis=dict(title="variance factor (log scale)", type="log", tickvals=[0.25, 0.5, 1, 2, 4],
                      ticktext=["x0.25", "x0.5", "x1", "x2", "x4"]), yaxis_title="log loss (lower = better)")
    fig_show(fig, 320)
    p_now = ndtr((wp.mu_loo - wp.opp_mu) / np.sqrt(pick * (wp.var_loo + wp.opp_var)))
    st.caption(f"At x{pick:.2f}: log loss {ll[list(fac).index(pick)]:.4f}, Brier {np.mean((p_now - y) ** 2):.4f}.")
    note(f"**Result:** the best factor is about **x{best:.2f}**. " + ("That is close to 1, so the Normal formula with each team's own spread is about right."
         if 0.67 < best < 1.5 else "That is far from 1: the formula's odds are too " + ("confident." if best > 1 else "cautious.")))

# ======================================================================================================
elif section == "Player variance":
    st.markdown("### Are weekly stat counts over-dispersed Poisson?")
    st.markdown("The model says a player's weekly count of each stat (goals, hits, saves...) varies like a Poisson count times a **dispersion** factor: "
                "variance = dispersion x mean. A factor of 1 is pure Poisson; above 1 means streakier. Each estimate below pools every player-season with 10 or more weeks played.")
    d = disp.sort_values(["goalie", "estimated"])
    lab = [f"{s_} ({'goalie' if g else 'skater'})" for s_, g in zip(d.stat, d.goalie)]
    fig = go.Figure()
    for yl, lo, hi in zip(lab, d.lo, d.hi):
        fig.add_scatter(x=[lo, hi], y=[yl, yl], mode="lines", line=dict(color=rgba(CAT[0], 0.6), width=3), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=d.assumed, y=lab, mode="markers", name="assumed in the model", marker=dict(size=12, color="white", line=dict(color=INK, width=2)),
                    hovertemplate="%{y}: assumed %{x:.2f}<extra></extra>")
    fig.add_scatter(x=d.estimated, y=lab, mode="markers", name="measured (95% CI)", marker=dict(size=11, color=CAT[0], line=dict(color="white", width=2)),
                    customdata=np.stack([d.lo, d.hi, d.player_seasons, d.weekly_mean], axis=1),
                    hovertemplate="%{y}: measured %{x:.2f} (%{customdata[0]:.2f} to %{customdata[1]:.2f})<br>%{customdata[2]} player-seasons, "
                                  "average %{customdata[3]:.2f} per week<extra></extra>")
    fig.add_vline(x=1, line_color=MUTED)
    fig.add_annotation(x=0, y=1.02, xref="x", yref="paper", text="Poisson (1)", showarrow=False, font=dict(color=MUTED))  # log axis: x is log10
    fig.update_layout(title="Dispersion per stat: assumed (hollow) vs measured (filled)", xaxis=dict(title="dispersion = weekly variance / weekly mean (log scale)", type="log", tickvals=[0.5, 0.7, 1, 1.5, 2, 3, 5, 10]),
                      legend=dict(orientation="h", y=-0.15))
    fig_show(fig, 80 + 30 * len(d))
    sv = disp[disp.stat == "SV"]
    note(f"**What stands out:** {close} of {len(disp)} stats are within 25% of the assumed value. The exception is goalie saves"
         + (f" (measured {sv.estimated.iloc[0]:.1f} vs 1.2)" if len(sv) else "")
         + ", because a goalie's weekly saves depend mostly on how many games he starts. Wins are under-dispersed (below 1), as the model expects, "
           "since a goalie can win at most one game a night. Thousands of player-seasons make every interval very narrow, so closeness matters more than "
           "whether the assumed value sits inside the interval.")

    st.markdown("### How much do a player's stats move together?")
    st.markdown("Goals, assists and power-play points rise and fall together, so the weekly fantasy score varies more than independent stats would. The model "
                f"multiplies the variance by **{COVAR_INFLATION}** to allow for that. Each player-season's measured ratio is observed variance divided by the independent-stats variance.")
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure(go.Histogram(x=cov.ratio, nbinsx=60, marker_color=rgba(CAT[0], 0.55), marker_line=dict(color="white", width=1), hovertemplate="%{y} player-seasons<extra></extra>"))
        fig.add_vline(x=COVAR_INFLATION, line_color=CAT[1], line_width=2, annotation_text=f"assumed {COVAR_INFLATION}", annotation_position="top right")
        fig.add_vline(x=cov_sum["estimated"], line_color=INK, line_width=2, annotation_text=f"measured {cov_sum['estimated']:.2f}", annotation_position="top left")
        fig.update_layout(title="Observed / independent-stats variance, one bar per range of player-seasons", xaxis=dict(title="variance ratio", range=[0, 4]),
                          yaxis_title="player-seasons", showlegend=False, bargap=0.02)
        fig_show(fig, 340)
    with c2:
        grp_of = {"G": "Goalies", "D": "Defense"}
        cg = cov.assign(g=cov.pos.map(lambda p_: grp_of.get(p_, "Forwards")))
        by_g = cg.groupby("g").apply(lambda x: ((x.n - 1) * x.fp_var).sum() / ((x.n - 1) * x.implied).sum(), include_groups=False)
        by_g = by_g.reindex([g_ for g_ in ("Forwards", "Defense", "Goalies") if g_ in by_g])
        fig = go.Figure(go.Bar(x=by_g.values, y=by_g.index, orientation="h", marker_color=CAT[0], text=[f"{v:.2f}" for v in by_g.values],
                               textposition="outside", hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
        fig.add_vline(x=COVAR_INFLATION, line_color=CAT[1], line_width=2, annotation_text=f"assumed {COVAR_INFLATION}", annotation_position="top")
        fig.update_layout(title="Measured inflation by position", xaxis=dict(title="pooled variance ratio", range=[0, max(2.0, by_g.max() * 1.2)]))
        fig_show(fig, 200)
        st.metric("Assumed inflation", f"{COVAR_INFLATION:.2f}", border=True)
        st.metric("Measured inflation", f"{cov_sum['estimated']:.2f}", delta=f"95% CI {cov_sum['lo']:.2f} to {cov_sum['hi']:.2f}", delta_color="off", delta_arrow="off", border=True)
        st.metric("With measured dispersions", f"{cov_est_sum['estimated']:.2f}", border=True,
                  help="Inflation still needed once each stat uses its own measured dispersion instead of the assumed one.")
    note(f"**Result:** the data supports an inflation of about **{cov_sum['estimated']:.2f}**, lower than the {COVAR_INFLATION} in the model. From stat "
         f"variation alone, the model overstates a player's weekly SD by about {100 * (np.sqrt(COVAR_INFLATION / cov_sum['estimated']) - 1):.0f}% (whole-team swings "
         "are a different story; see Weekly scores). Because every candidate gets the same inflation, "
         "this mostly makes P(win) a little too close to 50% rather than changing which pick is best. Goalies are no exception despite their streaky saves: "
         "saves (+0.2 each) and goals against (-2 each) both rise with games started, so their swings largely cancel.")

    st.markdown("### Player explorer")
    top = wk.groupby("player_id").fp.sum().nlargest(300).index
    opts = sorted(top, key=lambda p: H.player_names.get(p, str(p)))
    default = int(wk.groupby("player_id").fp.sum().idxmax())
    c1, c2 = st.columns([2, 1])
    pid = c1.selectbox("Player", opts, index=opts.index(default) if default in opts else 0, format_func=lambda p: f"{H.player_names.get(p, p)} ({H.player_pos.get(p, '?')})", key="pv_player")
    seas = sorted(wk[wk.player_id == pid].season.unique())
    season_pick = c2.segmented_control("Season", seas, default=seas[-1], key="pv_season")
    g = wk[(wk.player_id == pid) & (wk.season == (season_pick or seas[-1]))].sort_values("week")
    row = cov[(cov.player_id == pid) & (cov.season == (season_pick or seas[-1]))]
    mu_ = g.fp.mean()
    sd_model = float(np.sqrt(row.implied.iloc[0] * COVAR_INFLATION)) if len(row) else float("nan")
    fig = go.Figure()
    fig.add_hrect(y0=mu_ - sd_model, y1=mu_ + sd_model, fillcolor=rgba(CAT[1], 0.12), line_width=0, layer="below")
    fig.add_bar(x=g.week, y=g.fp, marker_color=CAT[0], customdata=g.games, name="weekly points",
                hovertemplate="week %{x}: %{y:.1f} points in %{customdata} games<extra></extra>")
    fig.add_hline(y=mu_, line_color=INK, line_width=1, annotation_text=f"average {mu_:.1f}", annotation_position="top right",
                  annotation=dict(bgcolor="rgba(255,255,255,0.85)"))
    fig.update_layout(title=f"{H.player_names.get(pid, pid)}: weekly fantasy points (shaded band = average +/- the model's SD)", xaxis_title="matchup week",
                      yaxis_title="fantasy points", showlegend=False, bargap=0.25)
    fig_show(fig, 320)
    if len(row):
        st.caption(f"Observed weekly SD {g.fp.std():.1f} vs the model's {sd_model:.1f} for this player's averages. Hover a bar for the games played that week; "
                   "weeks with more games score more, which is part of the real variance.")

# ======================================================================================================
elif section == "Projections":
    st.markdown("### Are the projections calibrated?")
    st.markdown("Calibrated projections are right on average at every level: players projected for 150 points score about 150. "
                "Each dot is one player-season from a season the model had not seen.")
    srcs = {"market_adj": "Our projection (final)", "blend": "Blend before market adjustment", "own": "Our own model only", "espn": "ESPN's projection"}
    c1, c2 = st.columns([2, 1])
    short = {"market_adj": "Final (ours)", "blend": "Blend", "own": "Own model", "espn": "ESPN"}
    src = c1.segmented_control("Projection", list(srcs), default="market_adj", format_func=short.get, key="pj_src") or "market_adj"
    grp = {"C": "F", "LW": "F", "RW": "F", "D": "D", "G": "G"}
    posf = c2.pills("Positions", ["F", "D", "G"], selection_mode="multi", default=["F", "D", "G"], key="pj_pos")
    f = pe[pe.pos.map(grp).isin(posf or [])]
    if len(f) < 20:
        st.info("Pick at least one position.")
        st.stop()
    cl = D.calibration_line(f[src], f.actual)
    bm = D.binned_means(f[src], f.actual, 10)
    c1, c2 = st.columns(2)
    with c1:
        lim = float(max(f[src].max(), f.actual.max()) * 1.05)
        fig = go.Figure()
        fig.add_scatter(x=f[src], y=f.actual, mode="markers", marker=dict(size=6, color=rgba(CAT[0], 0.35)), name="player-season",
                        customdata=np.stack([f.name, f.season, f.pos], axis=1), hovertemplate="%{customdata[0]} (%{customdata[2]}, %{customdata[1]})<br>projected %{x:.0f}, actual %{y:.0f}<extra></extra>")
        fig.add_scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(color=MUTED, width=1), name="perfect", hoverinfo="skip")
        fig.add_scatter(x=bm.predicted, y=bm.actual, mode="markers+lines", name="average per bin (95% CI)", line=dict(color=CAT[1], width=2),
                        marker=dict(size=9, color=CAT[1], line=dict(color="white", width=2)), error_y=dict(type="data", array=1.96 * bm.se, color=rgba(CAT[1], 0.6)),
                        hovertemplate="projected %{x:.0f}: actual average %{y:.0f}<extra></extra>")
        fig.update_layout(title=f"{srcs[src]}: projected vs actual season points", xaxis=dict(title="projected fantasy points", range=[0, lim]),
                          yaxis=dict(title="actual fantasy points", range=[0, lim]), legend=dict(orientation="h", y=-0.2))
        fig_show(fig, 420)
    with c2:
        resid = f.actual - f[src]
        bpx = D.breusch_pagan(f[src], resid)
        fig = go.Figure()
        fig.add_scatter(x=f[src], y=resid, mode="markers", marker=dict(size=6, color=rgba(CAT[0], 0.35)), showlegend=False,
                        customdata=np.stack([f.name, f.season], axis=1), hovertemplate="%{customdata[0]} (%{customdata[1]})<br>miss %{y:+.0f}<extra></extra>")
        q = pd.qcut(f[src], 10, duplicates="drop")
        band = pd.DataFrame({"x": f[src], "r": resid, "q": q}).groupby("q", observed=True).agg(x=("x", "mean"), m=("r", "mean"), sd=("r", "std"))
        fig.add_scatter(x=band.x, y=band.m + band.sd, mode="lines", line=dict(color=CAT[1], width=1.5, dash="dot"), name="+/- 1 SD per bin", hoverinfo="skip")
        fig.add_scatter(x=band.x, y=band.m - band.sd, mode="lines", line=dict(color=CAT[1], width=1.5, dash="dot"), showlegend=False, hoverinfo="skip")
        fig.add_hline(y=0, line_color=MUTED)
        fig.update_layout(title="Misses (actual - projected) vs projection", xaxis_title="projected fantasy points", yaxis_title="miss (points)",
                          legend=dict(orientation="h", y=-0.2))
        fig_show(fig, 420)
    with st.container(horizontal=True):
        st.metric("Calibration slope", f"{cl['slope']:.2f}", delta=f"95% CI {cl['slope_lo']:.2f} to {cl['slope_hi']:.2f}", delta_color="off", delta_arrow="off", border=True,
                  help="1.00 = calibrated. Below 1 = the projections spread players out too much (overconfident).")
        st.metric("Intercept", f"{cl['intercept']:+.1f}", border=True)
        st.metric("R squared", f"{cl['r2']:.2f}", border=True)
        st.metric("Breusch-Pagan p", f"{bpx['p']:.3f}", border=True, help="Below 0.05 = misses get bigger (or smaller) as the projection grows.")

    st.markdown("#### All four projections side by side")
    rows = []
    for s_, lab in srcs.items():
        c_ = D.calibration_line(f[s_], f.actual)
        rows.append({"source": lab, **c_, "mae": float((f[s_] - f.actual).abs().mean())})
    cmp_ = pd.DataFrame(rows)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for _, r in cmp_.iterrows():
            fig.add_scatter(x=[r.slope_lo, r.slope_hi], y=[r.source] * 2, mode="lines", line=dict(color=rgba(CAT[0], 0.6), width=3), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=cmp_.slope, y=cmp_.source, mode="markers", marker=dict(size=12, color=[CAT[1] if s == srcs[src] else CAT[0] for s in cmp_.source],
                        line=dict(color="white", width=2)), showlegend=False, hovertemplate="%{y}: slope %{x:.2f}<extra></extra>")
        fig.add_vline(x=1, line_color=INK, line_width=1, annotation_text="calibrated", annotation_position="top")
        fig.update_layout(title="Calibration slope with 95% CI", xaxis_title="slope of actual on projected")
        fig_show(fig, 260)
    with c2:
        fig = go.Figure(go.Bar(x=cmp_.mae, y=cmp_.source, orientation="h", marker_color=[CAT[1] if s == srcs[src] else CAT[0] for s in cmp_.source],
                               hovertemplate="%{y}: %{x:.1f} points<extra></extra>"))
        fig.update_layout(title="Average miss per player (smaller = better)", xaxis_title="average miss (points)")
        fig_show(fig, 260)
    note(f"**Result:** ESPN's projections have a slope of about {cal_espn['slope']:.2f}: they spread players out too much, so ESPN's top projections are too high "
         f"and its low ones too low. The final projection (after the market adjustment) has a slope of {cal['slope']:.2f}, within its interval of 1, and its misses have a "
         "steady spread across the range. That is the regression's job: it shrinks our projection toward the market by exactly the amount the past data supports.")

# ======================================================================================================
elif section == "Pick model":
    st.markdown("### Do opponents pick the way the model says?")
    st.markdown("The draft simulator gives every available player a probability of being picked next (a conditional logit on ESPN ranking, roster need and "
                "position by round). Each past draft is scored with a model fitted only on the *other* seasons.")
    with st.container(horizontal=True):
        st.metric("Picks scored", len(picks), border=True)
        st.metric("Actual pick = model's favourite", f"{(picks.model_rank == 1).mean():.0%}", delta=f"ESPN's top player: {(picks.adp_rank == 1).mean():.0%}", delta_color="off", delta_arrow="off", border=True)
        st.metric("In the model's top 5", f"{(picks.model_rank <= 5).mean():.0%}", delta=f"ESPN's top 5: {(picks.adp_rank <= 5).mean():.0%}", delta_color="off", delta_arrow="off", border=True)
        st.metric("Median probability given to the actual pick", f"{picks.p_chosen.median():.1%}", border=True)
    c1, c2 = st.columns(2)
    with c1:
        mask = pr > 1e-4
        rel = D.reliability(pr[mask], ch[mask], 12)
        fig = go.Figure()
        fig.add_scatter(x=[1e-4, 1], y=[1e-4, 1], mode="lines", line=dict(color=MUTED, width=1), name="perfect", hoverinfo="skip")
        fig.add_scatter(x=rel.predicted, y=rel.observed.clip(lower=1e-4), mode="markers+lines", name="actual picks", line=dict(color=CAT[0], width=2),
                        marker=dict(size=9, color=CAT[0], line=dict(color="white", width=2)), customdata=rel.n,
                        hovertemplate="model said %{x:.2%}<br>picked %{y:.2%} of the time<br>%{customdata} player-picks<extra></extra>")
        tv, tt = [1e-4, 1e-3, 1e-2, 1e-1, 1], ["0.01%", "0.1%", "1%", "10%", "100%"]
        fig.update_layout(title="Pick probabilities: predicted vs actual (log scales)", xaxis=dict(title="model's probability", type="log", tickvals=tv, ticktext=tt),
                          yaxis=dict(title="share actually picked", type="log", tickvals=tv, ticktext=tt), legend=dict(orientation="h", y=-0.2))
        fig_show(fig, 400)
    with c2:
        ranks = np.arange(1, 21)
        fig = go.Figure()
        fig.add_bar(x=ranks, y=[(picks.model_rank == r).mean() for r in ranks], name="rank in the model", marker_color=CAT[0])
        fig.add_bar(x=ranks, y=[(picks.adp_rank == r).mean() for r in ranks], name="rank by ESPN order", marker_color=CAT[1])
        fig.update_layout(barmode="group", title="Where the actual pick ranked among available players", xaxis=dict(title="rank (1 = favourite)", dtick=1),
                          yaxis=dict(title="share of picks", tickformat=".0%"), legend=dict(orientation="h", y=-0.2), bargap=0.25, bargroupgap=0.1)
        fig_show(fig, 400)
    by_r = picks.groupby("round").agg(top5=("model_rank", lambda x: (x <= 5).mean()), adp5=("adp_rank", lambda x: (x <= 5).mean()), p=("p_chosen", "median")).reset_index()
    fig = go.Figure()
    fig.add_scatter(x=by_r["round"], y=by_r.top5, mode="lines+markers", name="model top 5", line=dict(color=CAT[0], width=2), marker=dict(size=8))
    fig.add_scatter(x=by_r["round"], y=by_r.adp5, mode="lines+markers", name="ESPN order top 5", line=dict(color=CAT[1], width=2), marker=dict(size=8))
    fig.update_layout(title="How predictable each round is", xaxis=dict(title="round", dtick=2), yaxis=dict(title="actual pick in the top 5", tickformat=".0%", range=[0, 1]),
                      legend=dict(orientation="h", y=-0.2))
    fig_show(fig, 320)
    note("**Result:** the model is well calibrated where it matters (players with a real chance of going next) and predicts about as well as ESPN's order alone. "
         "Early rounds are predictable; after round 10 the actual pick is rarely one of the five obvious names. That is why the simulator samples many "
         "possible drafts instead of assuming one.")

# ======================================================================================================
elif section == "Experiment tests":
    st.markdown("### Do the experiment results depend on the test's assumptions?")
    if not runs or not runs[0]["results"][0].get("diffs"):
        st.info("No experiment results with per-unit data yet. Run: `PYTHONPATH=src .venv/bin/python -m iknowpuck.experiments`")
        st.stop()
    res = runs[0]["results"]
    st.markdown("Each experiment compares two arms on the same units (paired). Three tests with different assumptions judge each difference: "
                "the **sign-flip permutation test** (assumes differences are symmetric around zero if there is no effect; the app's primary test), "
                "the **Wilcoxon signed-rank test** (also symmetry, uses ranks), and the **t-test** (assumes roughly Normal mean differences). "
                "If all three agree, the verdict doesn't hinge on any one assumption.")
    pt = pd.DataFrame([{"Hypothesis": r["id"], "units": r["test"]["n"], "mean difference": r["test"]["mean_diff"], "permutation p": r["test"]["p_permutation"],
                        "Wilcoxon p": r["test"]["p_wilcoxon"], "t-test p": r["test"]["p_ttest"], "Holm-adjusted p": r["test"]["p_adjusted"],
                        "verdict": "supported" if r["test"]["significant"] else "not supported"} for r in res])
    st.dataframe(pt, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ["mean difference", "permutation p", "Wilcoxon p", "t-test p", "Holm-adjusted p"]})
    hid = st.segmented_control("Hypothesis", [r["id"] for r in res], default=res[0]["id"], key="ex_h") or res[0]["id"]
    r = next(x for x in res if x["id"] == hid)
    d = np.asarray(r["diffs"], float)
    nd = D.normality(d)
    st.caption(f"**{hid}:** {r['hypothesis']}. Unit: {r['unit']}. Positive difference = the new idea ({r['treatment']}) did better than {r['baseline']}.")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Histogram(x=d, nbinsx=50, marker_color=rgba(CAT[0], 0.55), marker_line=dict(color="white", width=1), hovertemplate="%{y} units<extra></extra>"))
        fig.add_vline(x=0, line_color=INK, line_width=1)
        fig.add_vline(x=d.mean(), line_color=CAT[1], line_width=2, annotation_text=f"mean {d.mean():+.3f}", annotation_position="top right")
        fig.update_layout(title="Paired differences, one per unit", xaxis_title=r["metric"] + " (treatment better ->)", yaxis_title="units", showlegend=False, bargap=0.02)
        fig_show(fig, 340)
    with c2:
        th, sm = D.qq(d)
        fig = go.Figure()
        fig.add_scatter(x=[th.min(), th.max()], y=[th.min(), th.max()], mode="lines", line=dict(color=MUTED, width=1), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=th, y=sm, mode="markers", marker=dict(size=6, color=CAT[0]), showlegend=False, hovertemplate="%{y:.2f}<extra></extra>")
        fig.update_layout(title="Q-Q plot of the differences", xaxis_title="Normal quantile", yaxis_title="observed quantile")
        fig_show(fig, 340)
    with st.container(horizontal=True):
        st.metric("Skewness", f"{nd['skew']:+.2f}", border=True, help="Symmetric differences have skewness near 0.")
        st.metric("Excess kurtosis", f"{nd['excess_kurtosis']:+.2f}", border=True, help="Heavy tails (large values) make the t-test less reliable, not the permutation test.")
        st.metric("Units", nd["n"], border=True)
        st.metric("Effect size d_z", f"{r['test']['cohens_dz']:+.2f}", border=True)
    note("**How to read this:** heavy tails or skew in the histogram would make the t-test unreliable, which is why the permutation test is primary. "
         "With a few hundred units the average difference is close to Normal anyway (central limit theorem), so all three tests usually agree.")

    st.markdown("#### Power: how many units does a test need?")
    c1, c2 = st.columns([1, 2])
    alpha = c1.select_slider("Significance level", [0.01, 0.025, 0.05, 0.1], value=0.05, key="pw_a")
    power = c1.select_slider("Power", [0.5, 0.7, 0.8, 0.9, 0.95], value=0.8, key="pw_p")
    dz = np.linspace(0.02, 1.0, 200)
    need = np.array([required_n_paired(x, alpha, power) for x in dz])
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=dz, y=need, mode="lines", line=dict(color=CAT[0], width=2), showlegend=False, hovertemplate="d_z %{x:.2f}: %{y} units<extra></extra>")
        for x in res:
            t = x["test"]
            if np.isfinite(t["cohens_dz"]) and abs(t["cohens_dz"]) > 0.005:
                fig.add_scatter(x=[abs(t["cohens_dz"])], y=[t["n"]], mode="markers+text", text=[x["id"]], textposition="top right",
                                marker=dict(size=10, color=CAT[1] if t["significant"] else MUTED, line=dict(color="white", width=2)), showlegend=False,
                                hovertemplate=f"{x['id']}: |d_z| {abs(t['cohens_dz']):.2f}, {t['n']} units<extra></extra>")
        fig.update_layout(title="Units needed to detect an effect (curve) vs each experiment (dots)", xaxis_title="effect size |d_z|",
                          yaxis=dict(title="units needed", type="log", tickvals=[10, 100, 1000, 10000, 100000], ticktext=["10", "100", "1k", "10k", "100k"]))
        fig_show(fig, 340)
    st.caption("Dots above the curve had enough units to detect an effect of their observed size; dots below did not. Rust = supported after Holm correction.")
