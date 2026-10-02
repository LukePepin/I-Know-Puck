"""Theory tests: a manager's hunch about hockey, tested the same way as the models.

Theory 1: when an NHL team beats its preseason expectations, all of its players benefit."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import CAT, FAINT, INK, MUTED, app_state, fig_show, get_panel, load_runs, note, rgba, selected_custom

from iknowpuck import team_effects as T
from iknowpuck.history import NHL_ABBREV

A = app_state()
b = A["b"]
H = b.history
st.markdown("## Theory tests")
st.caption("Your hunches about hockey, checked against real results with the same care as the app's own models.")
if H is None:
    st.warning("Theory tests need your league's game logs (ESPN cookies in the .env file).")
    st.stop()

GROUP_NAME = {"F": "Forwards", "D": "Defense", "G": "Goalies"}
GROUP_COLOR = {"F": CAT[0], "D": CAT[1], "G": CAT[2]}  # same forward / defense / goalie colours as the Optimization page


@st.cache_resource(show_spinner="Matching every game to the NHL team it was played for...")
def theory_data(_b, season: int):
    rates = T.projected_rates(get_panel(season), _b.settings, _b.history.seasons)
    return T.stints(_b.history.gamelogs, rates), T.halves(_b.history.gamelogs, _b.history.schedules, rates)


@st.cache_data(max_entries=64, show_spinner="Running the team-level permutation test...")
def effect(_d: pd.DataFrame, key: tuple, n: int = 2000) -> dict:
    return T.team_effect(_d.reset_index(drop=True), n_perm=n, n_boot=n)


@st.cache_data(max_entries=8, show_spinner="Testing whether a hot first half carries over...")
def carry_over(_w: pd.DataFrame, key: tuple) -> tuple[dict, dict]:
    return T.persistence(_w, n_perm=2000), T.loso_errors(_w)


def team_abbr(t) -> str:
    return NHL_ABBREV.get(int(t), f"team {t}")


stints, halves = theory_data(b, A["season"])
seasons_all = sorted(stints.season.unique())

st.markdown("### Theory 1: when an NHL team beats expectations, all its players benefit")
with st.container(border=True):
    st.markdown(
        "**How it is tested.** The expectation is **ESPN's preseason projection** for every player. A player's **surprise** is his fantasy points "
        "per game on that NHL team compared with his projection (+10% = he scored 10% more than projected). His **team's surprise** is the same "
        "comparison for all his **teammates** combined, leaving him out, so his own points can't make his team look good. Players on one team "
        "share one team result, so the p-value comes from shuffling whole teams, not players. Teams come from the game logs (the team he played "
        "each game for), which exist for your league's seasons " + ", ".join(map(str, seasons_all)) + "."
    )

with st.container(border=True):
    c1, c2 = st.columns(2)
    measure = c1.segmented_control("Team success measured by", ["scoring", "wins"], default="scoring",
                                   format_func={"scoring": "Scoring vs projection", "wins": "Wins vs projection"}.get, key="tt_measure") or "scoring"
    grp_opts = ["F", "D", "G"] if measure == "scoring" else ["F", "D"]
    groups = c2.pills("Players", grp_opts, selection_mode="multi", default=["F", "D"], format_func=GROUP_NAME.get, key=f"tt_groups_{measure}") or ["F", "D"]
    c3, c4 = st.columns(2)
    seasons = c3.pills("Seasons", seasons_all, selection_mode="multi", default=seasons_all, key="tt_seasons") or seasons_all
    min_games = c4.slider("Minimum games on the team", 10, 60, 20, 5, key="tt_min")
    st.caption("Wins vs projection compares the team's goalie wins with ESPN's projected wins for its goalies (skaters only, since a goalie's own wins are the measure). "
               "Goalies' relative surprise is much noisier than skaters', so they are off by default.")

d_all = T.add_team_surprise(stints, min_games, measure)
d = d_all[d_all.season.isin(seasons) & d_all.grp.isin(groups)].reset_index(drop=True)
if d.groupby(T.KEYS).ngroups < 10:
    st.info("Too few teams for this selection.")
    st.stop()
key = (measure, tuple(sorted(groups)), tuple(sorted(seasons)), min_games)
eff = effect(d, key)
ic = T.icc(d)
qt = T.by_quantile(d)
unit = "+10% team scoring" if measure == "scoring" else "+10% team wins"

with st.container(horizontal=True):
    st.metric(f"Player gain per {unit}", f"{10 * eff['slope']:+.1f}%", delta=f"95% CI {10 * eff['lo']:+.1f}% to {10 * eff['hi']:+.1f}%",
              delta_color="off", delta_arrow="off", border=True, help="Average change in a player's own points-per-game surprise.")
    st.metric("Team-level p-value", f"{eff['p']:.3f}", border=True, help="Chance of a link this strong if team results were handed out at random.")
    st.metric("Share of a player's surprise explained by his team", f"{max(ic, 0):.0%}", border=True, help="Intraclass correlation within team-seasons.")
    st.metric("Beat projection on the hottest fifth of teams", f"{qt.beat.iloc[-1]:.0%}", delta=f"coldest fifth: {qt.beat.iloc[0]:.0%}",
              delta_color="off", delta_arrow="off", border=True)
    st.metric("Player-seasons", f"{eff['n']:,}", delta=f"{eff['teams']} team-seasons", delta_color="off", delta_arrow="off", border=True)

real = eff["p"] < 0.05 and eff["lo"] > 0
strong = 10 * eff["slope"] >= 5 and ic >= 0.2
most = qt.beat.iloc[-1] >= 0.75
verdict = ("**Verdict: true, and strong.**" if real and strong and most else "**Verdict: partly true.**" if real else "**Verdict: not supported.**")
note(verdict + (f" Players on teams that beat expectations do score more than projected, and it isn't luck (p = {eff['p']:.3f}). But the lift is modest: "
                f"a team {'scoring' if measure == 'scoring' else 'winning'} 10% more than expected lifts each teammate "
                f"by about {10 * eff['slope']:.1f}% on average, and the team explains only about {max(ic, 0):.0%} of why a player beats or misses his projection. "
                f"Even on the hottest fifth of teams, {1 - qt.beat.iloc[-1]:.0%} of players still missed their projection, so it does not lift *all* players."
                if real else f" In this selection the link between a team's surprise and its players' surprise could be chance (p = {eff['p']:.3f})."))

section = st.segmented_control("Section", ["The evidence", "Team explorer", "Who benefits?", "Can you use it?"], default="The evidence",
                               key="tt_section", label_visibility="collapsed")

# ======================================================================================================
if section == "The evidence" or section is None:
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        for g in groups:
            x = d[d.grp == g]
            fig.add_scatter(x=x.team_surprise, y=x.surprise, mode="markers", name=GROUP_NAME[g], marker=dict(size=7, color=rgba(GROUP_COLOR[g], 0.5)),
                            customdata=np.stack([x.name, x.pro_team_id.map(team_abbr), x.season, x.games, x.fp / x.games, x.proj_rate], axis=1),
                            hovertemplate="%{customdata[0]} (%{customdata[1]}, %{customdata[2]})<br>%{customdata[3]} games: %{customdata[4]:.2f} pts/game vs "
                                          "%{customdata[5]:.2f} projected<br>player %{y:+.0%}, teammates %{x:+.0%}<extra></extra>")
        xs = np.linspace(d.team_surprise.quantile(0.01), d.team_surprise.quantile(0.99), 50)
        fig.add_scatter(x=xs, y=d.surprise.mean() + eff["slope"] * (xs - d.team_surprise.mean()), mode="lines", name="average trend",
                        line=dict(color=INK, width=2.5), hoverinfo="skip")
        fig.add_hline(y=0, line_color=MUTED, line_width=1)
        fig.add_vline(x=0, line_color=MUTED, line_width=1)
        fig.update_layout(title="Each dot is a player-season: his surprise vs his teammates' surprise",
                          xaxis=dict(title="teammates vs projection" if measure == "scoring" else "team wins vs projection", tickformat="+.0%"),
                          yaxis=dict(title="player's points per game vs projection", tickformat="+.0%", range=[-0.8, 1.0]), legend=dict(orientation="h", y=-0.2))
        fig_show(fig, 460)
        st.caption("The y-axis stops at -80% and +100%; a few extreme player-seasons sit outside it. Hover a dot for the player.")
    with c2:
        labels = ["coldest fifth", "2nd", "middle", "4th", "hottest fifth"][: len(qt)]
        fig = go.Figure(go.Bar(x=labels, y=qt.beat, marker_color=[CAT[1] if v < d.beat.mean() else CAT[0] for v in qt.beat],
                               error_y=dict(type="data", symmetric=False, array=qt.hi - qt.beat, arrayminus=qt.beat - qt.lo, color=MUTED, thickness=1.5),
                               customdata=np.stack([qt.team, qt.n, qt.surprise], axis=1),
                               hovertemplate="%{x} of teams (teammates %{customdata[0]:+.0%})<br>%{y:.0%} of %{customdata[1]} players beat their projection"
                                             "<br>average player surprise %{customdata[2]:+.0%}<extra></extra>"))
        fig.add_hline(y=d.beat.mean(), line_color=INK, line_width=1, annotation_text=f"all players {d.beat.mean():.0%}", annotation_position="top left",
                      annotation=dict(bgcolor="rgba(255,255,255,0.85)"))
        fig.update_layout(title="Share of players who beat their projection, by how their team did", yaxis=dict(title="beat projection", tickformat=".0%", range=[0, 1]),
                          xaxis_title="teams grouped by teammates' surprise")
        fig_show(fig, 460)
    st.caption("ESPN's per-game projections run a little high for most players, so fewer than half beat them in a typical season. The comparison that matters is "
               "hot teams vs cold teams.")

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure(go.Histogram(x=10 * eff["null"], nbinsx=50, marker_color=rgba(CAT[0], 0.5), marker_line=dict(color="white", width=1),
                                     hovertemplate="%{y} shuffles<extra></extra>"))
        fig.add_vline(x=10 * eff["slope"], line_color=CAT[1], line_width=2.5, annotation_text=f"real teams: {10 * eff['slope']:+.1f}%", annotation_position="top right")
        fig.update_layout(title="Player gain per +10% team surprise, real teams vs 2,000 random reshuffles of teams", xaxis_title="player gain per +10% team surprise (%)",
                          yaxis_title="shuffles", showlegend=False, bargap=0.02)
        fig_show(fig, 320)
    with c2:
        st.markdown(
            "**Why shuffle teams?** Within each season, the team results are handed out to teams at random, and every player on a team gets the same "
            "random team's result. If team success didn't matter, the real link would look like one of these shuffles. "
            f"Only **{(eff['null'] >= eff['slope']).mean():.1%}** of shuffles show a link as strong as the real one."
        )
    with st.expander("Technical details"):
        st.markdown(
            f"- Player surprise = points on the team / (games on the team x ESPN projected points per game) - 1. Per game, so injuries don't count as under-performance.\n"
            f"- Team surprise (scoring) = the same ratio pooled over the player's teammates with an ESPN projection (leave one out). Goalies are compared with "
            f"their whole team's skaters.\n"
            f"- Slope from ordinary least squares; 95% CI from 2,000 bootstrap resamples of whole team-seasons; one-sided permutation p-value (2,000 shuffles of "
            f"team-season results within season).\n"
            f"- Correlation r = {eff['r']:.3f}; intraclass correlation of player surprise within team-seasons = {ic:.3f}.\n"
            "- Some of the link is mechanical: a teammate's goal often gives the player an assist. That is part of the benefit the theory describes."
        )

# ======================================================================================================
elif section == "Team explorer":
    tt = T.team_table(stints)
    season_pick = st.segmented_control("Season", seasons_all, default=seasons_all[-1], key="tt_ex_season") or seasons_all[-1]
    t = tt[tt.season == season_pick].dropna(subset=["scoring_surprise"]).copy()
    league = t.t_fp.sum() / t.t_exp.sum() - 1  # ESPN's per-game projections miss the whole league by this much
    t["relative"] = (1 + t.scoring_surprise) / (1 + league) - 1
    t = t.sort_values("relative")
    fig = go.Figure(go.Bar(x=t.pro_team_id.map(team_abbr), y=t.relative, marker_color=[CAT[0] if v > 0 else CAT[1] for v in t.relative],
                           customdata=np.stack([t.pro_team_id, t.wins.fillna(np.nan), t.exp_wins.fillna(np.nan), t.skaters, t.scoring_surprise], axis=1),
                           hovertemplate="%{x}: %{y:+.0%} vs the average team (%{customdata[4]:+.0%} vs raw projection)<br>%{customdata[1]:.0f} goalie wins vs "
                                         "%{customdata[2]:.0f} projected<br>%{customdata[3]} projected skaters<extra></extra>"))
    fig.add_hline(y=0, line_color=INK, line_width=1)
    fig.update_layout(title=f"How each NHL team's skaters did against ESPN's projections in {season_pick}, compared with the average team (click a team)",
                      yaxis=dict(title="scoring vs projection, relative to league", tickformat="+.0%"), xaxis=dict(title="", tickangle=-45))
    ev = fig_show(fig, 380, key="tt_team_bar", on_select="rerun")
    st.caption(f"ESPN's per-game projections ran {abs(league):.0%} {'high' if league < 0 else 'low'} for the league as a whole in {season_pick}, so each team is "
               "shown relative to that league-wide miss: blue teams beat expectations more than the average team, rust teams less.")
    clicked = selected_custom(ev)
    teams_sorted = [int(x) for x in t.pro_team_id.tolist()[::-1]]
    # a new click on the chart selects that team; otherwise keep whatever was picked in the dropdown
    if clicked is not None and int(clicked) != st.session_state.get("tt_last_click"):
        st.session_state["tt_team"] = st.session_state["tt_last_click"] = int(clicked)
    if st.session_state.get("tt_team") not in teams_sorted:
        st.session_state["tt_team"] = teams_sorted[0]
    team = st.selectbox("Team", teams_sorted, format_func=team_abbr, key="tt_team")
    p = d_all[(d_all.season == season_pick) & (d_all.pro_team_id == team) & d_all.grp.isin(["F", "D", "G"])].sort_values("surprise")
    if len(p):
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure(go.Bar(x=p.surprise, y=p.name + " (" + p.pos + ")", orientation="h", marker_color=[CAT[0] if v > 0 else CAT[1] for v in p.surprise],
                                   customdata=np.stack([p.games, p.fp / p.games, p.proj_rate], axis=1),
                                   hovertemplate="%{y}<br>%{customdata[0]} games: %{customdata[1]:.2f} pts/game vs %{customdata[2]:.2f} projected<extra></extra>"))
            fig.add_vline(x=0, line_color=INK, line_width=1)
            fig.update_layout(title=f"{team_abbr(team)} {season_pick}: each player vs his projection", xaxis=dict(title="points per game vs projection", tickformat="+.0%"))
            fig_show(fig, 80 + 26 * len(p))
        with c2:
            row = t[t.pro_team_id == team].iloc[0]
            st.metric("Team scoring vs the average team", f"{row.relative:+.0%}", delta=f"{row.scoring_surprise:+.0%} vs raw projection",
                      delta_color="off", delta_arrow="off", border=True)
            if pd.notna(row.wins_surprise):
                st.metric("Goalie wins vs projection", f"{row.wins:.0f} vs {row.exp_wins:.0f}", delta=f"{row.wins_surprise:+.0%}", border=True)
            st.metric("Players who beat their projection", f"{p.beat.sum()} of {len(p)}", border=True)
            st.caption(f"Players with {min_games}+ games for this team and an ESPN projection.")
    st.markdown("#### Do teams that score more than expected also win more than expected?")
    ta = tt.dropna(subset=["scoring_surprise", "wins_surprise"])
    fig = go.Figure()
    for i, s_ in enumerate(seasons_all):
        x = ta[ta.season == s_]
        fig.add_scatter(x=x.scoring_surprise, y=x.wins_surprise, mode="markers", name=str(s_), marker=dict(size=10, color=CAT[i % len(CAT)], line=dict(color="white", width=1)),
                        customdata=x.pro_team_id.map(team_abbr), hovertemplate="%{customdata} " + str(s_) + "<br>scoring %{x:+.0%}, wins %{y:+.0%}<extra></extra>")
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_layout(title=f"Team scoring surprise vs wins surprise (correlation {ta.scoring_surprise.corr(ta.wins_surprise):+.2f})",
                      xaxis=dict(title="skater scoring vs projection", tickformat="+.0%"), yaxis=dict(title="goalie wins vs projection", tickformat="+.0%"),
                      legend=dict(orientation="h", y=-0.2))
    fig_show(fig, 380)

# ======================================================================================================
elif section == "Who benefits?":
    st.markdown("The theory says **all** players benefit. If it held, the lift would show up for every kind of player. Each dot below is the same test "
                "run on one kind of player, with its 95% interval.")
    dd = d_all[d_all.season.isin(seasons)].copy()
    sk = dd[dd.grp != "G"].copy()
    sk["role_rank"] = sk.groupby(T.KEYS + ["grp"]).proj_rate.rank(ascending=False, pct=True)
    subsets = [("Forwards", dd[dd.grp == "F"]), ("Defense", dd[dd.grp == "D"])]
    if measure == "scoring":
        subsets.append(("Goalies", dd[dd.grp == "G"]))
    subsets += [("Top of the lineup (top third by projection)", sk[sk.role_rank <= 1 / 3]), ("Middle of the lineup", sk[(sk.role_rank > 1 / 3) & (sk.role_rank <= 2 / 3)]),
                ("Depth players (bottom third)", sk[sk.role_rank > 2 / 3])]
    rows = []
    for lab, x in subsets:
        if x.groupby(T.KEYS).ngroups >= 10:
            e = effect(x, key + (lab,), 800)
            rows.append({"group": lab, "gain": 10 * e["slope"], "lo": 10 * e["lo"], "hi": 10 * e["hi"], "p": e["p"], "n": e["n"]})
    rt = pd.DataFrame(rows)
    sk_rows = rt[rt.group != "Goalies"]
    x0, x1 = min(sk_rows.lo.min(), 0) - 2, sk_rows.hi.max() + 2  # goalies' interval is far wider; keep the skater rows readable
    fig = go.Figure()
    for _, r in rt.iterrows():
        fig.add_scatter(x=[max(r.lo, x0), min(r.hi, x1)], y=[r.group] * 2, mode="lines", line=dict(color=rgba(CAT[0], 0.55), width=3), hoverinfo="skip", showlegend=False)
        if r.group == "Goalies":
            fig.add_annotation(x=x1, y=r.group, text=f"{r.gain:+.0f}% (CI {r.lo:+.0f} to {r.hi:+.0f}), off scale: too few goalies", showarrow=False,
                               xanchor="right", yshift=12, font=dict(size=11, color=INK))
    fig.add_scatter(x=rt.gain.clip(x0, x1), y=rt.group, mode="markers", marker=dict(size=12, color=[CAT[0] if lo > 0 else MUTED for lo in rt.lo], line=dict(color="white", width=2)),
                    customdata=np.stack([rt.lo, rt.hi, rt.p, rt.n], axis=1), showlegend=False,
                    hovertemplate="%{y}<br>%{x:+.1f}% per +10% team (95%% CI %{customdata[0]:+.1f} to %{customdata[1]:+.1f})<br>p = %{customdata[2]:.3f}, "
                                  "%{customdata[3]} player-seasons<extra></extra>")
    fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_layout(title="Player gain per +10% team surprise, by kind of player (blue = interval above zero)", xaxis=dict(title="player gain (%)", range=[x0, x1]),
                      yaxis=dict(autorange="reversed"))
    fig_show(fig, 90 + 46 * len(rt))
    hot = d[d.team_surprise > 0.05]
    role = rt.set_index("group")
    top_lab, dep_lab = "Top of the lineup (top third by projection)", "Depth players (bottom third)"
    role_txt = (f" The lift is not even: players at the **top of the lineup** gained {role.loc[top_lab, 'gain']:+.1f}% per +10% "
                f"(95% CI {role.loc[top_lab, 'lo']:+.1f} to {role.loc[top_lab, 'hi']:+.1f}), while **depth players** gained {role.loc[dep_lab, 'gain']:+.1f}%. "
                "Stars' output barely moves with their team's; the supporting cast's does."
                if top_lab in role.index and dep_lab in role.index else "")
    note(f"**All players?** On teams whose other players beat projections by more than 5%, **{(~hot.beat).mean():.0%}** of players still missed their own "
         f"projection ({len(hot)} player-seasons). A hot team raises the odds for its players; it doesn't guarantee anything for any one of them." + role_txt +
         " Wide intervals mean that group has too few player-seasons to be sure.")

# ======================================================================================================
elif section == "Can you use it?":
    st.markdown("Knowing a team had a great season only helps **after** the season. The useful question is whether a team that is hot at the halfway "
                "point lifts its players in the **second half**, beyond what each player's own first half already tells you. If it did, buying "
                "players on hot teams at the trade deadline would pay.")
    pz, er = carry_over(halves, ("halves", len(halves)))
    c1, c2 = st.columns([3, 2])
    with c1:
        xh = halves
        fig = go.Figure()
        for g in ["F", "D"]:
            x = xh[xh.grp == g]
            fig.add_scatter(x=x.team1, y=x.surprise2, mode="markers", name=GROUP_NAME[g], marker=dict(size=7, color=rgba(GROUP_COLOR[g], 0.45)),
                            customdata=np.stack([x.name, x.pro_team_id.map(team_abbr), x.season, x.own1], axis=1),
                            hovertemplate="%{customdata[0]} (%{customdata[1]}, %{customdata[2]})<br>teammates' first half %{x:+.0%}<br>his first half "
                                          "%{customdata[3]:+.0%}, second half %{y:+.0%}<extra></extra>")
        xs = np.linspace(xh.team1.quantile(0.01), xh.team1.quantile(0.99), 50)
        fig.add_scatter(x=xs, y=xh.surprise2.mean() + pz["team_alone"] * (xs - xh.team1.mean()), mode="lines", name="trend (team only)", line=dict(color=INK, width=2.5))
        fig.add_hline(y=0, line_color=MUTED, line_width=1)
        fig.update_layout(title="Teammates' first half vs the player's second half", xaxis=dict(title="teammates' first half vs projection", tickformat="+.0%"),
                          yaxis=dict(title="player's second half vs projection", tickformat="+.0%", range=[-0.8, 1.0]), legend=dict(orientation="h", y=-0.2))
        fig_show(fig, 420)
    with c2:
        st.markdown("**What predicts a player's second half?**")
        st.dataframe(pd.DataFrame([
            {"uses": "team only", "+10% team": f"{10 * pz['team_alone']:+.1f}%", "+10% own": ""},
            {"uses": "team + own", "+10% team": f"{10 * pz['team']:+.1f}%", "+10% own": f"{10 * pz['own']:+.1f}%"},
        ]), hide_index=True)
        st.caption("Change in the player's second half for +10% in his team's first half, and for +10% in his own first half.")
        st.metric("Team effect once his own first half is known", f"{10 * pz['team']:+.1f}%", delta=f"permutation p = {pz['p']:.2f}", delta_color="off",
                  delta_arrow="off", border=True)
        st.caption(f"{pz['n']} skater-seasons with 15+ games in each half on the same team.")
    means = {k: v.mean() for k, v in er.items()}
    ses = {k: v.std(ddof=1) / np.sqrt(len(v)) for k, v in er.items()}
    fig = go.Figure(go.Bar(x=list(means.values()), y=["ESPN projection", "+ his own first half", "+ his team's first half"], orientation="h",
                           marker_color=[MUTED, CAT[0], CAT[1]], error_x=dict(type="data", array=[1.96 * ses[k] for k in means], color=INK, thickness=1.5),
                           hovertemplate="%{y}: %{x:.3f} points per game<extra></extra>"))
    fig.update_layout(title="Average second-half miss, each season predicted from the other seasons (smaller = better)", xaxis_title="error (fantasy points per game)",
                      yaxis=dict(autorange="reversed"))
    fig_show(fig, 250)
    runs = load_runs(1)
    h5 = next((r for r in runs[0]["results"] if r["id"] == "H5"), None) if runs else None
    note(f"**Result:** on its own, a hot team does point to a better second half (+{10 * pz['team_alone']:.1f}% per +10%). But that is because hot teams are "
         f"full of players who are themselves hot. Once each player's own first half is known, the team adds {10 * pz['team']:+.1f}% (p = {pz['p']:.2f}), which is nothing. "
         f"Out of sample, adding the team makes predictions {'slightly worse' if means['own + team'] > means['own'] else 'no better'} "
         f"({means['own + team']:.3f} vs {means['own']:.3f} points per game)."
         + (f" Experiment H5 records this: difference {h5['test']['mean_diff']:+.4f} (95% CI {h5['test']['ci_low']:+.4f} to {h5['test']['ci_high']:+.4f}), "
            f"Holm-adjusted p = {h5['test']['p_adjusted']:.2f}." if h5 else "")
         + " **For your team:** judge a player by his own season so far, not by his team's.")
    st.caption("H5 was added after this exploratory look at the same three seasons, so it is a check, not a pre-registered test. Next season's data will be the clean test.")
