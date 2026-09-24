"""League history: who wins and how, managers, past drafts, trades and pickups, injuries."""

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from ui import GREY, NAVY, PALETTE, RUST, SAGE, SAND, app_state, fig_show, note, selected_custom

from iknowpuck.history import NHL_ABBREV
from iknowpuck.strategy import STRATEGY_COLS, beyond_draft
from iknowpuck.summaries import league_takeaways, manager_reports, ordinal

A = app_state()
b, S, names, my_team = A["b"], A["S"], A["names"], A["my_team"]
H = b.history
ms = b.strategy.copy()
mgr_names = b.manager_names
active = set(b.managers.loc[b.managers.active, "owner_id"].astype(str)) if len(b.managers) else set()
my_owner = str(b.manager_of_team.get(my_team))


def who(m) -> str:
    n_ = mgr_names.get(str(m), str(m))
    return n_ if str(m) in active else f"{n_} (former)"


group_of = {}
if b.spectral is not None and b.cluster_names:
    for m, c in b.spectral.groups().items():
        group_of[str(m)] = b.cluster_names.get(int(c), {}).get("name", f"Group {c + 1}")
ms["manager"] = ms["owner_id"].map(who)
ms["group"] = ms["owner_id"].map(group_of)


@st.cache_resource(max_entries=4, show_spinner="Running the stacking test...")
def stacking_test(_b, season: int):
    return _b.history.stacking_null(_b.drafts, n=400)


@st.cache_resource(max_entries=4, show_spinner=False)
def draft_table(_b, season: int) -> pd.DataFrame:
    """Every historical pick with player details and actual fantasy points that season."""
    h = _b.history
    d = _b.drafts.copy()
    d["player"] = d.player_id.map(h.player_names)
    d["pos"] = d.player_id.map(h.player_pos).fillna(d.get("pos"))
    d["nhl"] = [NHL_ABBREV.get(h.player_team.get((s, p)), "?") for s, p in zip(d.season, d.player_id)]
    pts = h.gamelogs.groupby(["season", "player_id"]).fp.sum().rename("actual_pts")
    gp = h.gamelogs.groupby(["season", "player_id"]).size().rename("games")
    d = d.merge(pts.reset_index(), on=["season", "player_id"], how="left").merge(gp.reset_index(), on=["season", "player_id"], how="left")
    d = d.fillna({"actual_pts": 0, "games": 0})
    return d


st.markdown("## League history")
st.caption("Three seasons of your league (2024 to 2026): drafts, trades, pickups, injuries and results.")
section = st.segmented_control(
    "Section", ["Overview", "Managers", "Past drafts", "Trades and pickups", "Injuries"],
    default="Overview", key="lh_section", label_visibility="collapsed",
)
if H is None:
    st.warning("League history needs your ESPN league cookies in the .env file.")
    st.stop()

# ======================================================================================================
if section == "Overview":
    tk = league_takeaways(b.strategy)
    st.markdown("### What separates the top teams")
    comp = pd.DataFrame({
        "": ["Weekly win rate", "Draft points above expected, per season", "Draft skill (injuries evened out)", "Round of first goalie",
             "Share of defensemen in rounds 1 to 6", "Waiver activity (percentile within season)"],
        "Top four finishers": [f"{tk['top_win']:.0%}", f"{tk['top_value']:+.0f}", f"{ms[ms.final_rank <= 4].draft_skill.mean():+.0f}",
                               f"{tk['top_goalie_round']:.1f}", f"{tk['top_d_share']:.0%}", f"{ms[ms.final_rank <= 4].pickups_pct_rank.mean():.0%}"],
        "Everyone else": [f"{tk['rest_win']:.0%}", f"{tk['rest_value']:+.0f}", f"{ms[ms.final_rank > 4].draft_skill.mean():+.0f}",
                          f"{tk['rest_goalie_round']:.1f}", f"{tk['rest_d_share']:.0%}", f"{ms[ms.final_rank > 4].pickups_pct_rank.mean():.0%}"],
    })
    st.dataframe(comp, hide_index=True)

    st.markdown("### Which habits go with winning?")
    cor = b.strategy_cor
    wp = cor[cor.outcome == "Win %"].copy()
    bd = beyond_draft(ms, ["pickups_pct_rank", "lineup_moves_pct_rank", "trades", "top_count", "injury_luck", "share_waiver / free agent"])
    for col, data, title, lab in ((st.container(), wp, "All habits vs win rate", "strategy"), (st.container(), bd, "In-season habits, after accounting for draft quality", "habit")):
        with col:
            data = data.sort_values("rho")
            fig = go.Figure()
            for _, r in data.iterrows():
                c = SAGE if r.ci_low > 0 else (RUST if r.ci_high < 0 else GREY)
                fig.add_scatter(x=[r.ci_low, r.ci_high], y=[r[lab]] * 2, mode="lines", line=dict(color=c, width=3), showlegend=False, hoverinfo="skip")
                fig.add_scatter(x=[r.rho], y=[r[lab]], mode="markers", marker=dict(color=c, size=10), showlegend=False,
                                hovertemplate=f"{r[lab]}<br>correlation %{{x:.2f}}<br>95% range {r.ci_low:.2f} to {r.ci_high:.2f}<extra></extra>")
            fig.add_vline(x=0, line_color="#444")
            fig.update_layout(title=title, xaxis_title="correlation with win rate (-1 to +1)")
            fig_show(fig, 60 + 34 * len(data))
    note(
        "**How to read this:** dots to the right go with winning, dots to the left with losing. Green means the whole 95% range is above zero; "
        "rust means it is entirely below zero; grey is unclear. **What stands out:** how well you draft matters most, and among in-season habits only "
        "**waiver activity** still goes with winning once draft quality is accounted for. Trades and stacking one NHL team show no link."
    )

    st.markdown("### Where each team's points came from")
    ps = H.points_by_source()
    ps["manager"] = ps.owner_id.map(who)
    season_pick = st.segmented_control("Season", sorted(ps.season.unique()), default=int(ps.season.max()), key="ps_season")
    d = ps[ps.season == season_pick].sort_values("total")
    fig = go.Figure()
    for src, colr in (("draft", NAVY), ("waiver / free agent", SAGE), ("trade", SAND)):
        if src in d:
            fig.add_bar(y=d.manager, x=d[src], name=src, orientation="h", marker_color=colr, hovertemplate="%{y}<br>%{x:.0f} pts<extra>" + src + "</extra>")
    fig.update_layout(barmode="stack", title=f"Fantasy points scored while on each roster, {season_pick}", xaxis_title="fantasy points (bench included)")
    fig_show(fig, 460)
    avg = ps.groupby("season")[[c for c in ps.columns if c.startswith("share_")]].mean()
    st.caption("League average share of points: " + "; ".join(f"{s}: draft {r.get('share_draft', 0):.0%}, pickups {r.get('share_waiver / free agent', 0):.0%}, trades {r.get('share_trade', 0):.0%}" for s, r in avg.iterrows()))

    st.markdown("### Drafting players from the same NHL team")
    obs, null_mean, null_hi, p = stacking_test(b, A["season"])
    note(
        f"On average a roster held **{obs:.1f}** players from its most-drafted NHL team. If managers ignored NHL teams we would expect about "
        f"**{null_mean:.1f}** (95% of random drafts stay below {null_hi:.1f}). Chance this is luck: **p = {p:.3f}**. So managers in your league do "
        "favour certain NHL teams, but stacking shows **no link with winning** (see the chart above). Stacking one team makes your weekly score swing more: "
        "great weeks when that team scores, bad weeks when it doesn't."
    )
    stk = H.stacking(b.drafts)
    stk["manager"] = stk.owner_id.map(who)
    fig = px.scatter(stk, x="season", y="top_count", color="manager", hover_data=["top_team", "picks"], color_discrete_sequence=PALETTE,
                     labels={"top_count": "players from one NHL team", "season": "season"})
    fig.update_traces(marker=dict(size=11))
    fig.update_xaxes(dtick=1)
    fig.update_layout(title="Biggest same-team stack on each roster (hover for the team)")
    fig_show(fig, 380)

    st.markdown("### Final standings")
    stand = ms.pivot_table(index="manager", columns="season", values="final_rank").sort_values(ms.season.max())
    st.dataframe(stand.map(lambda x: ordinal(x) if pd.notna(x) else ""))

# ======================================================================================================
elif section == "Managers":
    reports = manager_reports(b.strategy, mgr_names, active)
    tk = league_takeaways(b.strategy)
    mine_r = next((r for r in reports if r.owner_id == my_owner), None)
    if mine_r:
        st.markdown(f"### Your review: {mine_r.name}")
        mine_rows = ms[ms.owner_id == my_owner].sort_values("season")
        league_pickups = ms.groupby("season").pickups.median()
        st.dataframe(pd.DataFrame({
            "Season": mine_rows.season.astype(str), "Finish": mine_rows.final_rank.map(ordinal),
            "Weekly win rate": mine_rows.win_pct.map(lambda x: f"{x:.0%}"),
            "Draft skill": mine_rows.draft_skill.map(lambda x: f"{x:+.0f}"),
            "Injury luck": mine_rows.injury_luck.map(lambda x: f"{x:+.0f}"),
            "Pickups (league median)": [f"{int(p)} ({int(league_pickups[s])})" for p, s in zip(mine_rows.pickups.fillna(0), mine_rows.season)],
            "Biggest NHL stack": [f"{t} x{int(c)}" for t, c in zip(mine_rows.top_team.fillna("?"), mine_rows.top_count.fillna(0))],
        }), hide_index=True)
        insights = []
        skill_med = ms.groupby("owner_id").draft_skill.mean().median()
        my_skill = mine_rows.draft_skill.mean()
        insights.append(("Your drafting is " + ("above" if my_skill >= skill_med else "below") +
                         f" the league's typical manager (draft skill {my_skill:+.0f} per season vs a median of {skill_med:+.0f})."))
        if mine_rows.pickups.mean() > league_pickups.mean():
            insights.append("You are already one of the most active managers on the waiver wire, so activity is not what is holding you back.")
        worst = mine_rows.loc[mine_rows.injury_luck.idxmin()]
        if worst.injury_luck < -50:
            insights.append(f"Injuries hurt you most in {int(worst.season)}: your drafted players missed {int(worst.games_lost)} games, about {abs(worst.injury_luck):.0f} points more than an average team lost.")
        good_wins = mine_rows[(mine_rows.win_pct >= 0.6) & (mine_rows.final_rank > 6)]
        for _, r in good_wins.iterrows():
            insights.append(f"In {int(r.season)} you won {r.win_pct:.0%} of your weeks but finished {ordinal(r.final_rank)}: the playoffs, not the regular season, decided that year.")
        if mine_rows.top_count.max() >= 5:
            r = mine_rows.loc[mine_rows.top_count.idxmax()]
            insights.append(f"You stacked {int(r.top_count)} {r.top_team} players in {int(r.season)}. That makes your weekly score swing with one NHL team.")
        if mine_rows.first_goalie_round.mean() > tk["top_goalie_round"] + 0.5:
            insights.append(f"Top teams take their first goalie around round {tk['top_goalie_round']:.0f}; you averaged round {mine_rows.first_goalie_round.mean():.0f}.")
        note("**What the data says about you**\n\n" + "\n".join(f"- {x}" for x in insights))

    st.markdown("### Manager map")
    st.caption("Each dot is a manager; similar drafters sit close together. **Click a dot** to open that manager's profile below.")
    sel_default = None
    if b.spectral is not None:
        tbl = b.spectral.table().reset_index()
        tbl["who"] = tbl["manager"].map(who)
        tbl["group"] = tbl["manager"].map(group_of)
        groups_sorted = sorted(tbl.group.dropna().unique())
        pick_sel = alt.selection_point(name="pick", fields=["manager"], on="click", clear="dblclick")
        base = alt.Chart(tbl).encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(padding=40)), y=alt.Y("y:Q", axis=None, scale=alt.Scale(padding=30)),
        )
        dots = base.mark_circle(size=260, stroke="white", strokeWidth=1).encode(
            color=alt.Color("group:N", scale=alt.Scale(domain=groups_sorted, range=PALETTE[: len(groups_sorted)]), legend=alt.Legend(orient="bottom", title=None)),
            opacity=alt.condition(pick_sel, alt.value(1.0), alt.value(0.35)),
            tooltip=[alt.Tooltip("who:N", title="manager"), alt.Tooltip("group:N", title="style")],
        ).add_params(pick_sel)
        labels = base.mark_text(dy=-15, font="Georgia", fontSize=12, color="#1B1B1B").encode(text="who:N")
        chart = (dots + labels).properties(height=480, title="Drafting styles in your league").configure_view(strokeWidth=0).configure_title(font="Georgia", fontSize=15)
        event = st.altair_chart(chart, on_select="rerun", key="mgr_map_alt")
        try:
            picked = event["selection"]["pick"]
            sel_default = picked[0]["manager"] if picked else None
        except (KeyError, TypeError, IndexError):
            sel_default = None
        with st.expander("What the groups mean"):
            for c, v in b.cluster_names.items():
                st.markdown(f"**{v['name']}**: {', '.join(v['traits'])}. Members: {', '.join(who(m) for m in v['members'])}.")
            st.caption("Groups are found by graph spectral clustering of draft habits (goalie timing, reaching, defense share, NHL-team loyalty, autodraft use). "
                       "Names are assigned from each group's most distinctive habits.")

    st.markdown("### Manager profile")
    owners = sorted(ms.owner_id.unique(), key=lambda o: (o not in active, who(o)))
    # a new click on the map opens that manager; otherwise keep whatever was picked in the dropdown
    if sel_default in owners and sel_default != st.session_state.get("last_map_pick"):
        st.session_state["mgr_select"] = sel_default
        st.session_state["last_map_pick"] = sel_default
    pick = st.selectbox("Manager", owners, format_func=who, key="mgr_select")
    rows = ms[ms.owner_id == pick].sort_values("season")
    rep = next((r for r in reports if r.owner_id == pick), None)
    with st.container(horizontal=True):
        st.metric("Average finish", f"{rows.final_rank.mean():.1f}", border=True)
        st.metric("Weekly win rate", f"{rows.win_pct.mean():.0%}", border=True)
        st.metric("Draft skill per season", f"{rows.draft_skill.mean():+.0f}", border=True)
        st.metric("Injury luck per season", f"{rows.injury_luck.mean():+.0f}", border=True)
        st.metric("Pickups per season", f"{rows.pickups.mean():.0f}", border=True)
    st.caption(f"Style group: **{group_of.get(pick, 'n/a')}**" + (f" · {rep.headline}" if rep else "") +
               ". Group names describe most members; this manager's own habits are listed below and can differ.")
    if rep:
        for x in rep.bullets:
            st.markdown(f"- {x}")
    dt = draft_table(b, A["season"])
    mp = dt[dt.owner_id == pick]
    c1, c2 = st.columns(2)
    with c1:
        mp2 = mp.assign(phase=pd.cut(mp["round"], [0, 3, 6, 10, 15, 26], labels=["1-3", "4-6", "7-10", "11-15", "16+"]))
        mix = mp2.groupby(["phase", "pos"], observed=True).size().unstack(fill_value=0)
        mix = mix.div(mix.sum(axis=1), axis=0)
        fig = go.Figure()
        for i, pcol in enumerate([c for c in ["C", "LW", "RW", "D", "G"] if c in mix]):
            fig.add_bar(x=mix.index.astype(str), y=mix[pcol], name=pcol, marker_color=PALETTE[i])
        fig.update_layout(barmode="stack", title="Positions drafted by round", xaxis_title="rounds", yaxis=dict(title="share", tickformat=".0%"))
        fig_show(fig, 340)
    with c2:
        tm = mp[mp.nhl != "?"].nhl.value_counts().head(8)
        fig = go.Figure(go.Bar(x=tm.values, y=tm.index, orientation="h", marker_color=NAVY))
        fig.update_layout(title="Favourite NHL teams (all drafts)", xaxis_title="players drafted", yaxis=dict(autorange="reversed"))
        fig_show(fig, 340)
    with st.expander("Every pick this manager made"):
        st.dataframe(mp[["season", "round", "overall", "player", "pos", "nhl", "adp", "actual_pts", "games"]].sort_values(["season", "overall"]),
                     hide_index=True, column_config={"adp": st.column_config.NumberColumn("ADP", format="%.0f"),
                                                     "actual_pts": st.column_config.NumberColumn("actual pts", format="%.0f")})
    ts = H.trade_summary()
    if len(ts):
        mt = ts[(ts.owner_a == pick) | (ts.owner_b == pick)]
        if len(mt):
            st.markdown("**Trades**")
            st.dataframe(mt.assign(a=mt.owner_a.map(who), b=mt.owner_b.map(who), won=mt.winner.map(who))[
                ["season", "date", "a", "received_a", "points_a", "b", "received_b", "points_b", "won"]].round(0), hide_index=True)
    bp = H.best_pickups(40)
    mb = bp[bp.owner_id == pick].head(8)
    if len(mb):
        st.markdown("**Best pickups**")
        st.dataframe(mb[["season", "player", "pos", "added", "points", "games"]].round(0), hide_index=True)

    st.markdown("### Head-to-head comparison")
    cA, cB = st.columns(2)
    m1 = cA.selectbox("Manager A", owners, index=owners.index(my_owner) if my_owner in owners else 0, format_func=who, key="h2h_a")
    m2 = cB.selectbox("Manager B", owners, index=1 if len(owners) > 1 else 0, format_func=who, key="h2h_b")
    metrics = {"win_pct": "Win rate", "draft_skill": "Draft skill", "injury_luck": "Injury luck", "pickups": "Pickups",
               "first_goalie_round": "First goalie round", "d_share_r1_6": "Defense early", "reach_early": "Reaching", "top_count": "Same-team stack"}
    avg = ms.groupby("owner_id")[list(metrics)].mean()
    pct = avg.rank(pct=True)
    fig = go.Figure()
    for m_, colr in ((m1, NAVY), (m2, RUST)):
        fig.add_bar(x=[metrics[k] for k in metrics], y=pct.loc[m_].values, name=who(m_), marker_color=colr,
                    customdata=avg.loc[m_].values, hovertemplate="%{x}<br>league percentile %{y:.0%}<br>value %{customdata:.2f}<extra>" + who(m_) + "</extra>")
    fig.update_layout(barmode="group", title="Where each manager ranks in the league (percentile, hover for the raw value)", yaxis=dict(tickformat=".0%", range=[0, 1.05]))
    fig_show(fig, 380)

    st.markdown("### Scouting reports")
    act = [r for r in reports if r.active and r.owner_id != my_owner]
    cols = st.columns(2)
    for i, r in enumerate(act):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{r.name}**: {r.headline.lower()} · *{group_of.get(r.owner_id, '')}*")
                st.caption(f"Finishes: {r.finishes} · win rate {r.win_pct:.0%}")
                for x in r.bullets:
                    st.markdown(f"- {x}")
                st.markdown(f"*{r.sunday_tip}*")

# ======================================================================================================
elif section == "Past drafts":
    dt = draft_table(b, A["season"])
    dt["manager"] = dt.owner_id.map(who)
    with st.container(horizontal=True):
        seasons = st.pills("Seasons", sorted(dt.season.unique()), selection_mode="multi", default=sorted(dt.season.unique()), key="pd_seasons")
        posf = st.pills("Positions", ["C", "LW", "RW", "D", "G"], selection_mode="multi", default=["C", "LW", "RW", "D", "G"], key="pd_pos")
    c1, c2 = st.columns([2, 1])
    mgrs = c1.multiselect("Managers (empty = everyone)", sorted(dt.manager.unique()), key="pd_mgrs")
    rr = c2.slider("Rounds", 1, int(dt["round"].max()), (1, int(dt["round"].max())), key="pd_rounds")
    f = dt[dt.season.isin(seasons or []) & dt.pos.isin(posf or []) & dt["round"].between(*rr)]
    if mgrs:
        f = f[f.manager.isin(mgrs)]
    st.caption(f"{len(f)} picks shown")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(f, x="overall", y="adp", color="manager", hover_name="player", hover_data=["season", "round", "pos", "nhl"],
                         color_discrete_sequence=PALETTE, labels={"overall": "pick number", "adp": "ESPN ADP"})
        lim = float(max(f.overall.max() if len(f) else 1, 1))
        fig.add_scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(color=GREY, dash="dash"), name="picked at ADP", hoverinfo="skip")
        fig.update_layout(title="Pick number vs ESPN ranking (above the line = picked earlier than ESPN ranks him)")
        fig_show(fig, 460)
    with c2:
        fig = px.scatter(f, x="overall", y="actual_pts", color="manager", hover_name="player", hover_data=["season", "round", "games"],
                         color_discrete_sequence=PALETTE, labels={"overall": "pick number", "actual_pts": "actual fantasy points"})
        fig.update_layout(title="What each pick actually scored", showlegend=False)
        fig_show(fig, 460)
    st.dataframe(f[["season", "round", "overall", "manager", "player", "pos", "nhl", "adp", "actual_pts", "games"]].sort_values(["season", "overall"]),
                 hide_index=True, height=420, column_config={"adp": st.column_config.NumberColumn("ADP", format="%.0f"),
                                                            "actual_pts": st.column_config.NumberColumn("actual pts", format="%.0f")})

# ======================================================================================================
elif section == "Trades and pickups":
    ts = H.trade_summary()
    st.markdown("### Every completed trade")
    if len(ts):
        view = ts.assign(a=ts.owner_a.map(who), b=ts.owner_b.map(who), won=ts.winner.map(who))
        st.dataframe(view[["season", "date", "a", "received_a", "points_a", "b", "received_b", "points_b", "won", "margin"]].round(0), hide_index=True,
                     column_config={"a": "manager A", "received_a": "A received", "points_a": "A's points after", "b": "manager B",
                                    "received_b": "B received", "points_b": "B's points after", "won": "won the trade", "margin": "margin"})
        st.caption("Points after = fantasy points the received players scored for their new team for the rest of the season.")
        # trade network
        counts = pd.concat([ts[["owner_a", "owner_b"]]]).value_counts().reset_index(name="n")
        nodes = sorted(set(counts.owner_a) | set(counts.owner_b), key=who)
        ang = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
        pos = {n_: (np.cos(a), np.sin(a)) for n_, a in zip(nodes, ang)}
        deg = pd.concat([counts.owner_a, counts.owner_b]).value_counts()
        fig = go.Figure()
        for r in counts.itertuples():
            (x0, y0), (x1, y1) = pos[r.owner_a], pos[r.owner_b]
            fig.add_scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(width=2 + 3 * r.n, color=GREY), hoverinfo="text",
                            text=f"{who(r.owner_a)} and {who(r.owner_b)}: {r.n} trade(s)", showlegend=False)
        fig.add_scatter(x=[pos[n_][0] for n_ in nodes], y=[pos[n_][1] for n_ in nodes], mode="markers+text", text=[who(n_) for n_ in nodes],
                        textposition="top center", marker=dict(size=[14 + 8 * deg.get(n_, 0) for n_ in nodes], color=[RUST if n_ == my_owner else NAVY for n_ in nodes]),
                        hovertext=[f"{who(n_)}: {deg.get(n_, 0)} trade(s)" for n_ in nodes], hoverinfo="text", showlegend=False)
        fig.update_layout(title="Trade network (thicker line = more trades; bigger dot = more trades made)",
                          xaxis=dict(visible=False, range=[-1.5, 1.5]), yaxis=dict(visible=False, range=[-1.4, 1.4]))
        fig_show(fig, 460)
        wins = ts.winner.map(who).value_counts()
        note(f"**{len(ts)} trades** went through in three seasons, so trading is rare in your league. Trades winners: " +
             ", ".join(f"{k} {v}" for k, v in wins.items()) + ". Trades explain about 1% of all points scored.")
    else:
        st.info("No completed trades found.")

    st.markdown("### Best waiver-wire pickups")
    bp = H.best_pickups(10)
    bp["manager"] = bp.owner_id.map(who)
    season_pick = st.segmented_control("Season", sorted(bp.season.unique()), default=int(bp.season.max()), key="bp_season")
    d = bp[bp.season == season_pick].sort_values("points")
    fig = go.Figure(go.Bar(x=d.points, y=d.player + " (" + d.manager + ")", orientation="h", marker_color=SAGE,
                           customdata=np.stack([d.added.astype(str), d.games], axis=1),
                           hovertemplate="%{y}<br>%{x:.0f} pts in %{customdata[1]} games<br>added %{customdata[0]}<extra></extra>"))
    fig.update_layout(title=f"Top pickups of {season_pick}: points scored for the team that added them", xaxis_title="fantasy points")
    fig_show(fig, 420)
    finders = H.best_pickups(20).assign(manager=lambda x: x.owner_id.map(who)).manager.value_counts().head(5)
    st.caption("Most top-20 pickups found (all seasons): " + ", ".join(f"{k} {v}" for k, v in finders.items()))

    st.markdown("### Does being active win more?")
    resid_df = ms.dropna(subset=["pickups_pct_rank", "win_pct", "value_added_all"]).copy()
    from iknowpuck.history import partial_residual
    resid_df["win_beyond_draft"] = partial_residual(resid_df.win_pct, resid_df.value_added_all)
    fig = px.scatter(resid_df, x="pickups", y="win_beyond_draft", color="manager", hover_data=["season", "trades", "lineup_moves"],
                     color_discrete_sequence=PALETTE, labels={"pickups": "waiver pickups in the season", "win_beyond_draft": "win rate above what the draft predicted"})
    fig.add_hline(y=0, line_color=GREY)
    fig.update_layout(title="Pickups vs wins beyond what the draft predicted (each dot = one team-season)")
    fig_show(fig, 420)
    bd = beyond_draft(ms, ["pickups_pct_rank", "lineup_moves_pct_rank", "trades"])
    r0 = bd.iloc[0]
    note(f"**Answer:** managers who made more pickups than others in the same season won more than their draft predicted "
         f"(correlation {r0.rho:+.2f}, 95% range {r0.ci_low:+.2f} to {r0.ci_high:+.2f}). Lineup changes and trades showed no clear link. "
         "Caveat: ESPN kept many more lineup records for 2024 than later seasons, so lineup activity is compared within each season only.")

# ======================================================================================================
elif section == "Injuries":
    st.markdown("### Injury timelines")
    st.markdown(
        "We rebuilt every absence from game-by-game records: whenever a player's NHL team played and he did not, for **three or more games in a row**, "
        "that counts as an absence. Most are injuries; some are illness, suspensions or healthy scratches. Goalies are left out because backups sit by design."
    )
    sp = H.spells.copy()
    dt = draft_table(b, A["season"])
    c1, c2, c3 = st.columns(3)
    season_pick = c1.selectbox("Season", sorted(sp.season.unique()), index=len(sp.season.unique()) - 1, key="inj_season")
    scope = c2.segmented_control("Players", ["Drafted in your league", "All"], default="Drafted in your league", key="inj_scope")
    mgr_f = c3.selectbox("Drafted by", ["Everyone"] + sorted(dt.owner_id.unique(), key=who), format_func=lambda o: o if o == "Everyone" else who(o), key="inj_mgr")
    s = sp[sp.season == season_pick].copy()
    drafted = dt[dt.season == season_pick]
    if scope == "Drafted in your league":
        s = s[s.player_id.isin(drafted.player_id)]
    if mgr_f != "Everyone":
        s = s[s.player_id.isin(drafted[drafted.owner_id == mgr_f].player_id)]
    sched = H.schedules[H.schedules.season == season_pick]
    spd = sched.groupby("scoring_period").date.min()
    s["start"] = s.start_sp.map(spd)
    s["end"] = s.end_sp.map(spd) + pd.Timedelta(days=1)
    s["player"] = s.player_id.map(H.player_names)
    owner_map = dict(zip(drafted.player_id, drafted.owner_id))
    s["drafted_by"] = s.player_id.map(lambda p: who(owner_map[p]) if p in owner_map else "undrafted")
    top_players = s.groupby("player").games.sum().nlargest(35).index
    s = s[s.player.isin(top_players)].dropna(subset=["start", "end"])
    if len(s):
        fig = px.timeline(s, x_start="start", x_end="end", y="player", color="where", hover_data=["games", "drafted_by"],
                          color_discrete_map={"mid-season": RUST, "season start": SAND, "season end": GREY})
        fig.update_yaxes(autorange="reversed", title="")
        fig.update_layout(title=f"Absences of 3+ games, {season_pick} (35 players with the most games missed)", xaxis_title="date", legend_title="when")
        fig_show(fig, 60 + 18 * s.player.nunique())
        st.caption("Season-start absences can also mean the player started in the minors; mid-season absences are the clearest injury signal.")
    else:
        st.info("No absences for this selection.")

    st.markdown("### Injury luck by manager")
    il = ms[ms.season == season_pick].sort_values("injury_luck")
    fig = go.Figure(go.Bar(x=il.injury_luck, y=il.manager, orientation="h", marker_color=[SAGE if v > 0 else RUST for v in il.injury_luck],
                           customdata=il.games_lost, hovertemplate="%{y}<br>luck %{x:+.0f} pts<br>games lost %{customdata:.0f}<extra></extra>"))
    fig.update_layout(title=f"Injury luck in {season_pick}: points NOT lost to injuries, compared with an average team", xaxis_title="fantasy points (+ = luckier)")
    fig_show(fig, 440)
    note("**Why this matters:** a manager can draft well and still lose because their players got hurt. The League history numbers split each manager's "
         "draft result into **draft skill** (what they would have scored with average injury luck) and **injury luck**.")

    st.markdown("### Injury history of this year's players")
    f = A["pool"].frame
    risk = f[f.get("avg_missed").notna()] if "avg_missed" in f else f.iloc[0:0]
    risk = risk.nsmallest(200, "adp").sort_values("avg_missed", ascending=False).head(30)
    st.dataframe(risk[["name", "pos", "adp", "avg_missed", "injury_risk", "injury_status", "games_missed_assumed"]], hide_index=True,
                 column_config={"adp": st.column_config.NumberColumn("ADP", format="%.0f"),
                                "avg_missed": st.column_config.NumberColumn("avg games missed / season", format="%.1f"),
                                "games_missed_assumed": st.column_config.NumberColumn("games missed assumed this year", format="%.0f")})
    st.caption("Among the top 200 by ADP. A history of missed games already lowers each player's games-played projection; current injuries are handled on the Pre-draft plan page.")
