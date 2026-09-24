"""In-season moves: waiver-wire pickups and drops for your roster, plus what history says works."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import NAVY, SAGE, app_state, fig_show, note

from iknowpuck.data.espn import EspnClient
from iknowpuck.inseason import current_rosters, waiver_suggestions
from iknowpuck.strategy import beyond_draft

A = app_state()
b, ctx, pool, names, my_team = A["b"], A["ctx"], A["pool"], A["names"], A["my_team"]
frame = pool.frame


@st.cache_data(ttl="5m", max_entries=4, show_spinner="Reading current rosters from ESPN...")
def espn_rosters(season: int) -> dict[int, list[int]]:
    return current_rosters(EspnClient(A["creds"]), season)


st.markdown("## In-season moves")
st.caption("Use this page once a week after the draft. It looks at every free agent and every player on your bench to find swaps that raise your weekly points.")

rosters = espn_rosters(A["season"])
source = "ESPN"
if not any(rosters.values()):
    picks = st.session_state.get("picks", [])
    if picks:
        rosters = {}
        for t, p in picks:
            rosters.setdefault(t, []).append(int(p))
        source = "the draft room"
    else:
        rosters = {}

if not rosters.get(my_team):
    st.info("Your roster is empty right now. This page starts working as soon as your draft is done (or after you record picks in the Draft room).")
else:
    st.caption(f"Rosters read from {source}. Free agents = players in the pool that no team owns.")
    n_fa = st.slider("Free agents to consider", 20, 150, 60, key="fa_n")
    with st.spinner("Scoring every add and drop..."):
        sugg = waiver_suggestions(pool, ctx.val, rosters, my_team, n_free_agents=n_fa)
    if sugg.empty:
        st.success("No free agent would raise your expected weekly points right now. Your roster is in good shape.")
    else:
        top = sugg.iloc[0]
        st.success(f"**Best move: add {top.add_player} ({top.add_pos}), drop {top.drop_player} ({top.drop_pos}).** "
                   f"Expected weekly points go up by {top.gain_pts:.1f}" + (f" and your chance of winning a week by {top.gain_win:+.1%}." if "gain_win" in sugg else "."))
        d = sugg.head(12).iloc[::-1]
        fig = go.Figure(go.Bar(x=d.gain_pts, y="add " + d.add_player + " / drop " + d.drop_player, orientation="h", marker_color=SAGE,
                               hovertemplate="%{y}<br>+%{x:.2f} weekly points<extra></extra>"))
        fig.update_layout(title="Best add / drop swaps (gain in expected weekly points)", xaxis_title="extra fantasy points per week")
        fig_show(fig, 60 + 30 * len(d))
        cols = ["add_player", "add_pos", "add_status", "drop_player", "drop_pos", "gain_pts"] + (["gain_win"] if "gain_win" in sugg else [])
        st.dataframe(sugg[cols], hide_index=True, column_config={
            "add_player": "add", "add_pos": "pos", "add_status": "injury", "drop_player": "drop", "drop_pos": "pos ",
            "gain_pts": st.column_config.NumberColumn("weekly pts gained", format="%+.2f"),
            "gain_win": st.column_config.NumberColumn("win chance gained", format="%+.3f"),
        })
        note("Gains count only what the player adds to your **starting lineup** (bench players count about a third, since daily lineups let them fill in). "
             "Projections are season-long; check recent news before you drop anyone.")

st.markdown("---")
st.markdown("### What history says about in-season moves")
H = b.history
if H is not None and len(b.strategy):
    ms = b.strategy
    bd = beyond_draft(ms, ["pickups_pct_rank", "lineup_moves_pct_rank", "trades"])
    r0 = bd.iloc[0]
    c1, c2 = st.columns(2)
    with c1:
        ps = H.points_by_source()
        avg = ps.groupby("season")[[c for c in ps.columns if c.startswith("share_")]].mean().mean()
        st.metric("Share of points from waiver pickups", f"{avg.get('share_waiver / free agent', 0):.0%}", border=True)
        st.metric("Share of points from trades", f"{avg.get('share_trade', 0):.1%}", border=True)
    with c2:
        bp = H.best_pickups(20)
        bp["month"] = pd.to_datetime(bp["added"]).dt.strftime("%b")
        order_m = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]
        cnt = bp.month.value_counts().reindex(order_m).fillna(0)
        fig = go.Figure(go.Bar(x=cnt.index, y=cnt.values, marker_color=NAVY))
        fig.update_layout(title="When the best pickups were made (top 20 each season)", yaxis_title="pickups")
        fig_show(fig, 300)
        top_months = list(cnt.sort_values(ascending=False).index[:2])
    note(f"- About **{avg.get('share_waiver / free agent', 0):.0%}** of a typical team's points come from players added after the draft.\n"
         f"- Managers who made more pickups than others that season won more than their draft predicted (correlation {r0.rho:+.2f}, 95% range {r0.ci_low:+.2f} to {r0.ci_high:+.2f}).\n"
         f"- The best pickups were most often made in **{top_months[0]}** and **{top_months[1]}**: check the wire most often then.\n"
         "- Trades are rare in your league and explain very little of the scoring.")
