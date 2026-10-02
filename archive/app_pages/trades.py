"""ARCHIVED trade analyzer (was the 'Trades and pickups' section of League history).

Restore: add it back to app/app_pages/league_history.py, or copy this file to app/app_pages/ and add an
st.Page for it in app/streamlit_app.py. It expects the same globals as League history (b, H, who, my_owner).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import GREY, NAVY, RUST, app_state, fig_show, note

A = app_state()
b = A["b"]
H = b.history
mgr_names = b.manager_names
active = set(b.managers.loc[b.managers.active, "owner_id"].astype(str)) if len(b.managers) else set()
my_owner = str(b.manager_of_team.get(A["my_team"]))


def who(m) -> str:
    n_ = mgr_names.get(str(m), str(m))
    return n_ if str(m) in active else f"{n_} (former)"


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

