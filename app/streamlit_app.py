"""I Know Puck: draft decision support.   Run:  streamlit run app/streamlit_app.py"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui import DEFAULT_MISSED, get_bundle, injured_context  # noqa: E402

from iknowpuck.config import load_credentials  # noqa: E402
from iknowpuck.injuries import load_overrides  # noqa: E402

st.set_page_config(page_title="I Know Puck", page_icon=":material/sports_hockey:", layout="wide")

creds = load_credentials()
st.session_state.setdefault("refresh", 0)
st.session_state.setdefault("picks", [])  # list[(team_id, player_id)] recorded in the draft room
st.session_state.setdefault("injury_defaults", dict(DEFAULT_MISSED))
st.session_state.setdefault("injury_overrides", load_overrides())

with st.sidebar:
    st.markdown("### I Know Puck")
    st.caption("Draft decision support for an ESPN head-to-head fantasy hockey league")
    season = st.number_input("ESPN season (year the season ends)", value=creds.season, step=1, key="season")

b = get_bundle(int(season), st.session_state.refresh)
S = b.settings
names = b.team_labels() or {t: f"Team {t}" for t in range(1, S.n_teams + 1)}

with st.sidebar:
    order_default = S.pick_order or list(names)
    order_txt = st.text_input("Round 1 pick order (team ids)", ",".join(map(str, order_default)), key="order_txt")
    order = [int(x) for x in order_txt.split(",") if x.strip().isdigit()]
    my_team = st.selectbox("My team", list(names), index=list(names).index(S.my_team_id) if S.my_team_id in names else 0,
                           format_func=lambda t: names[t], key="my_team")
    st.caption(f"Draft slot {order.index(my_team) + 1 if my_team in order else '?'} of {len(order)} · {S.rounds} rounds · head-to-head points")
    with st.expander("Draft order by manager"):
        st.markdown("\n".join(f"{i}. {names.get(t, t)}" + (" (you)" if t == my_team else "") for i, t in enumerate(order, start=1)))
    if st.button("Rebuild data and models", icon=":material/refresh:", help="Re-download ESPN and MoneyPuck data and refit every model"):
        st.session_state.refresh += 1
        get_bundle.clear()
        st.rerun()
    for n in b.notes:
        st.warning(n)

pool, ctx = injured_context(
    b, int(season),
    tuple(sorted(st.session_state.injury_overrides.items())),
    tuple(sorted(st.session_state.injury_defaults.items())),
    tuple(order), int(my_team),
)
st.session_state["app"] = {
    "b": b, "S": S, "names": names, "order": order, "my_team": my_team, "season": int(season),
    "pool": pool, "ctx": ctx, "creds": creds,
}

page = st.navigation(
    {
        "Start here": [st.Page("app_pages/walkthrough.py", title="Walkthrough", icon=":material/menu_book:", default=True)],
        "Before the draft": [
            st.Page("app_pages/league_history.py", title="League history", icon=":material/history:"),
            st.Page("app_pages/pre_draft.py", title="Pre-draft plan", icon=":material/checklist:"),
        ],
        "Draft day": [st.Page("app_pages/draft.py", title="Draft room", icon=":material/sports_hockey:")],
        "After the draft": [st.Page("app_pages/in_season.py", title="In-season moves", icon=":material/swap_horiz:")],
        "Lab presentation": [st.Page("app_pages/systems_engineering.py", title="Systems engineering", icon=":material/account_tree:")],
    },
    position="sidebar",
)
page.run()
