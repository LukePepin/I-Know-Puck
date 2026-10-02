"""I Know Puck: public showcase.   Run:  streamlit run showcase/streamlit_app.py

Reads only the snapshot in showcase/data/ (made by scripts/export_showcase.py). No league credentials and no
live ESPN calls, so it can be hosted publicly.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="I Know Puck", page_icon=":material/sports_hockey:", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/home.py", title="Overview", icon=":material/home:", default=True),
        st.Page("app_pages/draft_2027.py", title="The 2027 draft", icon=":material/sports_hockey:"),
        st.Page("app_pages/model.py", title="How the model picks", icon=":material/tune:"),
        st.Page("app_pages/evidence.py", title="Does it work?", icon=":material/fact_check:"),
        st.Page("app_pages/history.py", title="League history", icon=":material/history:"),
    ],
    position="sidebar",
)
with st.sidebar:
    st.caption("A fantasy hockey draft assistant built with statistics and operations research. Other managers are anonymized.")
page.run()
