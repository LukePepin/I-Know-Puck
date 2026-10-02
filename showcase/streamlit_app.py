"""I Know Puck: public showcase.   Run:  streamlit run showcase/streamlit_app.py

Three tabs: History (2024-26), Present (the 2027 draft and this week), Future (season simulation, how the
models work, systems overview). Reads only the snapshot in showcase/data/ (made by scripts/export_showcase.py):
no league credentials and no live ESPN calls, so it can be hosted publicly.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="I Know Puck", page_icon=":material/sports_hockey:", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/history.py", title="History", icon=":material/history:"),
        st.Page("app_pages/present.py", title="Present", icon=":material/today:", default=True),
        st.Page("app_pages/future.py", title="Future", icon=":material/insights:"),
    ],
    position="top",
)
page.run()
