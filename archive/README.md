# Archive

Draft-day tools retired after the 2027 draft (27 September 2026). They still work; they are just out of the
app's menu.

| File | What it was | Restore |
|---|---|---|
| `app_pages/draft_room.py` | Draft room: suggested pick (Monte Carlo with common random numbers), search-depth presets, stack teams, practice mode, paste-the-ESPN-log loader | copy to `app/app_pages/draft.py`, copy `draft_log.py` to `app/`, add `st.Page("app_pages/draft.py", ...)` in `app/streamlit_app.py` |
| `draft_log.py` | Parser for ESPN's draft chat log and draft board (ESPN's API publishes no picks during a live draft) | see above |
| `app_pages/pre_draft.py` | Pre-draft plan: editable injury table, simulated drafts from your slot, availability at your first pick | copy to `app/app_pages/`, add an `st.Page` |
| `app_pages/trades.py` | Trade analyzer: every completed trade, who won it, trade network | copy to `app/app_pages/`, add an `st.Page` |

Removed rather than archived: the in-season add/drop notifier (`app/app_pages/in_season.py` and
`src/iknowpuck/inseason.py`); both are in git history before this change. The roster reader it used now lives
in `EspnClient.rosters()`.

Lesson from draft night: ESPN's read API (`lm-api-reads.fantasy.espn.com`, view `mDraftDetail`) returned
`playerId -1` for every pick and empty rosters while the draft was in progress, so live sync saw nothing until
the draft ended. Pasting the draft board into the Draft room was the working fallback.
