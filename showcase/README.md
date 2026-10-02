# I Know Puck: public showcase

A five-page, read-only version of the project for sharing: what it does, the 2027 draft, how the model picks,
whether it works, and the league's history. It reads only the snapshot in `data/`, makes no web requests and
needs no credentials, so it is safe to host publicly.

```bash
pip install -r showcase/requirements.txt
streamlit run showcase/streamlit_app.py
```

| Page | What it shows |
|---|---|
| Overview | The problem, the four-step method, headline numbers, what I learned |
| The 2027 draft | Every roster's expected weekly points, who beats whom, the draft board, steals and reaches, notes on each roster |
| How the model picks | Only the starting lineup scores (weekly lock), win chance from weekly points, simulated drafts with common random numbers, positional scarcity |
| Does it work? | The seven pre-registered tests in plain language, projection accuracy vs ESPN, assumption checks, a theory tested |
| League history | Habits that go with winning, drafting-style map, every 2024-26 pick vs its slot, my past seasons |

## What is in the snapshot (and what is not)

`data/` is written by `scripts/export_showcase.py`, which runs on my machine with the private league cache.

- **Included:** public NHL player data (names, positions, ESPN projections and ADP), the league's picks and
  rosters, model results and test results.
- **Anonymized:** every other manager ("Team 1".."Team 12" by 2027 draft slot, "Manager A".. in history).
  Fantasy team names are not included.
- **Never included:** ESPN cookies (`espn_s2`, `SWID`), the league id, ESPN member ids, other managers' names.
  The export script checks nothing of this kind is written; `git grep` for the league id and cookie values
  before publishing is a good second check.

Refresh it after new results: `PYTHONPATH=src .venv/bin/python scripts/export_showcase.py`.

## Hosting on Streamlit Community Cloud (free)

1. Push the repository to GitHub (the `.env` file is git-ignored and has never been committed).
2. At share.streamlit.io choose **New app**, pick the repository and branch, and set the main file to
   `showcase/streamlit_app.py`. Community Cloud installs `showcase/requirements.txt`.
3. No secrets are needed. Leave the Secrets box empty.

Alternatively copy the `showcase/` folder into its own public repository; it is self-contained
(`streamlit_app.py`, `ui.py`, `app_pages/`, `data/`, `requirements.txt`, `.streamlit/config.toml`).
