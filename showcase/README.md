# I Know Puck: public showcase

A three-tab website about a 12-team ESPN fantasy hockey league: the models, the data and the visualizations.
It reads only the snapshot in `data/`, makes no web requests and needs no credentials, so it is safe to host
publicly.

```bash
pip install -r showcase/requirements.txt
streamlit run showcase/streamlit_app.py
```

| Tab | What it shows |
|---|---|
| History | 2024-26 champions and standings, habits that go with winning, drafting-style map (graph spectral clustering), every past pick vs its slot, one theory tested |
| Present | This week's matchups, power rankings, who beats whom, the 2026-27 draft board, bargains and reaches, every roster |
| Future | A live season simulation (10,000 seasons on the real schedule, playoff and title odds), how the models work, the tests, a whole-system overview |

Every tab ends with **How this page is built** (where its data comes from and what is done to it), a short
**The math** box, and **Definitions**.

## What is in the snapshot (and what is not)

`data/` is written by `scripts/export_showcase.py`, which runs on my machine with the private league cache.

- **Included:** public NHL player data (names, positions, ESPN projections and ADP), the league's picks,
  rosters, schedule and scores, model results and test results. Managers appear by **first name only**.
- **Never included:** ESPN cookies (`espn_s2`, `SWID`), the league id, ESPN member ids, surnames, fantasy team names.
- `scripts/check_public.py` scans every file git would publish for those before each push.

Refresh it (for example once a week): `PYTHONPATH=src .venv/bin/python scripts/export_showcase.py`, then
`PYTHONPATH=src .venv/bin/python scripts/check_public.py`, then commit and push. The hosted site redeploys.

## Hosting on Streamlit Community Cloud (free)

1. Sign in at share.streamlit.io with GitHub.
2. **Create app**: repository `LukePepin/I-Know-Puck`, the branch to publish, main file `showcase/streamlit_app.py`.
   Community Cloud installs `showcase/requirements.txt`.
3. Leave **Secrets** empty and press **Deploy**.

Free apps sleep after a stretch without visitors; the next visitor wakes it in about half a minute.
