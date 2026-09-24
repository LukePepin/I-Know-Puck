# I Know Puck

An operations-research draft assistant for an ESPN fantasy hockey league. It combines ESPN league
history, ESPN projections and MoneyPuck advanced stats. Monte Carlo simulation of the draft picks the
player that maximizes your chance of winning weekly matchups. Every modeling claim is tested
statistically before it is trusted.

## Quick start (draft day)

```bash
python -m venv .venv
.venv\Scripts\pip install -e .[dev]
copy .env.example .env      # then fill in ESPN_LEAGUE_ID, ESPN_S2, ESPN_SWID
.venv\Scripts\streamlit run app/streamlit_app.py
```

The first launch takes about 2 minutes: it pulls 10 seasons of ESPN + MoneyPuck data, fits the models and
caches them. After that it starts instantly. In the **Draft room** tab, turn on **Live sync from ESPN
draft**. The app polls ESPN every 5 seconds and re-runs the simulation when you're on the clock. If
live sync fails, record picks manually with **Record a pick**.

| Tab | What it does |
|---|---|
| Draft room | Live board. Recommendations show simulated P(win weekly matchup), the gap to the best option with its standard error, and the chance each player lasts until your next pick |
| Pre-draft plan | Full simulated drafts from your slot: who you typically land each round, plus availability at your first pick |
| Player board | Blended vs own vs ESPN projections; value relative to ADP |
| League intel | Every manager across seasons, spectral-cluster archetypes, fitted pick tendencies |
| Research | Experiment reports (hypothesis tests) and projection blend weights |

## How it works

```
ESPN (league settings, drafts, projections, actuals)  ─┐
MoneyPuck (xG, TOI, PP usage)                          ─┼─> player panel 2018-2027
                                                        │
  projections.py  Marcel+ own model + MoneyPuck residual ridge, blended with ESPN (weights fit out-of-sample)
  valuation.py    roster -> weekly fantasy points ~ Normal(mu, var) with optimal slotting -> P(win matchup)
  opponents.py    conditional-logit pick model per manager, shrunk toward spectral clusters
  spectral.py     manager-behavior graph -> normalized Laplacian -> eigengap k -> clusters
  draft.py        snake-draft Monte Carlo; candidates compared with common random numbers
  experiments.py  pre-registered hypotheses H1-H4, paired tests, Holm correction
```

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the math and the experimental design.

## Layout

- `src/puckcore/`: **domain-agnostic** experiment framework (cache, paired statistics, experiment runner,
  plugin protocols). Reusable in other projects without changes.
- `src/iknowpuck/`: the hockey/ESPN plugin.
- `app/`: the Streamlit draft room.
- `tests/`: unit tests on synthetic data (`pytest`).

Run the experiment suite: `python -m iknowpuck.experiments` (add `--quick` to skip the draft backtest).
Reports are written to `runs/<timestamp>_ikp/report.md`.
