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

The first launch takes a few minutes: it pulls 10 seasons of ESPN + MoneyPuck data plus your league's
transactions and game logs, fits the models and caches everything. After that it starts instantly. Follow
the pages in the left menu from top to bottom. On draft day, open **Draft room** and turn on **Live sync
with ESPN draft**: the app reads the ESPN draft every 5 seconds and suggests a pick when you're on the
clock. If live sync fails, record picks manually with **Record a pick**.

| Page (left menu) | What it does |
|---|---|
| Walkthrough | Plain-language guide: the short version, a step-by-step game plan, how the model works, the evidence scorecard, glossary |
| League history | Who wins and how; manager review, clickable manager map, drill-down and head-to-head; past-draft explorer; trades (network and winners), best pickups, activity vs winning; injury timelines and injury luck |
| Pre-draft plan | Editable injury table (games missed), simulated drafts from your slot, availability at your first pick, player board |
| Draft room | Live ESPN sync, suggested pick with uncertainty, injury and same-team flags, roster, pick log |
| In-season moves | Weekly add/drop suggestions for your roster, plus what league history says about waiver activity |
| Systems engineering | Architecture, requirements traceability, V-model, critical findings, trade study, FMEA, a 7-minute lab-talk outline, transferable components |

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
  history.py      league history: roster stints, trades, pickups, injury absences, stacking
  injuries.py     current-injury discounts (editable) and injury-risk history
  inseason.py     waiver-wire add/drop optimiser
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
