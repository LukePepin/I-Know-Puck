# I Know Puck

An operations-research draft assistant for an ESPN fantasy hockey league. It combines ESPN league
history, ESPN projections and MoneyPuck advanced stats. Monte Carlo simulation of the draft picks the
player that maximizes your chance of winning weekly matchups. Every modeling claim is tested
statistically before it is trusted.

## Public showcase

`showcase/` is the public website: three tabs (History, Present, Future) on the models, data and visualizations,
each with a "how this page is built" overview, a short math box and definitions. It reads a cleaned snapshot
(managers by first name only, no credentials), so it can be hosted publicly:
`streamlit run showcase/streamlit_app.py`. See [showcase/README.md](showcase/README.md) for what the snapshot
contains and how to deploy it free on Streamlit Community Cloud.

## Private login data

This repository is public, so the ESPN login never goes into it:

| What | Where it lives | Who uses it |
|---|---|---|
| `ESPN_S2`, `ESPN_SWID` cookies and `ESPN_LEAGUE_ID` | `.env` on your own computer only (git-ignored, never committed) | the full research app and `scripts/export_showcase.py`, both run locally |
| League data downloaded with them | `data/cache/` (git-ignored) | same |
| The public website | `showcase/data/`: a cleaned snapshot, managers by first name only, no ids, surnames or cookies | the hosted showcase, which needs no secrets |

Updating the website is a local three-step loop: run `PYTHONPATH=src .venv/bin/python scripts/export_showcase.py`,
check with `PYTHONPATH=src .venv/bin/python scripts/check_public.py` (scans every file git would publish for the
cookies, the league id, ESPN member ids and other managers' names), then commit and push. The host redeploys.

To run the privacy check automatically before every push, install it as a git hook once:
`printf '#!/bin/sh\nPYTHONPATH=src .venv/bin/python scripts/check_public.py\n' > .git/hooks/pre-push && chmod +x .git/hooks/pre-push`.

`espn_s2` works like a password for your ESPN account. If it ever leaks, sign out of ESPN on all devices (or
change the ESPN password) to invalidate it, then copy the new cookies into `.env`.

## Quick start (full research app, needs your league's ESPN cookies)

Windows:

```bash
python -m venv .venv
.venv\Scripts\pip install -e .[dev]
copy .env.example .env      # then fill in ESPN_LEAGUE_ID, ESPN_S2, ESPN_SWID
.venv\Scripts\streamlit run app/streamlit_app.py
```

macOS (needs Python 3.11+; the system `python3` is 3.9, so use Homebrew's, e.g. `brew install python`):

```bash
/opt/homebrew/bin/python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp .env.example .env        # then fill in ESPN_LEAGUE_ID, ESPN_S2, ESPN_SWID
.venv/bin/streamlit run app/streamlit_app.py
```

The first launch takes a few minutes: it pulls 10 seasons of ESPN + MoneyPuck data plus your league's
transactions and game logs, fits the models and caches everything. After that it starts instantly.

The draft-day tools (Pre-draft plan, Draft room with its ESPN-log loader) and the trade analyzer were archived
after the 2027 draft; see [archive/README.md](archive/README.md) to restore them. The in-season add/drop
notifier was removed.

| Page (left menu) | What it does |
|---|---|
| Walkthrough | Plain-language guide: the short version, a step-by-step game plan, how the model works, the evidence scorecard, glossary |
| League history | Who wins and how; manager review, clickable manager map, drill-down and head-to-head; past-draft explorer; best pickups, activity vs winning; injury timelines and injury luck |
| Graph spectral analysis | Interactive similarity graph of managers, affinity matrix, Laplacian spectrum and eigengap, kernel-width sweep, Fiedler axis, 3-D spectral map, group profiles, and two tests of whether the groups are real (permutation test, bootstrap stability) |
| Assumption tests | A scorecard of every modelling assumption checked against your league's real results: Normal weekly scores, P(win) calibration, stat dispersion and covariance, projection calibration, pick-model calibration, and whether the experiment verdicts depend on test assumptions (with a power calculator) |
| Optimization | What the app maximises and how: interactive P(win) calculator and risk insight, the Hungarian lineup assignment, Monte Carlo rollouts with common random numbers vs independent futures, where each player gets drafted, positional scarcity and VONA |
| Theory tests | Your own hunches tested properly. Theory 1: when an NHL team beats expectations, do all its players benefit? Leave-one-out team surprise, team-level permutation test, team explorer, who benefits, and whether a hot first half carries over (experiment H5) |
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
  experiments.py  pre-registered hypotheses H1-H5, paired tests, Holm correction
  diagnostics.py  assumption tests against real matchups, weekly stat lines and held-out drafts
  team_effects.py theory test: does an NHL team's surprise spread to its players (and carry over)?
```

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the math and the experimental design.

## Layout

- `src/puckcore/`: **domain-agnostic** experiment framework (cache, paired statistics, experiment runner,
  plugin protocols). Reusable in other projects without changes.
- `src/iknowpuck/`: the hockey/ESPN plugin.
- `app/`: the full Streamlit research app (needs the league cookies).
- `showcase/`: the public, credential-free showcase; `scripts/export_showcase.py` writes its snapshot.
- `archive/`: retired draft-day pages.
- `tests/`: unit tests on synthetic data (`pytest`).

Run the experiment suite: `python -m iknowpuck.experiments` (add `--quick` to skip the draft backtest).
Reports are written to `runs/<timestamp>_ikp/report.md`; `results.json` also keeps every unit's paired
difference so the Assumption tests page can check the tests' own assumptions.

On macOS with Python 3.13+, `site` skips `.pth` files that carry the Finder "hidden" flag, and folders synced
by iCloud Drive (such as `~/Documents`) can set it on the editable-install file. If `import iknowpuck` fails
outside the app, run commands with `PYTHONPATH=src` (the app and `pytest` already add it), for example
`PYTHONPATH=src .venv/bin/python -m iknowpuck.experiments`.
