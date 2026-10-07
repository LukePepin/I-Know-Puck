# SE-01 Charter

**Status: DRAFT.** AI-drafted by Claude on 2026-10-06 from the repository and its in-app SE page. Nothing here is reviewed by Luke yet.

## 1. Problem

A 12-team snake draft is a sequential decision under uncertainty. Each pick changes what the 11 opponents can take, and their choices are random. The default is to draft by ADP, the market's average draft position. Projection-based tools rank players by value, but an optimiser that picks from noisy estimates tends to pick their errors (the winner's curse). The question is whether an operations-research assistant can do better than the market, and the project trusts no modelling claim until it has been tested on the league's own history.

A second aim: show the lab a verification discipline and reusable parts (SE page §9).

## 2. Question (`PROPOSED`, drafted from the SE page's mission)

> Does a draft assistant whose every modelling claim is tested on the league's own history pick rosters that win weekly head-to-head matchups more often than drafting by ADP?

**Answer so far: not shown.** The market-anchored policy ties ADP (H2, 6 of 6 runs). The projection-greedy policy loses to it (H2a). The full Monte Carlo recommender used on draft day has no backtest yet (H2b, G4).

## 3. Scope

**Must (built, `IMPLEMENTED`):**
- [x] Data: ESPN league history, projections and ADP, plus MoneyPuck advanced stats, in a disk cache.
- [x] Models: blended projections with a market adjustment; valuation by optimal slotting to P(win a weekly matchup); a league-wide opponent logit (D-006).
- [x] Decision engine: Monte Carlo rollouts with common random numbers and the market-anchored base policy (D-005).
- [x] Verification: experiment suite H1–H5 (paired tests, Holm), assumption tests, 35 unit tests.
- [x] Research app, run locally: Walkthrough, League history, Graph spectral analysis, Assumption tests, Optimization, Theory tests, Systems engineering.
- [x] Privacy controls and a public showcase (§4).

**Should (next, `PROPOSED`):** reliable evidence records (G2: R15, METHODOLOGY matched to the runs); re-fit the variance constants and re-run H2 (G3); backtest the full recommender (H2b, G4); restore the draft tools before the next draft (G5); re-test H5 on the 2027 season (G6); refresh the showcase on a set cadence (OD-8).
**Could:** reuse `puckcore` in other projects; other ESPN fantasy sports (SE page §9).
**Archived (restorable, D-008):** Draft room with its ESPN-log loader, Pre-draft plan, trade analyzer.
**Won't:** the in-season add/drop notifier in the research app (removed, D-009); anything from `app/private/` in the public repo or site (D-012); relying on ESPN's API for live picks during a draft, since it showed none on 9/27 (`PROPOSED`).

## 4. Privacy boundary

The repository is public. Everything git would publish is public.

| Side | What | Where |
|---|---|---|
| **Private, Luke's machine only** | ESPN cookies and the league ID | `.env` (git-ignored) |
| | League data downloaded with them | `data/cache/` (git-ignored) |
| | Luke's in-season lineup and waiver tools | `app/private/` (git-ignored, own local repo) |
| | Injury overrides and the pasted draft log | `data/` (git-ignored files) |
| **Local only, not secret** | Experiment run outputs | `runs/` (git-ignored; OD-3) |
| **Public** | Code, tests, docs, archived pages | `src/`, `app/`, `tests/`, `docs/`, `archive/` |
| | Cleaned snapshot: managers by first name only; no cookies, league ID, member IDs, surnames or team names | `showcase/data/` |

**One crossing:** `scripts/export_showcase.py` reads the private side and writes the snapshot. `scripts/check_public.py` (PRIV-1) and the pre-push hook guard every push (D-010, D-013).

## 5. Minimum evidence

1. Every `VERIFIED` requirement in SE-03 cites a stored run ID or a test file.
2. Each experiment claim has a seed-0/seed-1 pair with matching verdicts.
3. `pytest` passes (35 of 35 on 10/6).
4. PRIV-1 prints OK before every push.
5. Nothing is claimed about the live season until it is over. Backtests are not live results.

**Dates and decisions:** [SE-00](SE-00_project_plan.md).

---

**Explain-back (Luke):** In 30 seconds, without notes: what does I Know Puck claim today, what has the evidence actually shown against ADP, and what sits on each side of the privacy boundary?

>
