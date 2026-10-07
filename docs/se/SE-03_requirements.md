# SE-03 Requirements and verification traceability (RVTM)

**Status: DRAFT.** AI-drafted by Claude on 2026-10-06 from the repository and its in-app SE page. Nothing here is reviewed by Luke yet.

R1–R10 keep the IDs of the app's SE page (§3), so the page and this document stay traceable. Each one is rewritten as a single "shall". R11–R15 are new. Statuses describe the code and the evidence. Every requirement is still unreviewed by Luke, and none is `ACCEPTED`.

**Status rule here:** `VERIFIED` only where a stored run in `runs/` or a test file is cited. If the only evidence is the app's own label, the status is `IMPLEMENTED` and the evidence file is to be linked.

**Methods:** T = test, D = demonstration, I = inspection, A = analysis.

## Stakeholder needs

N1–N4 come from the SE page's stakeholder table. N5 and N6 were added by Claude (`PROPOSED`).

| Need | Statement |
|---|---|
| N1 | The manager gets a clear, fast recommendation during a live draft, in plain language. |
| N2 | The analyst gets reproducible experiments, honest statistics and a reusable framework. |
| N3 | The models follow the league's ESPN rules, rosters, schedule and opponent behavior. |
| N4 | The system respects the data providers' rate limits and handles format changes and data-quality problems. |
| N5 | Luke's ESPN login and the other managers' identities stay out of the public repo and site. |
| N6 | The public and the lab can see the work without credentials. |

## Requirements

| ID | Requirement | Method | Trace | Allocated | Verified by | Status |
|---|---|---|---|---|---|---|
| R1 | The blended projection shall have lower absolute fantasy-point error than ESPN's projection on the held-out seasons 2024–26. | T | N2; D-002 | `projections.py` | H1 | VERIFIED (H1, six runs: reject H0 in all) |
| R2 | Adding the MoneyPuck features shall lower the own model's absolute fantasy-point error on held-out seasons. | T | N2; D-002 | `projections.py` (residual ridge), `data/moneypuck.py` | H1b | VERIFIED (H1b, six runs: reject H0 in all). The effect is small; whether it "justifies the cost" (the app's wording) is Luke's call. |
| R3 | The draft policy shall win weekly matchups at least as often as drafting by ADP, in a backtest on 2024–26 scored with actual stats. | T, A | N1; D-005 | `draft.py` market-anchored policy | H2 | IMPLEMENTED. H2 shows no difference in six of six runs. "At least as good" needs a non-inferiority margin before it can be verified (OD-5). |
| R4 | The engine shall not use a draft policy that loses to ADP drafting in the backtest. | T, I | N1; D-005 | `draft.py` rollout | H2a; inspection of `draft.py` | VERIFIED (H2a, six runs: VONA worse than ADP, CI below zero in all; the rollout uses `market_choice`) |
| R5 | The simulator's opponent model shall be no more complex than held-out drafts support. | T, I | N2, N3; D-006 | `opponents.py`, `spectral.py`, `pipeline.py` | H3, H4; inspection of `pipeline.py` | VERIFIED (H3, H4, six runs: no gain in any; `pipeline.py` passes the league-wide model to the simulator). B-6: the `draft.py` docstring still says per-manager. |
| R6 | The recommender shall return a suggested pick in under 15 s during a live draft. | T | N1 | `draft.py` `recommend`; Draft room (archived, D-008) | Timing | IMPLEMENTED. The app says about 7 s for 12 × 20 rollouts. No timing record exists; evidence to be linked (OD-7). |
| R7 | Each page shall explain its results in plain language for non-experts. | I | N1, N6 | Walkthrough, notes, glossary | Usability review | IMPLEMENTED. The app says "Reviewed", but no review record exists; evidence to be linked (OD-7). |
| R8 | Player values shall reflect current injuries, with overrides Luke can edit. | T, I | N1, N3 | `injuries.py`; injury table | UT-HIST | VERIFIED (UT-HIST: `test_apply_injuries_scales_projection_and_respects_override`, `test_injury_history_flags_high_risk`). The app's "manual check on flagged stars" is not recorded. |
| R9 | Re-running the experiment suite with another seed shall give the same verdict for every hypothesis. | T | N2; D-002 | `puckcore/experiment.py`, `iknowpuck/experiments.py` | Seed pairs; UT-CORE | VERIFIED (seed pairs of 9/24, 9/26 and 9/27: identical verdicts; UT-CORE `test_run_suite_writes_report`). Exact-code reruns are not shown (R15). |
| R10 | The ESPN cookies and the league ID shall be kept only in the local, git-ignored `.env`, and shall appear in no file git would publish. | I, T | N5 | `.gitignore`, `config.py`, `data/espn.py` | PRIV-1 | VERIFIED (PRIV-1, passed 10/6; console output, not stored) |
| R11 | The showcase snapshot shall name managers by first name only (with a last initial when two share a first name), and shall contain no cookies, league ID, ESPN member IDs, surnames or fantasy team names. | I, T | N5; D-010 | `scripts/export_showcase.py` | PRIV-1; inspection | IMPLEMENTED. PRIV-1 passes, but it cannot see surnames under 5 letters or team names under 6 letters. |
| R12 | A push shall be refused if PRIV-1 finds private data, if the branch name starts with `private`, or if the pushed commit contains `app/private/`. | T | N5; D-013 | `.git/hooks/pre-push` | PRIV-2 | IMPLEMENTED. Installed locally on 10/5; not versioned (B-7); no recorded refusal test. |
| R13 | The showcase shall run with no credentials and no web requests, reading only `showcase/data/`. | I | N5, N6; D-010 | `showcase/` | Inspection | IMPLEMENTED. Claude's inspection on 10/6 (not stored): no `iknowpuck`, credential or HTTP-library imports; `requirements.txt` lists only streamlit, pandas, numpy and plotly. |
| R14 | No file under `app/private/` shall be among the files git would publish. | T | N5; D-012 | `.gitignore`; PRIV-1; pre-push hook | PRIV-1 | VERIFIED (PRIV-1, passed 10/6; console output, not stored) |
| R15 | Each run record shall identify the exact code that produced it: the commit plus a clean-tree flag. | T | N2; B-1 | `puckcore/experiment.py` | Unit test to be written | PROPOSED |

**Coverage:** N1, N2, N3, N5 and N6 each trace to at least one requirement. **N4 has none yet**; FMEA row 2 (API format changes) is the starting point (`PROPOSED`).

## App labels vs SE status

The SE page computes the labels for R1–R4 and R9 from the two newest runs, so they currently reflect the 9/27 pair. The labels for R5–R8 and R10 are fixed text.

| ID | The app says | Here | Why it differs |
|---|---|---|---|
| R1, R2 | Verified | VERIFIED | — |
| R3 | Met (no difference) | IMPLEMENTED | No difference is not the same as "at least as good" without a margin |
| R4 | Rejected as designed | VERIFIED | — |
| R5 | Simplest model kept | VERIFIED | — |
| R6 | Verified | IMPLEMENTED | No timing record |
| R7 | Reviewed | IMPLEMENTED | No review record. Here `REVIEWED` means Luke reviewed an SE claim. |
| R8 | Verified | VERIFIED | Unit tests only |
| R9 | Verified | VERIFIED | Seed replication only (R15) |
| R10 | Verified | VERIFIED | PRIV-1 |

## Verification IDs

| ID | What | Evidence |
|---|---|---|
| H1, H1b, H2, H2a, H3, H4 | Experiment suite: paired tests, Holm correction (FWER 0.05) | `runs/20260924-221645_ikp` (seed 0), `20260924-221831_ikp` (seed 1), `20260926-181644_ikp` (0), `20260926-181826_ikp` (1), `20260927-214035_ikp` (0), `20260927-214219_ikp` (1). All local only (OD-3). |
| H5 | Team-surprise theory test. It was registered after an exploratory look at the same seasons, so it is a replication check (METHODOLOGY §10). | The four runs of 9/26 and 9/27 |
| H2b | Backtest of the full Monte Carlo recommender | not run (G4) |
| UT-CORE | `tests/test_core.py` (4 tests) | `pytest`: 35 passed on 10/6 |
| UT-DIAG | `tests/test_diagnostics.py` (11) | same |
| UT-HIST | `tests/test_history.py` (5) | same |
| UT-HOCKEY | `tests/test_hockey.py` (7) | same |
| UT-SUM | `tests/test_summaries.py` (4) | same |
| UT-TEAM | `tests/test_team_effects.py` (4) | same |
| PRIV-1 | `scripts/check_public.py`: scans every publishable file for the cookies, the league ID, member IDs, other managers' names and team names, and anything under `app/private/` | passed 10/6 (console) |
| PRIV-2 | `.git/hooks/pre-push`: refuses private branches and `app/private/`, then runs PRIV-1 | local only, not versioned |

## Experiment results (from the stored runs)

A positive difference means the treatment is better. For the error metrics, that means lower error. Numbers are from the newest seed-0 run, `20260927-214035_ikp`.

| ID | Treatment vs baseline | Runs | Verdict in every run | Difference (95% CI) | Reading |
|---|---|---|---|---|---|
| H1 | Blend vs ESPN projection | 6 | reject H0 | +4.6633 (+3.8500, +5.4652) points | supported |
| H1b | With vs without MoneyPuck | 6 | reject H0 | +0.3232 (+0.0892, +0.5644) points | supported, small |
| H2 | Market-anchored vs ADP | 6 | fail to reject | −0.0082 (−0.0250, +0.0085) P(win); −0.0082 to +0.0094 across runs | no difference |
| H2a | VONA vs ADP | 6 | fail to reject | −0.1008 (−0.1206, −0.0804) P(win); −0.0800 to −0.1008 across runs | worse than ADP in all six |
| H3 | Per-manager + spectral vs league-wide logit | 6 | fail to reject | −0.0061 (−0.0226, +0.0107) log-likelihood | no difference |
| H4 | Spectral pooling vs independent shrinkage | 6 | fail to reject | −0.0101 (−0.0188, −0.0014) log-likelihood | slightly worse in all six (B-3) |
| H5 | Adding the team's first half | 4 | fail to reject | −0.0021 (−0.0036, −0.0007) points per game | slightly worse in all four |

---

**Explain-back (Luke):** Pick R3 and R11. For each one, say what it guarantees, why it is not `VERIFIED` yet, and what evidence would change that.

>
