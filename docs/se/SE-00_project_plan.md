# SE-00 Project plan, risks and decision log

**Status: DRAFT.** AI-drafted by Claude on 2026-10-06 from the repository and its in-app SE page. Nothing here is reviewed by Luke yet.

Luke owns every decision and date. Past decisions below are reconstructed from commits and files; Luke confirms or corrects each one.

## Current phase

**In-season, 2026-27 (ESPN season 2027).**

| Date | What happened | Source |
|---|---|---|
| 9/23 | Draft assistant built: `puckcore` core, `iknowpuck` plugin, experiments, SE page | commits 859927c to 3299476 |
| 9/24, 9/26, 9/27 | Experiment suite run in seed-0/seed-1 pairs | `runs/` (six runs, SE-03) |
| 9/27 | League draft | `archive/README.md` |
| 10/1 | Draft tools and trade analyzer archived; add/drop notifier removed; public showcase and privacy check added | 46bbe0f |
| 10/2 | Showcase reorganized into History, Present, Future; snapshot exported (week 1) | 828a8b3, `showcase/data/meta.json` |
| 10/5 | Private in-season tools moved to git-ignored `app/private/` | c9284b6 |
| 10/6 | SE light set drafted (this folder); `pytest`: 35 passed | D-014 |

## Capacity

Side work. It ranks behind the thesis and ISE coursework. No weekly hours are recorded (OD-2). Nothing here may take time from thesis deadlines.

## Gates and milestones

No gate is passed on expected results. Only 10/14 is a fixed date; Luke sets the others.

| Gate | Target | Minimum evidence | Stop or redesign if |
|---|---|---|---|
| G0 SE set reviewed | Luke sets (OD-2) | Luke has answered the explain-backs; OD-3 and OD-4 are closed | The evidence behind `VERIFIED` rows can't be cited publicly. Then cite run IDs only and keep the reports local. |
| G1 Lab talk | **Wed 10/14** | Talk cut to the time slot (OD-1); demo runs from the public showcase; PRIV-1 prints OK on what is pushed | PRIV-1 flags the snapshot. Then demo locally and push nothing. |
| G2 Evidence repair | Luke sets | R15 implemented with a test; METHODOLOGY results match stored runs (OD-4); R3 margin set (OD-5) | — |
| G3 Re-fit and re-run | Luke sets | Variance constants re-fit to the measured values, then H2 re-run as a seed pair (METHODOLOGY §11) | Verdicts change. Then record a decision before changing the engine. |
| G4 Full-recommender backtest (H2b) | Before the next draft | H2b seed pair stored in `runs/` | Monte Carlo lookahead does not beat the base policy. Then draft with the simpler policy. |
| G5 Next draft ready | Before the 2027-28 draft (date unknown) | Draft tools restored from `archive/`; tests pass; R6 timing recorded | ESPN's read API still shows no picks during a live draft. Then use the paste fallback (`archive/README.md`). |
| G6 Season validation | After the 2026-27 regular season | Season results compared with the league; H5 re-tested on 2027 data as the clean test (METHODOLOGY §10) | — |

## 10/14 lab meeting

- A 7-minute, 7-slide outline already exists on the SE page (§8): problem 45 s, architecture 60 s, method 75 s, verification 75 s, findings 90 s, live demo 60 s, transfer 30 s.
- Time per project is unknown (OD-1). Suggested cuts. **About 3 minutes:** problem (30 s), critical findings (90 s: the leakage that verification caught, the winner's curse in H2a, the simpler opponent model), showcase demo (60 s). **About 5 minutes:** add verification (60 s) and architecture (30 s). Cut method and transfer first.
- Demo from the public showcase, not the research app. It needs no cookies and shows no private data.
- The SE page's architecture diagram still shows the removed waiver optimiser and "6 pages" (there are 7). Fix it before 10/14 or skip that slide.

## Backlog (found while drafting)

| Item | Finding | Source |
|---|---|---|
| B-1 | All six stored runs record git `3299476` (9/23). H5 was committed on 10/1 (46bbe0f), so the 9/26 and 9/27 runs ran on uncommitted code. The runner records `HEAD` with no clean-tree flag. | `runs/*/report.md`; `puckcore/experiment.py` `_git_commit` → R15 |
| B-2 | METHODOLOGY §8 results don't all match the stored runs. Example: H1b is 0.35 (CI 0.12 to 0.59) there and 0.32 (CI 0.09 to 0.56) in every run. Verdicts agree. | OD-4 |
| B-3 | METHODOLOGY says H4 shows "no difference". In all six runs the CI is entirely below zero (pooling slightly worse). | `runs/*/report.md` |
| B-4 | The trade study's own weights rank ADP-only (3.95) above the chosen market-anchored policy (3.75). | SE page §6; OD-6 |
| B-5 | SE page diagram is stale (waiver optimiser, "6 pages"). | SE page §2 |
| B-6 | `draft.py`'s module docstring says opponents use per-manager models; `pipeline.py` passes the league-wide model to the simulator. | `draft.py` line 5; `pipeline.py` lines 167–169 |
| B-7 | The pre-push hook lives only in `.git/hooks/` and is not versioned, so a fresh clone has no hook. | R12 |

## Risk register

Likelihood and impact are each rated L, M or H. The ratings are Claude's first estimate; Luke sets them.

**Technical failure modes** are in the SE page's FMEA (§7), which these risks do not duplicate. Its highest risk priority numbers (S × O × D): projection error on a player (180), data leakage in backtests (168), stale injury information (120). Three of its seven rows (cookie expiry during the draft, pick order changes, Monte Carlo noise) are draft-day modes. They are dormant until G5.

| ID | Risk | Likelihood | Impact | Mitigation / trigger |
|---|---|---|---|---|
| PR-1 | Project time competes with the thesis and coursework. | H | H | Side work only. No new features before G0. **Trigger:** any thesis deadline at risk → pause this project. |
| PR-2 | Private data leaks through the public repo or site: cookies, the league ID, member IDs, other managers' names, team names. | M | H | `.env` git-ignored; first-name export (R11); PRIV-1 before every push; pre-push hook (R12). Known gaps: PRIV-1 misses surnames under 5 letters and team names under 6 letters, and the hook isn't versioned (B-7). **Trigger:** a cookie leaks → sign out of ESPN on all devices and replace the cookies (README). |
| PR-3 | Private in-season tools get published. | L | H | `app/private/` git-ignored with its own local repo; PRIV-1 and the hook both refuse it (R14). |
| PR-4 | ESPN changes its read API, or cookies expire. | M | M | Cached raw JSON; parsing isolated in one module (FMEA row 2). Live-draft lesson: the API showed no picks during the draft, so paste was the fallback. |
| PR-5 | The evidence can't be audited: `runs/` is git-ignored, run records name the wrong commit (B-1), and METHODOLOGY numbers drift from the runs (B-2). | H | M | G2. Cite run IDs in SE-03 and decide OD-3. |
| PR-6 | AI-drafted work goes unreviewed. 16 of the 17 commits are co-authored with Claude, and Claude drafted this SE set. | H | M | Explain-backs; Luke approves each section; no status above `IMPLEMENTED` without a cited run or test. |
| PR-7 | The showcase snapshot goes stale. The last export was 10/2 (week 1). | H | L | Set a refresh cadence (OD-8). |
| PR-8 | Model constants are known to be off: covariance inflation measured at 1.36 vs 1.6 assumed. | — (known) | M | G3: re-fit, then re-run H2 (METHODOLOGY §9, §11). |

## Decision log

Each decision records who made it and where the repo shows it. Status "reconstructed" means Claude inferred it from the repo and Luke has not confirmed it.

| ID | Date | Decision | By | Source | Status |
|---|---|---|---|---|---|
| D-001 | 09/23 | Split a domain-agnostic core (`puckcore`: cache, paired statistics, experiment runner, plugin interfaces) from the hockey/ESPN plugin (`iknowpuck`). `puckcore` imports nothing from `iknowpuck`. | Luke | 859927c; `src/puckcore/__init__.py` | reconstructed |
| D-002 | 09/23 | Every modelling claim gets a pre-registered paired test with Holm correction (FWER 0.05). Reports record the commit and seed, and each claim is re-run with a second seed. | Luke | 859927c, 1228e9b; METHODOLOGY §8 | reconstructed |
| D-003 | 09/23 | The objective is head-to-head **points**, not categories. The league had been described as categories; ESPN's settings showed points in every season. | Luke | SE page §5; METHODOLOGY §1 | reconstructed |
| D-004 | 09/23 | Backtest leakage fixed: ESPN's undrafted ADP sentinel (about 230) and the wiped 2026 ADP are replaced by ESPN's preseason rank, and player order is shuffled before tie-breaks. | Luke | 5156e61 | reconstructed |
| D-005 | 09/23 | The rollout base policy is **market-anchored** (best projected value among the next two roster-fitting players by ADP), not projection-greedy (VONA), because VONA lost to ADP in H2a. A market adjustment shrinks projections toward ADP. | Luke | 5156e61; `draft.py` `market_choice`; METHODOLOGY §3, §7 | reconstructed |
| D-006 | 09/23 | The simulator uses the **league-wide** opponent logit. Per-manager and spectral-pooled models did not predict held-out picks better (H3, H4); they are kept for scouting only. | Luke | 5156e61; `pipeline.py` lines 167–169 | reconstructed |
| D-007 | 10/01 | Bench usage follows the league's lineup lock: 0.25 for this weekly-lock league, measured from 756 team-weeks. | Luke | 46bbe0f; METHODOLOGY §4, §9 | reconstructed |
| D-008 | 10/01 | After the 9/27 draft, the Draft room, the Pre-draft plan, the ESPN-log loader and the trade analyzer are **archived** (restorable), not deleted. | Luke | 46bbe0f; `archive/README.md` | reconstructed |
| D-009 | 10/01 | The in-season add/drop notifier is **removed** (kept only in git history). Its roster reader moved to `EspnClient.rosters()`. | Luke | 46bbe0f; `archive/README.md` | reconstructed |
| D-010 | 10/01 | The public showcase is a separate, credential-free app that reads only a cleaned snapshot. `export_showcase.py` writes the snapshot locally (managers by first name only, no team singled out), and `check_public.py` scans before every push. | Luke | 46bbe0f, 828a8b3; `showcase/README.md` | reconstructed |
| D-011 | 10/02 | Host the showcase free on Streamlit Community Cloud with Secrets left empty. Claude's notes say it deploys from `main`; a stale `public-showcase` branch also exists (OD-8). | Luke | `showcase/README.md`; Claude's notes of 10/2 | reconstructed; branch to confirm |
| D-012 | 10/05 | Private in-season tools live in git-ignored `app/private/`. The research app shows them only where the folder exists; PRIV-1 and the pre-push hook refuse them. | Luke | c9284b6; `.gitignore`; `.git/hooks/pre-push` | reconstructed |
| D-013 | 10/05 | Install the privacy check as a local pre-push hook. | Luke | `.git/hooks/pre-push` (file date 10/5); README | reconstructed |
| D-014 | 10/06 | The SE set for this project is the **light set**: README, SE-00, SE-01, SE-02 and SE-03 in `docs/se/`. SE-04 to SE-06 are not written; the app's SE page covers them. | Luke | Luke's 10/6 instruction | decided |
| D-015 | 10/06 | `docs/se/` is committed to this public repo. The run reports in `runs/` stay git-ignored until OD-3 is decided. | Luke | Luke's 10/6 instruction | decided |

## Open decisions

| # | Decision | Owner | Needed by |
|---|---|---|---|
| OD-1 | Time per project at the 10/14 lab meeting, and which cut of the talk to use | Luke | 10/13 |
| OD-2 | Weekly hours for this project during the term, and review dates for SE-01 to SE-03 | Luke | after 10/14 |
| OD-3 | Publish the run reports or cite local-only evidence. `runs/` is git-ignored; on 10/6 Claude scanned all twelve run files with PRIV-1's terms and found nothing. (Committing `docs/se/` itself is decided: D-015.) | Luke | before the next push |
| OD-4 | Which stored runs are canonical, and update METHODOLOGY §8 to match them (B-2, B-3) | Luke | before the next push (the committed `docs/se/` cites runs that METHODOLOGY contradicts) |
| OD-5 | A non-inferiority margin for R3 (how much worse than ADP still counts as "at least as good") | Luke | before the next H2 run (G3) |
| OD-6 | The trade study ranks ADP-only first by its own weights. Revise the scores or weights, or state why market-anchored was chosen. | Luke | 10/14, if the trade study is shown |
| OD-7 | R6 (timing) and R7 (plain language) have no stored evidence. Record it, or retire them while the draft room is archived. | Luke | before G5 |
| OD-8 | Showcase refresh cadence (the README suggests weekly) and the deploy branch (`main` vs the stale `public-showcase`) | Luke | next push |
| OD-9 | Order of the next work: G3 re-fit, G4 (H2b), restoring draft tools, other features | Luke | after 10/14 |

## Change control

Changes to the question, the Must tier, the privacy boundary or the minimum evidence need a decision-log entry by Luke. AI assistants may propose changes but never make them. Any change to `export_showcase.py`, `check_public.py`, `.gitignore` or the showcase data needs a passing PRIV-1 before the push.

---

**Explain-back (Luke):** Which risk here could hurt someone other than you, what check stops it before a push, and which gap in that check (B-7, or the short-name blind spot) would you close first?

>
