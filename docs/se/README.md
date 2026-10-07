# Systems engineering documents: I Know Puck

**Status: DRAFT.** AI-drafted by Claude on 2026-10-06 from the repository and its in-app SE page. Nothing here is reviewed by Luke yet.

A light SE record for an operations-research assistant for one ESPN fantasy hockey league. The system is largely built, so these documents mostly **record what exists, with evidence**, and frame what comes next. The math and the experiment design are in [`docs/METHODOLOGY.md`](../METHODOLOGY.md). This set is the light set (D-014).

## The V

```
 Charter + ConOps (SE-01, SE-02) ..................... Validation: the 2026-27 season vs the league (under way, no result yet)
     Requirements (SE-03) ......................... System verification: backtests H2, H2a; privacy check PRIV-1
         Architecture (app SE page §2) ......... Integration: end-to-end pipeline build; every page loads
             Components (app SE page) ....... Component verification: H1, H1b, H3, H4, H5
                       \                       /
                    Implementation → unit tests (pytest, 35 tests)
```

Read each dotted line as "verified by". The left side below SE-03 is not written out as documents: it lives on the app's Systems engineering page (`app/app_pages/systems_engineering.py`, sections 2 and 4). Run evidence is in `runs/<run id>/report.md`. `runs/` is git-ignored, so that evidence exists only on Luke's machine (OD-3).

## Document set

| ID | Document | Who writes | Due | State |
|---|---|---|---|---|
| SE-00 | [Project plan, risks, decision log](SE-00_project_plan.md) | Claude drafts; Luke owns decisions and dates | ongoing | DRAFT: AI-drafted 10/6, not reviewed |
| SE-01 | [Charter](SE-01_charter.md) (about one page) | Claude drafts from the repo; Luke approves | Luke sets (OD-2) | DRAFT: AI-drafted 10/6, not reviewed |
| SE-02 | [ConOps](SE-02_conops.md): actors, two deployments, season lifecycle, use cases | Claude drafts from the repo; Luke approves | Luke sets (OD-2) | DRAFT: AI-drafted 10/6, not reviewed |
| SE-03 | [Requirements + RVTM](SE-03_requirements.md): R1–R10 from the app, plus privacy requirements | Claude drafts from the repo; Luke approves each requirement | Luke sets (OD-2) | DRAFT: AI-drafted 10/6, not reviewed |
| — | App SE page: mission, stakeholders, architecture, R1–R10, V-model, findings, trade study, FMEA, 7-minute lab talk | existing (commit 86e910e, updated 46bbe0f) | — | IMPLEMENTED; parts are stale (SE-00 backlog) |

**Not in the light set:** SE-04 (architecture), SE-05 (interfaces) and SE-06 (V&V plan). The architecture diagram, the V-model and the FMEA already live on the app's Systems engineering page. These documents reference that page and do not duplicate it. If the page and a document disagree, the code and the stored evidence decide, and the disagreement goes into the SE-00 backlog.

## Rules for every document

- **Status on every claim.**
  - `PROPOSED`: not yet decided.
  - `DECIDED (D-nnn)`: recorded in the SE-00 decision log.
  - `REVIEWED`: Luke has reviewed it; explain-back pending.
  - `ACCEPTED`: reviewed, and Luke has answered the explain-back.
  - `IMPLEMENTED`: the code exists.
  - `VERIFIED (test id, evidence file)`: a recorded test shows it.
- **The app's own labels are not evidence.** The SE page says "Verified", "Reviewed" or "Met" for some requirements. SE-03 uses `VERIFIED` only where a stored run or a test file shows it.
- **Who writes.** Luke directs and Claude drafts. Luke approves every section and makes every decision.
- **Explain-back.** Every AI-drafted section ends with an explain-back prompt. It is accepted only after Luke answers in his own words below the prompt.
- **Keep evidence types separate.** A backtest on past seasons is not a live-season result. A seed-0/seed-1 replication is not a re-run of the exact code (R15).
- **This repository is public.** Never write the league ID, cookie values, ESPN member IDs, other managers' names or team names, or anything from `.env`, `data/cache/` or `app/private/`. Refer to managers generically. Run `scripts/check_public.py` before any commit that touches this folder.
- **Scope changes** go through the SE-00 decision log. AI proposes; it never decides.

---

**Explain-back (Luke):** Why is this a light set, where do the architecture, V-model and FMEA live instead, and which evidence behind a `VERIFIED` row would a reader of the public repo not be able to see?

>
