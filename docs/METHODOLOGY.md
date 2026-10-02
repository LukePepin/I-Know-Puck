# Methodology

## 1. Problem

A 12-team, 26-round ESPN snake draft in an **H2H points** league. Scoring: skaters G 2, A 1, PPP 0.5,
SHP 0.5, BLK 0.5, SOG 0.1, HIT 0.1. Goalies: W 4, SO 3, OTL 1, SV 0.2, GA −2. Lineup: 4C / 4LW / 4RW /
6D / 2G / 6BN. Our decision at each of our picks is which available player to take. The objective is
the probability of winning a weekly matchup with the finished roster, averaged over the 11 opponents
the draft actually produces.

This is a sequential decision problem under uncertainty, and the opponents' choices are stochastic.
We solve it approximately with **Monte Carlo rollouts**, a one-step lookahead with a base policy.

## 2. Data

| Source | Content | Use |
|---|---|---|
| ESPN public `kona_player_info` | per-season actual totals, ESPN preseason projections, ADP (2018-2027) | targets, baseline projection, opponent features |
| ESPN private league | settings, team → manager per season, every historical pick (2024-26) | objective, opponent model, spectral graph |
| MoneyPuck season summaries | xG, shot attempts, TOI, 5v4 TOI | projection features |

Managers are keyed by ESPN account id, not team slot. Team slots changed hands (#3, #8, #9), and a
manager who moved slots keeps their history.

## 3. Projections

**Own model (Marcel+).** For stat k, the per-game rate is a recency-weighted history (weights 5/4/3,
weighted by games played), shrunk toward the position-group mean with constant R_k:

    rate_k = (sum_i w_i x_ik + R_k * mu_pos,k) / (sum_i w_i GP_i + R_k)

R_k is chosen by grid search on past seasons. A ridge regression on MoneyPuck features (xG − goals per
game, PP TOI, TOI, shot attempts) corrects the residual. Games played is a linear model on the previous
two seasons' share of the schedule.

**Blend.** final_k = w_k·own_k + (1 − w_k)·ESPN_k. w_k is fit by least squares on *out-of-sample*
own-model predictions for the three prior seasons, so the weights never see the season they are
applied to.

**Market adjustment (winner's-curse guard).** For forwards, defense and goalies separately, actual fantasy
points from past seasons are regressed on the blended projection and log(ADP), plus a no-ADP indicator.
Each player's stat line is then rescaled so that their fantasy points equal the fitted value. Projection
weights come out around 0.5–0.8, and lowest for goalies.

## 4. Valuation (objective function)

Starters are assigned to slots by the Hungarian algorithm (`linear_sum_assignment`). Usage is 1 for
starters. Bench usage depends on the league's lineup lock: 0.35 for skaters (0.2 for goalies) with daily lineups, and 0.25
for everyone with weekly locks. This league locks lineups weekly (ESPN `INDIVIDUAL_FIRSTGAME_WEEKLY`), so the bench only helps
through the weekly lineup choice and injury cover. The 0.25 comes from the league's own weekly scores (section 9).
Weekly team points are modeled as Normal:

    mu  = sum_i u_i pts_i / W
    var = c * sum_i u_i sum_k w_k^2 d_k S_ik / W

using over-dispersed Poisson stats with dispersion d_k and a covariance inflation c, with W = 26 weeks.
P(win vs opponent) = Φ((μ_me − μ_opp) / sqrt(var_me + var_opp)). Category leagues are also supported:
there the objective is a Poisson-binomial over per-category win probabilities.

## 5. Opponent model

For each historical pick, a McFadden conditional logit over the top 300 available players by ADP:

    u_mj = β_adp[m]·(−log ADP_j) + β_need·need_j + β_pos[m, group_j, round phase]

Forwards are the reference position group. Global coefficients are fit by MLE. Per-manager deviations
carry an L2 penalty toward a **cluster mean**, and the clusters come from spectral analysis (section 6).
With only 2–3 drafts per manager, this partial pooling is what keeps per-manager estimates stable.

## 6. Graph spectral analysis of managers

Each manager is described by a feature vector: mean reach vs ADP, early-round reach, reach variance,
ADP rank correlation, D and G share in rounds 1–6, first goalie round, goalies per draft, pro-team
loyalty ("homer index"), and autodraft share. A Gaussian-kernel affinity graph (bandwidth = median
pairwise distance) is built over the standardized features, followed by the normalized Laplacian
L = I − D^{-1/2} W D^{-1/2}. The number of clusters k comes from the largest eigengap, and k-means runs
on row-normalized eigenvectors (Ng–Jordan–Weiss). The Fiedler vector orders managers along the
dominant axis of behavioral difference. Whether the groups are real is tested in section 9: a permutation
test of the eigengap (each habit shuffled independently across managers) and bootstrap co-clustering
stability (seasons resampled within manager, adjusted Rand index vs the full-data partition).

## 7. Draft recommender

At pick t, candidates are the next roster-fitting players in the market (ADP) window, plus a few
highest-VONA players so that large model disagreements are still shown. For each candidate, M rollouts
finish the draft: opponents sample from the league-wide logit (Gumbel-max), and our later picks use the
**market-anchored policy**, which takes the highest projected value among the next two roster-fitting
players by ADP. (The projection-greedy VONA policy lost to ADP drafting in the backtest, see H2a.) The
final league is then scored. All candidates share the same random seeds (**common random numbers**),
so candidate differences are paired. The app reports the mean, the gap to the best candidate, the
paired SE of that gap, P(best), and each player's availability at our next pick. Its "Suggested pick"
is, among candidates within 2 SE of the best, the one least likely to be available at our next pick.

## 8. Experimental design (the "wheel")

Implemented generically in `puckcore.experiment`: hypothesis → paired design → run → analyze →
correct → report. Each experiment returns per-unit scores for a baseline arm and a treatment arm. We
report the mean paired difference with a 10k-bootstrap 95% CI, Cohen's d_z, a sign-flip permutation
p-value (primary), Wilcoxon and t-test p-values, and **Holm–Bonferroni** adjustment across the suite
(FWER α = 0.05). Every report records the git commit and seed.

| ID | H1 (alternative) | Unit | Metric |
|---|---|---|---|
| H1 | blend < ESPN error | player-season, held out 2024–26 | abs fantasy-point error |
| H1b | MoneyPuck features reduce own-model error | player-season | abs error |
| H2 | market-anchored policy > ADP drafting | (season, slot, seed), rosters scored on **actual** stats | P(win weekly matchup) |
| H2a | projection-greedy (VONA) policy > ADP drafting | same as H2 | P(win weekly matchup) |
| H3 | per-manager + spectral logit > league-wide logit | held-out pick (leave-one-season-out) | log-likelihood |
| H4 | spectral pooling > independent shrinkage | held-out pick | log-likelihood |
| H5 | adding the NHL team's first-half surprise to a player's own first half lowers second-half error | skater-season on one NHL team (2024–26), leave-one-season-out | abs fantasy points per game |

### Results (two independent runs, seeds 0 and 1, identical verdicts)

| ID | Result |
|---|---|
| H1 | Supported: 4.65 fewer fantasy points of absolute error per player (95% CI 3.8 to 5.4), Holm p < 0.001 |
| H1b | Supported but small: 0.35 points (CI 0.12 to 0.59) |
| H2 | No difference: +0.005 and -0.001 P(win) in the two runs (CIs within about +/-0.02) |
| H2a | Rejected: -0.083 and -0.100 P(win); the policy was removed from the engine |
| H3 | No difference: per-manager models do not predict held-out picks better |
| H4 | No difference: spectral pooling does not help |
| H5 | Rejected: −0.0021 points per game (CI −0.0036 to −0.0007) in both runs; the team's first half makes predictions slightly worse (added later, see section 10) |

Two data problems were found and fixed along the way: ESPN's undrafted ADP sentinel (about 230) had let
end-of-season % owned leak into tie-breaks, and 2026's historical ADP had been wiped, so ESPN's
preseason rank stands in for missing ADP.

## 9. Assumption tests

The modelling choices above rest on assumptions. `diagnostics.py` checks each one against the league's own
history (2024-26), and the **Assumption tests** page in the app shows every test interactively. Results
from the current data:

| Assumption | Test | Result | Verdict |
|---|---|---|---|
| Weekly team scores are Normal (section 4) | 756 regular-season ESPN matchup scores (`mMatchupScore`), z-scored within team-season; Shapiro-Wilk, D'Agostino K² | skew +0.14, excess kurtosis −0.03, Shapiro-Wilk p = 0.30 | holds |
| P(win) = Φ(Δμ / √(σ²ₐ + σ²ᵦ)) gives honest odds | 378 matchups, each team's mean and SD from its *other* weeks (leave one week out); reliability diagram, Brier score, log loss | Brier 0.224 (coin 0.250), log loss 0.638 (0.693), 64% winners called; best variance multiplier ×0.88 | holds |
| Weekly stat counts are over-dispersed Poisson with dispersion d_k | weekly counts per player-season (≥ 10 weeks); pooled Σ(n−1)s² / Σ(n−1)m, bootstrap CI over player-seasons | 11 of 12 stats within 25% of the assumed d_k; saves 8.3 vs 1.2 (driven by games started); wins 0.62 vs 0.7 | mostly holds |
| Bench usage 0.35 (daily-lineup prior) | league settings (`lineupLocktimeType`) and 756 team-weeks: real score on likely starters' and bench points | the league locks lineups weekly; real ≈ 0.92 × starters + 0.25 × bench (0.20 to 0.29) | wrong for this league, now 0.25 |
| Covariance inflation c = 1.6 | observed weekly fantasy-point variance / Σ w_k² d_k m_k per player-season | 1.36 (95% CI 1.33 to 1.39); F 1.38, D 1.47, G 1.31; 1.14 with measured d_k | too high |
| Projections are calibrated | OLS of actual on projected, held-out seasons | final projection slope 0.98 (0.92 to 1.03); ESPN 0.73 (0.68 to 0.79) | holds (ESPN overconfident) |
| Constant error variance in the market regression | Breusch-Pagan | p = 0.72 (ESPN's projections: p < 0.001) | holds |
| Opponents pick by the conditional logit | leave-one-season-out choice probabilities for 924 picks; reliability, rank of the actual pick | actual pick in the model's top 5: 40% (ESPN order: 41%) | roughly holds |
| Managers have distinct drafting styles (section 6) | eigengap permutation test (1,000 independent column shuffles); bootstrap co-clustering over seasons | p ≈ 0.10; median adjusted Rand index 0.44 | not supported |
| Experiment verdicts don't hinge on test assumptions | sign-flip permutation, Wilcoxon and t-test on the stored per-unit paired differences | all three agree for all 6 hypotheses | holds |

Two findings pull in opposite directions. At the player level the covariance inflation is too high, yet whole
teams still swing more in real life (median within-season SD 20.4 points) than the model's simulated rosters
(about 17), because real weeks also contain injuries, roster moves and schedule effects the model leaves out.
**Bench usage.** The league locks lineups weekly, so the original daily-lineup prior (0.35) was wrong in kind. Regressing each
team-week's real score on the points of its 20 likely starters (top players by position on the manager's own season
average) and of its bench gives real ≈ 0.92 × starters + 0.25 × bench (756 team-weeks; bench 95% CI 0.20 to 0.29), so weekly-lock leagues
now use 0.25. All of these shift every candidate alike, so they
change how confident P(win) is more than which pick ranks first.

The **Optimization** page also measures the variance reduction from common random numbers (section 7): at the
first pick, with 5 candidates and 30 rollouts, the root-mean-square standard error of candidate gaps was 2.6 to 4.7
times smaller than with independent futures across the runs tried (it depends on the seed and the injury settings),
which is roughly 7 to 22 times fewer rollouts for equal precision.

## 10. Theory test: does a team that beats expectations lift all its players?

A manager's theory, tested on the **Theory tests** page (`team_effects.py`). Expectation = ESPN's preseason
projection. Player surprise = fantasy points per game on an NHL team / projected points per game − 1. Team
surprise = the same ratio pooled over his **teammates** only (leave one out), so a player's own points never
appear on both sides. Teams come from the game logs (the team each game was played for), because ESPN's season
records list a player's *current* team (it matched the game-log team for only 81–91% of player-seasons). That
limits the test to the league's game-log seasons, 2024–26: 1,051 skater-seasons (20+ games) on 96 team-seasons.
Inference is at the team level: permutation shuffles team-season results among teams within a season, and the
bootstrap resamples whole team-seasons.

| Question | Result |
|---|---|
| Do players on teams that beat projections score more than projected? | Yes, modestly: +4.3% per +10% teammate surprise (95% CI +2.0% to +5.7%), permutation p = 0.011; measured by team wins vs projected wins instead, p = 0.001 |
| How much of a player's surprise is his team? | About 6% (intraclass correlation 0.06) |
| Do **all** players benefit? | No: on the hottest fifth of teams 50% of players beat their projection (27% on the coldest fifth). Top-of-lineup players (top third by projection within team) gain +0.7% per +10% (CI −1.4% to +2.5%); depth players +6.6% |
| Is it usable in-season? | No: a team's first half predicts its players' second half only through the players' own first halves; with both in the model the team coefficient is +0.1% per +10% (p = 0.79), and out of sample it adds error (H5) |

H5 was registered after this exploratory look at the same seasons, so it is a replication check, not a
pre-registered test; the 2027 season will be the clean test.

## 11. Known limitations / next experiments

- The weekly variance constants are now tested (section 9) but not yet re-fitted. Re-fitting covariance
  inflation, goalie save dispersion and bench usage to the measured values, then re-running H2, is the next step.
- The H2 backtest compares the base policy against ADP. The full MC recommender costs about 26x
  more per draft and deserves its own backtest (H2b).
- League strategy associations (strategy.py) come from 36 manager-seasons with about 24 comparisons;
  they are descriptive and would not all survive a family-wise correction.
- Historical ADP is ESPN's season-level ADP, not what was shown on each draft day.
- Injuries and preseason depth-chart news reach the model only through ESPN's projections.
