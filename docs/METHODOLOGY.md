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

## 4. Valuation (objective function)

Starters are assigned to slots by the Hungarian algorithm (`linear_sum_assignment`). Usage is 1 for
starters, 0.35 for bench skaters and 0.2 for bench goalies (daily lineups let the bench fill idle days).
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
dominant axis of behavioral difference.

## 7. Draft recommender

At pick t, candidates are the top-K players by VONA (value over the k-th best same-group player left at
our next pick) plus the top ADP players. For each candidate, M rollouts finish the draft: opponents
sample from their logit model (Gumbel-max), and our later picks use the VONA greedy policy. The final
league is then scored. All candidates share the same random seeds (**common random numbers**), so
candidate differences are paired. The app reports the mean, the gap to the best candidate, the paired
SE of that gap, P(best), and each player's availability at our next pick.

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
| H2 | VONA policy > ADP drafting | (season, slot, seed), rosters scored on **actual** stats | P(win weekly matchup) |
| H3 | per-manager + spectral logit > league-wide logit | held-out pick (leave-one-season-out) | log-likelihood |
| H4 | spectral pooling > independent shrinkage | held-out pick | log-likelihood |

## 9. Known limitations / next experiments

- The weekly variance constants (dispersion, covariance inflation, bench usage) are priors. They should
  be calibrated against the league's own weekly matchup scores (ESPN `mMatchup`).
- The H2 backtest compares the base policy against ADP. The full MC recommender costs about 26×
  more per draft and deserves its own backtest (H2b).
- Historical ADP is ESPN's season-level ADP, not what was shown on each draft day.
- Injuries and preseason depth-chart news reach the model only through ESPN's projections.
