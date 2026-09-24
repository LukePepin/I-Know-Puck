"""Statistical tests used to decide whether a treatment beats a baseline.

All comparisons are *paired*: the same experimental unit (a player, a simulated
season, a held-out draft pick) is scored under both arms, which removes
between-unit variance and is what makes small fantasy samples testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats as sps


@dataclass
class PairedTestResult:
    metric: str
    n: int
    baseline_mean: float
    treatment_mean: float
    mean_diff: float  # treatment - baseline, oriented so positive = treatment better
    ci_low: float
    ci_high: float
    cohens_dz: float
    p_permutation: float
    p_wilcoxon: float
    p_ttest: float
    alternative: str
    p_adjusted: float | None = None
    significant: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of ``stat`` over units."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = stat(x[idx], axis=1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_permutation_test(d: np.ndarray, n_perm: int = 20_000, alternative: str = "greater", seed: int = 0) -> float:
    """Sign-flip permutation test on paired differences (exact under H0: symmetric about 0)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=float)
    observed = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * d).mean(axis=1)
    if alternative == "greater":
        return float((np.sum(null >= observed) + 1) / (n_perm + 1))
    if alternative == "less":
        return float((np.sum(null <= observed) + 1) / (n_perm + 1))
    return float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1))


def compare_paired(
    baseline: np.ndarray,
    treatment: np.ndarray,
    metric: str,
    higher_is_better: bool = True,
    alternative: str = "greater",
    seed: int = 0,
) -> PairedTestResult:
    """Full paired comparison. Differences are oriented so positive means treatment is better."""
    b = np.asarray(baseline, dtype=float)
    t = np.asarray(treatment, dtype=float)
    mask = np.isfinite(b) & np.isfinite(t)
    b, t = b[mask], t[mask]
    d = (t - b) if higher_is_better else (b - t)
    sd = d.std(ddof=1) if len(d) > 1 else np.nan
    lo, hi = bootstrap_ci(d, seed=seed)
    try:
        p_w = float(sps.wilcoxon(d, alternative=alternative, zero_method="zsplit").pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p_w = float("nan")
    p_t = float(sps.ttest_1samp(d, 0.0, alternative=alternative).pvalue) if len(d) > 1 else float("nan")
    return PairedTestResult(
        metric=metric,
        n=int(len(d)),
        baseline_mean=float(b.mean()),
        treatment_mean=float(t.mean()),
        mean_diff=float(d.mean()),
        ci_low=lo,
        ci_high=hi,
        cohens_dz=float(d.mean() / sd) if sd and sd > 0 else float("nan"),
        p_permutation=paired_permutation_test(d, alternative=alternative, seed=seed),
        p_wilcoxon=p_w,
        p_ttest=p_t,
        alternative=alternative,
    )


def holm_bonferroni(results: list[PairedTestResult], alpha: float = 0.05, p_field: str = "p_permutation") -> list[PairedTestResult]:
    """Holm step-down correction across a family of hypotheses (controls FWER)."""
    order = sorted(range(len(results)), key=lambda i: getattr(results[i], p_field))
    m = len(results)
    running_max = 0.0
    still_rejecting = True
    for rank, i in enumerate(order):
        p = getattr(results[i], p_field)
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)
        results[i].p_adjusted = running_max
        still_rejecting = still_rejecting and running_max <= alpha
        results[i].significant = still_rejecting
    return results


def required_n_paired(effect_dz: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Approximate units needed for a one-sided paired test to detect effect size ``dz``."""
    if effect_dz <= 0:
        return -1
    z_a = sps.norm.ppf(1 - alpha)
    z_b = sps.norm.ppf(power)
    return int(np.ceil(((z_a + z_b) / effect_dz) ** 2))
