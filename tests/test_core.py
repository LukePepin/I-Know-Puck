import numpy as np
import pandas as pd

from puckcore.experiment import Experiment, ExperimentOutput, run_suite
from puckcore.stats import compare_paired, holm_bonferroni, paired_permutation_test


def test_permutation_detects_shift_and_respects_null():
    rng = np.random.default_rng(0)
    assert paired_permutation_test(rng.normal(0.5, 1, 200)) < 0.01
    assert paired_permutation_test(rng.normal(0.0, 1, 200)) > 0.01


def test_compare_paired_orientation():
    base = np.full(50, 10.0) + np.random.default_rng(1).normal(0, 0.1, 50)
    better_low = base - 1  # an error metric: lower is better
    r = compare_paired(base, better_low, "err", higher_is_better=False)
    assert r.mean_diff > 0.9 and r.ci_low > 0 and r.p_permutation < 0.01


def test_holm_is_monotone_and_step_down():
    rs = [compare_paired(np.zeros(30), np.random.default_rng(i).normal(m, 1, 30), f"m{i}") for i, m in enumerate([1.0, 0.0, 0.8])]
    holm_bonferroni(rs)
    by_p = sorted(rs, key=lambda r: r.p_permutation)
    assert all(a.p_adjusted <= b.p_adjusted for a, b in zip(by_p, by_p[1:]))
    assert by_p[0].significant and not by_p[-1].significant


def test_run_suite_writes_report(tmp_path):
    exp = Experiment("T", "t better", "no diff", "score", "a", "b",
                     lambda s: ExperimentOutput(np.zeros(20), np.ones(20), unit="u"))
    results, run_dir = run_suite([exp], out_dir=tmp_path, suite_name="t")
    assert (run_dir / "report.md").exists() and results[0].test.significant
