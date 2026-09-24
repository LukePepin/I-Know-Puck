"""puckcore: a domain-agnostic experiment framework.

Nothing in this package knows about hockey or ESPN. It provides:

- ``cache``      : content-addressed disk cache for any data source
- ``stats``      : paired hypothesis tests, bootstrap CIs, effect sizes, multiplicity control
- ``experiment`` : the experiment "wheel" (hypothesis -> design -> run -> analyze -> report)
- ``interfaces`` : Protocols that domain plugins implement (DataSource, Model, Policy, Evaluator)

A domain plugin (e.g. ``iknowpuck``) implements the interfaces and registers experiments.
"""

from .experiment import Experiment, ExperimentResult, run_suite
from .stats import PairedTestResult, compare_paired, holm_bonferroni

__all__ = [
    "Experiment",
    "ExperimentResult",
    "run_suite",
    "PairedTestResult",
    "compare_paired",
    "holm_bonferroni",
]
