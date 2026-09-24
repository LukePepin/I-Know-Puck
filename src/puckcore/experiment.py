"""The experiment wheel.

    1. Hypothesis  - state H0/H1, the metric, and the unit of analysis up front
    2. Design      - baseline arm vs treatment arm scored on the same units (paired)
    3. Run         - a plugin-supplied ``run(seed)`` returns per-unit scores for both arms
    4. Analyze     - paired permutation / Wilcoxon / t tests, bootstrap CI, effect size
    5. Correct     - Holm-Bonferroni across every hypothesis in the suite
    6. Report      - results.json + report.md with git commit, seed and config for reproduction

Plugins define experiments; this module never needs to change when the domain does.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .stats import PairedTestResult, compare_paired, holm_bonferroni


@dataclass
class ExperimentOutput:
    """What a plugin's run function returns."""

    baseline: np.ndarray
    treatment: np.ndarray
    unit: str
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    id: str
    hypothesis: str  # H1 in plain language
    null: str  # H0 in plain language
    metric: str
    baseline_name: str
    treatment_name: str
    run: Callable[[int], ExperimentOutput]
    higher_is_better: bool = True
    alternative: str = "greater"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    experiment: Experiment
    test: PairedTestResult
    unit: str
    extras: dict[str, Any]
    seconds: float

    def to_dict(self) -> dict:
        e = self.experiment
        return {
            "id": e.id,
            "hypothesis": e.hypothesis,
            "null": e.null,
            "metric": e.metric,
            "baseline": e.baseline_name,
            "treatment": e.treatment_name,
            "unit": self.unit,
            "config": e.config,
            "test": self.test.to_dict(),
            "extras": _jsonable(self.extras),
            "seconds": round(self.seconds, 2),
        }


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def run_experiment(exp: Experiment, seed: int = 0) -> ExperimentResult:
    t0 = time.time()
    out = exp.run(seed)
    test = compare_paired(
        out.baseline,
        out.treatment,
        metric=exp.metric,
        higher_is_better=exp.higher_is_better,
        alternative=exp.alternative,
        seed=seed,
    )
    return ExperimentResult(exp, test, out.unit, out.extras, time.time() - t0)


def run_suite(
    experiments: list[Experiment],
    out_dir: str | Path = "runs",
    seed: int = 0,
    alpha: float = 0.05,
    suite_name: str = "suite",
) -> tuple[list[ExperimentResult], Path]:
    results = [run_experiment(e, seed) for e in experiments]
    holm_bonferroni([r.test for r in results], alpha=alpha)

    run_dir = Path(out_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}_{suite_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"suite": suite_name, "seed": seed, "alpha": alpha, "git_commit": _git_commit(), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    payload = {"meta": meta, "results": [r.to_dict() for r in results]}
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(payload), encoding="utf-8")
    return results, run_dir


def render_report(payload: dict) -> str:
    m = payload["meta"]
    lines = [
        f"# Experiment report: {m['suite']}",
        "",
        f"- created: {m['created']}  |  git: `{m['git_commit']}`  |  seed: {m['seed']}  |  alpha (FWER, Holm): {m['alpha']}",
        "",
        "| id | treatment vs baseline | metric | n | diff (95% CI) | d_z | p perm | p Holm | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in payload["results"]:
        t = r["test"]
        verdict = "reject H0" if t["significant"] else "fail to reject"
        lines.append(
            f"| {r['id']} | {r['treatment']} vs {r['baseline']} | {r['metric']} | {t['n']} | "
            f"{t['mean_diff']:+.4f} ({t['ci_low']:+.4f}, {t['ci_high']:+.4f}) | {t['cohens_dz']:.2f} | "
            f"{t['p_permutation']:.4f} | {t['p_adjusted']:.4f} | {verdict} |"
        )
    lines.append("")
    for r in payload["results"]:
        lines += [f"## {r['id']}", "", f"- **H1:** {r['hypothesis']}", f"- **H0:** {r['null']}", f"- **unit:** {r['unit']}", f"- **config:** `{json.dumps(r['config'])}`", ""]
    return "\n".join(lines)
