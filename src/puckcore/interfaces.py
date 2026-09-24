"""Protocols a domain plugin implements to plug into the core.

They are structural (duck-typed) so plugins need not import or subclass anything.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd


class DataSource(Protocol):
    """Fetches and caches raw data; returns tidy frames."""

    name: str

    def load(self, **kwargs: Any) -> pd.DataFrame: ...


class Forecaster(Protocol):
    """Predicts per-unit outcomes (e.g. player season stats)."""

    def fit(self, history: pd.DataFrame) -> "Forecaster": ...

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame: ...


class ChoiceModel(Protocol):
    """Probability that an agent chooses each available option (e.g. an opponent's draft pick)."""

    def probabilities(self, agent: Any, available: np.ndarray, state: Any) -> np.ndarray: ...


class Policy(Protocol):
    """Our decision rule inside a sequential decision process (e.g. which player to draft)."""

    def choose(self, available: np.ndarray, state: Any, rng: np.random.Generator) -> int: ...


class Evaluator(Protocol):
    """Scores a terminal state (e.g. a finished roster) against competitors."""

    def score(self, mine: Sequence[int], others: Sequence[Sequence[int]]) -> float: ...
