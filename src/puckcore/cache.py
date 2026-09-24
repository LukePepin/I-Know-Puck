"""Disk cache keyed by a namespace + arbitrary JSON-serialisable key."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd


class DiskCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, key: Any, ext: str) -> Path:
        digest = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]
        d = self.root / namespace
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{digest}.{ext}"

    def _fresh(self, path: Path, max_age_s: float | None) -> bool:
        if not path.exists():
            return False
        return max_age_s is None or (time.time() - path.stat().st_mtime) < max_age_s

    def json(self, namespace: str, key: Any, fetch: Callable[[], Any], max_age_s: float | None = None) -> Any:
        path = self._path(namespace, key, "json")
        if self._fresh(path, max_age_s):
            return json.loads(path.read_text(encoding="utf-8"))
        value = fetch()
        path.write_text(json.dumps(value), encoding="utf-8")
        return value

    def frame(self, namespace: str, key: Any, fetch: Callable[[], pd.DataFrame], max_age_s: float | None = None) -> pd.DataFrame:
        path = self._path(namespace, key, "parquet")
        if self._fresh(path, max_age_s):
            return pd.read_parquet(path)
        df = fetch()
        df.to_parquet(path, index=False)
        return df
