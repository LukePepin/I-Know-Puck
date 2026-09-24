"""Assemble a multi-season player panel: ESPN actuals + ESPN projections + MoneyPuck advanced stats."""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from ..config import load_credentials
from .espn import EspnClient
from .moneypuck import MoneyPuck


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower().replace("-", " "))
    return re.sub(r"\s+", " ", s).strip()


def _is_d(pos: str) -> bool:
    return pos == "D"


def attach_moneypuck(espn: pd.DataFrame, mp: pd.DataFrame) -> pd.DataFrame:
    """Join by normalised name, disambiguating same-name players by skater/goalie/defense group."""
    mp = mp.copy()
    mp["key"] = mp["mp_name"].map(norm_name)
    mp["grp"] = np.where(mp["position"] == "G", "G", np.where(mp["position"] == "D", "D", "F"))
    mp = mp.drop_duplicates(["key", "grp"])
    e = espn.copy()
    e["key"] = e["name"].map(norm_name)
    e["grp"] = np.where(e["pos"] == "G", "G", np.where(e["pos"] == "D", "D", "F"))
    out = e.merge(mp.drop(columns=["position"]), on=["key", "grp"], how="left")
    # fall back to name-only match for position-group mismatches (e.g. F listed as D)
    miss = out["nhl_id"].isna()
    if miss.any():
        uniq = mp.drop_duplicates("key", keep=False).drop(columns=["position", "grp"])
        fb = e.loc[miss.values, ["key"]].merge(uniq, on="key", how="left")
        for c in uniq.columns:
            if c != "key":
                out.loc[miss.values, c] = fb[c].values
    return out.drop(columns=["key", "grp"])


def build_panel(seasons: list[int], client: EspnClient | None = None, with_moneypuck: bool = True, limit: int = 1200) -> pd.DataFrame:
    """One row per (ESPN season, player). ESPN season = year the NHL season ends."""
    client = client or EspnClient(load_credentials())
    mp_src = MoneyPuck(client.cache) if with_moneypuck else None
    frames = []
    for s in seasons:
        df = client.players(s, limit=limit)
        if mp_src is not None:
            try:
                df = attach_moneypuck(df, mp_src.load(s - 1))
            except Exception:  # current season has no MoneyPuck data yet
                pass
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
