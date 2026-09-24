"""Player season projections.

Own model ("Marcel+"):
  1. per-game rate for each stat = recency-weighted history (weights 5/4/3 by GP),
     regressed toward the position mean with a per-stat shrinkage constant R fit on past seasons
  2. optional MoneyPuck residual correction (ridge on xG-luck, PP time, TOI) -> ablation-testable
  3. games-played model: linear on prior two seasons' share of schedule

Blend: final_k = w_k * own_k + (1 - w_k) * espn_k, with w_k fit by least squares on past seasons.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import GOALIE_STATS, RATIO_STATS, LeagueSettings

SCHEDULE_GAMES = {2020: 70, 2021: 56}  # shortened seasons; everything else 82
HIST_WEIGHTS = (5.0, 4.0, 3.0)
R_GRID = (0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
MP_FEATURES = ["f_xg_luck", "f_pp_toi", "f_toi", "f_isa"]


def season_games(season: int) -> int:
    return SCHEDULE_GAMES.get(season, 82)


def modeled_stats(settings: LeagueSettings) -> list[int]:
    """Counting stats needed to compute every league category (ratio stats via components)."""
    out: set[int] = set()
    for c in settings.scoring_categories:
        if c.stat_id in RATIO_STATS:
            num, den, _ = RATIO_STATS[c.stat_id]
            out |= {num, den}
        else:
            out.add(c.stat_id)
    return sorted(out)


def _group(pos: pd.Series) -> pd.Series:
    return pos.map(lambda p: "G" if p == "G" else ("D" if p == "D" else "F"))


def _mp_features(df: pd.DataFrame) -> pd.DataFrame:
    gp = df.get("mp_gp", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    f = pd.DataFrame(index=df.index)
    f["f_xg_luck"] = (df.get("mp_ixg", np.nan) - df.get("mp_goals", np.nan)) / gp
    f["f_pp_toi"] = df.get("mp_pp_toi", np.nan) / gp / 60.0
    f["f_toi"] = df.get("mp_toi", np.nan) / gp / 60.0
    f["f_isa"] = df.get("mp_isa", np.nan) / gp
    return f


@dataclass
class OwnModel:
    stats: list[int]
    use_moneypuck: bool = True
    r_: dict[int, float] = field(default_factory=dict)
    ridge_: dict[int, Ridge] = field(default_factory=dict)
    mp_mu_: pd.Series | None = None
    mp_sd_: pd.Series | None = None
    gp_coef_: dict[str, np.ndarray] = field(default_factory=dict)

    # --- feature construction -----------------------------------------------------------------
    def history_features(self, panel: pd.DataFrame, target: int) -> pd.DataFrame:
        """Weighted history sums for every player who appeared in any of the 3 prior seasons."""
        hist = panel[(panel.season < target) & (panel.season >= target - len(HIST_WEIGHTS))].copy()
        hist = hist[hist.get("act_30", pd.Series(0, index=hist.index)).fillna(0) > 0]
        if hist.empty:
            return pd.DataFrame()
        hist["lag"] = target - hist.season
        hist["w"] = hist["lag"].map(lambda l: HIST_WEIGHTS[l - 1])
        hist["gp"] = hist["act_30"].fillna(0)
        agg = {"wgp": ("wgp", "sum")}
        hist["wgp"] = hist.w * hist.gp
        for k in self.stats:
            col = f"act_{k}"
            hist[f"w{k}"] = hist.w * hist.get(col, 0).fillna(0) if col in hist else 0.0
            agg[f"w{k}"] = (f"w{k}", "sum")
        out = hist.groupby("player_id").agg(**agg)
        last = hist.sort_values("season").groupby("player_id").tail(1).set_index("player_id")
        out["pos"] = last["pos"]
        out["name"] = last["name"]
        for lag in (1, 2):
            s = hist[hist.lag == lag].set_index("player_id")
            out[f"gpfrac_{lag}"] = (s["gp"] / season_games(target - lag)).reindex(out.index)
        mp_last = hist[hist.lag == 1].set_index("player_id")
        feats = _mp_features(mp_last).reindex(out.index)
        return out.join(feats)

    def _pos_means(self, feats: pd.DataFrame) -> pd.DataFrame:
        g = _group(feats["pos"])
        rates = pd.DataFrame({k: feats[f"w{k}"] / feats["wgp"].replace(0, np.nan) for k in self.stats}, index=feats.index)
        w = feats["wgp"]
        means = {}
        for grp in ("F", "D", "G"):
            m = g == grp
            if m.any():
                means[grp] = (rates[m].mul(w[m], axis=0).sum() / w[m].sum()).fillna(0)
        return pd.DataFrame(means).T

    def _rates(self, feats: pd.DataFrame, r: dict[int, float], only: list[int] | None = None, pm: pd.DataFrame | None = None) -> pd.DataFrame:
        pm = self._pos_means(feats) if pm is None else pm
        g = _group(feats["pos"])
        out = {}
        for k in only or self.stats:
            prior = g.map(lambda x: pm.loc[x, k] if x in pm.index else 0.0)
            rk = r.get(k, 20.0)
            out[k] = (feats[f"w{k}"] + rk * prior) / (feats["wgp"] + rk)
            # goalie stats are meaningless for skaters and vice versa
            is_g = g == "G"
            out[k] = out[k].where(is_g == (k in GOALIE_STATS), 0.0)
        return pd.DataFrame(out, index=feats.index)

    def _gp_design(self, feats: pd.DataFrame) -> np.ndarray:
        g1 = feats["gpfrac_1"].fillna(0).clip(0, 1)
        g2 = feats["gpfrac_2"].fillna(0).clip(0, 1)
        return np.column_stack([np.ones(len(feats)), g1, g2, feats["gpfrac_1"].isna(), feats["gpfrac_2"].isna()]).astype(float)

    # --- fit / predict ------------------------------------------------------------------------
    def fit(self, panel: pd.DataFrame, targets: list[int]) -> "OwnModel":
        blocks = []
        for t in targets:
            f = self.history_features(panel, t)
            if f.empty:
                continue
            act = panel[panel.season == t].set_index("player_id")
            f = f.join(act[[c for c in act.columns if c.startswith("act_")]], how="inner")
            f = f[f["act_30"].fillna(0) > 0]
            f["target"] = t
            blocks.append(f)
        train = pd.concat(blocks)
        gp_t = train["act_30"]
        # 1) shrinkage constant per stat by grid search on GP-weighted squared rate error
        pm = self._pos_means(train)
        for k in self.stats:
            y = train.get(f"act_{k}", pd.Series(0, index=train.index)).fillna(0) / gp_t
            best = min(R_GRID, key=lambda rk: float(np.average((self._rates(train, {k: rk}, [k], pm)[k] - y) ** 2, weights=gp_t)))
            self.r_[k] = best
        # 2) MoneyPuck residual ridge
        if self.use_moneypuck:
            X = train[MP_FEATURES]
            self.mp_mu_, self.mp_sd_ = X.mean(), X.std().replace(0, 1)
            Xs = ((X - self.mp_mu_) / self.mp_sd_).fillna(0).values
            base = self._rates(train, self.r_, pm=pm)
            for k in self.stats:
                if k in GOALIE_STATS:
                    continue
                y = train.get(f"act_{k}", pd.Series(0, index=train.index)).fillna(0) / gp_t - base[k]
                m = (_group(train["pos"]) != "G").values
                self.ridge_[k] = Ridge(alpha=50.0).fit(Xs[m], y.values[m], sample_weight=gp_t.values[m])
        # 3) games-played share model, separate for skaters and goalies
        for grp, mask in (("S", _group(train["pos"]) != "G"), ("G", _group(train["pos"]) == "G")):
            X = self._gp_design(train[mask])
            y = (train.loc[mask, "act_30"] / train.loc[mask, "target"].map(season_games)).clip(0, 1).values
            self.gp_coef_[grp] = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, panel: pd.DataFrame, target: int) -> pd.DataFrame:
        f = self.history_features(panel, target)
        if f.empty:
            return pd.DataFrame()
        rates = self._rates(f, self.r_)
        if self.use_moneypuck and self.ridge_:
            Xs = ((f[MP_FEATURES] - self.mp_mu_) / self.mp_sd_).fillna(0).values
            skater = (_group(f["pos"]) != "G").values
            for k, model in self.ridge_.items():
                rates.loc[skater, k] = rates.loc[skater, k] + model.predict(Xs[skater])
        for k in rates.columns:
            if k != 15:  # only +/- can be negative
                rates[k] = rates[k].clip(lower=0)
        gp = pd.Series(0.0, index=f.index)
        for grp, mask in (("S", _group(f["pos"]) != "G"), ("G", _group(f["pos"]) == "G")):
            if mask.any() and grp in self.gp_coef_:
                gp[mask] = np.clip(self._gp_design(f[mask]) @ self.gp_coef_[grp], 0, 1) * season_games(target)
        out = rates.mul(gp, axis=0)
        out.columns = [f"own_{k}" for k in out.columns]
        out["own_30"] = gp
        out["pos"] = f["pos"]
        return out


@dataclass
class Blender:
    """Per-stat convex weight between own model and ESPN projection."""

    stats: list[int]
    w_: dict[int, float] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> "Blender":
        for k in [*self.stats, 30]:
            o, e, a = frame.get(f"own_{k}"), frame.get(f"proj_{k}"), frame.get(f"act_{k}")
            if o is None or e is None or a is None:
                self.w_[k] = 0.0
                continue
            m = o.notna() & e.notna() & a.notna()
            d = (o - e)[m]
            denom = float((d**2).sum())
            self.w_[k] = float(np.clip(((a - e)[m] * d).sum() / denom, 0, 1)) if denom > 0 else 0.0
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for k in [*self.stats, 30]:
            o = out.get(f"own_{k}", pd.Series(np.nan, index=out.index))
            e = out.get(f"proj_{k}", pd.Series(np.nan, index=out.index))
            w = self.w_.get(k, 0.0)
            out[f"blend_{k}"] = np.where(o.notna() & e.notna(), w * o + (1 - w) * e, np.where(e.notna(), e, o))
        return out


def season_frame(panel: pd.DataFrame, model: OwnModel, target: int) -> pd.DataFrame:
    """Players of ``target`` season with ESPN proj/actual columns plus own-model projections."""
    base = panel[panel.season == target].set_index("player_id")
    own = model.predict(panel, target).drop(columns=["pos"], errors="ignore")
    return base.join(own, how="left")


def fill_projection(frame: pd.DataFrame, stats: list[int], source: str) -> pd.DataFrame:
    """Return a frame with ``p_<k>`` columns from the chosen source ('proj', 'own', 'blend', 'act')."""
    out = frame.copy()
    for k in [*stats, 30]:
        col = f"{source}_{k}"
        out[f"p_{k}"] = out[col].fillna(0.0) if col in out else 0.0
    return out
