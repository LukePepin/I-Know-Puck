"""Graph spectral analysis of league-manager draft behaviour.

Each manager-season is a vector of draft-behaviour features (reach vs ADP, positional timing,
goalie timing, pro-team loyalty, ...). Managers become nodes of a similarity graph with
Gaussian-kernel edge weights (bandwidth = median pairwise distance). We take the normalised
Laplacian  L = I - D^{-1/2} W D^{-1/2},  pick the number of clusters k by the largest eigengap,
and run k-means on the row-normalised bottom-k eigenvectors (Ng-Jordan-Weiss). The Fiedler
vector gives a 1-D ordering of managers (e.g. "ADP followers" <-> "contrarians").

The clusters are used as shrinkage targets for per-manager opponent models (opponents.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

from .opponents import group_of


def manager_features(drafts: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """One row per manager (pooled across seasons). drafts: season, overall, round, owner_id, player_id."""
    pl = players[["season", "player_id", "adp", "pos", "pro_team_id"]].drop_duplicates(["season", "player_id"])
    d = drafts.merge(pl, on=["season", "player_id"], how="left")
    d["adp"] = d["adp"].fillna(d["overall"].max() + 30)
    d["reach"] = np.log(d["adp"]) - np.log(d["overall"])  # >0: took the player earlier than ADP
    d["grp"] = d["pos"].map(group_of)
    rows = []
    for mgr, g in d.groupby("owner_id"):
        early = g[g["round"] <= 6]
        goalies = g[g.grp == 2]
        team_share = g.groupby(["season", "pro_team_id"]).size().groupby("season").max() / g.groupby("season").size()
        rows.append(
            {
                "manager": str(mgr),
                "n_picks": len(g),
                "mean_reach": g["reach"].mean(),
                "reach_early": early["reach"].mean(),
                "reach_sd": g["reach"].std(),
                "adp_rank_corr": g[["overall", "adp"]].corr(method="spearman").iloc[0, 1],
                "d_share_early": (early.grp == 1).mean(),
                "g_share_early": (early.grp == 2).mean(),
                "first_goalie_round": goalies["round"].min() if len(goalies) else g["round"].max() + 1,
                "n_goalies": len(goalies) / g["season"].nunique(),
                "homer_index": team_share.mean(),
                "auto_share": g["auto"].mean() if "auto" in g else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("manager")


@dataclass
class SpectralResult:
    features: pd.DataFrame
    affinity: np.ndarray
    eigenvalues: np.ndarray
    embedding: np.ndarray  # (n, 2) first two non-trivial eigenvectors, for plotting
    fiedler: np.ndarray
    k: int
    labels: np.ndarray

    def groups(self) -> dict[str, int]:
        return dict(zip(self.features.index, self.labels.tolist()))

    def table(self) -> pd.DataFrame:
        out = self.features.copy()
        out["cluster"] = self.labels
        out["fiedler"] = self.fiedler
        out["x"], out["y"] = self.embedding[:, 0], self.embedding[:, 1]
        return out


def spectral_clusters(features: pd.DataFrame, k: int | None = None, k_max: int = 4, seed: int = 0) -> SpectralResult:
    X = features.drop(columns=["n_picks"], errors="ignore").astype(float)
    X = X.fillna(X.mean())
    Z = ((X - X.mean()) / X.std(ddof=0).replace(0, 1)).to_numpy()
    D = squareform(pdist(Z))
    sigma = np.median(D[D > 0]) if (D > 0).any() else 1.0
    W = np.exp(-(D**2) / (2 * sigma**2))
    np.fill_diagonal(W, 0.0)
    deg = W.sum(axis=1)
    Dm = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    L = np.eye(len(W)) - Dm @ W @ Dm
    vals, vecs = np.linalg.eigh(L)
    n = len(W)
    if k is None:
        k_hi = max(2, min(k_max, n - 1))
        gaps = np.diff(vals[: k_hi + 1])
        k = int(np.argmax(gaps[1:]) + 2) if len(gaps) > 1 else 2
    U = vecs[:, :k]
    U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(U) if n > k else np.arange(n)
    emb = vecs[:, 1:3] if n > 2 else np.column_stack([vecs[:, 1], np.zeros(n)])
    return SpectralResult(features, W, vals, emb, vecs[:, 1], k, labels)
