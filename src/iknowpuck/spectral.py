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
    d = drafts.drop(columns=["pos", "adp"], errors="ignore").merge(pl, on=["season", "player_id"], how="left")
    d["adp"] = d["adp"].fillna(d["overall"].max() + 30)
    d["reach"] = np.log(d["adp"]) - np.log(d["overall"])  # >0: took the player earlier than ADP
    d["grp"] = d["pos"].map(group_of)
    rows = []
    for mgr, g in d.groupby("owner_id"):
        per_season = []
        for _, gs in g.groupby("season"):
            early = gs[gs["round"] <= 6]
            goalies = gs[gs.grp == 2]
            per_season.append({
                "mean_reach": gs["reach"].mean(),
                "reach_early": early["reach"].mean(),
                "reach_sd": gs["reach"].std(),
                "adp_rank_corr": gs[["overall", "adp"]].corr(method="spearman").iloc[0, 1],
                "d_share_early": (early.grp == 1).mean(),
                "g_share_early": (early.grp == 2).mean(),
                "first_goalie_round": goalies["round"].min() if len(goalies) else gs["round"].max() + 1,
                "n_goalies": len(goalies),
                "homer_index": gs.groupby("pro_team_id").size().max() / len(gs),
                "auto_share": gs["auto"].mean() if "auto" in gs else 0.0,
            })
        row = pd.DataFrame(per_season).mean().to_dict()  # average over seasons, not pooled extremes
        row.update({"manager": str(mgr), "n_picks": len(g)})
        rows.append(row)
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


# Plain-language description of each behaviour feature: (high phrase, low phrase)
FEATURE_PHRASES = {
    "first_goalie_round": ("waits on goalies", "grabs goalies early"),
    "g_share_early": ("goalies early", "skaters early"),
    "reach_early": ("reaches ahead of the rankings", "sticks to the rankings"),
    "mean_reach": ("reaches ahead of the rankings", "sticks to the rankings"),
    "d_share_early": ("defensemen early", "forwards early"),
    "homer_index": ("favours one NHL team", "spreads across NHL teams"),
    "auto_share": ("often autodrafts", "drafts by hand"),
    "n_goalies": ("stockpiles goalies", "carries few goalies"),
}


def name_clusters(res: SpectralResult) -> dict[int, dict]:
    """Name each cluster from what distinguishes it (centroid z-scores vs the league).

    Returns {cluster: {"name", "traits", "members"}}; names are chosen by rules on the two most
    distinctive traits so they stay honest to the data.
    """
    X = res.features.drop(columns=["n_picks"], errors="ignore").astype(float)
    X = X.fillna(X.mean())
    z = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    out = {}
    for c in sorted(set(res.labels)):
        cz = z[res.labels == c].mean()
        traits: list[str] = []
        for f in cz.abs().sort_values(ascending=False).index:
            if f in FEATURE_PHRASES:
                t = FEATURE_PHRASES[f][0] if cz[f] > 0 else FEATURE_PHRASES[f][1]
                if t not in traits:
                    traits.append(t)
            if len(traits) == 3:
                break
        members = z[res.labels == c]

        def shared(feature: str, sign: int, share: float = 0.7) -> bool:
            """True if the group leans this way on average AND most members lean the same way."""
            if feature not in members:
                return False
            return sign * cz[feature] > 0.3 and float(((sign * members[feature]) > 0).mean()) >= share

        if shared("first_goalie_round", -1) and shared("reach_early", 1):
            name = "Aggressive Reachers"
        elif shared("first_goalie_round", -1):
            name = "Goalie Grabbers"
        elif shared("first_goalie_round", 1) and shared("d_share_early", 1):
            name = "Blue-line Builders"
        elif shared("first_goalie_round", 1):
            name = "Patient Builders"
        elif shared("reach_early", 1):
            name = "Reachers"
        elif shared("reach_early", -1, 0.6):
            name = "By-the-Book Drafters"
        elif shared("homer_index", 1):
            name = "Team Loyalists"
        else:
            name = "Balanced Drafters"
        out[int(c)] = {"name": name, "traits": traits, "members": list(res.features.index[res.labels == c])}
    # make duplicate names unique
    seen: dict[str, int] = {}
    for c, v in out.items():
        seen[v["name"]] = seen.get(v["name"], 0) + 1
        if seen[v["name"]] > 1:
            v["name"] = f"{v['name']} ({v['traits'][0]})"
    return out
