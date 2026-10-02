"""Graph spectral analysis of league-manager draft behaviour.

Each manager-season is a vector of draft-behaviour features (reach vs ADP, positional timing,
goalie timing, pro-team loyalty, ...). Managers become nodes of a similarity graph with
Gaussian-kernel edge weights (bandwidth = median pairwise distance). We take the normalised
Laplacian  L = I - D^{-1/2} W D^{-1/2},  pick the number of clusters k by the largest eigengap,
and run k-means on the row-normalised bottom-k eigenvectors (Ng-Jordan-Weiss). The Fiedler
vector gives a 1-D ordering of managers (e.g. "ADP followers" <-> "contrarians").

The clusters are used as shrinkage targets for per-manager opponent models (opponents.py).

Two checks ask whether the structure is real rather than assumed:
  eigengap_null          shuffle each habit independently across managers (keeps every habit's spread,
                         destroys any joint "style") and recompute the eigengap -> permutation p-value
  bootstrap_coclustering resample each manager's draft seasons, re-cluster, and count how often every pair
                         of managers lands in the same group (plus adjusted Rand index vs the full-data labels)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from .opponents import group_of


def manager_season_features(drafts: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """One row per (manager, season) draft. drafts: season, overall, round, owner_id, player_id."""
    pl = players[["season", "player_id", "adp", "pos", "pro_team_id"]].drop_duplicates(["season", "player_id"])
    d = drafts.drop(columns=["pos", "adp"], errors="ignore").merge(pl, on=["season", "player_id"], how="left")
    d["adp"] = d["adp"].fillna(d["overall"].max() + 30)
    d["reach"] = np.log(d["adp"]) - np.log(d["overall"])  # >0: took the player earlier than ADP
    d["grp"] = d["pos"].map(group_of)
    rows = []
    for (mgr, season), gs in d.groupby(["owner_id", "season"]):
        early = gs[gs["round"] <= 6]
        goalies = gs[gs.grp == 2]
        rows.append({
            "manager": str(mgr),
            "season": season,
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
            "n_picks": len(gs),
        })
    return pd.DataFrame(rows)


def manager_features(drafts: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """One row per manager (pooled across seasons). drafts: season, overall, round, owner_id, player_id."""
    per_season = manager_season_features(drafts, players)
    feats = per_season.drop(columns=["season", "n_picks"]).groupby("manager").mean()  # average over seasons, not pooled extremes
    feats["n_picks"] = per_season.groupby("manager").n_picks.sum()
    return feats


@dataclass
class SpectralResult:
    features: pd.DataFrame
    affinity: np.ndarray
    eigenvalues: np.ndarray
    embedding: np.ndarray  # (n, 2) first two non-trivial eigenvectors, for plotting
    fiedler: np.ndarray
    k: int
    labels: np.ndarray
    sigma: float = float("nan")  # Gaussian-kernel bandwidth actually used

    def groups(self) -> dict[str, int]:
        return dict(zip(self.features.index, self.labels.tolist()))

    def table(self) -> pd.DataFrame:
        out = self.features.copy()
        out["cluster"] = self.labels
        out["fiedler"] = self.fiedler
        out["x"], out["y"] = self.embedding[:, 0], self.embedding[:, 1]
        return out


def _standardise(features: pd.DataFrame) -> np.ndarray:
    X = features.drop(columns=["n_picks"], errors="ignore").astype(float)
    X = X.fillna(X.mean())
    return ((X - X.mean()) / X.std(ddof=0).replace(0, 1)).to_numpy()


def laplacian_spectrum(Z: np.ndarray, bandwidth_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(affinity W, eigenvalues, eigenvectors, sigma) of the normalised Laplacian for standardised rows Z."""
    D = squareform(pdist(Z))
    sigma = (np.median(D[D > 0]) if (D > 0).any() else 1.0) * bandwidth_scale
    W = np.exp(-(D**2) / (2 * sigma**2))
    np.fill_diagonal(W, 0.0)
    deg = W.sum(axis=1)
    Dm = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    L = np.eye(len(W)) - Dm @ W @ Dm
    vals, vecs = np.linalg.eigh(L)
    return W, vals, vecs, float(sigma)


def eigengap_k(vals: np.ndarray, k_max: int = 4) -> tuple[int, float]:
    """k at the largest gap lambda_{k+1} - lambda_k for k >= 2, and the size of that gap."""
    n = len(vals)
    k_hi = max(2, min(k_max, n - 1))
    gaps = np.diff(vals[: k_hi + 1])
    if len(gaps) <= 1:
        return 2, float(gaps[-1]) if len(gaps) else 0.0
    i = int(np.argmax(gaps[1:]))
    return i + 2, float(gaps[1:][i])


def spectral_clusters(features: pd.DataFrame, k: int | None = None, k_max: int = 4, seed: int = 0,
                      bandwidth_scale: float = 1.0, n_init: int = 20) -> SpectralResult:
    W, vals, vecs, sigma = laplacian_spectrum(_standardise(features), bandwidth_scale)
    n = len(W)
    if k is None:
        k, _ = eigengap_k(vals, k_max)
    U = vecs[:, :k]
    U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    labels = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(U) if n > k else np.arange(n)
    emb = vecs[:, 1:3] if n > 2 else np.column_stack([vecs[:, 1], np.zeros(n)])
    return SpectralResult(features, W, vals, emb, vecs[:, 1], k, labels, sigma)


def eigengap_null(features: pd.DataFrame, n_perm: int = 500, k_max: int = 4, bandwidth_scale: float = 1.0,
                  seed: int = 0) -> tuple[float, np.ndarray, float]:
    """Permutation test of cluster structure: (observed eigengap, null eigengaps, p-value).

    Each habit column is shuffled independently across managers, which keeps every habit's
    distribution but removes any tendency of habits to go together (the thing a "style" is)."""
    Z = _standardise(features)
    _, vals, _, _ = laplacian_spectrum(Z, bandwidth_scale)
    observed = eigengap_k(vals, k_max)[1]
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        Zp = np.column_stack([rng.permutation(col) for col in Z.T])
        null[i] = eigengap_k(laplacian_spectrum(Zp, bandwidth_scale)[1], k_max)[1]
    return observed, null, float((np.sum(null >= observed) + 1) / (n_perm + 1))


def bootstrap_coclustering(season_features: pd.DataFrame, columns: list[str], k: int, n_boot: int = 200,
                           bandwidth_scale: float = 1.0, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Resample each manager's draft seasons with replacement, re-cluster with the same k.

    Returns (co-assignment share for every pair of managers, adjusted Rand index of each
    bootstrap clustering vs the clustering of the full data)."""
    sf = season_features.copy()
    ref_feats = sf.groupby("manager")[columns].mean()
    ref = spectral_clusters(ref_feats, k=k, bandwidth_scale=bandwidth_scale, seed=seed).labels
    managers = ref_feats.index.tolist()
    by_mgr = {m: g[columns].to_numpy(float) for m, g in sf.groupby("manager")}
    rng = np.random.default_rng(seed)
    co = np.zeros((len(managers), len(managers)))
    ari = np.empty(n_boot)
    for b in range(n_boot):
        rows = [np.nanmean(by_mgr[m][rng.integers(0, len(by_mgr[m]), len(by_mgr[m]))], axis=0) for m in managers]
        feats = pd.DataFrame(rows, index=managers, columns=columns)
        lab = spectral_clusters(feats, k=k, bandwidth_scale=bandwidth_scale, seed=seed, n_init=5).labels
        co += lab[:, None] == lab[None, :]
        ari[b] = adjusted_rand_score(ref, lab)
    return pd.DataFrame(co / n_boot, index=managers, columns=managers), ari


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
