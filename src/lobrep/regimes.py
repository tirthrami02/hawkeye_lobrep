"""Unsupervised regime discovery: cluster window embeddings, then score against the true regimes.

IDEA: if the learned representation captures market state, windows from the same regime should
land close together, so clustering WITHOUT labels should recover the regimes. We compare
embeddings from the SSL encoder, a random (untrained) encoder, and raw rolling statistics.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


def cluster_eval(Z: np.ndarray, truth: np.ndarray | None, k: int = 4, seed: int = 0) -> dict:
    """Cluster Z (N, d) and report quality.

    Metrics:
      * silhouette (-1..1): how well separated the clusters are; needs no labels.
      * ARI (adjusted Rand index; 0 = random, 1 = perfect match) and NMI: agreement with the
        simulator's TRUE regimes (only possible because the data is simulated).
    K-Means needs k (we use the true number, 4). HDBSCAN is density-based: it finds the number
    of clusters itself and labels outliers as noise (-1), so we also report cluster count/noise share.
    """
    Zs = StandardScaler().fit_transform(Z)
    out = {}
    km = KMeans(k, n_init=10, random_state=seed).fit(Zs)
    out["kmeans_sil"] = float(silhouette_score(Zs, km.labels_, sample_size=3000, random_state=seed))
    if truth is not None:
        out["kmeans_ari"] = float(adjusted_rand_score(truth, km.labels_))
        out["kmeans_nmi"] = float(normalized_mutual_info_score(truth, km.labels_))
    hd = HDBSCAN(min_cluster_size=max(30, len(Zs) // 100)).fit(Zs)
    lab = hd.labels_
    out["hdbscan_clusters"] = int(len(set(lab)) - (1 if -1 in lab else 0))
    out["hdbscan_noise"] = float((lab == -1).mean())
    if truth is not None and out["hdbscan_clusters"] > 1:
        out["hdbscan_ari"] = float(adjusted_rand_score(truth, lab))
    return out
