"""Five 95% intervals for balanced accuracy.

Balanced accuracy is a mean over classes of per-class accuracy. With the class counts n_k held fixed it is a sum of
per-cluster contributions c_g = (1/K) * sum over the cluster's samples of correct_i / n_{k(i)}. That makes a
cluster-robust (sandwich) variance exact up to the G/(G-1) factor.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .sim import EvalSet, balanced_accuracy

Z = norm.ppf(0.975)


def _contrib(s: EvalSet):
    classes, inv = np.unique(s.label, return_inverse=True)
    n_k = np.bincount(inv)
    K = len(classes)
    per_sample = s.correct / n_k[inv] / K
    clusters, cinv = np.unique(s.cluster, return_inverse=True)
    return np.bincount(cinv, weights=per_sample), inv, n_k, K, cinv


def wald_iid(s: EvalSet):
    """Per-class binomial variance, samples treated as independent."""
    _, inv, n_k, K, _ = _contrib(s)
    p = np.bincount(inv, weights=s.correct) / n_k
    se = np.sqrt((p * (1 - p) / n_k).sum()) / K
    ba = p.mean()
    return ba - Z * se, ba + Z * se


def wald_cluster(s: EvalSet):
    """Cluster-robust variance of the sum of per-cluster contributions."""
    c, *_ = _contrib(s)
    G = len(c)
    ba = c.sum()
    se = np.sqrt(G / (G - 1) * ((c - ba / G) ** 2).sum())
    return ba - Z * se, ba + Z * se


def bootstrap_samples(s: EvalSet, rng, B: int = 1000):
    """Percentile bootstrap over samples, stratified by class."""
    _, inv, n_k, K, _ = _contrib(s)
    stats = np.zeros(B)
    for k in range(K):
        x = s.correct[inv == k]
        stats += rng.binomial(len(x), x.mean(), B) / len(x) if len(x) else 0
    stats /= K
    return tuple(np.quantile(stats, [0.025, 0.975]))


def bootstrap_clusters(s: EvalSet, rng, B: int = 1000):
    """Percentile bootstrap over clusters: resample whole prompts and recompute balanced accuracy."""
    _, inv, n_k, K, cinv = _contrib(s)
    G = cinv.max() + 1
    right = np.zeros((G, K))
    count = np.zeros((G, K))
    np.add.at(right, (cinv, inv), s.correct)
    np.add.at(count, (cinv, inv), 1)
    W = rng.multinomial(G, np.full(G, 1 / G), size=B)  # B x G cluster multiplicities
    num, den = W @ right, W @ count
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = np.nanmean(num / den, axis=1)
    return tuple(np.quantile(stats, [0.025, 0.975]))


def jackknife_clusters(s: EvalSet):
    """Delete-one-cluster jackknife with a normal interval."""
    _, inv, n_k, K, cinv = _contrib(s)
    G = cinv.max() + 1
    right = np.zeros((G, K))
    count = np.zeros((G, K))
    np.add.at(right, (cinv, inv), s.correct)
    np.add.at(count, (cinv, inv), 1)
    R, C = right.sum(0), count.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        loo = np.nanmean((R - right) / (C - count), axis=1)
    se = np.sqrt((G - 1) / G * ((loo - loo.mean()) ** 2).sum())
    ba = balanced_accuracy(s)
    return ba - Z * se, ba + Z * se


METHODS = {
    "wald_iid": lambda s, rng: wald_iid(s),
    "bootstrap_samples": bootstrap_samples,
    "wald_cluster": lambda s, rng: wald_cluster(s),
    "jackknife_clusters": lambda s, rng: jackknife_clusters(s),
    "bootstrap_clusters": bootstrap_clusters,
}
