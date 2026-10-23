"""Simulated evaluation sets: G clusters (prompts), each answered by m of K classes (authors).

Whether a classifier gets a sample right is Bernoulli with logit a_k + u_g: a_k is the class difficulty and u_g a
cluster effect shared by every sample of the cluster (some prompts are easy for all authors, some hard). sigma is
the standard deviation of u_g, so sigma = 0 means independent samples. The true balanced accuracy is the mean over
classes of E[sigmoid(a_k + u)], computed by Gauss-Hermite quadrature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Design:
    K: int = 18
    G: int = 150
    m: int = 18  # classes answering each cluster (m = K: every class answers every cluster)
    sigma: float = 1.0
    base: float = 0.3  # mean logit of being right (about 0.57 accuracy at sigma 0)
    spread: float = 1.0  # standard deviation of the class difficulties a_k

    def classes(self, rng: np.random.Generator) -> np.ndarray:
        return self.base + self.spread * np.linspace(-1.5, 1.5, self.K)  # fixed, evenly spread difficulties


@dataclass
class EvalSet:
    correct: np.ndarray  # 0/1 per sample
    label: np.ndarray  # class per sample
    cluster: np.ndarray  # cluster id per sample


def truth(design: Design) -> float:
    a = design.classes(None)
    x, w = np.polynomial.hermite_e.hermegauss(64)
    w = w / w.sum()
    return float(np.mean([(w / (1 + np.exp(-(ak + design.sigma * x)))).sum() for ak in a]))


def sample(design: Design, rng: np.random.Generator) -> EvalSet:
    a = design.classes(rng)
    labels, clusters = [], []
    for g in range(design.G):
        ks = np.arange(design.K) if design.m == design.K else rng.choice(design.K, design.m, replace=False)
        labels.append(ks)
        clusters.append(np.full(len(ks), g))
    label, cluster = np.concatenate(labels), np.concatenate(clusters)
    u = rng.normal(0.0, design.sigma, design.G)
    p = 1 / (1 + np.exp(-(a[label] + u[cluster])))
    return EvalSet((rng.random(len(p)) < p).astype(float), label, cluster)


def balanced_accuracy(s: EvalSet) -> float:
    return float(np.mean([s.correct[s.label == k].mean() for k in np.unique(s.label)]))
