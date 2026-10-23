"""The experiments. Each returns a long table; summarise_* reduce it.

1. coverage: how often each 95% interval contains the true balanced accuracy, by cluster effect, number of clusters
   and classes per cluster.
2. design_effect: the spread of the estimate across replicates, against the standard errors the iid and the
   cluster-robust formulas report.
3. cv_vs_lb: how far apart a train-layout CV estimate and a test-layout leaderboard estimate of the same classifier
   fall by chance alone, for a competition-like setup (150 prompts x 18 authors in CV, 1,200 public test rows with 1
   to 3 authors per prompt).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import intervals as iv
from .sim import Design, balanced_accuracy, sample, truth

SEED = 20261023
SIGMAS = (0.0, 0.5, 1.0, 1.5)
GS = (30, 150, 500)
MS = (1, 3, 18)


def coverage(reps: int = 300, sigmas=SIGMAS, gs=GS, ms=MS) -> pd.DataFrame:
    rows = []
    for sigma in sigmas:
        for G in gs:
            for m in ms:
                if G * m < 4 * 18:  # too few samples to see every class
                    continue
                d = Design(G=G, m=m, sigma=sigma)
                t = truth(d)
                rng = np.random.default_rng([SEED, 1, int(sigma * 10), G, m])
                for r in range(reps):
                    s = sample(d, rng)
                    if len(np.unique(s.label)) < d.K:
                        continue
                    for name, f in iv.METHODS.items():
                        lo, hi = f(s, rng)
                        rows.append(dict(sigma=sigma, G=G, m=m, rep=r, method=name, lo=lo, hi=hi,
                                         covered=lo <= t <= hi, width=hi - lo))
    return pd.DataFrame(rows)


def summarise_coverage(df: pd.DataFrame) -> pd.DataFrame:
    t = df.pivot_table(index=["sigma", "G", "m"], columns="method", values="covered", aggfunc="mean")
    order = list(iv.METHODS)
    return t[order].reset_index()


def design_effect(reps: int = 400, sigmas=SIGMAS, ms=MS, G: int = 150) -> pd.DataFrame:
    rows = []
    for sigma in sigmas:
        for m in ms:
            d = Design(G=G, m=m, sigma=sigma)
            rng = np.random.default_rng([SEED, 2, int(sigma * 10), m])
            est, se_iid, se_cl = [], [], []
            for _ in range(reps):
                s = sample(d, rng)
                if len(np.unique(s.label)) < d.K:
                    continue
                est.append(balanced_accuracy(s))
                lo, hi = iv.wald_iid(s)
                se_iid.append((hi - lo) / (2 * iv.Z))
                lo, hi = iv.wald_cluster(s)
                se_cl.append((hi - lo) / (2 * iv.Z))
            sd = float(np.std(est, ddof=1))
            rows.append(dict(sigma=sigma, m=m, G=G, true_sd=sd, iid_se=float(np.mean(se_iid)),
                             cluster_se=float(np.mean(se_cl)), iid_ratio=sd / np.mean(se_iid),
                             cluster_ratio=sd / np.mean(se_cl)))
    return pd.DataFrame(rows)


def cv_vs_lb(reps: int = 2000, sigmas=SIGMAS) -> pd.DataFrame:
    """Gap between a CV estimate (150 x 18) and a public-LB estimate (1,200 rows, 1-3 authors per prompt)."""
    rows = []
    for sigma in sigmas:
        rng = np.random.default_rng([SEED, 3, int(sigma * 10)])
        cv_d = Design(G=150, m=18, sigma=sigma)
        gaps = []
        for _ in range(reps):
            cv = balanced_accuracy(sample(cv_d, rng))
            m = int(rng.integers(1, 4))
            lb = balanced_accuracy(sample(Design(G=1200 // m, m=m, sigma=sigma), rng))
            gaps.append(cv - lb)
        g = np.array(gaps)
        rows.append(dict(sigma=sigma, gap_sd=g.std(ddof=1), gap_q95=np.quantile(np.abs(g), 0.95),
                         p_gap_ge_0_04=float(np.mean(g >= 0.04))))
    return pd.DataFrame(rows)
