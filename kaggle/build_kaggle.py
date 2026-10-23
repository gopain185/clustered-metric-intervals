"""Builds the Kaggle dataset and notebook for this project.

    python kaggle/build_kaggle.py [--public] [--notebook-only] [--reuse-results]

writes kaggle/dataset/ (simulated evaluation sets with their true balanced accuracy, the result tables of the three
experiments and the datasheet) and kaggle/notebook/ (a CPU notebook that embeds the package source and reruns the
experiments).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cmibench import experiments as ex  # noqa: E402
from cmibench.sim import Design, sample, truth  # noqa: E402

OUT = ROOT / "kaggle"
DATASET_ID = "locdunhat/clustered-eval-interval-simulations"
NOTEBOOK_ID = "locdunhat/confidence-intervals-for-clustered-eval-sets"
TITLE = "Confidence Intervals for Clustered Eval Sets"


def build_dataset(public: bool, rerun: bool = True) -> Path:
    out = OUT / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    for sigma in ex.SIGMAS:
        for m in ex.MS:
            d = Design(G=150, m=m, sigma=sigma)
            rng = np.random.default_rng([ex.SEED, 9, int(sigma * 10), m])
            for r in range(20):
                s = sample(d, rng)
                parts.append(pd.DataFrame(dict(sigma=sigma, m=m, set_id=r, cluster=s.cluster, label=s.label,
                                               correct=s.correct.astype(int), true_balanced_accuracy=round(truth(d), 6))))
    pd.concat(parts).to_csv(out / "eval_sets.csv", index=False)
    if rerun:
        for name, fn in [("exp1_coverage", ex.coverage), ("exp2_design_effect", ex.design_effect),
                         ("exp3_cv_vs_lb", ex.cv_vs_lb)]:
            fn().to_csv(out / f"{name}.csv", index=False)
    else:  # reuse the tables written by `python -m cmibench results`
        for f in sorted((ROOT / "results").glob("exp*.csv")):
            (out / f.name).write_bytes(f.read_bytes())
    datasheet = (OUT / "DATASHEET.md").read_text(encoding="utf-8")
    (out / "DATASHEET.md").write_text(datasheet, encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "Clustered Eval Interval Simulations",
        "id": DATASET_ID,
        "subtitle": "Eval sets with prompt clusters and a known balanced accuracy",
        "description": datasheet,
        "isPrivate": not public,
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["benchmark", "classification"],
    }, indent=2), encoding="utf-8")
    return out


def _sources() -> dict[str, str]:
    files = sorted((ROOT / "cmibench").glob("*.py"))
    return {f.relative_to(ROOT).as_posix(): f.read_text(encoding="utf-8") for f in files}


SETUP = """

import os, sys
ROOT = "/kaggle/working/clustered-metric-intervals" if os.path.isdir("/kaggle/working") else os.path.abspath("clustered-metric-intervals")
for path, text in SOURCES.items():
    os.makedirs(os.path.dirname(os.path.join(ROOT, path)), exist_ok=True)
    with open(os.path.join(ROOT, path), "w", encoding="utf-8") as f:
        f.write(text)
sys.path.insert(0, ROOT)
"""

IMPORTS = """import time, warnings
warnings.simplefilter("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from cmibench import experiments as ex
pd.set_option("display.precision", 3); pd.set_option("display.width", 160)
"""

EXP1 = """t = time.time(); e1 = ex.coverage(); print(round(time.time() - t), "s")
s1 = ex.summarise_coverage(e1); s1
"""

PLOT1 = """fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, m in zip(axes, ex.MS):
    g = s1[(s1.G == 150) & (s1.m == m)]
    for name in ["wald_iid", "bootstrap_samples", "wald_cluster", "jackknife_clusters", "bootstrap_clusters"]:
        ax.plot(g.sigma, g[name], marker="o", label=name)
    ax.axhline(0.95, color="k", lw=0.8); ax.set_title(f"150 clusters, {m} classes per cluster")
    ax.set_xlabel("cluster effect sigma (logit scale)")
axes[0].set_ylabel("coverage of the 95% interval"); axes[-1].legend()
plt.tight_layout(); plt.show()
"""

EXP2 = """t = time.time(); e2 = ex.design_effect(); print(round(time.time() - t), "s")
e2
"""

EXP3 = """t = time.time(); e3 = ex.cv_vs_lb(); print(round(time.time() - t), "s")
e3
"""

SAMPLE = """import glob
from cmibench import intervals as iv
from cmibench.sim import EvalSet
hits = glob.glob("/kaggle/input/**/eval_sets.csv", recursive=True)
if hits:
    df = pd.read_csv(hits[0]); rng = np.random.default_rng(0); rows = []
    for (sigma, m), g in df.groupby(["sigma", "m"]):
        cover = {k: [] for k in iv.METHODS}
        for _, e in g.groupby("set_id"):
            s = EvalSet(e.correct.to_numpy(float), e.label.to_numpy(), e.cluster.to_numpy())
            t = e.true_balanced_accuracy.iloc[0]
            for k, f in iv.METHODS.items():
                lo, hi = f(s, rng); cover[k].append(lo <= t <= hi)
        rows.append({"sigma": sigma, "m": m, **{k: np.mean(v) for k, v in cover.items()}})
    display(pd.DataFrame(rows))
else:
    print("dataset not attached")
"""


def _cells() -> list[tuple[str, str]]:
    return [
        ("markdown", (OUT / "notebook_intro.md").read_text(encoding="utf-8")),
        ("code", "SOURCES = " + json.dumps(_sources(), indent=0) + SETUP),
        ("code", IMPORTS),
        ("markdown", "## 1. Coverage\n\nShare of 95% intervals that contain the true balanced accuracy (18 classes, "
                     "300 evaluation sets per row). `G` is the number of clusters, `m` the classes answering each "
                     "cluster, `sigma` the standard deviation of the shared cluster effect on the logit of being "
                     "right."),
        ("code", EXP1),
        ("code", PLOT1),
        ("markdown", "## 2. How far off is the iid standard error?\n\nThe spread of the estimate across 400 "
                     "evaluation sets (`true_sd`) against the average standard error from the iid and the "
                     "cluster-robust formulas, with 150 clusters."),
        ("code", EXP2),
        ("markdown", "## 3. CV against a public leaderboard, by chance alone\n\nThe same classifier, scored once on "
                     "a train-layout CV (150 prompts, all 18 classes each) and once on a public leaderboard of 1,200 "
                     "rows with 1 to 3 classes per prompt. `gap_sd` is the spread of CV minus LB, `gap_q95` the 95th "
                     "percentile of its size, and `p_gap_ge_0_04` the chance that CV beats LB by 0.04 or more. "
                     "2,000 draws per row."),
        ("code", EXP3),
        ("markdown", "## 4. The evaluation sets in the dataset\n\nThe companion dataset holds 20 simulated "
                     "evaluation sets per (sigma, m) with 150 clusters, every sample's correctness, class and cluster, "
                     "and the true balanced accuracy. The cell recomputes coverage on them."),
        ("code", SAMPLE),
        ("markdown", (OUT / "notebook_findings.md").read_text(encoding="utf-8")),
    ]


def build_notebook(public: bool) -> Path:
    out = OUT / "notebook"
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for i, (kind, src) in enumerate(_cells()):
        c = {"cell_type": kind, "id": f"cell-{i}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(outputs=[], execution_count=None)
        cells.append(c)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (out / "clustered-metric-intervals.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": NOTEBOOK_ID, "title": TITLE, "code_file": "clustered-metric-intervals.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": not public, "enable_gpu": False, "enable_internet": False,
        "dataset_sources": [DATASET_ID], "competition_sources": [], "kernel_sources": [],
        "keywords": ["benchmark"],
    }, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    public = "--public" in sys.argv
    if "--notebook-only" not in sys.argv:
        print("wrote", build_dataset(public, rerun="--reuse-results" not in sys.argv))
    print("wrote", build_notebook(public))
