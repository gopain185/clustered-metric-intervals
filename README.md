# clustered-metric-intervals

When the samples of an evaluation set come in clusters, for example passages that answer the same prompt, how
should the confidence interval of a balanced-accuracy score be built? And how far apart can a CV score and a public
leaderboard fall by chance? This repository simulates evaluation sets with a known balanced accuracy and a shared
cluster effect, then measures the coverage of five 95% intervals.

## Results

Coverage of the 95% interval with 150 clusters (300 evaluation sets per cell, 18 classes):

| classes per cluster | sigma | iid Wald | bootstrap over samples | cluster-robust Wald | cluster jackknife | bootstrap over clusters |
| --- | --- | --- | --- | --- | --- | --- |
| 18 | 0 | 0.93 | 0.94 | 0.95 | 0.95 | 0.94 |
| 18 | 0.5 | 0.84 | 0.83 | 0.95 | 0.95 | 0.94 |
| 18 | 1.0 | 0.73 | 0.72 | 0.94 | 0.94 | 0.94 |
| 18 | 1.5 | 0.58 | 0.58 | 0.96 | 0.96 | 0.96 |
| 3 | 1.5 | 0.87 | 0.86 | 0.93 | 0.93 | 0.92 |
| 1 | 1.5 | 0.93 | 0.93 | 0.98 | 0.97 | 0.96 |

With every class in every cluster, the spread of the estimate is 1.4 to 2.3 times the iid standard error
(sigma 0.5 to 1.5). For a CV of 150 x 18 against a 1,200-row public leaderboard with 1 to 3 classes per prompt,
CV − LB has a standard deviation of 0.016 (sigma 0) to 0.027 (sigma 1.5). It reaches 0.04 or more in 0.5% to 7% of
draws.

On a real grouped OOF (my Name That LLM models, 150 prompts x 18 authors), the cluster-robust and iid half-widths
agreed within 0.001, at about 0.017. There, prompt difficulty barely clusters correctness.

The full write-up is in `kaggle/notebook_findings.md`.

## Use

```
pip install -r requirements.txt
python -m pytest                 # 8 tests
python -m cmibench results       # all three experiments, about 6 minutes on a laptop CPU
python kaggle/build_kaggle.py --reuse-results   # Kaggle dataset and notebook folders
```

- `cmibench/sim.py`: the generator and the true balanced accuracy.
- `cmibench/intervals.py`: the five intervals.
- `cmibench/experiments.py`: coverage, design effect, CV against LB.

Kaggle notebook: https://www.kaggle.com/code/locdunhat/confidence-intervals-for-clustered-eval-sets
Kaggle dataset: https://www.kaggle.com/datasets/locdunhat/clustered-eval-interval-simulations

## License

MIT for the code, CC0 for the generated data.
