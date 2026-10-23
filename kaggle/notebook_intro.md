# Confidence Intervals for Clustered Eval Sets

Many evaluation sets come in clusters. In *Name That LLM*, every passage answers a prompt, and train has 150 prompts
answered by all 18 models. If some prompts are easy for every model and others hard, the passages of one prompt are
not independent. A standard error that treats them as independent is then too small, and a CV score looks more
precise than it is.

This notebook simulates such evaluation sets with a known truth. Whether a classifier is right on a sample follows a
logistic model with a class difficulty and a shared cluster effect of standard deviation `sigma`. It compares five
95% intervals for balanced accuracy:

- `wald_iid`: per-class binomial variance, samples independent.
- `bootstrap_samples`: percentile bootstrap over samples, stratified by class.
- `wald_cluster`: cluster-robust (sandwich) variance. Balanced accuracy is a sum of per-cluster contributions once
  the class counts are fixed.
- `jackknife_clusters`: delete-one-cluster jackknife.
- `bootstrap_clusters`: percentile bootstrap over whole clusters.

It then asks how large a gap between a train-layout CV and a public leaderboard chance alone can produce.

Everything runs on CPU in about 10 minutes. The package source is embedded in the next cell, so the notebook is
self-contained.
