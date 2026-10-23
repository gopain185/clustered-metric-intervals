## Findings

Numbers are from the run committed with the package (`results/`). The notebook's own run uses the same seeds.

1. **When every class answers every cluster, the iid interval fails as soon as there is a cluster effect.** With 150
   clusters of 18, the iid Wald interval covered 93% at sigma 0, 84% at sigma 0.5, 73% at sigma 1 and 58% at
   sigma 1.5. The bootstrap over samples was no better. The spread of the estimate was 1.4, 2.0 and 2.3 times the
   iid standard error at those three sigmas. Adding clusters does not fix it: with 500 clusters coverage at
   sigma 1.5 was still 55%.
2. **The three cluster methods stay close to 95%.** Across all 32 conditions the cluster-robust Wald interval
   covered between 93% and 98%, the cluster jackknife between 93% and 97%, and the cluster bootstrap between 90% and
   97%. With only 30 clusters, the cluster bootstrap fell to 90% to 96%. The sandwich and the jackknife held up
   better there.
3. **With one class per cluster, clustering does not matter.** Each sample is then its own cluster, and the iid
   interval is close to nominal: 90% to 96%, a little low because class counts vary. The cluster-robust formula
   treats the class counts as fixed, so it is a little conservative here (96% to 98%).
4. **On a real grouped OOF the cluster effect turned out to be small.** For my Name That LLM models (150 prompts x 18
   authors), the cluster-robust and iid 95% half-widths were within 0.001 of each other: 0.017 for a LoRA classifier
   at 0.576 and 0.016 to 0.018 for two n-gram SVMs. Prompt difficulty barely clusters correctness there. Check
   before assuming either way.
5. **Chance alone rarely puts a CV 0.04 above a public LB.** For a CV on 150 x 18 and a public LB of 1,200 rows with
   1 to 3 classes per prompt, the gap CV − LB had a standard deviation of 0.016 at sigma 0 and 0.023 at sigma 1. It
   reached 0.04 or more in 0.5% and 4% of draws. Every submission is scored on the same public rows, so one unlucky
   split moves all of them together. A gap of that size seen in several submissions is still one draw, not several.

For a CV score on a clustered set, report the cluster-robust (or cluster-jackknife) interval: it costs nothing and
it is right in both layouts. When CV and LB disagree by more than about two gap standard deviations, look for a
difference between the CV pipeline and the test pipeline before blaming noise.

It is a simulation with a logistic random-intercept model. Real cluster effects can also differ by class, which
this setup does not cover.
