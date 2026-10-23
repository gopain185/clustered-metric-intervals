import numpy as np

from cmibench import intervals as iv
from cmibench.sim import Design, EvalSet, balanced_accuracy, sample


def test_cluster_contributions_sum_to_balanced_accuracy():
    s = sample(Design(G=40, m=3), np.random.default_rng(0))
    c, *_ = iv._contrib(s)
    assert np.isclose(c.sum(), balanced_accuracy(s))


def test_all_intervals_bracket_the_estimate():
    s = sample(Design(G=150, m=18, sigma=1.0), np.random.default_rng(0))
    ba = balanced_accuracy(s)
    for name, f in iv.METHODS.items():
        lo, hi = f(s, np.random.default_rng(1))
        assert lo < ba < hi, name


def test_cluster_interval_is_wider_when_a_shared_effect_hits_few_classes_per_cluster():
    s = sample(Design(G=150, m=3, sigma=2.0), np.random.default_rng(0))
    assert np.diff(iv.wald_cluster(s))[0] > 1.2 * np.diff(iv.wald_iid(s))[0]


def test_jackknife_matches_sandwich_closely():
    s = sample(Design(G=300, m=18, sigma=1.0), np.random.default_rng(3))
    a = np.diff(iv.wald_cluster(s))[0]
    b = np.diff(iv.jackknife_clusters(s))[0]
    assert abs(a - b) / a < 0.05


def test_perfect_classifier_has_zero_width_iid():
    s = EvalSet(np.ones(36), np.repeat(np.arange(18), 2), np.arange(36))
    lo, hi = iv.wald_iid(s)
    assert lo == hi == 1.0
