import numpy as np

from cmibench.sim import Design, balanced_accuracy, sample, truth


def test_layout_and_truth_matches_large_sample():
    d = Design(G=4000, m=18, sigma=1.0)
    s = sample(d, np.random.default_rng(0))
    assert len(s.correct) == 4000 * 18
    assert np.bincount(s.label).tolist() == [4000] * 18
    assert abs(balanced_accuracy(s) - truth(d)) < 0.005


def test_sigma_zero_truth_is_mean_of_sigmoids():
    d = Design(sigma=0.0)
    a = d.classes(None)
    assert np.isclose(truth(d), np.mean(1 / (1 + np.exp(-a))))


def test_partial_layout_uses_m_classes_per_cluster():
    s = sample(Design(G=50, m=3), np.random.default_rng(1))
    assert len(s.correct) == 150
    assert all(len(np.unique(s.label[s.cluster == g])) == 3 for g in range(50))
