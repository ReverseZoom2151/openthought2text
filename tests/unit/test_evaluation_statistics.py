import pytest

from openthought2text.evaluation import (
    cluster_bootstrap_ci,
    paired_permutation_test,
    stimulus_bootstrap_ci,
    subject_bootstrap_ci,
)


def test_subject_bootstrap_is_seeded_cluster_aware_and_contains_estimate() -> None:
    scores = {"s1": [0.0, 0.0, 0.0, 0.0], "s2": [1.0]}
    first = subject_bootstrap_ci(scores, resamples=500, seed=17)
    second = subject_bootstrap_ci(scores, resamples=500, seed=17)
    assert first == second
    assert first.cluster_unit == "subject"
    assert first.clusters == 2
    assert first.estimate == pytest.approx(0.5)  # Equal subject, not window, weighting.
    assert first.lower <= first.estimate <= first.upper


def test_stimulus_and_generic_bootstraps_validate_and_label_units() -> None:
    stimulus = stimulus_bootstrap_ci({"sentence-a": [0.4], "sentence-b": [0.8]}, resamples=20)
    generic = cluster_bootstrap_ci({"day-1": [0.4], "day-2": [0.8]}, resamples=20)
    assert stimulus.cluster_unit == "stimulus"
    assert generic.cluster_unit == "cluster"
    with pytest.raises(ValueError, match="at least one score"):
        subject_bootstrap_ci({"s1": []})


def test_paired_permutation_exact_result_and_error_direction() -> None:
    higher = paired_permutation_test([4.0] * 4, [0.0] * 4, permutations=7, seed=99)
    assert higher.exact
    assert higher.permutations == 16
    assert higher.observed_difference == 4.0
    assert higher.p_value == pytest.approx(1 / 16)

    lower = paired_permutation_test([0.2, 0.2], [0.8, 0.8], higher_is_better=False)
    assert lower.observed_difference == pytest.approx(0.6)


def test_paired_permutation_monte_carlo_is_seeded() -> None:
    full = [0.8 + index / 1_000 for index in range(20)]
    control = [0.3] * 20
    first = paired_permutation_test(full, control, permutations=500, exact_max_pairs=4, seed=13)
    second = paired_permutation_test(full, control, permutations=500, exact_max_pairs=4, seed=13)
    assert first == second
    assert not first.exact
    assert first.permutations == 500
    assert first.p_value < 0.05
