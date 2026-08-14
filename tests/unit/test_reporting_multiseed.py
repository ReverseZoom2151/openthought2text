import pytest

from openthought2text.evaluation import BenchmarkRowLabel
from openthought2text.reporting import (
    MultiSeedBenchmarkPlan,
    SeedExpectation,
    SeedMetricResult,
    aggregate_multi_seed,
)


def _plan():
    return MultiSeedBenchmarkPlan(
        "tiny",
        BenchmarkRowLabel("zuco", "eeg", "read", "trial", "loso", "open", "greedy"),
        ("wer",),
        (SeedExpectation(1, "p1", "v1"), SeedExpectation(2, "p2", "v2")),
    )


def test_multiseed_aggregate_requires_complete_paired_plan_and_stores_all_seeds():
    aggregate = aggregate_multi_seed(
        _plan(),
        (
            SeedMetricResult(2, {"wer": 0.4}, "p2", "v2"),
            SeedMetricResult(1, {"wer": 0.2}, "p1", "v1"),
        ),
    )
    assert [item.seed for item in aggregate.per_seed] == [1, 2]
    assert aggregate.mean_metrics == {"wer": pytest.approx(0.3)}
    assert "no confidence interval" in aggregate.no_statistical_claim


def test_multiseed_rejects_missing_seed_and_tampered_artifact_pairing():
    with pytest.raises(ValueError, match="all and only"):
        aggregate_multi_seed(_plan(), (SeedMetricResult(1, {"wer": 0.2}, "p1", "v1"),))
    with pytest.raises(ValueError, match="must match"):
        aggregate_multi_seed(
            _plan(),
            (
                SeedMetricResult(1, {"wer": 0.2}, "wrong", "v1"),
                SeedMetricResult(2, {"wer": 0.4}, "p2", "v2"),
            ),
        )


def test_multiseed_json_bindings_detect_tampering():
    plan = _plan()
    payload = plan.to_dict()
    assert MultiSeedBenchmarkPlan.from_dict(payload) == plan
    payload["name"] = "tampered"
    with pytest.raises(ValueError, match="binding"):
        MultiSeedBenchmarkPlan.from_dict(payload)
