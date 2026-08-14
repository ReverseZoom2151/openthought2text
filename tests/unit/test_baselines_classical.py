import pytest

from openthought2text.baselines import ConstantTextBaseline, FrequencyTextBaseline, NearestNeighborTextRetrieval, RidgeRegressor
from openthought2text.evaluation import assert_target_free_signature


def test_text_baselines_fit_only_train_text_and_infer_without_targets() -> None:
    frequency = FrequencyTextBaseline().fit(["b", "a", "b"])
    constant = ConstantTextBaseline("fixed")
    assert_target_free_signature(frequency.predict)
    assert frequency.predict([[0.1], [0.2]]) == ("b", "b")
    assert constant.predict([[0.1]]) == ("fixed",)
    assert len(frequency.artifact().checksum_sha256) == 64


def test_nearest_neighbor_uses_neural_embeddings_at_target_free_inference() -> None:
    baseline = NearestNeighborTextRetrieval().fit([[1, 0], [0, 1]], ["left", "right"])
    assert_target_free_signature(baseline.predict)
    assert baseline.predict([[0.9, 0.1], [0.1, 0.9]]) == ("left", "right")


def test_ridge_train_only_targets_and_serializable_fit_artifact() -> None:
    ridge = RidgeRegressor(alpha=0.1).fit([[0], [1], [2]], [1, 3, 5])
    assert_target_free_signature(ridge.predict)
    assert ridge.predict([[3]])[0] == pytest.approx(6.8, abs=0.3)
    assert ridge.artifact().payload["fit_intercept"] is True
    with pytest.raises(ValueError, match="align"):
        RidgeRegressor().fit([[0], [1]], [1])
