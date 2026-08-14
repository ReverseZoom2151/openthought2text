import inspect

import pytest
import torch

from openthought2text.models import (
    EvidenceFactorizedCandidateScorer,
    FactorizedScoringControl,
    ValidationFittedScoreWeights,
    fit_factorized_score_weights,
)


def _weights():
    return ValidationFittedScoreWeights(
        2.0, 3.0, -1.0, validation_examples=4, ridge_regularization=0.1
    )


def test_factorized_scoring_accounts_for_each_component_masks_and_ranking():
    scorer = EvidenceFactorizedCandidateScorer(_weights())
    output = scorer(
        neural_scores=torch.tensor([[1.0, 2.0, 3.0]]),
        lm_scores=torch.tensor([[0.5, 1.0, 0.0]]),
        length_scores=torch.tensor([[1.0, 2.0, 1.0]]),
        candidate_ids=torch.tensor([10, 20, 30]),
        candidate_mask=torch.tensor([True, False, True]),
    )
    torch.testing.assert_close(output.combined_scores[:, [0, 2]], torch.tensor([[2.5, 5.0]]))
    assert torch.isneginf(output.combined_scores[0, 1])
    assert output.effective_weights == (2.0, 3.0, -1.0)
    assert output.ranked_candidate_ids[0, 0].item() == 30


def test_factorized_scoring_exposes_zero_neural_and_lm_controls_and_has_no_targets():
    scorer = EvidenceFactorizedCandidateScorer(
        _weights(), FactorizedScoringControl(lambda_neural=0.0, lambda_lm=0.0)
    )
    signature = inspect.signature(scorer.forward)
    assert set(signature.parameters) == {
        "neural_scores",
        "lm_scores",
        "length_scores",
        "candidate_ids",
        "candidate_mask",
    }
    output = scorer(
        torch.tensor([[100.0, -100.0]]),
        torch.tensor([[-100.0, 100.0]]),
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([1, 2]),
    )
    torch.testing.assert_close(output.combined_scores, torch.tensor([[-1.0, -2.0]]))


def test_validation_fit_is_frozen_and_rejects_invalid_validation_inputs():
    neural = torch.tensor([1.0, 0.0, 2.0, 3.0])
    lm = torch.tensor([0.0, 1.0, 1.0, -1.0])
    length = torch.tensor([1.0, 2.0, 0.0, 1.0])
    utility = 2 * neural - 3 * lm + 0.5 * length
    weights = fit_factorized_score_weights(neural, lm, length, utility, ridge_regularization=1e-6)
    assert weights.validation_examples == 4
    assert weights.fit_method == "ridge_least_squares"
    assert weights.lambda_neural == pytest.approx(2.0, abs=1e-4)
    assert weights.lambda_lm == pytest.approx(-3.0, abs=1e-4)
    assert weights.lambda_length == pytest.approx(0.5, abs=1e-4)
    with pytest.raises(ValueError, match="equal length"):
        fit_factorized_score_weights(neural, lm[:-1], length, utility)
    with pytest.raises(ValueError, match="positive"):
        fit_factorized_score_weights(neural, lm, length, utility, ridge_regularization=0)
    with pytest.raises(ValueError, match="finite"):
        fit_factorized_score_weights(
            neural, lm, length, torch.tensor([1.0, 2.0, float("nan"), 4.0])
        )
