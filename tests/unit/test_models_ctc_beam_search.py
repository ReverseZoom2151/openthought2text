import inspect

import pytest
import torch

from openthought2text.models import (
    CTCBeamSearchConfig,
    TargetFreeCTCBeamSearch,
    ValidationFittedLanguageScorer,
)


def _logits(token_path):
    logits = torch.full((1, len(token_path), 3), -10.0)
    logits[0, torch.arange(len(token_path)), torch.tensor(token_path)] = 10.0
    return logits


def test_ctc_beam_search_handles_blank_repetition_and_valid_lengths():
    decoder = TargetFreeCTCBeamSearch(CTCBeamSearchConfig(beam_width=4, blank_token_id=0))
    # 1,1 collapse; blank then 1 permits a second emitted 1. Final strong 2 is padding.
    logits = _logits([1, 1, 0, 1, 2])
    output = decoder.decode(logits, torch.tensor([4]))
    assert output.best_token_ids == ((1, 1),)
    assert output.hypothesis_mask[0, 0]
    assert output.language_scores is None


def test_ctc_beam_search_exposes_neural_and_validation_fitted_language_components():
    scorer = ValidationFittedLanguageScorer(
        lambda prefix: 30.0 if prefix == (2,) else 0.0, "held-out calibration", 5
    )
    decoder = TargetFreeCTCBeamSearch(
        CTCBeamSearchConfig(beam_width=3, blank_token_id=0, language_weight=1.0)
    )
    logits = _logits([1])
    output = decoder.decode(logits, torch.tensor([1]), scorer)
    assert output.best_token_ids == ((2,),)
    assert output.language_scores is not None
    assert output.language_scores[0, 0].item() == 30.0
    assert output.combined_scores[0, 0].item() == pytest.approx(
        output.neural_scores[0, 0].item() + 30.0
    )


def test_ctc_beam_search_is_target_free_deterministic_and_validates_inputs():
    decoder = TargetFreeCTCBeamSearch(
        CTCBeamSearchConfig(beam_width=2, blank_token_id=0, input_is_log_probs=True)
    )
    signature = inspect.signature(decoder.decode)
    assert set(signature.parameters) == {"logits_or_log_probs", "valid_lengths", "language_scorer"}
    log_probs = torch.log_softmax(_logits([1, 0]), dim=-1)
    first = decoder.decode(log_probs, torch.tensor([2]))
    second = decoder.decode(log_probs, torch.tensor([2]))
    assert first.hypotheses == second.hypotheses
    with pytest.raises(ValueError, match="normalized"):
        decoder.decode(torch.zeros(1, 2, 3), torch.tensor([2]))
    with pytest.raises(ValueError, match=r"\[1, time\]"):
        decoder.decode(log_probs, torch.tensor([3]))
    with pytest.raises(ValueError, match="ValidationFitted"):
        decoder.decode(log_probs, torch.tensor([2]), lambda _: 0.0)
