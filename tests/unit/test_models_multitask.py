import inspect

import pytest
import torch

from openthought2text.models import (
    BalancedTaskSchedule,
    MultiParadigmTaskConfig,
    MultiParadigmTaskHead,
    TaskTrackConfig,
    gradient_conflict_log,
)


def _head():
    return MultiParadigmTaskHead(
        MultiParadigmTaskConfig(
            4,
            (TaskTrackConfig("words", ("eeg", "meg"), 3), TaskTrackConfig("anchors", ("eeg",), 2)),
        )
    )


def test_multitask_target_free_forward_masks_and_training_labels_are_separate():
    head = _head()
    features = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, True]])
    assert set(inspect.signature(head.forward).parameters) == {
        "features",
        "mask",
        "task_name",
        "modality",
    }
    logits = head(features, mask, "words", "eeg")
    assert logits.shape == (2, 3)
    trained = head.training_loss(features, mask, "words", "eeg", torch.tensor([0, 2]))
    trained.loss.backward()
    assert features.grad is not None
    with pytest.raises(ValueError, match="incompatible"):
        head(features.detach(), mask, "anchors", "meg")


def test_balanced_schedule_and_gradient_conflict_contracts():
    schedule = BalancedTaskSchedule(("words", "anchors"))
    assert [schedule.task_at(x) for x in range(4)] == ["words", "anchors", "words", "anchors"]
    log = gradient_conflict_log({"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([-1.0, 0.0])})
    assert log.negative_pairs == 1 and log.pairwise_cosines[("a", "b")] == -1.0
    with pytest.raises(ValueError, match="equal length"):
        gradient_conflict_log({"a": torch.ones(2), "b": torch.ones(3)})
