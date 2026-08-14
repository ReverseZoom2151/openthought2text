import pytest
import torch

from openthought2text.models import ReducedChannelDistillationConfig, ReducedChannelDistillationLoss


def test_reduced_channel_distillation_matches_only_valid_features_and_backpropagates_student():
    torch.manual_seed(21)
    criterion = ReducedChannelDistillationLoss()
    student = torch.randn(2, 4, 3, requires_grad=True)
    teacher = torch.randn(2, 4, 3, requires_grad=True)
    mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
    first = criterion(student, teacher, mask)
    changed = student.detach().clone()
    changed[~mask] = torch.randn_like(changed[~mask]) * 1_000
    second = criterion(changed, teacher, mask)
    torch.testing.assert_close(first.representation_loss, second.representation_loss, rtol=0, atol=0)
    assert first.logits_loss is None
    assert first.valid_token_count.item() == 4
    first.loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert teacher.grad is None


def test_reduced_channel_distillation_optional_logits_is_temperature_scaled_and_masks_padding():
    torch.manual_seed(22)
    criterion = ReducedChannelDistillationLoss(
        ReducedChannelDistillationConfig(representation_weight=0.5, logits_weight=2.0, temperature=2.0)
    )
    student_features = torch.randn(1, 3, 4, requires_grad=True)
    teacher_features = torch.randn(1, 3, 4, requires_grad=True)
    student_logits = torch.randn(1, 3, 5, requires_grad=True)
    teacher_logits = torch.randn(1, 3, 5, requires_grad=True)
    output = criterion(
        student_features,
        teacher_features,
        torch.tensor([[True, True, False]]),
        student_logits,
        teacher_logits,
    )
    assert output.logits_loss is not None and torch.isfinite(output.logits_loss)
    torch.testing.assert_close(output.loss, 0.5 * output.representation_loss + 2.0 * output.logits_loss)
    output.loss.backward()
    assert student_features.grad is not None and student_features.grad.abs().sum() > 0
    assert student_logits.grad is not None and student_logits.grad.abs().sum() > 0
    assert teacher_features.grad is None and teacher_logits.grad is None


def test_reduced_channel_distillation_rejects_misaligned_masks_and_partial_logits():
    criterion = ReducedChannelDistillationLoss()
    student = torch.randn(1, 2, 3)
    teacher = torch.randn(1, 2, 3)
    with pytest.raises(ValueError, match="token_mask"):
        criterion(student, teacher, torch.ones(1, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="together"):
        criterion(student, teacher, torch.ones(1, 2, dtype=torch.bool), torch.randn(1, 2, 4))
    with pytest.raises(ValueError, match="identical shapes"):
        criterion(
            student,
            teacher,
            torch.ones(1, 2, dtype=torch.bool),
            torch.randn(1, 2, 4),
            torch.randn(1, 2, 5),
        )
