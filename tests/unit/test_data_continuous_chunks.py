from __future__ import annotations

from dataclasses import replace

import torch

from openthought2text.data import (
    SIGNAL_TIMELINE_ALIGNMENT,
    TensorBackedSample,
    build_continuous_chunk_view,
)

from .test_data_schema import sample


def source(target=True):
    item = sample()
    if not target:
        item = replace(item, target=None)
    return TensorBackedSample(
        sample=item,
        values=torch.tensor([[1.0, 2.0, 3.0, 4.0, 99.0], [5.0, 6.0, 7.0, 8.0, 99.0]]),
        time_mask=torch.tensor([True, True, True, True, False]),
    )


def test_continuous_chunks_use_signal_timestamps_and_fixed_windows_only() -> None:
    view = build_continuous_chunk_view((source(),), duration_s=0.008, stride_s=0.008)
    assert [row.sample.sample_id for row in view.chunks] == ["s-1:chunk:0", "s-1:chunk:2"]
    assert [row.values.shape for row in view.chunks] == [torch.Size([2, 2]), torch.Size([2, 2])]
    assert view.chunks[1].values.tolist() == [[3.0, 4.0], [7.0, 8.0]]
    assert all(row.sample.target is None for row in view.chunks)
    assert view.provenance.alignment_regime == SIGNAL_TIMELINE_ALIGNMENT


def test_target_deletion_or_change_cannot_affect_chunk_boundaries_or_masks() -> None:
    with_target = build_continuous_chunk_view((source(),), duration_s=0.008, stride_s=0.008)
    without_target = build_continuous_chunk_view(
        (source(target=False),), duration_s=0.008, stride_s=0.008
    )
    changed = replace(
        source().sample, target=replace(source().sample.target, text="many unrelated words")
    )
    changed_view = build_continuous_chunk_view(
        (replace(source(), sample=changed),), duration_s=0.008, stride_s=0.008
    )

    for other in (without_target, changed_view):
        assert [row.sample.sample_id for row in other.chunks] == [
            row.sample.sample_id for row in with_target.chunks
        ]
        assert [row.time_mask.tolist() for row in other.chunks] == [
            row.time_mask.tolist() for row in with_target.chunks
        ]
        assert [row.values.shape for row in other.chunks] == [
            row.values.shape for row in with_target.chunks
        ]
