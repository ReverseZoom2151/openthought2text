from openthought2text.controls import (
    ControlCondition,
    build_control,
    gaussian_noise_like,
    length_only_signal,
    mask_only_signal,
    phase_randomized_surrogate,
    shuffle_batch,
    timing_only_signal,
    zero_signal,
)


def test_zero_and_shuffled_controls_preserve_shape() -> None:
    signal = [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]
    assert zero_signal(signal) == [[[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]]
    assert shuffle_batch(signal, permutation=[2, 0, 1]) == [signal[2], signal[0], signal[1]]
    assert shuffle_batch(signal, seed=9) != signal


def test_noise_and_surrogate_are_seeded_and_distribution_preserving() -> None:
    signal = [[1.0, 2.0], [3.0, 4.0]]
    assert gaussian_noise_like(signal, seed=7) == gaussian_noise_like(signal, seed=7)
    surrogate = phase_randomized_surrogate(signal, seed=5)
    assert sorted(item for row in surrogate for item in row) == [1.0, 2.0, 3.0, 4.0]


def test_structure_only_controls_expose_only_declared_side_information() -> None:
    assert mask_only_signal([[True, False, True]], channels=2) == [
        [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
    ]
    assert length_only_signal([2, 1], channels=1, max_length=3) == [
        [[1.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]]
    ]
    assert timing_only_signal([[0, 3], [1]], time_steps=4, channels=1) == [
        [[1.0, 0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0, 0.0]]
    ]


def test_named_control_dispatcher_requires_declared_metadata() -> None:
    assert build_control(ControlCondition.ZERO, [[3.0]]) == [[0.0]]
    assert build_control("length", [[3.0]], valid_lengths=[1], time_steps=2) == [
        [[1.0, 0.0]]
    ]
