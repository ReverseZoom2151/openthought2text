import pytest

from openthought2text.evaluation import benchmark_target_free_inference


def test_benchmark_uses_explicit_inputs_warmup_and_injected_clock() -> None:
    inputs, calls = [], []
    timestamps = iter([0.0, 0.1, 0.2, 0.5, 0.5, 1.1])

    def factory(index):
        inputs.append(index)
        return {"neural": index}

    def generate(neural_input):
        calls.append(neural_input)
        return "generated"

    report = benchmark_target_free_inference(
        generate,
        factory,
        warmup_count=1,
        measured_count=3,
        samples_per_input=2,
        clock=lambda: next(timestamps),
    )
    assert inputs == [0, 1, 2, 3]
    assert calls == [{"neural": 0}, {"neural": 1}, {"neural": 2}, {"neural": 3}]
    assert report.elapsed_wall_s == pytest.approx(1.0)
    assert report.samples_per_second == pytest.approx(6.0)
    assert report.latency_p50_s == pytest.approx(0.3)
    assert report.metadata["inference_path"] == "target_free"


def test_benchmark_rejects_target_accepting_generator_and_invalid_counts() -> None:
    def unsafe(neural_input, labels=None):
        return neural_input

    with pytest.raises(AssertionError, match="forbidden target"):
        benchmark_target_free_inference(unsafe, lambda index: index, measured_count=1)
    with pytest.raises(ValueError, match="positive"):
        benchmark_target_free_inference(
            lambda neural: neural, lambda index: index, measured_count=0
        )
