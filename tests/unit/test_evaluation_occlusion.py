from openthought2text.evaluation import (
    MetricSpec,
    OcclusionMode,
    occlude_channel_time,
    occlude_channels,
    occlude_time,
    run_occlusion_suite,
)


def _exact(predictions, references) -> float:
    return float(tuple(predictions) == tuple(references))


def test_channel_time_variants_zero_values_preserve_input_and_record_metadata() -> None:
    signal = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
    channel = occlude_channels(signal, [1])
    assert signal[0][1] == [4.0, 5.0, 6.0]  # Never mutate the source signal.
    assert channel.signal[0][1] == [0.0, 0.0, 0.0]
    assert channel.signal_mask is None
    assert channel.metadata.control_label == "occlusion/zero/channel/channels=1/time=0:3"

    rectangle = occlude_channel_time(signal, [0], 1, 3, mode="mask")
    assert rectangle.signal[0][0] == [1.0, 0.0, 0.0]
    assert rectangle.signal_mask[0][0] == [True, False, False]
    assert rectangle.signal_mask[0][1] == [True, True, True]


def test_time_occlusion_and_suite_report_target_free_metric_drops() -> None:
    signal = [[[1.0, 1.0], [2.0, 2.0]]]

    def generate(neural_input, signal_mask=None):
        total = sum(value for channel in neural_input[0] for value in channel)
        return (str(int(total)),)

    variants = [occlude_channels(signal, [1]), occlude_time(signal, 0, 1, mode=OcclusionMode.MASK)]
    report = run_occlusion_suite(
        generate,
        signal,
        ("6",),
        [MetricSpec("exact", _exact)],
        variants,
        pass_signal_mask=True,
    )
    assert report.baseline_scores == {"exact": 1.0}
    assert [result.metric_drops["exact"] for result in report.results] == [1.0, 1.0]
    assert report.mean_metric_drops == {"exact": 1.0}


def test_occlusion_suite_rejects_target_accepting_generator() -> None:
    def unsafe(signal, labels=None):
        return ("x",)

    variant = occlude_channels([[[1.0]]], [0])
    try:
        run_occlusion_suite(unsafe, [[[1.0]]], ("x",), [MetricSpec("exact", _exact)], [variant])
    except AssertionError as error:
        assert "forbidden target" in str(error)
    else:
        raise AssertionError("target-accepting generator must be rejected")
