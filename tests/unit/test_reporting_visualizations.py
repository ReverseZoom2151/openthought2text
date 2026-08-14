from openthought2text.reporting import (
    render_control_leaderboard,
    render_data_quality_summary,
    render_failure_case_gallery,
    render_subject_benchmark_table,
)


def test_visualizations_are_safe_deterministic_and_mark_missing():
    view = render_data_quality_summary({"dataset": "<x>", "subjects": None})
    assert "Missing" in view.markdown and "&lt;x&gt;" in view.html
    controls = render_control_leaderboard([{"control": "zero"}, {"control": "full"}])
    assert controls.markdown.index("full") < controls.markdown.index("zero")


def test_empty_subject_and_failure_artifacts_render():
    assert "Missing" in render_subject_benchmark_table([]).markdown
    assert "&lt;x&gt;" in render_failure_case_gallery([{"sample_id": "a", "reference": "<x>"}]).html
