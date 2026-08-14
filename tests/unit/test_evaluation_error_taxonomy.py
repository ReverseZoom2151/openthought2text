from openthought2text.evaluation import (
    TextErrorCategory,
    classify_text_error,
    classify_text_errors,
    word_edit_operations,
)


def test_taxonomy_assigns_all_categories_deterministically() -> None:
    cases = [
        ("Hello, brain!", "hello brain", TextErrorCategory.EXACT),
        ("alpha beta", "", TextErrorCategory.EMPTY),
        ("alpha beta", "alpha alpha beta", TextErrorCategory.REPETITION),
        ("alpha beta", "alpha", TextErrorCategory.OMISSION),
        ("alpha beta", "alpha gamma", TextErrorCategory.SUBSTITUTION),
        ("alpha beta", "alpha beta gamma", TextErrorCategory.HALLUCINATION),
    ]
    for reference, hypothesis, expected in cases:
        category, _ = classify_text_error(reference, hypothesis)
        assert category is expected


def test_word_operations_and_hallucination_precedence_are_exposed() -> None:
    operations = word_edit_operations(["one", "two"], ["one", "three", "four"])
    assert operations.total == 2
    category, operations = classify_text_error("cat dog", "mouse bird")
    assert category is TextErrorCategory.HALLUCINATION
    assert operations.substitutions == 2


def test_report_has_per_sample_records_and_zero_filled_aggregate_counts() -> None:
    report = classify_text_errors(["a", "b", "c"], ["a", "", "x"], sample_ids=["s1", "s2", "s3"])
    assert [record.sample_id for record in report.records] == ["s1", "s2", "s3"]
    assert [record.category for record in report.records] == [
        TextErrorCategory.EXACT,
        TextErrorCategory.EMPTY,
        TextErrorCategory.HALLUCINATION,
    ]
    assert report.counts[TextErrorCategory.SUBSTITUTION] == 0
    assert sum(report.rates.values()) == 1.0
