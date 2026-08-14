from openthought2text.evaluation import (
    character_error_rate,
    corpus_word_error_rate,
    edit_distance,
    retrieval_metrics,
    word_error_rate,
)


def test_edit_distance_and_character_error_rate() -> None:
    assert edit_distance(list("kitten"), list("sitting")) == 3
    result = character_error_rate("A  neural signal", "a neural singal")
    assert result.errors == 2
    assert result.reference_units == len("a neural signal")
    assert result.rate == 2 / len("a neural signal")


def test_word_error_rate_aggregates_counts_not_example_means() -> None:
    single = word_error_rate("Hello, world!", "hello there world")
    assert (single.errors, single.reference_units) == (1, 2)

    corpus = corpus_word_error_rate(["a", "one two three"], ["x", "one two three"])
    assert (corpus.errors, corpus.reference_units, corpus.rate) == (1, 4, 0.25)


def test_retrieval_metrics_use_conservative_tie_rank() -> None:
    report = retrieval_metrics(
        [[0.9, 0.1, 0.2], [0.5, 0.5, 0.1]], [0, 0], ks=(1, 2, 5)
    )
    assert report.queries == 2
    assert report.mean_rank == 1.5  # tie gets rank two
    assert report.mean_reciprocal_rank == 0.75
    assert report.recall_at == {1: 0.5, 2: 1.0, 5: 1.0}
