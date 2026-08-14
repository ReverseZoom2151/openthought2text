import pytest

from openthought2text.evaluation import corpus_bleu, corpus_meteor_unigram_approx, corpus_rouge_l


def test_corpus_bleu_is_effective_order_and_tokenization_explicit() -> None:
    assert corpus_bleu(["alpha"], ["alpha"]) == pytest.approx(1.0)
    assert corpus_bleu(["a b c"], ["x y z"]) == 0.0
    assert corpus_bleu(["A|B"], ["a|b"], tokenizer=lambda text: text.casefold().split("|")) == pytest.approx(1.0)


def test_rouge_l_and_meteor_approx_known_and_empty_cases() -> None:
    assert corpus_rouge_l(["a b c"], ["a c"]) == pytest.approx(0.8)
    assert corpus_rouge_l([""], [""]) == 1.0
    assert corpus_meteor_unigram_approx(["a b"], ["a b"]) == pytest.approx(0.9375)
    assert corpus_meteor_unigram_approx([""], ["x"]) == 0.0


def test_similarity_metrics_reject_ambiguous_or_empty_corpora() -> None:
    with pytest.raises(ValueError, match="equally sized"):
        corpus_bleu(["a"], [])
    with pytest.raises(ValueError, match="non-empty"):
        corpus_rouge_l([], [])
