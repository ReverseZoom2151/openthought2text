import pytest

from openthought2text.evaluation import (
    generate_target_free_prediction_records,
    token_ids_to_prediction_records,
)


def test_target_free_generator_serializes_token_ids_without_labels() -> None:
    calls = []

    def generate(neural_input):
        calls.append(neural_input)
        return [[10, 11], [12]]

    records = generate_target_free_prediction_records(
        generate,
        neural_input=[[0.1], [0.2]],
        sample_ids=["sample-a", "sample-b"],
        decode_token_ids=lambda ids: " ".join(f"t{token}" for token in ids),
        run_id="run-1",
        control="shuffled",
        metadata={"decoder": "unit-tokenizer"},
    )
    assert calls == [[[0.1], [0.2]]]
    assert [record.prediction_text for record in records] == ["t10 t11", "t12"]
    assert all(record.target_free for record in records)
    assert all(record.metadata["generation_path"] == "target_free_token_ids" for record in records)
    assert records[0].control.value == "shuffled"


def test_target_accepting_generator_and_bad_alignment_are_rejected() -> None:
    def unsafe(neural_input, labels=None):
        return [[1]]

    with pytest.raises(AssertionError, match="forbidden target"):
        generate_target_free_prediction_records(
            unsafe, [[0.1]], ["sample"], lambda ids: "x", run_id="run-1"
        )
    with pytest.raises(ValueError, match="sample_ids"):
        token_ids_to_prediction_records([[1], [2]], ["same", "same"], lambda ids: "x", run_id="run-1")
    with pytest.raises(ValueError, match="shape"):
        token_ids_to_prediction_records([1, 2], ["a", "b"], lambda ids: "x", run_id="run-1")


def test_single_item_token_row_and_explicit_decoder_contract() -> None:
    records = token_ids_to_prediction_records([1, 2, 3], ["sample"], lambda ids: str(list(ids)), run_id="run-1")
    assert records[0].prediction_text == "[1, 2, 3]"
    with pytest.raises(TypeError, match="text strings"):
        token_ids_to_prediction_records([[1]], ["sample"], lambda ids: ids, run_id="run-1")
