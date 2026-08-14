from __future__ import annotations

import json

import pytest

from openthought2text.data import (
    UnknownTokenPolicy,
    fit_train_text_tokenizer,
    load_train_text_tokenizer,
    write_train_text_tokenizer,
)
from openthought2text.data.schema import TextTarget

from .test_data_schema import sample


def train_row(sample_id: str, text: str):
    return sample(sample_id=sample_id, split="train", target=TextTarget(text))


def test_train_text_tokenizer_is_deterministic_and_encodes_decodes() -> None:
    first = train_row("a", "Hello, world!")
    second = train_row("b", "hello there")
    tokenizer = fit_train_text_tokenizer((second, first), unknown_policy="error")

    assert tokenizer.vocabulary == ("<pad>", "<bos>", "<eos>", "<unk>", "hello", "!", ",", "there", "world")
    assert tokenizer.special_ids == {"pad": 0, "bos": 1, "eos": 2, "unk": 3}
    assert tokenizer.decode(tokenizer.encode("Hello, world!")) == "hello, world!"
    assert tokenizer.fit_sample_ids == ("a", "b")


def test_train_text_tokenizer_rejects_non_train_and_unknown_policy_ambiguity() -> None:
    with pytest.raises(ValueError, match="explicitly declared"):
        fit_train_text_tokenizer((train_row("a", "hello"),))
    with pytest.raises(ValueError, match="non-train"):
        fit_train_text_tokenizer(
            (train_row("a", "hello"), sample(sample_id="b", split="test")),
            unknown_policy="unk",
        )
    with pytest.raises(ValueError, match="error' or 'unk"):
        fit_train_text_tokenizer((train_row("a", "hello"),), unknown_policy="ignore")


def test_unknown_policy_and_artifact_checksum_validation(tmp_path) -> None:
    error_tokenizer = fit_train_text_tokenizer((train_row("a", "known"),), unknown_policy="error")
    unk_tokenizer = fit_train_text_tokenizer(
        (train_row("a", "known"),), unknown_policy=UnknownTokenPolicy.UNK
    )
    with pytest.raises(ValueError, match="unknown token"):
        error_tokenizer.encode("new")
    assert unk_tokenizer.encode("new", add_bos=False, add_eos=False) == (unk_tokenizer.unk_id,)

    output = tmp_path / "tokenizer.json"
    write_train_text_tokenizer(output, unk_tokenizer)
    assert load_train_text_tokenizer(output).checksum == unk_tokenizer.checksum
    content = json.loads(output.read_text(encoding="utf-8"))
    content["vocabulary"][-1] = "changed"
    output.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_train_text_tokenizer(output)
