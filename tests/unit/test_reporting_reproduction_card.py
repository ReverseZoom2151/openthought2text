import json

import pytest

from openthought2text.reporting import (
    ArtifactBinding,
    ProvenanceError,
    ReproductionProvenanceCard,
    SourceReference,
    read_reproduction_card,
    write_reproduction_card,
)


def _hash(value):
    return value * 64


def _card():
    bind = lambda name, value: ArtifactBinding(name, f"runs/{name}.json", _hash(value))
    return ReproductionProvenanceCard(
        "corrected-braintranslator",
        SourceReference("Paper", "https://example/paper", "v2"),
        SourceReference("Repository", "https://example/repo", "abc123"),
        "Clean-room corrected target-free inference reproduction.",
        ("Teacher-forced decoding removed.",),
        ("neural_features", "sample_mask"),
        bind("split", "a"),
        bind("config", "b"),
        bind("checkpoint", "c"),
        _hash("d"),
    )


def test_reproduction_card_round_trips_with_mandatory_disclosures(tmp_path):
    card, path = _card(), tmp_path / "reproduction.json"
    write_reproduction_card(path, card)
    assert read_reproduction_card(path) == card
    assert json.loads(path.read_text())["performance_claims"] == "none"


def test_reproduction_card_rejects_target_input_and_tampering():
    with pytest.raises(ProvenanceError, match="cannot include"):
        ReproductionProvenanceCard(
            "x",
            SourceReference("p", "u", "v"),
            SourceReference("r", "u", "v"),
            "f",
            ("none",),
            ("target_text",),
            ArtifactBinding("s", "u", _hash("a")),
            ArtifactBinding("c", "u", _hash("b")),
            ArtifactBinding("k", "u", _hash("c")),
            _hash("d"),
        )
    payload = _card().to_dict()
    payload["fidelity_summary"] = "tampered"
    with pytest.raises(ProvenanceError, match="does not match"):
        ReproductionProvenanceCard.from_dict(payload)
