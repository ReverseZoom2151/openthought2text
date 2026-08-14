from __future__ import annotations

import json

from openthought2text.data.dataset_card import (
    DatasetCard,
    load_dataset_card,
    validate_dataset_card,
    write_dataset_card,
)


def card() -> DatasetCard:
    return DatasetCard(
        dataset_id="synthetic_neural_text_v1",
        source="OpenThought2Text deterministic fixture",
        license="Apache-2.0",
        consent="No participant data; synthetic fixture only.",
        access="Public source-code fixture.",
        modality=("eeg",),
        splits={"protocol": "subject_disjoint", "unit": "subject"},
        preprocessing={"description": "Deterministic JSON fixture", "version": "1"},
    )


def test_dataset_card_round_trip_and_checksum(tmp_path) -> None:
    output = tmp_path / "dataset_card.json"
    write_dataset_card(output, card())

    report = validate_dataset_card(output)
    assert report.passed
    assert load_dataset_card(output).checksum == card().checksum


def test_dataset_card_reports_missing_disclosure_and_checksum_mismatch(tmp_path) -> None:
    output = tmp_path / "dataset_card.json"
    write_dataset_card(output, card())
    data = json.loads(output.read_text(encoding="utf-8"))
    del data["consent"]
    output.write_text(json.dumps(data), encoding="utf-8")
    report = validate_dataset_card(output)
    assert not report.passed
    assert report.issues[0].code == "MISSING_DISCLOSURE"

    write_dataset_card(output, card())
    data = json.loads(output.read_text(encoding="utf-8"))
    data["access"] = "changed access"
    output.write_text(json.dumps(data), encoding="utf-8")
    assert validate_dataset_card(output).issues[0].code == "INVALID_CARD_CHECKSUM"


def test_dataset_card_rejects_yaml_and_invalid_json(tmp_path) -> None:
    yaml_path = tmp_path / "dataset_card.yaml"
    yaml_path.write_text("dataset_id: no-yaml\n", encoding="utf-8")
    assert validate_dataset_card(yaml_path).issues[0].code == "UNSUPPORTED_CARD_FORMAT"

    json_path = tmp_path / "dataset_card.json"
    json_path.write_text("not json", encoding="utf-8")
    assert validate_dataset_card(json_path).issues[0].code == "INVALID_CARD_JSON"
