import json

import pytest

from openthought2text.reporting import (
    ArtifactBinding,
    InformationAccessContract,
    ProvenanceError,
    RunArtifactProvenance,
    read_provenance_report,
    write_provenance_report,
)


def _hash(character: str) -> str:
    return character * 64


def _report() -> RunArtifactProvenance:
    return RunArtifactProvenance(
        run_id="zuco-loso-seed-7",
        model=ArtifactBinding("continuous-conformer", "models/conformer.yaml", _hash("a")),
        checkpoint=ArtifactBinding("epoch-12", "checkpoints/epoch-12.pt", _hash("b")),
        data_manifest=ArtifactBinding("zuco-v1-prepared", "data/manifest.json", _hash("c")),
        split_plan=ArtifactBinding("loso-unique-text", "splits/loso.json", _hash("d")),
        config=ArtifactBinding("resolved-config", "runs/config.json", _hash("e")),
        code_revision="abc123def",
        information_access=InformationAccessContract(
            train_target_text=True,
            validation_target_text=True,
            inference_target_text=False,
            inference_text_context=False,
            inference_token_boundaries=False,
            inference_event_boundaries=False,
            inference_stimulus_audio=False,
            split_definition="LOSO with unique-text partition",
            alignment_source="signal-derived fixed windows",
        ),
    )


def test_provenance_round_trip_is_hash_bound_and_versioned(tmp_path) -> None:
    report = _report()
    path = tmp_path / "run-provenance.json"
    write_provenance_report(path, report)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == "1.0"
    assert payload["binding_sha256"] == report.binding_sha256
    assert read_provenance_report(path) == report


def test_provenance_rejects_ambiguous_values_and_missing_access_fields() -> None:
    with pytest.raises(ProvenanceError, match="non-ambiguous"):
        ArtifactBinding("unknown", "checkpoint.pt", _hash("a"))
    with pytest.raises(ProvenanceError, match="SHA-256"):
        ArtifactBinding("checkpoint", "checkpoint.pt", "too-short")

    payload = _report().to_dict()
    del payload["information_access"]["inference_target_text"]
    with pytest.raises(ProvenanceError, match="missing"):
        RunArtifactProvenance.from_dict(payload)


def test_provenance_rejects_tampered_artifact_bindings() -> None:
    payload = _report().to_dict()
    payload["checkpoint"]["sha256"] = _hash("f")
    with pytest.raises(ProvenanceError, match="does not match"):
        RunArtifactProvenance.from_dict(payload)
