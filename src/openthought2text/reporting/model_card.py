"""Evidence-bound Markdown model cards generated from immutable run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class ModelCardError(ValueError):
    """A model-card artifact lacks the evidence needed to make its status clear."""


class ModelCardStatus(str, Enum):
    CLAIMED = "claimed"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class ModelCardArtifact:
    """A deterministic Markdown card and its machine-readable evidence status."""

    run_id: str
    status: ModelCardStatus
    markdown: str


def generate_model_card(
    evaluation: Any,
    provenance: Any,
    release_gate: Any,
    *,
    artifact_references: Mapping[str, str],
) -> ModelCardArtifact:
    """Render a strict card from saved evaluation/provenance/gate objects.

    Required reference keys are ``evaluation_report`` and ``provenance_report``.
    The prediction artifact, checkpoint, data manifest, split plan, and resolved
    configuration are always rendered from their respective evidence objects.
    """
    _validate_inputs(evaluation, provenance, release_gate, artifact_references)
    status = ModelCardStatus.CLAIMED if release_gate.passed else ModelCardStatus.UNSUPPORTED
    markdown = _render_markdown(evaluation, provenance, release_gate, status, artifact_references)
    return ModelCardArtifact(run_id=evaluation.run_id, status=status, markdown=markdown)


def write_model_card(path: str | Path, card: ModelCardArtifact) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(card.markdown, encoding="utf-8", newline="\n")


def _validate_inputs(
    evaluation: Any, provenance: Any, release_gate: Any, artifact_references: Mapping[str, str]
) -> None:
    if evaluation.run_id != provenance.run_id:
        raise ModelCardError("evaluation and provenance artifacts must share run_id")
    required = {"evaluation_report", "provenance_report"}
    missing = required.difference(artifact_references)
    if missing:
        raise ModelCardError(f"missing required artifact reference(s): {', '.join(sorted(missing))}")
    for name, reference in artifact_references.items():
        if not isinstance(reference, str) or not reference.strip():
            raise ModelCardError(f"artifact reference {name!r} must be a non-empty string")
    if not isinstance(release_gate.passed, bool):
        raise ModelCardError("release_gate must provide an explicit boolean passed status")


def _render_markdown(
    evaluation: Any,
    provenance: Any,
    gate: Any,
    status: ModelCardStatus,
    references: Mapping[str, str],
) -> str:
    title = _escape(provenance.model.identifier)
    lines = [f"# Model Card: {title}", "", "## Evidence status", ""]
    if status is ModelCardStatus.CLAIMED:
        lines.extend(
            [
                "**CLAIMED — evidence gate passed.**",
                "",
                "This status supports only the constrained benchmark result documented below; it is not a claim of unrestricted thought decoding.",
            ]
        )
    else:
        lines.extend(
            [
                "**UNSUPPORTED — do not present this run as an evidence-backed decoding claim.**",
                "",
                "The measured values below are retained for audit only until every listed evidence-gate failure is resolved.",
            ]
        )
    lines.extend(["", "## Run and scope", ""])
    lines.extend(
        [
            f"- Run ID: `{_escape(evaluation.run_id)}`",
            f"- Benchmark row: `{_escape(evaluation.benchmark.value)}`",
            f"- Model artifact: `{_escape(provenance.model.identifier)}`",
            f"- Code revision: `{_escape(provenance.code_revision)}`",
            f"- Prediction count: {evaluation.prediction_count}",
            f"- Inference target text visible: **{_boolean(provenance.information_access.inference_target_text)}**",
            f"- Inference text context visible: **{_boolean(provenance.information_access.inference_text_context)}**",
            f"- Alignment source: {_escape(provenance.information_access.alignment_source)}",
            f"- Split definition: {_escape(provenance.information_access.split_definition)}",
        ]
    )
    lines.extend(["", "## Artifact references", "", "| Artifact | Reference | SHA-256 |", "| --- | --- | --- |"])
    rows = (
        ("Evaluation report", references["evaluation_report"], "—"),
        ("Provenance report", references["provenance_report"], provenance.binding_sha256),
        ("Predictions", evaluation.prediction_artifact, "—"),
        ("Model", provenance.model.uri, provenance.model.sha256),
        ("Checkpoint", provenance.checkpoint.uri, provenance.checkpoint.sha256),
        ("Data manifest", provenance.data_manifest.uri, provenance.data_manifest.sha256),
        ("Split plan", provenance.split_plan.uri, provenance.split_plan.sha256),
        ("Resolved config", provenance.config.uri, provenance.config.sha256),
    )
    lines.extend(f"| {_escape(name)} | `{_escape(reference)}` | `{checksum}` |" for name, reference, checksum in rows)
    lines.extend(["", "## Measured evaluation", "", "| Metric | Value | Grounded gain | Neural contribution |", "| --- | ---: | ---: | ---: |"])
    for metric, value in sorted(evaluation.metrics.items()):
        grounding = evaluation.grounding.get(metric)
        gain = "—" if grounding is None else _number(grounding.grounded_gain)
        contribution = "—" if grounding is None else _number(grounding.neural_contribution)
        lines.append(f"| `{_escape(metric)}` | {_number(value)} | {gain} | {contribution} |")
    lines.extend(["", "## Release-evidence gate", ""])
    if gate.passed:
        lines.append("- **PASS:** required provenance, target-free audit, controls, and grounded evidence are present.")
    else:
        lines.append("- **FAIL:** the following evidence is missing, mismatched, or insufficient:")
        lines.append("")
        for failure in gate.failures:
            lines.append(f"  - `{_escape(failure.code.value)}` — {_escape(failure.message)}")
    lines.extend(
        [
            "",
            "## Limitations and responsible use",
            "",
            "- This card describes a constrained recorded-task benchmark only.",
            "- Fluent output is not by itself evidence of neural grounding; consult the control results and grounded-gain fields above.",
            "- Do not infer private, unrestricted, clinical, or real-time capabilities from this artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def _escape(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("|", "\\|")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _number(value: float) -> str:
    return f"{float(value):.6g}"


def _boolean(value: bool) -> str:
    return "yes" if value else "no"
