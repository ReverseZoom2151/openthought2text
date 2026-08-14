"""Small, dependency-light CLI for reproducible project operations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import torch

from openthought2text.data import (
    AdapterRegistry,
    audit_authorized_preflight_plan,
    DatasetManifest,
    validate_dataset_card,
    SplitProtocol,
    SyntheticNeuralTextAdapter,
    ZuCoDiscoveryAdapter,
    Brain2QwertyDiscoveryAdapter,
    T15DiscoveryAdapter,
    audit_splits,
    build_split_plan,
    collate_tensor_backed_samples,
    load_json_tensor_samples,
    load_manifest,
    validate_split_plan,
    write_manifest,
)
from openthought2text.evaluation import (
    BenchmarkRowLabel,
    evaluate_saved_predictions,
    read_evaluation_report,
    write_evaluation_report,
    token_ids_to_prediction_records,
    write_prediction_jsonl,
)
from openthought2text.models import NeuralToTextModelConfig, build_neural_to_text_model
from openthought2text.training import (
    CheckpointMetadata,
    build_training_inputs,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)
from openthought2text.config.run import RunManifest
from openthought2text.version import __version__


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ott", description="OpenThought2Text research CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data = subparsers.add_parser("data", help="Dataset discovery and validation")
    data_subparsers = data.add_subparsers(dest="data_command", required=True)
    for name in ("discover", "validate", "prepare"):
        command = data_subparsers.add_parser(name)
        command.add_argument("--dataset", required=True)
        command.add_argument("--root", type=_path, required=True)
    card = data_subparsers.add_parser("card-validate", help="Validate a checksummed dataset card")
    card.add_argument("--card", type=_path, required=True)
    preflight = data_subparsers.add_parser(
        "preflight-audit", help="Audit authorized metadata bindings without loading participant signals"
    )
    preflight.add_argument("--plan", type=_path, required=True)

    splits = subparsers.add_parser("splits", help="Split audits")
    splits_subparsers = splits.add_subparsers(dest="splits_command", required=True)
    audit = splits_subparsers.add_parser("audit")
    audit.add_argument("--artifact", type=_path, required=True)
    audit.add_argument("--protocol", required=True)
    build_split = splits_subparsers.add_parser("build", help="Build a derived, leakage-aware split")
    build_split.add_argument("--manifest", type=_path, required=True)
    build_split.add_argument("--output", type=_path, required=True)
    build_split.add_argument("--protocol", choices=[item.value for item in SplitProtocol], required=True)
    build_split.add_argument("--seed", type=int, default=0)
    build_split.add_argument("--held-out-subject")
    build_split.add_argument("--validation-fraction", type=float, default=0.1)
    build_split.add_argument("--test-fraction", type=float, default=0.2)

    train = subparsers.add_parser("train", help="Reproducible local training paths")
    train_subparsers = train.add_subparsers(dest="train_command", required=True)
    synthetic = train_subparsers.add_parser("synthetic", help="Run the non-participant synthetic trace")
    synthetic.add_argument("--root", type=_path, required=True, help="prepared synthetic artifact root")
    synthetic.add_argument("--output", type=_path, required=True, help="new run directory")
    synthetic.add_argument("--epochs", type=int, default=1)
    synthetic.add_argument("--seed", type=int, default=7)

    evaluate = subparsers.add_parser("evaluate", help="Evaluation and audit tools")
    evaluate_subparsers = evaluate.add_subparsers(dest="evaluate_command", required=True)
    generation = evaluate_subparsers.add_parser("audit-generation")
    generation.add_argument("--checkpoint", type=_path, required=True)

    controls = evaluate_subparsers.add_parser("compare-controls")
    controls.add_argument("--run", type=_path, required=True)
    controls.add_argument("--controls", required=True)

    saved = evaluate_subparsers.add_parser("saved-predictions")
    saved.add_argument("--predictions", type=_path, required=True)
    saved.add_argument("--benchmark", required=True)
    saved.add_argument("--output", type=_path, required=True)

    report = subparsers.add_parser("report", help="Read saved evaluation artifacts")
    report_subparsers = report.add_subparsers(dest="report_command", required=True)
    build = report_subparsers.add_parser("build")
    build.add_argument("--report", type=_path, required=True)
    return parser


def _registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register("synthetic", SyntheticNeuralTextAdapter)
    registry.register("zuco_discovery", ZuCoDiscoveryAdapter)
    registry.register("brain2qwerty_discovery", Brain2QwertyDiscoveryAdapter)
    registry.register("t15_discovery", T15DiscoveryAdapter)
    return registry


def _manifest_path(artifact: Path) -> Path:
    return artifact if artifact.is_file() else artifact / "synthetic_manifest.jsonl"


def _emit(value: object) -> None:
    print(json.dumps(value, sort_keys=True, default=str))


def _run_data(args: argparse.Namespace) -> int:
    if args.data_command == "card-validate":
        report = validate_dataset_card(args.card)
        _emit({"passed": report.passed, "card": None if report.card is None else report.card.to_dict(),
               "issues": [{"code": item.code, "message": item.message} for item in report.issues]})
        return 0 if report.passed else 1
    if args.data_command == "preflight-audit":
        report = audit_authorized_preflight_plan(args.plan)
        _emit(
            {
                "passed": report.passed,
                "dataset_id": None if report.plan is None else report.plan.dataset_id,
                "requested_protocols": [] if report.plan is None else [item.value for item in report.plan.requested_protocols],
                "issues": [{"code": item.code, "message": item.message, "path": None if item.path is None else str(item.path)} for item in report.issues],
            }
        )
        return 0 if report.passed else 1
    registry = _registry()
    if args.dataset not in registry:
        available = ", ".join(registry.names())
        raise ValueError(f"adapter {args.dataset!r} is not installed; available: {available}")
    adapter = registry.create(args.dataset)
    source = str(args.root)
    if args.data_command == "discover":
        manifest = adapter.build_manifest(source)
        _emit({"dataset_id": manifest.dataset_id, "sample_count": len(manifest.samples),
               "information_access": manifest.information_access.to_dict()})
        return 0
    if args.data_command == "prepare":
        if not isinstance(adapter, SyntheticNeuralTextAdapter):
            raise ValueError("the selected adapter does not implement local preparation yet")
        manifest = adapter.generate(source)
        _emit({"dataset_id": manifest.dataset_id, "sample_count": len(manifest.samples),
               "artifact": str(args.root / "synthetic_manifest.jsonl")})
        return 0
    if args.data_command == "validate":
        report = adapter.validate(source)
        if isinstance(adapter, SyntheticNeuralTextAdapter):
            _emit({"passed": report.passed, "sample_count": len(report.manifest.samples),
                   "finding_codes": [finding.code for finding in report.split_audit.findings],
                   "missing_signal_files": [str(path) for path in report.missing_signal_files],
                   "invalid_signal_files": [str(path) for path in report.invalid_signal_files]})
        else:
            issues = getattr(report, "issues", ())
            _emit({"passed": bool(getattr(report, "passed", False)),
                   "issues": [{"code": item.code, "message": item.message,
                               "path": None if getattr(item, "path", None) is None else str(item.path)}
                              for item in issues]})
        return 0 if report.passed else 1
    raise ValueError(f"unsupported data command: {args.data_command}")


def _run_split_audit(args: argparse.Namespace) -> int:
    manifest = load_manifest(_manifest_path(args.artifact))
    report = audit_splits(manifest.samples, information_access=manifest.information_access)
    _emit({"protocol": args.protocol, "passed": report.passed, "sample_count": report.sample_count,
           "findings": [{"code": item.code, "severity": item.severity.value} for item in report.findings]})
    return 0 if report.passed else 1


def _split_plan_path(derived_manifest_path: Path) -> Path:
    """Return the required, deterministic sidecar location for a split plan."""
    return derived_manifest_path.with_suffix(".split_plan.json")


def _run_split_build(args: argparse.Namespace) -> int:
    source_path = args.manifest
    output_path = args.output
    plan_path = _split_plan_path(output_path)
    if output_path.exists() or plan_path.exists():
        raise ValueError("refusing to overwrite derived manifest or split-plan sidecar")
    source_manifest = load_manifest(source_path)
    plan = build_split_plan(
        source_manifest.samples,
        args.protocol,
        seed=args.seed,
        held_out_subject=args.held_out_subject,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    validation = validate_split_plan(source_manifest.samples, plan)
    validation.require_valid()
    metadata = dict(source_manifest.metadata)
    metadata["derived_from_manifest"] = str(source_path)
    metadata["split_plan"] = plan.to_dict()
    derived_manifest = DatasetManifest(
        dataset_id=source_manifest.dataset_id,
        samples=plan.materialize(source_manifest.samples),
        information_access=source_manifest.information_access,
        source_url=source_manifest.source_url,
        license=source_manifest.license,
        description=source_manifest.description,
        schema_version=source_manifest.schema_version,
        metadata=metadata,
    )
    write_manifest(output_path, derived_manifest)
    plan_path.write_text(json.dumps(plan.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _emit(
        {
            "source_manifest": str(source_path),
            "derived_manifest": str(output_path),
            "split_plan": str(plan_path),
            "protocol": plan.protocol.value,
            "sample_count": len(derived_manifest.samples),
            "excluded_sample_count": len(plan.excluded_sample_ids),
        }
    )
    return 0


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _run_synthetic_training(args: argparse.Namespace) -> int:
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.output.exists():
        raise ValueError("refusing to overwrite an existing run directory")
    manifest_path = args.root / "synthetic_manifest.jsonl"
    manifest = load_manifest(manifest_path)
    seed_everything(args.seed)
    rows = load_json_tensor_samples(manifest, args.root)
    train_rows = tuple(row for row in rows if row.sample.split == "train")
    test_rows = tuple(row for row in rows if row.sample.split == "test")
    if not train_rows or not test_rows:
        raise ValueError("synthetic run requires both train and test samples")
    inputs = build_training_inputs(train_rows, unknown_policy="unk")
    config = NeuralToTextModelConfig(
        hidden_size=32, temporal_kernel=5, stride_samples=4, encoder_layers=1, decoder_layers=1,
        encoder_heads=4, decoder_heads=4, vocabulary_size=len(inputs.tokenizer.vocabulary),
        max_sequence_length=32, encoder_dropout=0.0, decoder_dropout=0.0,
    )
    model = build_neural_to_text_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    steps = tuple(step for _ in range(args.epochs) for step in train_one_epoch(
        model, inputs.rows, inputs.tokenizer, optimizer=optimizer, batch_size=2,
        sample_rate_hz=train_rows[0].sample.signal.sampling_rate_hz,
    ))
    args.output.mkdir(parents=True)
    (args.output / "tokenizer.json").write_text(json.dumps(inputs.tokenizer.to_dict(), sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "normalizer.json").write_text(json.dumps(inputs.normalizer.to_dict(), sort_keys=True) + "\n", encoding="utf-8")
    run_id = f"synthetic-seed-{args.seed}"
    run_manifest = RunManifest(
        experiment_name=run_id, dataset_artifact_checksum=_sha256_file(manifest_path),
        split_manifest_checksum=_sha256_file(manifest_path), seed=args.seed,
        resolved_config={"model": asdict(config), "tokenizer_checksum": inputs.tokenizer.checksum,
                         "normalizer_checksum": inputs.normalizer.checksum},
    )
    (args.output / "run_manifest.json").write_text(
        json.dumps(run_manifest.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    checkpoint = args.output / "checkpoint.pt"
    save_checkpoint(checkpoint, model=model, optimizer=optimizer, metadata=CheckpointMetadata(
        epoch=args.epochs, step=len(steps), selection_metric="synthetic_train_loss",
        selection_value=steps[-1].loss, run_manifest=run_manifest,
    ))
    batch = collate_tensor_backed_samples(test_rows)
    model.eval()
    generated = model.generate(batch.signals, channel_mask=batch.channel_mask, token_mask=batch.time_mask,
                               sample_rate_hz=test_rows[0].sample.signal.sampling_rate_hz)
    # Random/untrained models may emit only special tokens; preserve that as an
    # explicit evaluable empty prediction rather than failing serialization.
    records = token_ids_to_prediction_records(
        generated.token_ids,
        batch.sample_ids,
        lambda ids: inputs.tokenizer.decode(ids) or "<empty>",
        run_id=run_id,
    )
    predictions = args.output / "predictions.jsonl"
    write_prediction_jsonl(predictions, records)
    _emit({"run_id": run_id, "output": str(args.output), "checkpoint": str(checkpoint),
           "predictions": str(predictions), "steps": len(steps), "final_train_loss": steps[-1].loss})
    return 0


def _run_evaluation(args: argparse.Namespace) -> int:
    if args.evaluate_command == "saved-predictions":
        report = evaluate_saved_predictions(
            args.predictions,
            benchmark=BenchmarkRowLabel.parse(args.benchmark),
            prediction_artifact=str(args.predictions),
        )
        write_evaluation_report(args.output, report)
        _emit({"output": str(args.output), "metrics": report.metrics,
               "grounding": {name: value.grounded_gain for name, value in report.grounding.items()}})
        return 0
    if args.evaluate_command == "compare-controls":
        report = read_evaluation_report(args.run)
        requested = set(args.controls.split(","))
        selected = [row.to_dict() for row in report.control_results if row.condition.value in requested]
        _emit({"run_id": report.run_id, "controls": selected,
               "grounding": {name: value.grounded_gain for name, value in report.grounding.items()}})
        return 0
    raise ValueError(
        "audit-generation cannot load an arbitrary checkpoint; construct a trusted model and use "
        "the Python target-free audit API"
    )


def _run_report(args: argparse.Namespace) -> int:
    report = read_evaluation_report(args.report)
    _emit({"run_id": report.run_id, "benchmark": report.benchmark.value, "metrics": report.metrics,
           "prediction_count": report.prediction_count,
           "grounded_gain": {name: value.grounded_gain for name, value in report.grounding.items()}})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "data":
            return _run_data(args)
        if args.command == "splits":
            if args.splits_command == "audit":
                return _run_split_audit(args)
            if args.splits_command == "build":
                return _run_split_build(args)
        if args.command == "train" and args.train_command == "synthetic":
            return _run_synthetic_training(args)
        if args.command == "evaluate":
            return _run_evaluation(args)
        if args.command == "report" and args.report_command == "build":
            return _run_report(args)
        print(f"OpenThought2Text command accepted: {args.command}")
        print("Use the dataset/model modules to execute the selected reproducible workflow.")
        return 0
    except ValueError as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
