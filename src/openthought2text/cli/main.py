"""Small, dependency-light CLI for reproducible project operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openthought2text.data import AdapterRegistry, SyntheticNeuralTextAdapter, audit_splits, load_manifest
from openthought2text.evaluation import (
    BenchmarkRowLabel,
    evaluate_saved_predictions,
    read_evaluation_report,
    write_evaluation_report,
)
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

    splits = subparsers.add_parser("splits", help="Split audits")
    splits_subparsers = splits.add_subparsers(dest="splits_command", required=True)
    audit = splits_subparsers.add_parser("audit")
    audit.add_argument("--artifact", type=_path, required=True)
    audit.add_argument("--protocol", required=True)

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
    return registry


def _manifest_path(artifact: Path) -> Path:
    return artifact if artifact.is_file() else artifact / "synthetic_manifest.jsonl"


def _emit(value: object) -> None:
    print(json.dumps(value, sort_keys=True, default=str))


def _run_data(args: argparse.Namespace) -> int:
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
        if not isinstance(adapter, SyntheticNeuralTextAdapter):
            raise ValueError("the selected adapter does not implement local validation yet")
        report = adapter.validate(source)
        _emit({"passed": report.passed, "sample_count": len(report.manifest.samples),
               "finding_codes": [finding.code for finding in report.split_audit.findings],
               "missing_signal_files": [str(path) for path in report.missing_signal_files],
               "invalid_signal_files": [str(path) for path in report.invalid_signal_files]})
        return 0
    raise ValueError(f"unsupported data command: {args.data_command}")


def _run_split_audit(args: argparse.Namespace) -> int:
    manifest = load_manifest(_manifest_path(args.artifact))
    report = audit_splits(manifest.samples, information_access=manifest.information_access)
    _emit({"protocol": args.protocol, "passed": report.passed, "sample_count": report.sample_count,
           "findings": [{"code": item.code, "severity": item.severity.value} for item in report.findings]})
    return 0 if report.passed else 1


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
    print("audit-generation requires an explicitly configured model loader; use the Python audit API.")
    return 0


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
        if args.command == "splits" and args.splits_command == "audit":
            return _run_split_audit(args)
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
