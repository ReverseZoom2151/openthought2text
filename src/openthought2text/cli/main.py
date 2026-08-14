"""Small, dependency-light CLI for reproducible project operations."""

from __future__ import annotations

import argparse
from pathlib import Path

from openthought2text.version import __version__


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ott", description="OpenThought2Text research CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data = subparsers.add_parser("data", help="Dataset discovery and validation")
    data_subparsers = data.add_subparsers(dest="data_command", required=True)
    for name in ("discover", "validate"):
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    print(f"OpenThought2Text command accepted: {args.command}")
    print("Use the dataset/model modules to execute the selected reproducible workflow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
