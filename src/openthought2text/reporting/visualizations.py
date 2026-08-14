"""Deterministic, HTML-safe descriptive release-artifact fragments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import Any


@dataclass(frozen=True, slots=True)
class VisualizationFragment:
    markdown: str
    html: str


def render_data_quality_summary(summary: Mapping[str, Any]) -> VisualizationFragment:
    return _table(
        "Data-quality summary",
        [
            {"field": key, "value": _value(summary.get(key))}
            for key in ("dataset", "subjects", "samples", "split", "missingness", "notes")
        ],
    )


def render_subject_benchmark_table(rows: Sequence[Mapping[str, Any]]) -> VisualizationFragment:
    return _rows(
        "Per-subject benchmark table", rows, ("subject", "metric", "value", "artifact"), "subject"
    )


def render_control_leaderboard(rows: Sequence[Mapping[str, Any]]) -> VisualizationFragment:
    return _rows("Control leaderboard", rows, ("control", "metric", "value", "run_id"), "control")


def render_failure_case_gallery(rows: Sequence[Mapping[str, Any]]) -> VisualizationFragment:
    return _rows(
        "Failure-case gallery",
        rows,
        ("sample_id", "error_category", "reference", "full_prediction", "control_predictions"),
        "sample_id",
    )


def _rows(title, rows, keys, sort):
    return _table(
        title,
        [
            {key: _value(row.get(key)) for key in keys}
            for row in sorted(rows, key=lambda item: str(item.get(sort, "")))
        ],
    )


def _table(title, rows):
    headers = list(rows[0]) if rows else ["status"]
    rows = list(rows) or [{"status": "Missing"}]
    md = (
        f"## {title}\n\n| "
        + " | ".join(headers)
        + " |\n| "
        + " | ".join("---" for _ in headers)
        + " |\n"
        + "\n".join(
            "| " + " | ".join(_md(row.get(h, "Missing")) for h in headers) + " |" for row in rows
        )
        + "\n\n_Descriptive artifact view only; no empirical claim is made._\n"
    )
    html = (
        "<section><h2>"
        + escape(title)
        + "</h2><table><thead><tr>"
        + "".join(f"<th>{escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{escape(row.get(h, 'Missing'))}</td>" for h in headers) + "</tr>"
            for row in rows
        )
        + "</tbody></table><p>Descriptive artifact view only; no empirical claim is made.</p></section>"
    )
    return VisualizationFragment(md, html)


def _value(v):
    return "Missing" if v is None or v == "" else str(v)


def _md(v):
    return v.replace("|", "\\|").replace("\n", " ")
