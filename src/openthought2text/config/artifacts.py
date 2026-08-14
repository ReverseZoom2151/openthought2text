"""Strict, checksummed JSON configuration artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any


def load_json_config(path: str | Path) -> dict[str, Any]:
    """Load a non-empty JSON object, rejecting duplicate-key ambiguity."""
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON configuration: {source}") from error
    if not isinstance(data, dict) or not data:
        raise ValueError("configuration must be a non-empty JSON object")
    return data


def config_checksum(config: Mapping[str, Any]) -> str:
    """Stable content identity for a JSON-serializable resolved config."""
    try:
        payload = json.dumps(dict(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError) as error:
        raise ValueError("configuration must contain JSON-serializable values") from error
    return sha256(payload.encode("utf-8")).hexdigest()


def resolve_named_configs(**configs: Mapping[str, Any]) -> dict[str, Any]:
    """Nest independently named config artifacts without lossy key merging."""
    if not configs or any(not name.strip() for name in configs):
        raise ValueError("at least one non-empty configuration name is required")
    result = {name: dict(value) for name, value in configs.items()}
    config_checksum(result)
    return result


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate configuration key: {key}")
        result[key] = value
    return result
