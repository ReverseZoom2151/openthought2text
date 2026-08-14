from __future__ import annotations

import pytest

from openthought2text.config import config_checksum, load_json_config, resolve_named_configs


def test_load_config_checksum_and_named_resolution(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"seed":7,"nested":{"x":true}}\n', encoding="utf-8")
    config = load_json_config(path)
    assert config_checksum(config) == config_checksum({"nested": {"x": True}, "seed": 7})
    assert resolve_named_configs(model={"x": 1}, task={"x": 2}) == {
        "model": {"x": 1}, "task": {"x": 2}
    }


def test_config_rejects_duplicate_or_empty_objects(tmp_path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x":1,"x":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_json_config(duplicate)
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        load_json_config(empty)
