from __future__ import annotations

from openthought2text.cli.main import main


def test_synthetic_prepare_validate_and_audit(tmp_path) -> None:
    assert main(["data", "prepare", "--dataset", "synthetic", "--root", str(tmp_path)]) == 0
    assert main(["data", "validate", "--dataset", "synthetic", "--root", str(tmp_path)]) == 0
    assert (
        main(["splits", "audit", "--artifact", str(tmp_path), "--protocol", "subject_disjoint"])
        == 0
    )
