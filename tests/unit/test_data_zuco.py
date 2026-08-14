from openthought2text.data import DatasetAdapter, ZuCoDiscoveryAdapter


def add_mat_file(root, task: str, filename: str) -> None:
    directory = root / task / "Matlab_files"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_bytes(b"MATLAB fixture only; not parsed")


def test_zuco_discovery_inventories_valid_layout_without_reading_matlab(tmp_path) -> None:
    add_mat_file(tmp_path, "task1-SR", "resultsZAB.mat")
    add_mat_file(tmp_path, "task2-NR", "resultsZDM_NR.mat")
    add_mat_file(tmp_path, "task3-TSR", "resultsZGW.mat")
    adapter = ZuCoDiscoveryAdapter()

    report = adapter.discover(str(tmp_path))
    manifest = adapter.build_manifest(str(tmp_path))

    assert isinstance(adapter, DatasetAdapter)
    assert report.passed
    assert [task.subject_ids for task in report.tasks] == [("ZAB",), ("ZDM",), ("ZGW",)]
    assert manifest.samples == ()
    assert manifest.metadata["inventory_only"] is True
    assert list(adapter.iter_samples(str(tmp_path))) == []


def test_zuco_discovery_reports_missing_required_layout_fields(tmp_path) -> None:
    (tmp_path / "task1-SR").mkdir()
    report = ZuCoDiscoveryAdapter().validate(str(tmp_path))

    assert not report.passed
    assert {issue.code for issue in report.missing_fields} == {"MISSING_MATLAB_FILES_DIRECTORY"}
    assert "PARTIAL_TASK_LAYOUT" in {issue.code for issue in report.warnings}


def test_zuco_discovery_reports_ambiguous_subject_and_task_versions(tmp_path) -> None:
    add_mat_file(tmp_path, "task1-SR", "not-a-results-name.mat")
    add_mat_file(tmp_path, "task2-NR", "resultsZAB.mat")
    add_mat_file(tmp_path, "task2-NR-2.0", "resultsZDM_NR.mat")
    report = ZuCoDiscoveryAdapter().discover(str(tmp_path))

    codes = {issue.code for issue in report.ambiguous_fields}
    assert {"AMBIGUOUS_SUBJECT_ID", "AMBIGUOUS_TASK_VERSION"} <= codes
    assert not report.passed


def test_zuco_discovery_reports_missing_source_directory(tmp_path) -> None:
    report = ZuCoDiscoveryAdapter().discover(str(tmp_path / "not-present"))
    assert [issue.code for issue in report.errors] == ["MISSING_SOURCE_DIRECTORY"]
