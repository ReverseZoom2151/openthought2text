from openthought2text.reporting import (
    CleanEnvironmentAuditPlan,
    audit_clean_environment,
    render_clean_environment_markdown,
)


def _hash(c):
    return c * 64


def test_clean_environment_audit_binds_plan_and_caller_observations_without_run_claim():
    plan = CleanEnvironmentAuditPlan(
        "Python 3.12",
        "openthought2text==0.0.1",
        _hash("a"),
        "python -m pytest",
        {"config.json": _hash("b"), "model.pt": _hash("c")},
    )
    record = audit_clean_environment(
        plan,
        {"python": "3.12.3", "platform": "linux"},
        {"config.json": _hash("b"), "model.pt": _hash("d")},
    )
    assert record.matching_artifacts == ("config.json",)
    assert record.mismatched_artifacts == ("model.pt",)
    assert "no successful real-data run" in render_clean_environment_markdown(record)


def test_clean_environment_plan_rejects_missing_hash_contract():
    try:
        CleanEnvironmentAuditPlan("py", "pkg", "bad", "test", {"x": "bad"})
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("invalid checksum plan should fail")
