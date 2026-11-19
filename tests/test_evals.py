from evals.run_evals import run_provider


def test_stub_eval_passes(tmp_path):
    report = run_provider("stub", state_dir=tmp_path / "state")
    failed = [item.scenario_id for item in report.results if not item.passed]
    assert failed == []
    assert report.passed_count == report.total
    assert report.total >= 24
