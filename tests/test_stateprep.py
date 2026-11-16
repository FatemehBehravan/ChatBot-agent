from src.stateprep import prepare_state


def test_prepare_state_is_idempotent(tmp_path):
    prepare_state(tmp_path)
    kept = tmp_path / "sessions" / "keep.txt"
    kept.write_text("stay", encoding="utf-8")
    prepare_state(tmp_path)
    assert kept.read_text(encoding="utf-8") == "stay"
    assert (tmp_path / "prepared").read_text(encoding="utf-8") == "ok\n"
    for name in ("sessions", "db", "index", "audit"):
        assert (tmp_path / name).is_dir()
