from pathlib import Path

from . import research_store


def test_configured_rooted_path_is_not_double_prefixed(tmp_path, monkeypatch):
    monkeypatch.setattr(research_store, "ROOT", tmp_path)
    configured = Path("cloud-backtest/research_lab_data/observations.jsonl")

    research_store.append_jsonl(configured, {"ok": True})

    expected = tmp_path / configured
    bad_nested = tmp_path / configured.parent / configured
    assert expected.exists()
    assert expected.read_text(encoding="utf-8").strip() == '{"ok":true}'
    assert not bad_nested.exists()

    rows = research_store.read_jsonl(configured)
    assert rows == [{"ok": True}]


def test_simple_relative_filename_is_still_rooted(tmp_path, monkeypatch):
    monkeypatch.setattr(research_store, "ROOT", tmp_path)

    research_store.append_jsonl("simple.jsonl", {"value": 7})

    assert (tmp_path / "simple.jsonl").exists()
    assert research_store.read_jsonl("simple.jsonl") == [{"value": 7}]


def test_absolute_path_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(research_store, "ROOT", tmp_path)
    absolute = tmp_path / "absolute.jsonl"

    research_store.append_jsonl(absolute, {"absolute": True})

    assert absolute.exists()
    assert research_store.read_jsonl(absolute) == [{"absolute": True}]
