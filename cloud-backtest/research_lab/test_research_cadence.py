import json
import pandas as pd

from . import run as runmod


def _write_state(path, last_attempt, last_count):
    path.write_text(
        json.dumps(
            {
                "last_analysis_attempt_at": last_attempt,
                "last_analysis_outcome_count": last_count,
            }
        ),
        encoding="utf-8",
    )


def test_hourly_gate_ignores_cron_phase(monkeypatch, tmp_path):
    state = tmp_path / "state.json"
    _write_state(
        state,
        "2026-08-18T05:44:00+00:00",
        1000,
    )

    monkeypatch.setattr(runmod, "STATE_FILE", state)
    monkeypatch.setattr(runmod, "ANALYSIS_INTERVAL_MIN", 60)
    monkeypatch.setattr(runmod, "MIN_NEW_OUTCOMES_FOR_ANALYSIS", 1000)

    # The five-minute cron can be phase-shifted. :44 -> :49 -> :54
    # is still exactly five-minute cadence.
    assert runmod._analysis_due(
        pd.Timestamp(
            "2026-08-18T06:44:00+00:00"
        ),
        outcome_count=2500,
    )


def test_hourly_gate_blocks_before_sixty_minutes(monkeypatch, tmp_path):
    state = tmp_path / "state.json"
    _write_state(
        state,
        "2026-08-18T05:44:00+00:00",
        1000,
    )

    monkeypatch.setattr(runmod, "STATE_FILE", state)
    monkeypatch.setattr(runmod, "ANALYSIS_INTERVAL_MIN", 60)
    monkeypatch.setattr(runmod, "MIN_NEW_OUTCOMES_FOR_ANALYSIS", 1000)

    assert not runmod._analysis_due(
        pd.Timestamp(
            "2026-08-18T06:39:00+00:00"
        ),
        outcome_count=2500,
    )


def test_hourly_gate_requires_new_matured_outcomes(monkeypatch, tmp_path):
    state = tmp_path / "state.json"
    _write_state(
        state,
        "2026-08-18T05:44:00+00:00",
        2000,
    )

    monkeypatch.setattr(runmod, "STATE_FILE", state)
    monkeypatch.setattr(runmod, "ANALYSIS_INTERVAL_MIN", 60)
    monkeypatch.setattr(runmod, "MIN_NEW_OUTCOMES_FOR_ANALYSIS", 1000)

    assert not runmod._analysis_due(
        pd.Timestamp(
            "2026-08-18T06:44:00+00:00"
        ),
        outcome_count=2500,
    )


def test_successful_analysis_state_is_distinguishable(monkeypatch, tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text(
        "{}",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        runmod,
        "STATE_FILE",
        state_file,
    )

    now = pd.Timestamp(
        "2026-08-18T06:44:00+00:00"
    )

    runmod._mark_analysis(
        now,
        "ok",
        rows=5000,
        outcomes=2500,
        successful_at=now,
    )

    state = json.loads(
        state_file.read_text(
            encoding="utf-8"
        )
    )

    assert (
        state["last_analysis_attempt_at"]
        == now.isoformat()
    )
    assert state[
        "last_analysis_attempt_status"
    ] == "ok"
    assert (
        state["last_successful_analysis_at"]
        == now.isoformat()
    )
    assert (
        state["last_successful_analysis_outcome_count"]
        == 2500
    )
