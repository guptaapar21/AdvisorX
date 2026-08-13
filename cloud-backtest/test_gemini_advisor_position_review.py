import json

import gemini_advisor as ga


def _pos(coin):
    return {
        "coin": coin,
        "direction": "long",
        "entry_price": 100.0,
        "stop_loss": 99.0,
        "target_price": 102.0,
        "original_reasoning": "test thesis",
        "minutes_open": 6,
        "current_price": 100.5,
        "unrealized_pnl_inr": 75.0,
    }


def _update(coin):
    return {
        "coin": coin,
        "direction": "long",
        "action": "hold",
        "updated_stop_loss": None,
        "updated_target_price": None,
        "reasoning": "thesis remains valid",
    }


def test_full_position_review_is_accepted():
    text = json.dumps({"new_signals": [], "position_updates": [_update("BTC"), _update("ETH")]})
    result, updates = ga._parse_position_response(text, set(), {"BTC", "ETH"}, {})
    assert result == {}
    assert set(updates) == {"BTC", "ETH"}


def test_bare_list_is_rejected_when_positions_exist():
    text = json.dumps([_update("BTC")])
    try:
        ga._parse_position_response(text, set(), {"BTC"}, {})
    except ValueError as exc:
        assert "bare-list" in str(exc)
    else:
        raise AssertionError("bare-list response must not be accepted with open positions")


def test_partial_position_review_is_rejected():
    text = json.dumps({"new_signals": [], "position_updates": [_update("BTC")]})
    try:
        ga._parse_position_response(text, set(), {"BTC", "ETH"}, {})
    except ValueError as exc:
        assert "ETH" in str(exc)
    else:
        raise AssertionError("partial position review must not be accepted")


def test_unexpected_coin_does_not_count_as_review():
    text = json.dumps({"new_signals": [], "position_updates": [_update("DOGE")]})
    try:
        ga._parse_position_response(text, set(), {"BTC"}, {})
    except ValueError as exc:
        assert "BTC" in str(exc)
    else:
        raise AssertionError("unexpected coin must not satisfy BTC review")


def test_response_schema_requires_exact_position_count():
    schema = ga._response_schema(position_count=3)
    pos = schema["properties"]["position_updates"]
    assert pos["minItems"] == 3
    assert pos["maxItems"] == 3


def test_max_notional_cap_is_not_in_runtime_advisor():
    assert not hasattr(ga, "MAX_NOTIONAL_INR")


def test_retry_response_extracts_from_parsed_api_json():
    raw = {"candidates": [{"content": {"parts": [{"text": json.dumps({"new_signals": [], "position_updates": [_update("ETH")]})}]}}]}
    assert ga.extract_text(raw) is not None

def test_source_contains_recovery_merge_and_no_response_json_bug():
    from pathlib import Path
    src = Path(ga.__file__).read_text(encoding="utf-8")
    assert "retry_text = extract_text(retry_raw)" in src
    assert "final_updates = dict(initial_updates)" in src
    assert "final_updates.update(recovered_updates)" in src
    assert "retry_raw.json()" not in src
