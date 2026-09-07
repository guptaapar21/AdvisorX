"""Production launcher for AdvisorX's trend scanner.

This wrapper preserves the existing scanner/V2 telemetry architecture while
fixing three concrete live-path problems identified in the current ledger:

1) V2 entry-quality diagnostics are observational; the wrapper no longer
   hard-vetoes Gemini TAKE decisions with entry_quality_gate.py.
2) Gemini tighten_stop requests are executable only after the trade has earned
   at least +0.50R. Genuine exit_now requests are never blocked by this rule.
3) MFE/MAE telemetry stops at the first target/stop event visible in the 1m
   data, preventing post-exit candles from contaminating outcome analytics.

The hard deterministic geometry gate remains in gemini_advisor.py.
"""
from __future__ import annotations

import html
import importlib.util
import os
import re
from pathlib import Path

import requests
import gemini_advisor
from advisorx_trade_management_policy import (
    DEFAULT_MIN_TIGHTEN_R,
    add_signal_provenance,
    apply_management_policy,
    update_mfe_mae_until_exit,
)
from v2_live_integration import record_cycle, record_position_exits

V2_CYCLE = {"signals": None, "flagged": None, "open_positions": None, "recorded": False}


def _management_prompt_addendum() -> str:
    return """
TRADE-MANAGEMENT DISCIPLINE — IMPORTANT:
Treat an open trade in four mental states: VALID, CAUTION, PROFIT_PROTECTION,
and INVALIDATED.

- VALID: the original structural thesis still holds. Prefer HOLD.
- CAUTION: indicators may be weakening, but there is no structural failure.
  CAUTION is NOT a reason to exit merely because momentum, ADX, EMA position,
  RVOL, or candle shape became less favorable.
- PROFIT_PROTECTION: meaningful favorable excursion has already been earned.
  Protect it mechanically and progressively rather than repeatedly tightening
  on small fluctuations.
- INVALIDATED: the original directional thesis has genuinely failed through
  structural evidence, e.g. an opposing confirmed BOS/CHoCH or decisive loss of
  the level that made the trade thesis valid. This is where exit_now belongs.

Do not convert ordinary drawdown or indicator deterioration into exit_now.
Do not request tighten_stop before the trade has earned meaningful profit.
A tighten_stop is normally appropriate only after at least +0.50R gross has
been earned; before that, HOLD unless a genuine structural invalidation calls
for exit_now.

For new entries, judge combinations rather than isolated signals. A BOS is
not automatically strong when it is already near a range extreme, materially
extended from the break, accompanied by repeated failed breaks/liquidity
sweeps, or high exhaustion. These are contextual risk factors for Gemini's
judgment, not Python hard filters.

CONVICTION CALIBRATION:
6 = acceptable/defendable; 7 = strong; 8 = exceptional; 9-10 = rare.
Do not cluster almost every trade at 6 merely because a trade is possible.
"""


def _relax_gemini_selectivity() -> None:
    prompt = gemini_advisor.SYSTEM_PROMPT
    marker = "REGIME-ADAPTIVE SELECTIVITY:"
    if marker not in prompt:
        prompt += (
            "\n\nREGIME-ADAPTIVE SELECTIVITY: Use TREND_UP/TREND_DOWN as environments "
            "where valid continuation trades can occur without textbook perfection. "
            "Use RANGE for boundary trades and BREAKOUT_TRANSITION/BREAKDOWN_TRANSITION "
            "for fresh structural breaks. Use EXHAUSTION/UNCLEAR more selectively. "
            "The goal is selective trading, not zero trading.\n"
        )
    prompt += "\n" + _management_prompt_addendum()
    gemini_advisor.SYSTEM_PROMPT = prompt


_relax_gemini_selectivity()
_original_get_trade_suggestions_batch = gemini_advisor.get_trade_suggestions_batch


def _quality_checked_batch(signals, scorecard=None, open_positions=None):
    ok, flagged, position_updates = _original_get_trade_suggestions_batch(
        signals, scorecard, open_positions
    )
    if not ok:
        V2_CYCLE.update({"signals": None, "flagged": None, "open_positions": None, "recorded": False})
        return ok, flagged, position_updates

    # Entry-quality V2 diagnostics are observational by architecture. The
    # deterministic risk/geometry gate inside gemini_advisor remains active.
    add_signal_provenance(flagged)
    position_updates = apply_management_policy(
        position_updates,
        open_positions or [],
        min_tighten_r=float(os.environ.get("MIN_PROFIT_R_TO_TIGHTEN", DEFAULT_MIN_TIGHTEN_R)),
    )

    V2_CYCLE["signals"] = signals
    V2_CYCLE["flagged"] = flagged
    V2_CYCLE["open_positions"] = open_positions
    V2_CYCLE["recorded"] = False
    return ok, flagged, position_updates


_SCANNER_PATH = Path(__file__).with_name("trend_alignment_scanner.py")
_SPEC = importlib.util.spec_from_file_location(
    "_advisorx_trend_alignment_scanner_impl",
    _SCANNER_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load scanner from {_SCANNER_PATH}")
_scanner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_scanner)

# Gemini call interception: observational V2 telemetry + management policy.
_scanner.get_trade_suggestions_batch = _quality_checked_batch

# Replace the pre-resolution MFE/MAE update so it cannot see candles after the
# first actual target/stop event in the available 1m series. The production
# scanner calls this hook with (ledger, fetched, now); keep that exact runtime
# signature while the policy implementation remains a two-argument helper.
def _update_position_telemetry_compat(ledger, fetched, now=None):
    return update_mfe_mae_until_exit(ledger, fetched)


_scanner.update_position_telemetry = _update_position_telemetry_compat

_original_apply_position_updates = _scanner.apply_position_updates


def _telemetry_position_updates(ledger, position_updates, current_prices, now):
    try:
        open_positions = _scanner.build_open_position_context(ledger, current_prices, now, {})
        record_position_exits(open_positions, position_updates)
    except Exception as exc:
        print(f"  V2 exit telemetry WARNING: {exc}")
    return _original_apply_position_updates(
        ledger, position_updates, current_prices, now
    )


_scanner.apply_position_updates = _telemetry_position_updates

_original_build_message = _scanner._build_message


def _record_v2_before_message():
    if V2_CYCLE["recorded"] or V2_CYCLE["signals"] is None:
        return
    try:
        summary = record_cycle(
            V2_CYCLE["signals"],
            V2_CYCLE["flagged"] or {},
            V2_CYCLE["open_positions"] or [],
        )
        V2_CYCLE["recorded"] = True
        print(
            "  V2 funnel: "
            f"records={len(summary['records'])} | buckets={summary['buckets']} | "
            f"portfolio={summary['portfolio']}"
        )
    except Exception as exc:
        print(f"  V2 telemetry WARNING: {exc}")


def _build_message_with_v2(*args, **kwargs):
    _record_v2_before_message()
    return _original_build_message(*args, **kwargs)


_scanner._build_message = _build_message_with_v2


def _safe_send_telegram(text, reply_markup=None, parse_mode="HTML"):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID env vars")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    safe_html = re.sub(r"<(?!/?b>)", "&lt;", str(text))
    payload = {"chat_id": chat_id, "text": safe_html}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    response = requests.post(url, json=payload, timeout=15)
    if response.ok:
        return
    detail = response.text[:2000]
    print(f"Telegram HTML delivery failed ({response.status_code}): {detail}")
    plain = html.unescape(re.sub(r"</?b>", "", str(text), flags=re.IGNORECASE))
    fallback = {"chat_id": chat_id, "text": plain}
    if reply_markup:
        fallback["reply_markup"] = reply_markup
    fallback_response = requests.post(url, json=fallback, timeout=15)
    if fallback_response.ok:
        print("Telegram plain-text fallback delivered.")
        return
    fallback_detail = fallback_response.text[:2000]
    raise RuntimeError(
        "Telegram HTML and plain-text fallback both failed: "
        f"HTML={response.status_code} {detail}; "
        f"fallback={fallback_response.status_code} {fallback_detail}"
    )


_scanner.send_telegram = _safe_send_telegram
TELEGRAM_CHUNK_LIMIT = 3800


def _split_telegram_message(text: str, max_chars: int = TELEGRAM_CHUNK_LIMIT):
    lines = str(text).splitlines()
    chunks = []
    current = ""
    for line in lines:
        candidate = line if not current else f"{current}\n{line}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        while len(line) > max_chars:
            chunks.append(line[:max_chars])
            line = line[max_chars:]
        current = line
    if current:
        chunks.append(current)
    return chunks or [""]


def _flush_pending_telegram_chunked(state):
    pending = state.get("pending_telegram")
    if not pending:
        return True
    text = pending.get("text", "")
    markup = pending.get("reply_markup")
    chunks = _split_telegram_message(text)
    for index, chunk in enumerate(chunks):
        try:
            _safe_send_telegram(chunk, markup if index == len(chunks) - 1 else None)
        except Exception as exc:
            state["pending_telegram"] = {
                "text": "\n".join(chunks[index:]),
                "reply_markup": markup,
                "chunk_index": index,
                "chunk_count": len(chunks),
            }
            try:
                _scanner.save_state(state)
            except Exception as save_exc:
                print(f"Failed to persist Telegram retry state: {save_exc}")
            raise RuntimeError(
                f"Telegram delivery failed on chunk {index + 1}/{len(chunks)}: {exc}"
            ) from exc
    state["pending_telegram"] = None
    state["last_sent_at"] = _scanner.now_utc().isoformat()
    state["last_telegram_chunk_count"] = len(chunks)
    return True


_scanner._flush_pending_telegram = _flush_pending_telegram_chunked
main = _scanner.main

if __name__ == "__main__":
    main()
