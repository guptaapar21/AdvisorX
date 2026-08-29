"""Production launcher with observational V2 telemetry integration.

The underlying scanner, Gemini advisor, entry-quality gate, Telegram delivery,
ledger/state handling, and external scheduling architecture remain intact.

IMPORTANT: the Gemini selectivity rewrite in ``_relax_gemini_selectivity`` is
existing production behavior already present in the current ``main`` branch.
It is preserved here verbatim and is NOT a V2 telemetry change. Any decision
to alter that prompt policy must be reviewed and shipped as a separate
strategy change. The V2 additions below only observe the resulting production
decisions and never veto, create, size, or modify trades.
"""
from __future__ import annotations

import html
import importlib.util
import os
import re
from pathlib import Path

import requests
import gemini_advisor
from entry_quality_gate import apply_entry_quality_gate
from v2_live_integration import record_cycle, record_position_exits


# References captured during one production cycle. The production scanner later
# enriches `flagged` in-place (entry-location telemetry, recent-signal context,
# etc.). We intentionally wait until _build_message(), after those mutations,
# before writing V2 telemetry.
_V2_CYCLE = {
    "signals": None,
    "flagged": None,
    "open_positions": None,
    "recorded": False,
}


def _relax_gemini_selectivity() -> None:
    prompt = gemini_advisor.SYSTEM_PROMPT
    replacements = [
        (
            'If recent performance has been poor (more stops/expiries than targets, negative realized P&L), '
            "that's a real reason to be MORE selective this cycle, not something to disregard because "
            '"this setup is different."',
            'If recent performance has been poor, become more selective about marginal setups, but do not '
            'suppress otherwise valid fresh continuation setups. Recent performance is a weighting factor, '
            'not a blanket no-trade condition.',
        ),
        (
            'Mixed or contradictory evidence should result in SKIP.',
            'Material contradiction should result in SKIP, especially when the execution timeframe (3m) '
            'and confirmation timeframe (15m) directly oppose the proposed direction. A neutral or lagging '
            '1h timeframe alone is not a reason to SKIP a fresh 3m/15m continuation.',
        ),
        (
            'Only flag genuinely high-quality, high-conviction opportunities. Do not flag marginal, borderline, '
            'or small setups just because one number happens to look elevated - that produces noise, not useful '
            'signals. Flagging nothing is the correct, expected outcome most cycles; only flag when you\'d actually '
            'stand behind it. take_trade: true is a higher bar than simply being worth mentioning - if you\'re '
            'flagging a coin mainly because something looks unusual but you\'re not genuinely confident, set '
            'take_trade: false and say so, rather than defaulting to true.',
            'Only flag setups with a real directional edge, but do not require every timeframe and indicator '
            'to be perfect. In a clear TREND_UP/TREND_DOWN regime, a fresh continuation with aligned 3m '
            'structure, supportive 15m structure (bullish/bearish or neutral), momentum/volume confirmation, '
            'and adequate room is a valid trade even when the 1h is lagging. In BREAKOUT_TRANSITION or '
            'BREAKDOWN_TRANSITION, a fresh structural break with confirmation is valid even if higher-timeframe '
            'structure has not caught up yet. In RANGE and EXHAUSTION regimes remain more selective. Do not '
            'manufacture trades, but do not turn a strong trend into SKIP merely because it is not a textbook '
            'perfect alignment. take_trade: true should mean the setup has a defendable edge, not that every '
            'possible feature agrees.',
        ),
        (
            'new_signals is empty if nothing this cycle meets your own bar for quality (the normal case).',
            'new_signals is empty when nothing this cycle has a defendable edge. In strong directional regimes, '
            'valid continuation opportunities are expected when the important evidence aligns.',
        ),
    ]
    for old, new in replacements:
        if old in prompt:
            prompt = prompt.replace(old, new)
    marker = "REGIME-ADAPTIVE SELECTIVITY:"
    if marker not in prompt:
        prompt += (
            "\n\nREGIME-ADAPTIVE SELECTIVITY: "
            "Use TREND_UP/TREND_DOWN as environments where good continuation trades "
            "should be allowed rather than requiring rare textbook perfection. "
            "Use RANGE as a boundary-trading environment. Use BREAKOUT_TRANSITION/"
            "BREAKDOWN_TRANSITION for fresh structural breaks. Use EXHAUSTION or "
            "UNCLEAR as high-selectivity environments. The goal is selective trading, "
            "not zero trading.\n"
        )
    gemini_advisor.SYSTEM_PROMPT = prompt


_relax_gemini_selectivity()
_original_get_trade_suggestions_batch = gemini_advisor.get_trade_suggestions_batch


def _quality_checked_batch(signals, scorecard=None, open_positions=None):
    """Run the existing Gemini + Python quality gate without changing decisions."""
    ok, flagged, position_updates = _original_get_trade_suggestions_batch(
        signals, scorecard, open_positions
    )
    if not ok:
        _V2_CYCLE["signals"] = None
        _V2_CYCLE["flagged"] = None
        _V2_CYCLE["open_positions"] = None
        _V2_CYCLE["recorded"] = False
        return ok, flagged, position_updates

    rejected = apply_entry_quality_gate(flagged, signals)
    if rejected:
        reasons = {}
        for item in flagged.values():
            reason = item.get("_entry_quality_reject_reason")
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        print(f"  Entry-quality gate: rejected {rejected} Gemini TAKE(s) | reasons={reasons}")

    # Keep object references. The scanner mutates flagged later in main() with
    # entry-location/recent-signal telemetry; _build_message() consumes it only
    # after those mutations have happened.
    _V2_CYCLE["signals"] = signals
    _V2_CYCLE["flagged"] = flagged
    _V2_CYCLE["open_positions"] = open_positions
    _V2_CYCLE["recorded"] = False
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
_scanner.get_trade_suggestions_batch = _quality_checked_batch

_original_apply_position_updates = _scanner.apply_position_updates


def _telemetry_position_updates(ledger, position_updates, current_prices, now):
    try:
        open_positions = _scanner.build_open_position_context(
            ledger, current_prices, now, {}
        )
        record_position_exits(open_positions, position_updates)
    except Exception as exc:
        print(f"  V2 exit telemetry WARNING: {exc}")
    return _original_apply_position_updates(
        ledger, position_updates, current_prices, now
    )


_scanner.apply_position_updates = _telemetry_position_updates


def _record_v2_before_message():
    if _V2_CYCLE["recorded"] or _V2_CYCLE["signals"] is None:
        return
    try:
        summary = record_cycle(
            _V2_CYCLE["signals"],
            _V2_CYCLE["flagged"] or {},
            _V2_CYCLE["open_positions"] or [],
        )
        _V2_CYCLE["recorded"] = True
        print(
            "  V2 funnel: "
            f"records={len(summary['records'])} | buckets={summary['buckets']} | "
            f"portfolio={summary['portfolio']}"
        )
    except Exception as exc:
        # Research telemetry must never disable the trading decision path.
        print(f"  V2 telemetry WARNING: {exc}")


_original_build_message = _scanner._build_message


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
