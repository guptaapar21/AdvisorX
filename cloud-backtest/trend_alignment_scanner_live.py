"""Production launcher for the live trend scanner.

Controlled opportunity-frequency revision:
- loads the real trend_alignment_scanner.py without circular imports;
- applies the entry-quality gate after Gemini;
- relaxes only stale/extension/entry-distance filters;
- allows a neutral 15m while requiring 3m alignment;
- treats 1h opposition as caution during transitions;
- makes Gemini selectivity regime-aware rather than globally "near-perfect";
- preserves the scanner's existing Telegram/state/ledger architecture.
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
        if old not in prompt:
            print("Gemini prompt note: expected phrase not found; leaving that phrase unchanged.")
            continue
        prompt = prompt.replace(old, new)

    # Add one explicit regime-frequency paragraph if it was not already added.
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
    ok, flagged, position_updates = _original_get_trade_suggestions_batch(
        signals, scorecard, open_positions
    )
    if not ok:
        return ok, flagged, position_updates

    rejected = apply_entry_quality_gate(flagged, signals)

    if rejected:
        reasons = {}
        for item in flagged.values():
            reason = item.get("_entry_quality_reject_reason")
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        print(
            "  Entry-quality gate: rejected "
            f"{rejected} Gemini TAKE(s) | reasons={reasons}"
        )

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

# The scanner imported the function directly; patch the LOCAL reference.
_scanner.get_trade_suggestions_batch = _quality_checked_batch


def _safe_send_telegram(text, reply_markup=None, parse_mode="HTML"):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID env vars"
        )

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

    plain = html.unescape(
        re.sub(r"</?b>", "", str(text), flags=re.IGNORECASE)
    )
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


# Telegram Bot API limits sendMessage text to 4096 characters. Split large
# reports into bounded chunks and retain only the unsent remainder on failure.
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
            current = ""

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
            _safe_send_telegram(
                chunk,
                markup if index == len(chunks) - 1 else None,
            )
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


# main() resolves _flush_pending_telegram inside the loaded scanner module,
# so replace that reference for both retry-before-scan and post-scan delivery.
_scanner._flush_pending_telegram = _flush_pending_telegram_chunked

main = _scanner.main

if __name__ == "__main__":
    main()
