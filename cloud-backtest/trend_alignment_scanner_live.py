"""Production launcher for the live trend scanner.

Loads the known-good trend_alignment_scanner.py without circular imports,
patches its local Gemini function so the entry-quality gate actually runs,
and patches its existing Telegram sender to safely handle literal '<' text.
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

# IMPORTANT: the real scanner imported get_trade_suggestions_batch directly,
# so patch the scanner's local reference, not only gemini_advisor.
_scanner.get_trade_suggestions_batch = _quality_checked_batch


def _safe_send_telegram(text, reply_markup=None, parse_mode="HTML"):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID env vars"
        )

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    # Preserve deliberate <b> tags while escaping all other literal '<'.
    safe_html = re.sub(r"<(?!/?b>)", "&lt;", str(text))
    payload = {"chat_id": chat_id, "text": safe_html}
    if reply_markup:
        payload["reply_markup"] = reply_markup

    response = requests.post(url, json=payload, timeout=15)
    if response.ok:
        return

    detail = response.text[:2000]
    print(
        f"Telegram HTML delivery failed ({response.status_code}): {detail}"
    )

    # Plain-text fallback keeps the same message and buttons.
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


# Keep the scanner's existing pending_telegram -> flush -> save flow.
_scanner.send_telegram = _safe_send_telegram

main = _scanner.main

if __name__ == "__main__":
    main()
