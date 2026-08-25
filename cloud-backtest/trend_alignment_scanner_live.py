"""Production launcher for the live trend scanner.

Keeps the existing scanner/Telegram/state architecture intact while adding
only the deterministic entry-quality gate and a defensive Telegram sender.
"""

from __future__ import annotations

import html
import os
import re

import requests
import gemini_advisor

from entry_quality_gate import apply_entry_quality_gate

_original = gemini_advisor.get_trade_suggestions_batch


def _wrapped(signals, scorecard=None, open_positions=None):
    ok, flagged, updates = _original(signals, scorecard, open_positions)
    if not ok:
        return ok, flagged, updates

    rejected = apply_entry_quality_gate(flagged, signals)
    if rejected:
        reasons = {}
        for item in flagged.values():
            reason = item.get("_entry_quality_reject_reason")
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        print(f"  Entry-quality gate: rejected {rejected} TAKE(s) | reasons={reasons}")
    return ok, flagged, updates


gemini_advisor.get_trade_suggestions_batch = _wrapped

from trend_alignment_scanner import main  # noqa: E402
import trend_alignment_scanner as _scanner  # noqa: E402


def _safe_send_telegram(text, reply_markup=None, parse_mode="HTML"):
    """Send scanner output without letting literal '<...' text break Telegram.

    The scanner intentionally contains HTML <b> tags, while metrics such as
    '<0.5R' are plain text. Telegram rejects the message when a literal '<'
    starts an unsupported HTML tag. Escape stray '<', log Telegram's response
    body on failure, and retry once as plain text.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID env vars")

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    # Preserve intentional <b> formatting; escape every other literal '<'.
    safe_html = re.sub(r"<(?!/?b>)", "&lt;", str(text))
    payload = {"chat_id": chat_id, "text": safe_html}
    if reply_markup:
        payload["reply_markup"] = reply_markup

    try:
        response = requests.post(url, json=payload, timeout=15)
        if not response.ok:
            detail = response.text[:2000]
            print(f"Telegram HTML delivery failed ({response.status_code}): {detail}")
            raise RuntimeError(f"Telegram {response.status_code}: {detail}")
        return
    except Exception as html_error:
        # Formatting failure fallback: plain text, same message and buttons.
        plain = html.unescape(re.sub(r"</?b>", "", str(text), flags=re.I))
        fallback = {"chat_id": chat_id, "text": plain}
        if reply_markup:
            fallback["reply_markup"] = reply_markup
        try:
            response = requests.post(url, json=fallback, timeout=15)
            if not response.ok:
                detail = response.text[:2000]
                raise RuntimeError(
                    f"Telegram fallback {response.status_code}: {detail}"
                )
            print(
                "Telegram delivered using plain-text fallback after HTML failure: "
                f"{html_error}"
            )
        except Exception as fallback_error:
            raise RuntimeError(
                f"Telegram HTML and fallback delivery failed: {fallback_error}"
            ) from fallback_error


# Patch the scanner module's existing sender. This does NOT create a second
# Telegram path; _flush_pending_telegram() continues to be the only caller.
_scanner.send_telegram = _safe_send_telegram


if __name__ == "__main__":
    main()
