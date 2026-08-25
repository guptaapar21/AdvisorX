"""Production launcher for the CoinDCX trend scanner.

This module intentionally does NOT import ``trend_alignment_scanner`` by its
module name.  The production file is loaded under a private module name so
there is no circular-import collision with this launcher.

Order:
1. wrap Gemini's batch decision function;
2. load the real scanner under a private module name;
3. replace only the scanner's existing Telegram sender;
4. call the scanner's existing main().

The scanner keeps ownership of state, ledger, Telegram queue, and persistence.
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


# ---------------------------------------------------------------------------
# 1) Wrap Gemini batch decisions with the deterministic entry-quality gate.
# ---------------------------------------------------------------------------
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


gemini_advisor.get_trade_suggestions_batch = _quality_checked_batch


# ---------------------------------------------------------------------------
# 2) Load the real scanner under a PRIVATE module name.
#
#    Do not use:
#        from trend_alignment_scanner import main
#
#    because the launcher itself is executed next to that file and that
#    creates the circular import shown in the failed GitHub Actions run.
# ---------------------------------------------------------------------------
_SCANNER_PATH = Path(__file__).with_name("trend_alignment_scanner.py")
_SPEC = importlib.util.spec_from_file_location(
    "_trend_alignment_scanner_impl",
    _SCANNER_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load scanner module from {_SCANNER_PATH}")

_scanner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_scanner)


# ---------------------------------------------------------------------------
# 3) Patch the scanner's EXISTING Telegram sender.
#
#    The scanner already owns pending_telegram and _flush_pending_telegram().
#    We do not add another Telegram workflow/path.
#
#    The concrete failure observed in Actions was Telegram HTTP 400 caused by
#    literal '<0.5R ...' being parsed as malformed HTML.
# ---------------------------------------------------------------------------
def _safe_send_telegram(text, reply_markup=None, parse_mode="HTML"):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID env vars"
        )

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    # Keep deliberate <b> / </b> tags but escape every other '<'.
    safe_html = re.sub(r"<(?!/?b>)", "&lt;", str(text))
    payload = {
        "chat_id": chat_id,
        "text": safe_html,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup

    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.ok:
            return

        detail = response.text[:2000]
        print(
            f"Telegram HTML delivery failed "
            f"({response.status_code}): {detail}"
        )
        raise RuntimeError(
            f"Telegram {response.status_code}: {detail}"
        )

    except Exception as html_error:
        # Formatting fallback: remove HTML tags and send as plain text.
        plain = html.unescape(
            re.sub(r"</?b>", "", str(text), flags=re.IGNORECASE)
        )

        fallback_payload = {
            "chat_id": chat_id,
            "text": plain,
        }
        if reply_markup:
            fallback_payload["reply_markup"] = reply_markup

        response = requests.post(
            url,
            json=fallback_payload,
            timeout=15,
        )

        if response.ok:
            print(
                "Telegram plain-text fallback delivered after HTML failure: "
                f"{html_error}"
            )
            return

        detail = response.text[:2000]
        raise RuntimeError(
            "Telegram HTML and plain-text fallback both failed: "
            f"{response.status_code}: {detail}"
        ) from html_error


_scanner.send_telegram = _safe_send_telegram


# ---------------------------------------------------------------------------
# 4) Run the REAL scanner.
# ---------------------------------------------------------------------------
main = _scanner.main


if __name__ == "__main__":
    main()
