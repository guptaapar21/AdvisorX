"""Production launcher for AdvisorX's trend scanner.

V4 keeps Gemini as the directional/setup decision-maker while Python owns the
final execution safety layer. New TAKE decisions pass deterministic entry-
quality and portfolio-concentration gates, and open positions receive the
progressive profit-protection policy before normal management actions.
"""
from __future__ import annotations

import html
import importlib.util
import os
import re
from pathlib import Path

import requests
import gemini_advisor
import portfolio_risk
from entry_quality_gate import apply_entry_quality_gate
from advisorx_trade_management_policy import (
    DEFAULT_MIN_TIGHTEN_R,
    add_signal_provenance,
    apply_management_policy,
    apply_profit_ladder,
    update_mfe_mae_until_exit,
)
from v2_live_integration import record_cycle, record_position_exits

V2_CYCLE = {"signals": None, "flagged": None, "open_positions": None, "recorded": False}


def _management_prompt_addendum() -> str:
    return """
V4 ENTRY-QUALITY DISCIPLINE:
A strong trend is not automatically a strong entry. Judge direction and entry
location separately. Do not buy merely because ADX, EMA alignment, RVOL,
positive momentum, or a BOS looks strong. Before TAKE, explicitly consider:
- current position inside the recent range;
- distance/extension from the latest structural break;
- break freshness and follow-through;
- failed breaks and liquidity sweeps;
- exhaustion evidence;
- room to the next opposing structure.
A fresh BOS can still be a bad entry when it is the terminal expansion of an
already exhausted move. Conversely, an extreme range location may still be
acceptable only when the break is genuinely fresh, clean, near the break, and
free of meaningful exhaustion/failure evidence.

CONVICTION CALIBRATION:
6 = acceptable/defendable, 7 = strong, 8 = very strong, 9-10 = rare.
Do not cluster at 6 merely because a trade is possible. High conviction does
not override a poor entry location.

TRADE-MANAGEMENT DISCIPLINE:
Treat an open trade in four mental states: VALID, CAUTION, PROFIT_PROTECTION,
and INVALIDATED. CAUTION is not an exit by itself. exit_now is for genuine
structural invalidation. tighten_stop should be used only after meaningful
profit; Python may suppress premature or non-monotonic adjustments.
"""


def _augment_gemini_prompt() -> None:
    marker = "V4 ENTRY-QUALITY DISCIPLINE:"
    prompt = gemini_advisor.SYSTEM_PROMPT
    # Remove the old instruction that entry-location/freshness variables are
    # strictly observational; V4 promotes the repeatedly validated failure
    # modes into the live decision process.
    prompt = prompt.replace(
        "The entry_quality_context, recent_signal_context, and entry_location_telemetry are observational diagnostics: use them as evidence, but do not apply a hard BOS-age, extension, re-entry, session, or entry-location rule. We are collecting this telemetry to test which variables actually predict outcomes.",
        "The entry_quality_context, recent_signal_context, and entry_location_telemetry are now decision-relevant evidence. Treat BOS age, extension, re-entry, entry location, exhaustion, failed breaks, and liquidity sweeps as material parts of the TAKE/SKIP judgment. Python still performs the final deterministic execution gate."
    )
    prompt = prompt.replace(
        "ENTRY-QUALITY DIAGNOSTICS: do not treat positive momentum + high RVOL + BOS as sufficient by themselves, but also do not turn the current freshness/extension/re-entry/session measurements into a hard rule yet. Treat them as additional evidence when judging the setup and record the relevant supporting/risk tags. We are explicitly testing whether entry location and continuation freshness improve expectancy before changing conviction or filtering trades.",
        "ENTRY-QUALITY: positive momentum + high RVOL + BOS are not sufficient by themselves. Treat location, freshness, extension, re-entry, exhaustion, failed-break and sweep evidence as material decision inputs. A trend can be correct while the current entry is poor."
    )
    if marker not in prompt:
        prompt += "\n\n" + _management_prompt_addendum()
    else:
        prompt += "\n" + _management_prompt_addendum()
    gemini_advisor.SYSTEM_PROMPT = prompt


_augment_gemini_prompt()
_original_get_trade_suggestions_batch = gemini_advisor.get_trade_suggestions_batch


def _quality_checked_batch(signals, scorecard=None, open_positions=None):
    ok, flagged, position_updates = _original_get_trade_suggestions_batch(signals, scorecard, open_positions)
    if not ok:
        V2_CYCLE.update({"signals": None, "flagged": None, "open_positions": None, "recorded": False})
        return ok, flagged, position_updates

    entry_rejected = apply_entry_quality_gate(flagged, signals)
    add_signal_provenance(flagged)
    portfolio_rejected = portfolio_risk.apply_execution_gate(flagged, open_positions or [])
    if entry_rejected or portfolio_rejected:
        print(
            "  V4 execution gate: "
            f"entry_quality_rejects={entry_rejected} | portfolio_rejects={portfolio_rejected}"
        )

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
_SPEC = importlib.util.spec_from_file_location("_advisorx_trend_alignment_scanner_impl", _SCANNER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load scanner from {_SCANNER_PATH}")
_scanner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_scanner)
_scanner.get_trade_suggestions_batch = _quality_checked_batch


def _update_position_telemetry_compat(ledger, fetched, now=None):
    return update_mfe_mae_until_exit(ledger, fetched)


_scanner.update_position_telemetry = _update_position_telemetry_compat
_original_apply_position_updates = _scanner.apply_position_updates


def _telemetry_position_updates(ledger, position_updates, current_prices, now):
    try:
        open_positions = _scanner.build_open_position_context(ledger, current_prices, now, {})
        record_position_exits(open_positions, position_updates)
    except Exception as exc:
        print(f"  V4 exit telemetry WARNING: {exc}")
    try:
        profit_actions = apply_profit_ladder(ledger, current_prices, now, fetched=None)
        for item in profit_actions:
            print(f"  V4 profit management: {item}")
    except Exception as exc:
        print(f"  V4 profit management WARNING: {exc}")
    return _original_apply_position_updates(ledger, position_updates, current_prices, now)


_scanner.apply_position_updates = _telemetry_position_updates
_original_build_message = _scanner._build_message


def _record_v2_before_message():
    if V2_CYCLE["recorded"] or V2_CYCLE["signals"] is None:
        return
    try:
        summary = record_cycle(V2_CYCLE["signals"], V2_CYCLE["flagged"] or {}, V2_CYCLE["open_positions"] or [])
        V2_CYCLE["recorded"] = True
        print(
            "  V2/V4 funnel: "
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
        fallback["reply_markup"] = markup if False else reply_markup
    fallback_response = requests.post(url, json=fallback, timeout=15)
    if fallback_response.ok:
        print("Telegram plain-text fallback delivered.")
        return
    fallback_detail = fallback_response.text[:2000]
    raise RuntimeError(
        "Telegram HTML and plain-text fallback both failed: "
        f"HTML={response.status_code} {detail}; fallback={fallback_response.status_code} {fallback_detail}"
    )


_scanner.send_telegram = _safe_send_telegram
TELEGRAM_CHUNK_LIMIT = 3800


def _split_telegram_message(text: str, max_chars: int = TELEGRAM_CHUNK_LIMIT):
    lines = str(text).splitlines(); chunks = []; current = ""
    for line in lines:
        candidate = line if not current else f"{current}\n{line}"
        if len(candidate) <= max_chars:
            current = candidate; continue
        if current: chunks.append(current)
        while len(line) > max_chars:
            chunks.append(line[:max_chars]); line = line[max_chars:]
        current = line
    if current: chunks.append(current)
    return chunks or [""]


def _flush_pending_telegram_chunked(state):
    pending = state.get("pending_telegram")
    if not pending: return True
    text = pending.get("text", ""); markup = pending.get("reply_markup"); chunks = _split_telegram_message(text)
    for index, chunk in enumerate(chunks):
        try:
            _safe_send_telegram(chunk, markup if index == len(chunks)-1 else None)
        except Exception as exc:
            state["pending_telegram"] = {"text":"\n".join(chunks[index:]),"reply_markup":markup,"chunk_index":index,"chunk_count":len(chunks)}
            try: _scanner.save_state(state)
            except Exception as save_exc: print(f"Failed to persist Telegram retry state: {save_exc}")
            raise RuntimeError(f"Telegram delivery failed on chunk {index+1}/{len(chunks)}: {exc}") from exc
    state["pending_telegram"] = None; state["last_sent_at"] = _scanner.now_utc().isoformat(); state["last_telegram_chunk_count"] = len(chunks)
    return True


_scanner._flush_pending_telegram = _flush_pending_telegram_chunked
main = _scanner.main

if __name__ == "__main__":
    main()
