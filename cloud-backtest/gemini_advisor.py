"""Calls Gemini ONCE per scan cycle with EVERY tracked coin's raw data
batched into a single request - no pre-filtering, no staged
qualification, no direction supplied. Gemini decides everything
itself: whether a coin is worth mentioning at all, which direction,
and (if flagged) entry/stop/target/size.

Explicitly built this way per direct instruction: "don't give our
logic to it... let gemini itself decide what to do." Called once per
new 3m candle (~480 times/day across a 24h day), well within the
budget sized against the available keys (12 keys x ~1000/day free
tier each, per Flash-Lite's published limits).

IMPORTANT CONTEXT for whoever reads this: none of this has been
backtested for what happens *after* Gemini flags something - no known
win rate, no known typical move size. This is Gemini's independent
judgment on raw data, not a validated edge. See
trend_alignment_scanner_issue_summary.md and the XRP incident in this
project's history for why that distinction matters.

Model and key rotation deliberately mirror geminiKeys.js /
geminiAgent.js in this same repo - same GEMINI_API_KEYS secret, same
model, same key-rotation-on-429 behavior - reusing infrastructure
already proven to work here.
"""
import json
import os
import time

import requests

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_TIMEOUT_SECONDS = 30  # longer than the single-signal version's 30s - one call now covers multiple coins' worth of candle data
MAX_LOSS_RUPEES = 1500
# CoinDCX standard futures taker fee - confirmed directly from
# coindcx.com's own blog (not a third-party estimate), checked August
# 2026: "The standard maker fee for futures trading on CoinDCX is
# 0.025%, while the standard taker fee is 0.075%." Taker rate used
# here (not maker) since a fast intraday reversal trade needs
# immediate fills on both entry and exit, not resting limit orders
# that might not get filled in time. Re-verify if CoinDCX changes
# pricing - this directly affects whether a low-conviction trade's
# potential profit even survives round-trip costs.
TAKER_FEE_RATE = float(os.environ.get("TAKER_FEE_RATE", "0.00075"))
# Crypto futures prices/contract values are quoted in USDT, while the
# risk budget and user-facing amounts are INR. Never silently assume a
# 1:1 conversion. Configure this explicitly in the workflow environment.
USDT_INR_RATE = float(os.environ.get("USDT_INR_RATE", "0"))
MAX_NOTIONAL_INR = float(os.environ.get("MAX_NOTIONAL_INR", "100000"))
MIN_RR = float(os.environ.get("MIN_RR", "1.5"))
MAX_STOP_PCT = float(os.environ.get("MAX_STOP_PCT", "0.08"))

SYSTEM_PROMPT = """Trade-scanning assistant for crypto futures on CoinDCX. You receive a JSON object with up to three parts every scan cycle (roughly every 3 minutes): "coins" - an array of ALL currently tracked coins, NOT pre-filtered by any trend/strength logic, "open_positions" - trades you (a prior call, not this one - you have no memory) previously flagged that are still open and unresolved, needing a hold/exit/adjust decision this cycle, and optionally "recent_performance_last_24h" - real, code-verified outcomes of your own recent calls over the trailing 24 hours. This performance data is NOT something you tracked yourself - you have no memory between calls - it's computed independently from actual subsequent price action and current market prices, so trust it completely; it is ground truth about how your recent judgment has actually performed, not a self-report.

Each entry in "open_positions" gives you: coin, direction, the entry/stop/target it was opened with, your own original reasoning for that call (verbatim, so you can judge whether that thesis still holds), how many minutes it's been open, current price, and current unrealized P&L in INR. This is a genuinely different judgment from scanning "coins" for new setups: here you're asking "is my original thesis still valid, given what price has actually done since" - not "is this a good entry right now."

Fields in recent_performance_last_24h: total (all calls in the window), target_hit (genuinely reached the target price), stop_hit (genuinely reached the stop price), expired (never reached either within 2 hours, closed at whatever price it was at when the window ran out - a real but different kind of outcome than a clean target/stop hit), pending (still open, unresolved), abandoned_or_invalid (calls superseded by a later reversal on the same coin, or excluded due to malformed data - not a performance signal either way). realized_pnl_inr is money already locked in (target_hit + stop_hit + expired combined, net of round-trip fees). unrealized_pnl_inr is a live mark-to-market estimate on still-pending positions - not locked in, can still move either way. total_pnl_inr is both combined.

If recent performance has been poor (more stops/expiries than targets, negative realized P&L), that's a real reason to be MORE selective this cycle, not something to disregard because "this setup is different."

For each coin in "coins" you get only raw, descriptive data: latest close, ATR (volatility), RVOL and its percentile rank against that coin's own history, plain momentum (% price change over the last 5/20/60 3m candles - just arithmetic, not an indicator), and TIERED candle history at decreasing resolution the further back in time it goes - candles_3m_last_1h (full 3m detail, most recent hour), candles_15m_last_7h (next several hours), candles_1h_rest_of_24h (rest of the day), and candles_1d_last_30d (daily candles, 30-day context). No direction, trend verdict, or strategy signal is supplied, and no rule is given for which numbers should agree or how - that judgment, entirely, is yours to make.

Only flag genuinely high-quality, high-conviction opportunities. Do not flag marginal, borderline, or small setups just because one number happens to look elevated - that produces noise, not useful signals. Flagging nothing is the correct, expected outcome most cycles; only flag when you'd actually stand behind it. take_trade: true is a higher bar than simply being worth mentioning - if you're flagging a coin mainly because something looks unusual but you're not genuinely confident, set take_trade: false and say so, rather than defaulting to true.

Some coins will include a "prior_call" field - your own most recent flag on that coin, with direction, whether you took it, how long ago, and how price has actually moved since (price_change_pct_since, moved_favorably). You have no memory of issuing that call - this is the only way you can see your own track record. When present, genuinely weigh it: if price has moved favorably, that's real evidence your prior thesis may still be playing out; if unfavorably, treat that as a real reason to reconsider, not something to ignore. If you reverse a recent call, say so explicitly in reasoning and explain what changed your view - don't reverse silently or contradict yourself without acknowledging it.

For each coin you DO include:
1. coin: the coin symbol, exactly as given
2. direction: "long" or "short" - your own call, nothing was supplied
3. take_trade: true/false
4. conviction: integer 1-10, how strong you genuinely believe this setup is - NOT a formality. 10 is reserved for the rare case where everything lines up cleanly; most real flags should be well below that. This directly controls how much capital gets risked, so an inflated conviction score puts real money at risk on a call you're not actually confident about.
5. reasoning: 1-2 sentences, reference specific numbers/candle structure actually given for THIS coin
6. entry_price, stop_loss, target_price: numbers - fill these even if take_trade is false (a best-guess reference level), so a skip is still comparable data next cycle

Do NOT compute trade_amount_inr yourself - position sizing is calculated separately from your conviction score, scaled down from a maximum of {max_loss} INR risk. A low-conviction flag should risk meaningfully less than a high-conviction one; that scaling is handled outside your response, driven entirely by the conviction number you give.

No backtested win-rate exists for any of this - it's your independent judgment on raw data, not a validated edge.

For each entry in "open_positions" you get, decide one action:
1. "hold" - thesis still looks valid, no change. Leave updated_stop_loss/updated_target_price null.
2. "exit_now" - thesis has broken down (invalidated by what price/candles have actually done since entry, not just "it hasn't hit target yet") - this closes the position immediately at current market price, code-side, the moment you return this. Only use this when you'd genuinely rather be flat than keep holding - it is a real, immediate exit, not a soft warning.
3. "tighten_stop" - thesis still valid but you want to reduce risk; give updated_stop_loss (updated_target_price can stay null).
4. "move_target" - thesis still valid but you want to adjust the profit objective; give updated_target_price (updated_stop_loss can stay null).
Always include reasoning for the action, referencing what's actually changed since entry (or explicitly that nothing has and it's still holding). Do not use exit_now or tighten_stop just because a position is currently at a small unrealized loss - that's normal noise, not thesis invalidation; use it only when the specific reason you entered no longer holds.

Respond with ONLY a JSON object, no markdown fences, no other text, in this exact shape:
{{"new_signals": [{{"coin": "string", "direction": "long"|"short", "take_trade": bool, "conviction": integer, "reasoning": "string", "entry_price": number, "stop_loss": number, "target_price": number}}, ...], "position_updates": [{{"coin": "string", "direction": "long"|"short", "action": "hold"|"exit_now"|"tighten_stop"|"move_target", "updated_stop_loss": number|null, "updated_target_price": number|null, "reasoning": "string"}}, ...]}}
new_signals is empty if nothing this cycle meets your own bar for quality (the normal case). position_updates must include exactly one entry for every coin given in "open_positions" - never omit one, since a missing entry there means it silently keeps whatever it already has with no signal either way.""".format(max_loss=MAX_LOSS_RUPEES)


def get_gemini_keys():
    """Same key-source pattern as geminiKeys.js: GEMINI_API_KEYS
    (comma-separated) is the one actually wired into this repo's
    GitHub Actions secrets (confirmed via agent.yml), plus the
    individually-numbered and single-key fallbacks for parity."""
    keys = []
    if os.environ.get("GEMINI_API_KEYS"):
        keys.extend(k.strip() for k in os.environ["GEMINI_API_KEYS"].split(",") if k.strip())
    for i in range(1, 31):
        k = os.environ.get(f"GEMINI_API_KEY_{i}")
        if k:
            keys.append(k.strip())
    if os.environ.get("GEMINI_API_KEY"):
        keys.append(os.environ["GEMINI_API_KEY"].strip())
    return list(dict.fromkeys(keys))  # de-duplicate, preserve order


def build_batch_prompt(signals, scorecard=None, open_positions=None):
    """signals: list of raw coin snapshots from build_coin_snapshot -
    every tracked coin, every cycle, NOT pre-filtered. No direction,
    no trend verdict, no prescribed combination of which numbers
    should agree - just raw descriptive numbers, the tiered candle
    context, and (when available) this coin's own prior-call context
    so Gemini can genuinely self-correct rather than reason blind.
    scorecard: optional aggregate stats over the trailing 24h -
    computed deterministically by code, not something Gemini tracks
    or is asked to remember itself. open_positions: optional list of
    dicts (from build_open_position_context in the caller) - the
    still-pending ledger entries Gemini itself previously flagged,
    given back with their original entry/stop/target/reasoning plus
    current price/P&L so Gemini can judge whether to hold, exit, or
    adjust - a genuinely separate decision from scanning "coins" for
    fresh setups."""
    payload = []
    for s in signals:
        entry = {
            "coin": s["coin"],
            "close": s.get("close"),
            "atr14_3m": s.get("atr14_3m"),
            "rvol": s.get("rvol"),
            "rvol_label": s.get("rvol_label"),
            "rvol_percentile": s.get("rvol_percentile"),
            "momentum_pct_5_3m": s.get("momentum_pct_5_3m"),
            "momentum_pct_20_3m": s.get("momentum_pct_20_3m"),
            "momentum_pct_60_3m": s.get("momentum_pct_60_3m"),
            "candles_3m_last_1h": s.get("ctx_3m", []),
            "candles_15m_last_7h": s.get("ctx_15m", []),
            "candles_1h_rest_of_24h": s.get("ctx_1h", []),
            "candles_1d_last_30d": s.get("ctx_daily_30d", []),
        }
        if s.get("prior_call"):
            entry["prior_call"] = s["prior_call"]
        payload.append(entry)
    wrapped = {"coins": payload}
    if open_positions:
        wrapped["open_positions"] = open_positions
    if scorecard:
        wrapped["recent_performance_last_24h"] = scorecard
    return json.dumps(wrapped)


def call_gemini_once(api_key, user_prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    resp = requests.post(
        url,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        json={
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"thinkingConfig": {"thinkingLevel": "medium"}},
        },
        timeout=GEMINI_TIMEOUT_SECONDS,
    )
    if resp.status_code == 429:
        err = RuntimeError(f"Gemini rate limited (429): {resp.text[:200]}")
        err.rate_limited = True
        raise err
    if resp.status_code == 503:
        err = RuntimeError("Gemini temporarily overloaded (503)")
        err.rate_limited = True
        raise err
    resp.raise_for_status()
    return resp.json()


def extract_text(response_json):
    try:
        return response_json["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return None


def _normalize_item(item):
    """Coerces a single parsed response item into the types the rest
    of the pipeline actually assumes - confirmed as real gaps, not
    theoretical: a numeric field returned as a string (e.g.
    entry_price: "55.12") crashes downstream arithmetic; take_trade
    returned as the STRING "false" is truthy in Python, silently
    displaying TAKE when SKIP was meant; and direction returned with
    different casing (e.g. "LONG" vs "long") silently inverts the
    favorable/unfavorable and whipsaw-warning comparisons downstream,
    which both use exact string equality against "long"."""
    normalized = dict(item)
    for field in ("entry_price", "stop_loss", "target_price"):
        if normalized.get(field) is not None:
            normalized[field] = float(normalized[field])
    if "take_trade" in normalized:
        val = normalized["take_trade"]
        normalized["take_trade"] = val if isinstance(val, bool) else str(val).strip().lower() == "true"
    if normalized.get("direction"):
        normalized["direction"] = str(normalized["direction"]).strip().lower()
    return normalized


def _validate_trade_geometry(parsed):
    """Deterministic safety gate for Gemini's proposed trade levels.

    Gemini supplies judgment; Python owns the non-negotiable geometry.
    Returns (ok, reason)."""
    direction = parsed.get("direction")
    entry = parsed.get("entry_price")
    stop = parsed.get("stop_loss")
    target = parsed.get("target_price")

    if direction not in {"long", "short"}:
        return False, "invalid direction"
    if any(v is None for v in (entry, stop, target)):
        return False, "entry/stop/target missing"
    if any(not isinstance(v, (int, float)) or not (v > 0) for v in (entry, stop, target)):
        return False, "entry/stop/target must be positive numbers"

    if direction == "long":
        if not (stop < entry < target):
            return False, "LONG requires stop < entry < target"
    else:
        if not (target < entry < stop):
            return False, "SHORT requires target < entry < stop"

    stop_pct = abs(entry - stop) / entry
    if stop_pct > MAX_STOP_PCT:
        return False, f"stop distance {stop_pct:.2%} exceeds MAX_STOP_PCT {MAX_STOP_PCT:.2%}"

    reward = abs(target - entry)
    risk = abs(entry - stop)
    rr = reward / risk if risk > 0 else 0
    if rr < MIN_RR:
        return False, f"risk/reward {rr:.2f} below MIN_RR {MIN_RR:.2f}"

    return True, "ok"


def _compute_position_size(parsed):
    """Convert INR risk into crypto quantity using an explicit USDT/INR rate.

    Risk is always calculated in INR, quantity is in coin units, and
    notional/fees are converted back to INR. If the FX rate is not configured,
    the trade is forced to SKIP rather than silently using a wrong unit.
    """
    conviction = parsed.get("conviction")
    try:
        conviction = int(conviction)
    except (TypeError, ValueError):
        conviction = None
    if conviction is None or not (1 <= conviction <= 10):
        print(f"  Gemini WARNING: {parsed.get('coin')} conviction missing/invalid "
              f"({parsed.get('conviction')}), defaulting to lowest (1) rather than risking the full amount")
        conviction = 1
    parsed["conviction"] = conviction

    max_loss_inr = round(MAX_LOSS_RUPEES * conviction / 10, 2)
    parsed["max_loss_this_trade_inr"] = max_loss_inr

    ok, reason = _validate_trade_geometry(parsed)
    if not ok:
        parsed["take_trade"] = False
        parsed["risk_validation_error"] = reason
        parsed["trade_amount_inr"] = None
        print(f"  Risk gate: {parsed.get('coin')} forced to SKIP - {reason}")
        return parsed

    if USDT_INR_RATE <= 0:
        parsed["take_trade"] = False
        parsed["risk_validation_error"] = "USDT_INR_RATE is not configured"
        parsed["trade_amount_inr"] = None
        print(f"  Risk gate: {parsed.get('coin')} forced to SKIP - USDT_INR_RATE is not configured")
        return parsed

    entry = float(parsed["entry_price"])
    stop = float(parsed["stop_loss"])
    target = float(parsed["target_price"])
    risk_per_unit_inr = abs(entry - stop) * USDT_INR_RATE
    quantity = max_loss_inr / risk_per_unit_inr
    trade_amount_inr = quantity * entry * USDT_INR_RATE
    parsed["quantity"] = quantity
    parsed["trade_amount_inr"] = round(trade_amount_inr, 2)

    if trade_amount_inr > MAX_NOTIONAL_INR:
        parsed["take_trade"] = False
        parsed["risk_validation_error"] = (
            f"notional ₹{trade_amount_inr:,.2f} exceeds MAX_NOTIONAL_INR ₹{MAX_NOTIONAL_INR:,.2f}"
        )
        print(f"  Risk gate: {parsed.get('coin')} forced to SKIP - {parsed['risk_validation_error']}")

    gross_profit_inr = quantity * abs(target - entry) * USDT_INR_RATE
    round_trip_fee_inr = trade_amount_inr * TAKER_FEE_RATE * 2
    net_profit_inr = gross_profit_inr - round_trip_fee_inr
    parsed["gross_profit_at_target_inr"] = round(gross_profit_inr, 2)
    parsed["estimated_fee_inr"] = round(round_trip_fee_inr, 2)
    parsed["net_profit_at_target_inr"] = round(net_profit_inr, 2)

    if gross_profit_inr > 0:
        fee_drag_pct = round(round_trip_fee_inr / gross_profit_inr * 100, 1)
        parsed["fee_drag_pct"] = fee_drag_pct
        if fee_drag_pct > 50 and parsed.get("take_trade"):
            parsed["take_trade"] = False
            parsed["fee_override"] = True
            parsed["risk_validation_error"] = "round-trip fees exceed 50% of gross target profit"
            print(f"  Fee check: {parsed.get('coin')} forced to SKIP - fees consume {fee_drag_pct}% of target profit")

    return parsed

def _normalize_position_update(item):
    """Same coercion purpose as _normalize_item, for the smaller
    position_updates schema: action as a bare lowercase string
    (guards against 'Exit_Now' / 'EXIT_NOW' silently failing an exact
    equality check downstream), updated_stop_loss/updated_target_price
    as floats or None."""
    normalized = dict(item)
    if normalized.get("action"):
        normalized["action"] = str(normalized["action"]).strip().lower()
    if normalized.get("direction"):
        normalized["direction"] = str(normalized["direction"]).strip().lower()
    for field in ("updated_stop_loss", "updated_target_price"):
        val = normalized.get(field)
        normalized[field] = float(val) if val not in (None, "", "null") else None
    return normalized


def get_trade_suggestions_batch(signals, scorecard=None, open_positions=None):
    """ONE Gemini call covering every signal given, PLUS a hold/exit/
    adjust review of every still-open position handed in via
    open_positions. Returns (new_signals, position_updates) - two
    dicts keyed by the 'coin' field Gemini echoed back, matching by
    name rather than trusting array order, since a model could in
    principle reorder or drop an entry.

    new_signals: {coin: suggestion} for fresh setups from "coins" -
    same shape/meaning as this function returned before open-position
    review existed.
    position_updates: {coin: update} for entries from open_positions -
    always expected to cover every coin passed in, but callers should
    still treat a missing coin key as 'no update available' (default
    to holding), never as an implicit exit.

    Returns (success, new_signals, position_updates). success=False means
    Gemini was unavailable/invalid and the caller must not treat that as
    a successful no-signal cycle. scorecard: optional dict from compute_scorecard -
    real trades-in/target-hit/stop-hit/net-P&L over the trailing
    window, computed deterministically by code (not asked of Gemini,
    which has no memory and no way to verify it)."""
    if not signals:
        return True, {}, {}
    keys = get_gemini_keys()
    if not keys:
        print("  Gemini: no keys configured (GEMINI_API_KEYS), skipping batch suggestion")
        return False, {}, {}

    user_prompt = build_batch_prompt(signals, scorecard, open_positions)
    expected_coins = {s["coin"] for s in signals}
    expected_position_coins = {p["coin"] for p in open_positions} if open_positions else set()
    last_error = None

    # Overall time budget across ALL key attempts combined - not just
    # a per-key timeout. With a 30s per-call timeout, the 90s budget
    # permits up to three full attempts before stopping key rotation.
    # This prevents a transient outage from consuming the workflow timeout.
    # workflow's 10-minute total timeout. A hard GH Actions kill at
    # that point skips save_state entirely, which is exactly the
    # failure mode the send/save-ordering fix elsewhere was built to
    # survive - this closes the gap for the case a timeout kill
    # bypasses that protection altogether.
    RETRY_BUDGET_SECONDS = 90
    batch_start = time.time()

    for key in keys:
        if time.time() - batch_start > RETRY_BUDGET_SECONDS:
            print(f"  Gemini: retry budget ({RETRY_BUDGET_SECONDS}s) exhausted, stopping key rotation early")
            break
        try:
            raw = call_gemini_once(key, user_prompt)
            text = extract_text(raw)
            if not text:
                last_error = "empty response"
                continue
            text = text.strip()
            if text.startswith("```"):
                text = text.strip("`").lstrip("json").strip()
            parsed = json.loads(text)

            # Backward-compat: a bare array (the pre-open-position-
            # review response shape) is treated as new_signals only,
            # with no position_updates - so an old/misbehaving model
            # response doesn't hard-fail the whole cycle.
            if isinstance(parsed, list):
                new_signals_raw, position_updates_raw = parsed, []
            elif isinstance(parsed, dict):
                new_signals_raw = parsed.get("new_signals", [])
                position_updates_raw = parsed.get("position_updates", [])
                if not isinstance(new_signals_raw, list) or not isinstance(position_updates_raw, list):
                    last_error = "new_signals/position_updates were not arrays"
                    continue
            else:
                last_error = f"expected a JSON object or array, got {type(parsed).__name__}"
                continue

            result = {}
            for item in new_signals_raw:
                coin = item.get("coin")
                if coin not in expected_coins:
                    print(f"  Gemini: response included unexpected coin '{coin}', ignoring")
                    continue
                try:
                    item = _normalize_item(item)
                    result[coin] = _compute_position_size(item)
                except (TypeError, ValueError) as e:
                    # Isolated per-item - one malformed item (e.g. a
                    # numeric field returned as an unparseable string)
                    # no longer discards the WHOLE batch response for
                    # this key attempt, confirmed directly as a real
                    # risk: a string-typed number used to crash
                    # _math_check, and since this was inside the outer
                    # per-key try/except, it silently threw away every
                    # OTHER valid coin's suggestion too.
                    print(f"  Gemini: skipping malformed item for '{coin}' ({e})")
                    continue

            position_updates = {}
            for item in position_updates_raw:
                coin = item.get("coin")
                if coin not in expected_position_coins:
                    print(f"  Gemini: position update for unexpected coin '{coin}', ignoring")
                    continue
                try:
                    position_updates[coin] = _normalize_position_update(item)
                except (TypeError, ValueError) as e:
                    print(f"  Gemini: skipping malformed position update for '{coin}' ({e})")
                    continue
            missing_positions = expected_position_coins - set(position_updates)
            if missing_positions:
                # Not fatal - caller defaults a missing coin to "hold"
                # - but surfaced since the prompt explicitly asks for
                # one entry per open position and a gap here is worth
                # knowing about.
                print(f"  Gemini: no position update returned for {sorted(missing_positions)}, defaulting to hold")

            # No "missing coins" warning for new_signals - Gemini is
            # EXPECTED to omit most coins every cycle under this
            # design (only flagging ones worth mentioning), so an
            # empty or partial result relative to the full input list
            # is normal, not a sign of a failed/incomplete response.
            print(f"  Gemini: flagged {len(result)} of {len(expected_coins)} coins this cycle, "
                  f"reviewed {len(position_updates)} of {len(expected_position_coins)} open positions")
            return True, result, position_updates
        except json.JSONDecodeError as e:
            last_error = f"unparseable JSON: {e}"
            continue
        except Exception as e:
            last_error = str(e)
            if getattr(e, "rate_limited", False):
                time.sleep(1)
            continue

    print(f"  Gemini: all {len(keys)} key(s) failed for batch call - {last_error}")
    return False, {}, {}
