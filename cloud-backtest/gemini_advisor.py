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
USDT_INR_RATE = float(os.environ.get("USDT_INR_RATE", "99.44"))
# There is deliberately NO maximum notional/capital cap. Position size is
# derived solely from the conviction-scaled INR loss budget and the structural
# stop distance. A wider/narrower stop therefore changes quantity naturally.
MIN_RR = float(os.environ.get("MIN_RR", "1.5"))
MAX_STOP_PCT = float(os.environ.get("MAX_STOP_PCT", "0.08"))
MIN_STOP_ATR_MULTIPLIER = float(os.environ.get("MIN_STOP_ATR_MULTIPLIER", "1.2"))

SYSTEM_PROMPT = """You are the final discretionary trade decision-maker for CoinDCX futures.

Python does NOT choose the trade direction for you. It supplies deterministic market evidence for every tracked coin. You must synthesize that evidence, decide whether a trade exists, choose long/short yourself, and say SKIP when the evidence is not strong enough.

WHAT YOU RECEIVE FOR EACH COIN
- Current price, ATR, RVOL and that coin's historical RVOL percentile.
- Raw 3m/15m/1h/daily candles.
- Confirmed swing highs/lows with HH/HL/LH/LL labels.
- Exact BOS/CHoCH events, confirmation times, break levels and break closes.
- Post-break follow-through measurements.
- Trend phase, exhaustion flags and multi-timeframe structure.
- Range high/low/midpoint/width, touch counts, location and efficiency ratio.
- Liquidity sweeps/reclaims and failed breaks.
- Candle body/range, upper/lower wicks, close location, volume acceleration and momentum acceleration.
- BTC reference context, universe breadth and each coin's relative strength versus BTC.
- Prior call context and the independently computed trailing-24h performance scorecard.

STRUCTURE FIRST
Inspect confirmed pivots rather than arbitrary visible highs/lows. Use HH/HL/LH/LL sequence, BOS and CHoCH, break timing, follow-through and multi-timeframe agreement. A 3m move against 15m/1h structure is countertrend and needs substantially stronger evidence. Do not invent levels.

TREND MODE
For continuation trades, require coherent structure and enough room to the next opposing structural level. EMA/ADX/VWAP/RSI/volume are evidence, not automatic triggers. Do not chase an extended move merely because ADX or RVOL is high.

BREAKOUT MODE
Distinguish wick/sweep from a genuine close through structure. Prefer a close beyond the level with meaningful volume/momentum and actual follow-through, or a successful retest. A wick alone is not a breakout. If follow-through is weak or the move immediately fails, downgrade or SKIP.

RANGE MODE
A genuine range is a valid swing-trading regime. LONGs are preferred near the lower range boundary after rejection/sweep-and-reclaim; SHORTs near the upper boundary after rejection/sweep-and-rejection. Avoid the middle of the range unless there is a very specific structural edge. Targets may be midpoint first and opposite boundary second, but the final target must be realistic and fee-aware. The stop should sit beyond the actual invalidation/sweep level with sensible volatility room, not at an arbitrary distance. If the range is too narrow, unstable, or fees consume too much of the expected move, SKIP.

EXHAUSTION/REVERSAL
Look for new extremes with weakening momentum/ADX, momentum acceleration turning against the move, rejection wicks, failed breaks, loss of EMA structure, or an opposite structural break. A reversal trade needs actual evidence of failure/reclaim; do not fade a strong trend simply because RSI is overbought/oversold.

TARGET AND STOP
Choose entry, stop and target from structure. The target must have genuine room. Minimum RR is enforced by Python, but a mathematically acceptable RR does not make a random target valid. For range trades, use the actual range/swing levels where appropriate.

RISK
Maximum loss budget is ₹1500 at conviction 10 and scales linearly with conviction. There is NO ₹1,00,000 maximum-notional cap. Do not invent a capital limit. Python calculates quantity from the INR risk budget and actual stop distance. Python enforces direction geometry, maximum stop percentage, minimum stop distance of {atr_mult}x 3m ATR, minimum RR {min_rr}, and fee-drag protection.

FEES
Judge whether the expected gross move is large enough to justify round-trip taker fees. Do not prefer tiny range moves that are mostly consumed by costs.

CROSS-MARKET CONTEXT
BTC and breadth are context, not hard filters. A coin materially outperforming BTC can support a long thesis; materially underperforming can support a short thesis. But strong coin-specific structure can override broad-market direction when the evidence is clear.

RECENT PERFORMANCE
The supplied 24h scorecard is ground truth for this tool's recent calls. If recent performance is poor, become more selective. Do not blindly increase activity to recover losses.

OPEN POSITIONS
For every supplied open position, return exactly one action: hold, exit_now, tighten_stop, or move_target. Judge whether the original thesis still holds using current structure, range, momentum and level validity. Do not exit merely because of normal noise. An exit_now is a real immediate close. A stop can only be tightened in the safe direction; a target must remain beyond current price.

OUTPUT
For every NEW signal include coin, direction, take_trade, conviction 1-10, reasoning, entry_price, stop_loss, target_price, market_regime, trade_type, market_location, key_level_used, invalidation_reason and setup_quality.
For every OPEN position include coin, direction, action, updated_stop_loss, updated_target_price and reasoning.
Return ONLY the required JSON object. Empty new_signals is normal and preferred over weak trades.
""".format(atr_mult=MIN_STOP_ATR_MULTIPLIER, min_rr=MIN_RR)


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
            "momentum_acceleration_5_3m": s.get("momentum_acceleration_5_3m"),
            "momentum_acceleration_20_3m": s.get("momentum_acceleration_20_3m"),
            "candle_body_pct": s.get("candle_body_pct"),
            "candle_range_pct": s.get("candle_range_pct"),
            "upper_wick_pct": s.get("upper_wick_pct"),
            "lower_wick_pct": s.get("lower_wick_pct"),
            "body_to_range": s.get("body_to_range"),
            "close_location_in_range": s.get("close_location_in_range"),
            "volume_acceleration_3": s.get("volume_acceleration_3"),
            "market_structure": s.get("market_structure", {}),
            "relative_strength_vs_btc": s.get("relative_strength_vs_btc", {}),
            "market_context": s.get("market_context", {}),
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


def _validate_trade_geometry(parsed, atr14_3m=None):
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

    if atr14_3m is not None:
        try:
            atr = float(atr14_3m)
        except (TypeError, ValueError):
            atr = 0.0
        if atr > 0:
            min_stop_distance = atr * MIN_STOP_ATR_MULTIPLIER
            actual_stop_distance = abs(entry - stop)
            if actual_stop_distance < min_stop_distance:
                return False, (
                    f"stop distance {actual_stop_distance:.8g} is below "
                    f"{MIN_STOP_ATR_MULTIPLIER:.2f}x 3m ATR ({min_stop_distance:.8g})"
                )

    return True, "ok"


def _compute_position_size(parsed, atr14_3m=None):
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

    ok, reason = _validate_trade_geometry(parsed, atr14_3m=atr14_3m)
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
    atr_by_coin = {s["coin"]: s.get("atr14_3m") for s in signals}
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
                    result[coin] = _compute_position_size(item, atr14_3m=atr_by_coin.get(coin))
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
