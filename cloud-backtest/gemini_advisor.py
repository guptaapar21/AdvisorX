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
GEMINI_TIMEOUT_SECONDS = 45  # longer than the single-signal version's 30s - one call now covers multiple coins' worth of candle data
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
TAKER_FEE_RATE = 0.00075

SYSTEM_PROMPT = """Trade-scanning assistant for crypto futures on CoinDCX. You receive a JSON object with two parts every scan cycle (roughly every 3 minutes): "coins" - an array of ALL currently tracked coins, NOT pre-filtered by any trend/strength logic, and optionally "recent_performance_last_1h" - real, code-verified outcomes of your own recent calls (total flagged, how many actually hit target, how many hit stop, how many are still pending, and net INR result). This performance data is NOT something you tracked yourself - you have no memory between calls - it's computed independently from actual subsequent price action, so trust it completely; it is ground truth about how your recent judgment has actually performed, not a self-report. If recent performance has been poor (more stops than targets, negative net), that's a real reason to be MORE selective this cycle, not something to disregard because "this setup is different."

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

Respond with ONLY a JSON array - empty if nothing this cycle meets your own bar for quality, containing only the coins worth mentioning otherwise, no markdown fences, no other text:
[{{"coin": "string", "direction": "long"|"short", "take_trade": bool, "conviction": integer, "reasoning": "string", "entry_price": number, "stop_loss": number, "target_price": number}}, ...]""".format(max_loss=MAX_LOSS_RUPEES)


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


def build_batch_prompt(signals, scorecard=None):
    """signals: list of raw coin snapshots from build_coin_snapshot -
    every tracked coin, every cycle, NOT pre-filtered. No direction,
    no trend verdict, no prescribed combination of which numbers
    should agree - just raw descriptive numbers, the tiered candle
    context, and (when available) this coin's own prior-call context
    so Gemini can genuinely self-correct rather than reason blind.
    scorecard: optional aggregate stats over the trailing 1h -
    computed deterministically by code, not something Gemini tracks
    or is asked to remember itself."""
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
    if scorecard:
        wrapped["recent_performance_last_1h"] = scorecard
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


def _compute_position_size(parsed):
    """Position size is calculated here, not by Gemini - it no longer
    even attempts trade_amount_inr. Directly addresses a real design
    flaw: previously every flagged trade was sized to the SAME full
    max loss regardless of how confident Gemini actually was, which
    doesn't make sense as risk management - a marginal, low-conviction
    flag was risking exactly as much capital as a high-conviction one.
    Now risk scales with Gemini's own stated conviction (1-10): a
    conviction of 3 risks 30% of MAX_LOSS_RUPEES, not the full amount."""
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

    entry = parsed.get("entry_price")
    stop = parsed.get("stop_loss")
    max_loss_this_trade = round(MAX_LOSS_RUPEES * conviction / 10, 2)
    parsed["max_loss_this_trade_inr"] = max_loss_this_trade

    trade_amount_inr = None
    if entry is not None and stop is not None and entry != stop:
        # quantity (units) needed so a stop-loss hit loses
        # ~max_loss_this_trade, converted to the rupee position size
        # by multiplying by entry price - not the quantity number
        # itself. Confirmed directly in an earlier round: quantity
        # alone (e.g. ~375,000 units) is not a rupee amount; the real
        # position size is quantity * entry.
        quantity = max_loss_this_trade / abs(entry - stop)
        trade_amount_inr = round(quantity * entry, 2)
    parsed["trade_amount_inr"] = trade_amount_inr

    # Fee-drag check: does the potential profit even survive real
    # transaction costs? This is EXACTLY the recurring pattern that
    # has killed every backtested idea in this project's history -
    # "gross positive, net negative due to fees" - and low-conviction
    # trades are the most exposed, since a small position size can
    # mean the round-trip fee eats most or all of a modest profit
    # target regardless of whether the direction call is even right.
    # CoinDCX's own published standard futures taker fee is 0.075%
    # (confirmed directly from coindcx.com) - applied on BOTH entry
    # and exit since a fast intraday reversal trade needs immediate
    # fills, not resting limit orders that might not get filled.
    target = parsed.get("target_price")
    if entry is not None and target is not None and trade_amount_inr and quantity:
        gross_profit_inr = round(quantity * abs(target - entry), 2)
        round_trip_fee_inr = round(trade_amount_inr * TAKER_FEE_RATE * 2, 2)
        net_profit_inr = round(gross_profit_inr - round_trip_fee_inr, 2)
        parsed["gross_profit_at_target_inr"] = gross_profit_inr
        parsed["estimated_fee_inr"] = round_trip_fee_inr
        parsed["net_profit_at_target_inr"] = net_profit_inr
        if gross_profit_inr > 0:
            fee_drag_pct = round(round_trip_fee_inr / gross_profit_inr * 100, 1)
            parsed["fee_drag_pct"] = fee_drag_pct
            if fee_drag_pct > 50 and parsed.get("take_trade"):
                print(f"  Fee check: {parsed.get('coin')} take_trade forced to False - "
                      f"round-trip fees (\u20b9{round_trip_fee_inr}) would consume {fee_drag_pct}% "
                      f"of the \u20b9{gross_profit_inr} gross profit at target")
                parsed["take_trade"] = False
                parsed["fee_override"] = True
    return parsed


def get_trade_suggestions_batch(signals, scorecard=None):
    """ONE Gemini call covering every signal given. Returns a dict
    {coin: suggestion}, keyed by the 'coin' field Gemini echoed back -
    matching by name rather than trusting array order, since a model
    could in principle reorder or drop an entry. Returns {} if signals
    is empty, no keys are configured, or every key fails - callers
    should treat a missing coin key as 'no suggestion available',
    never as an implicit skip/take decision. scorecard: optional dict
    from compute_scorecard - real trades-in/target-hit/stop-hit/net-P&L
    over the trailing window, computed deterministically by code (not
    asked of Gemini, which has no memory and no way to verify it)."""
    if not signals:
        return {}
    keys = get_gemini_keys()
    if not keys:
        print("  Gemini: no keys configured (GEMINI_API_KEYS), skipping batch suggestion")
        return {}

    user_prompt = build_batch_prompt(signals, scorecard)
    expected_coins = {s["coin"] for s in signals}
    last_error = None

    # Overall time budget across ALL key attempts combined - not just
    # a per-key timeout. Confirmed directly: 12 keys x 45s each could
    # take up to 540s (9 min) worst case, dangerously close to this
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
            parsed_list = json.loads(text)
            if not isinstance(parsed_list, list):
                last_error = f"expected a JSON array, got {type(parsed_list).__name__}"
                continue

            result = {}
            for item in parsed_list:
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

            # No "missing coins" warning here - Gemini is EXPECTED to
            # omit most coins every cycle under this design (only
            # flagging ones worth mentioning), so an empty or partial
            # result relative to the full input list is normal, not a
            # sign of a failed/incomplete response.
            print(f"  Gemini: flagged {len(result)} of {len(expected_coins)} coins this cycle")
            return result
        except json.JSONDecodeError as e:
            last_error = f"unparseable JSON: {e}"
            continue
        except Exception as e:
            last_error = str(e)
            if getattr(e, "rate_limited", False):
                time.sleep(1)
            continue

    print(f"  Gemini: all {len(keys)} key(s) failed for batch call - {last_error}")
    return {}
