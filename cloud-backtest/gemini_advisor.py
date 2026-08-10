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

SYSTEM_PROMPT = """Trade-scanning assistant for crypto futures on CoinDCX. You receive an ARRAY of ALL currently tracked coins, every single scan cycle (roughly every 3 minutes) - NOT pre-filtered by any trend/strength logic. For each coin you get only raw, descriptive data: latest close, ATR (volatility), RVOL and its percentile rank against that coin's own history, plain momentum (% price change over the last 5/20/60 3m candles - just arithmetic, not an indicator), and TIERED candle history at decreasing resolution the further back in time it goes - candles_3m_last_1h (full 3m detail, most recent hour), candles_15m_last_7h (next several hours), candles_1h_rest_of_24h (rest of the day), and candles_1d_last_30d (daily candles, 30-day context). No direction, trend verdict, or strategy signal is supplied - decide everything yourself from the raw numbers and candle structure.

For each coin, decide independently whether it's worth flagging at all. Most coins, most cycles, will NOT be worth flagging - OMIT those entirely from your response array. Only include a coin if you genuinely see something worth a person's attention.

For each coin you DO include:
1. coin: the coin symbol, exactly as given
2. direction: "long" or "short" - your own call, nothing was supplied
3. take_trade: true/false
4. reasoning: 1-2 sentences, reference specific numbers/candle structure actually given for THIS coin
5. entry_price, stop_loss, target_price: numbers
6. trade_amount_inr: position size in INR sized so a stop_loss hit loses as close to but not exceeding {max_loss} INR as possible, computed from entry_price and stop_loss - show the arithmetic in reasoning if take_trade is true

No backtested win-rate exists for any of this - it's your independent judgment on raw data, not a validated edge.

Respond with ONLY a JSON array - empty if nothing this cycle is worth flagging, containing only the coins worth mentioning otherwise, no markdown fences, no other text:
[{{"coin": "string", "direction": "long"|"short", "take_trade": bool, "reasoning": "string", "entry_price": number, "stop_loss": number, "target_price": number, "trade_amount_inr": number}}, ...]""".format(max_loss=MAX_LOSS_RUPEES)


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


def build_batch_prompt(signals):
    """signals: list of raw coin snapshots from build_coin_snapshot -
    every tracked coin, every cycle, NOT pre-filtered. No direction,
    no trend verdict - just raw descriptive numbers plus the tiered
    candle context."""
    payload = []
    for s in signals:
        payload.append({
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
        })
    return json.dumps(payload)


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


def _math_check(parsed):
    entry = parsed.get("entry_price")
    stop = parsed.get("stop_loss")
    math_check_inr = None
    if entry is not None and stop is not None and entry != stop:
        # quantity (units) needed so a stop-loss hit loses ~MAX_LOSS_RUPEES,
        # then converted to the actual rupee position size by multiplying
        # by entry price - NOT the quantity number itself. Confirmed
        # directly: for entry=0.242, stop=0.238, quantity alone is
        # ~375,000 (units), which is NOT a rupee amount - the real
        # position size is quantity * entry = ~INR 90,750.
        quantity = MAX_LOSS_RUPEES / abs(entry - stop)
        math_check_inr = round(quantity * entry, 2)
    parsed["math_check_trade_amount"] = math_check_inr
    if math_check_inr is not None and parsed.get("trade_amount_inr") is not None:
        deviation_pct = abs(parsed["trade_amount_inr"] - math_check_inr) / math_check_inr * 100
        parsed["math_check_deviation_pct"] = round(deviation_pct, 1)
        if deviation_pct > 15:
            print(f"  Gemini WARNING: {parsed.get('coin')} trade_amount_inr "
                  f"({parsed['trade_amount_inr']}) deviates {deviation_pct:.0f}% "
                  f"from independent math check ({math_check_inr})")
    return parsed


def get_trade_suggestions_batch(signals):
    """ONE Gemini call covering every signal given. Returns a dict
    {coin: suggestion}, keyed by the 'coin' field Gemini echoed back -
    matching by name rather than trusting array order, since a model
    could in principle reorder or drop an entry. Returns {} if signals
    is empty, no keys are configured, or every key fails - callers
    should treat a missing coin key as 'no suggestion available',
    never as an implicit skip/take decision."""
    if not signals:
        return {}
    keys = get_gemini_keys()
    if not keys:
        print("  Gemini: no keys configured (GEMINI_API_KEYS), skipping batch suggestion")
        return {}

    user_prompt = build_batch_prompt(signals)
    expected_coins = {s["coin"] for s in signals}
    last_error = None

    for key in keys:
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
                result[coin] = _math_check(item)

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
