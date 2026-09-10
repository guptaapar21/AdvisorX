# AdvisorX Market Behaviour Research

AdvisorX no longer runs an autonomous ResearchLab. The production scanner remains the decision engine. This module only records market behaviour and waits for the future to label what happened next.

## What is captured

The recorder fetches closed 1-minute CoinDCX futures candles for the live scanner universe and identifies deterministic events such as:

- long lower/upper wick rejection near a recent 30-minute support/resistance level;
- failed downside/upside breaks and reclaim/failure;
- unusually high activity around a level;
- a candle-direction signed-volume **proxy** for delta and possible absorption.

The recorder stores the underlying OHLC, volume, wick geometry, ATR, activity percentile, recent levels, event family and proxy-flow fields. The flow proxy is explicitly labelled `candle_direction_signed_volume`; it is not trade-level aggressor delta/CVD.

## Future labelling

Each event is kept pending until enough closed candles exist to measure 5m, 15m, 30m, 60m, 120m and 180m outcomes. Each outcome records:

- forward return;
- direction-signed return;
- maximum favourable excursion;
- maximum adverse excursion;
- whether the reference level held;
- whether price reclaimed the reference level.

This produces evidence such as: “high-activity negative-flow rejection at support led to +0.62% median 30-minute signed return across N observations”, rather than asking a model to invent a strategy before the evidence exists.

## Storage model

Data is written to:

```text
cloud-backtest/market_behavior_research/
  observations/YYYY-MM-DD/HH.jsonl
  outcomes/YYYY-MM-DD/HH.jsonl
cloud-backtest/market_behavior_state.json
```

Observation and outcome files are append-only hourly shards. Pending events are kept only in the small state file. Old shards are pruned from the active tree after 90 days. No monolithic research JSONL is rewritten, so the previous GitHub 100 MB failure mode is avoided.

The recorder is intentionally fail-soft: a market-data fetch failure is logged in the state output and does not make the production scanner fail.
