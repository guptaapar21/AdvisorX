# AdvisorX CoinDCX Pre-Filter Lab

This is a separate research-only workflow.

It reuses the existing AdvisorX backtesting components:
- `cloud-backtest/coindcx_fetcher.py`
- `cloud-backtest/indicators.py` as the indicator reference
- `cloud-backtest/fee_model.py` as the fee reference

It does not change:
- Trend Alignment production scanner
- AdvisorX bot decisions
- existing ResearchLab
- Gemini prompts
- research memory
- risk/positions/ledger

## Purpose

Find robust three-condition pre-filters that can later be evaluated as gates before Gemini.

## Data

Fresh CoinDCX futures 1-minute candles are fetched for 15 days by default, with a configurable maximum of 30 days.

## Anti-lookahead

Features at time T use candles <= T only.

The future 15-minute close is used only as the label/outcome.

Thresholds are learned only from the discovery partition.

Chronological split:
- 60% discovery
- purge equal to the horizon
- 20% validation
- purge again
- 20% holdout

Persistent consecutive matches on the same coin are de-duplicated into separate horizon windows so the system cannot manufacture hundreds of pseudo-trades from a single continuous condition.

No candidate is automatically promoted into production.
