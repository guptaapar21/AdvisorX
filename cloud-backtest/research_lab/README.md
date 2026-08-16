# AdvisorX ResearchLab

Research-only subsystem. It is deliberately isolated from AdvisorX's live/advisory trading path.

## Hard boundaries

- No imports/calls to `trend_alignment_scanner.py`, `gemini_advisor.py`, Telegram, risk/trading functions, or `trend_scanner_state.json`.
- Research hypotheses never change AdvisorX decisions.
- ResearchLab has independent state and data storage.
- Features at observation time `T` use only candles whose timestamps are `<= T`.
- Outcomes use only strictly later candles and are created only after the endpoint candle is closed.
- Maximum forward horizon is 60 minutes; the walk-forward purge is also 60 minutes.
- Discovery data is the only market/outcome data shown to Gemini while hypotheses are generated.
- Validation and holdout are evaluated by Python after discovery; holdout is never sent to Gemini for hypothesis generation.

## Runtime

The workflow runs every 5 minutes because GitHub Actions cron does not provide a 1-minute schedule. Every run fetches a rolling 75-minute 1m window and backfills all closed minutes in that window, so a 5-minute trigger still captures one-minute observations. A separate external `workflow_dispatch` trigger can increase invocation frequency without changing the code.

Candle storage is retained for `RESEARCH_CANDLE_RETENTION_DAYS` days (default 7). Unresolved observations are revisited from stored candles, so delayed CI does not silently lose matured outcomes.

## Research learning loop

Every minute observation becomes one immutable observation row. Each observation can later gain 5/10/15/30/60-minute outcomes. Discovery groups these horizons back into one observation before Gemini research, preventing five horizons from masquerading as five independent observations.

Gemini receives historical research memory as context, but it does not update model weights. The persistent memory contains only prior research conclusions; new hypotheses are generated from the current discovery split and are evaluated separately on validation and holdout data.

A hypothesis can be `VALIDATION_FAILED`, `VALIDATION_PASSED`, or `HOLDOUT_PASSED`. Only `HOLDOUT_PASSED` is retained as validated research memory. Nothing is connected to AdvisorX trading.

## Operational persistence

Research data is intended for the dedicated `researchlab-data` branch rather than `main`, to avoid high-frequency research-data commits colliding with AdvisorX source/state commits. The workflow serializes ResearchLab runs with a concurrency group.
