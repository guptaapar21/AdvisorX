# Market Behaviour Research

AdvisorX records closed 1-minute CoinDCX futures behaviour for later empirical analysis.

The recorder is evidence-only. It does not alter live trading decisions and is not sent to Gemini.

It stores deterministic observations for support/resistance rejection, long-wick rejection, failed breaks and volume/delta-proxy absorption, then measures future outcomes at fixed forward horizons.

`delta_proxy` is candle-direction signed volume. It is not exchange-provided aggressor-side delta/CVD.

Research data is sharded by UTC hour under `market_behavior_research/` to avoid monolithic-file growth.