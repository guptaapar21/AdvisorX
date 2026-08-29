"""Observational portfolio concentration diagnostics.

No notional cap. No automatic risk reduction. This only exposes correlated
position clusters so later research can measure whether apparent trade count
is really one market-beta bet repeated across coins.
"""
from __future__ import annotations

BETA_CLUSTERS = {
    "BTC_BETA": {
        "BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "LTC", "LINK", "TRX",
        "AVAX", "HEI", "BICO", "HYPE", "ZEC", "ZBT", "ADA", "ACE",
    },
    "GOLD": {"PAXG"},
}


def cluster_for_coin(coin: str) -> str:
    value = str(coin).upper()
    for name, coins in BETA_CLUSTERS.items():
        if value in coins:
            return name
    return "OTHER"


def summarize(open_positions):
    counts = {}
    for position in open_positions or []:
        key = (cluster_for_coin(position.get("coin")), str(position.get("direction", "")).upper())
        counts[key] = counts.get(key, 0) + 1
    return [
        {"cluster": key[0], "direction": key[1], "positions": value}
        for key, value in sorted(counts.items())
    ]
