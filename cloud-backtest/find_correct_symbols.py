"""
Diagnostic tool, not a guess: uses the real, published `coindcx`
PyPI package (an independent, community-maintained wrapper around
CoinDCX's actual public API) to pull the FULL list of active futures
instruments directly from CoinDCX, then searches it for anything
matching the 6 symbols that failed under our own fetcher's default
"B-{symbol}_USDT" format (BZ, BLESS, XAU, XAG, CL, NATGAS).

This settles the question with real data instead of another guessed
symbol string: if a matching instrument exists, this prints its EXACT
pair string as CoinDCX itself defines it. If nothing matches, that's
real evidence these products aren't available through this public
instruments list at all (consistent with the PAXG-succeeded-but-XAU-
failed pattern already found - a different product line, not a naming
typo).

Usage: python3 find_correct_symbols.py
"""
from coindcx import Client

TARGETS = ["BZ", "BLESS", "XAU", "XAG", "CL", "NATGAS"]

client = Client()

print("Fetching full active USDT-margined futures instrument list from CoinDCX...")
instruments = client.get_active_instruments(["USDT"])
print(f"Total active USDT instruments returned: {len(instruments)}\n")

for target in TARGETS:
    print(f"=== Searching for: {target} ===")
    matches = [inst for inst in instruments if target.upper() in str(inst).upper()]
    if matches:
        for m in matches:
            print(f"  MATCH FOUND: {m}")
    else:
        print(f"  No match found in the active instruments list for '{target}'.")
    print()

print("If a symbol shows no match above, it's not in CoinDCX's public active-instruments")
print("list at all right now - that's real evidence it's either delisted, not yet listed,")
print("or served through a completely separate product/endpoint (matches the pattern")
print("already found: PAXG - a real crypto token - worked fine, XAU/XAG/CL/NATGAS did not).")

# SECOND TEST, added after seeing a real DAILY chart for XAU with months
# of genuine price history - that directly contradicts "not enough
# history yet" as the explanation for the 1m fetch failures. New
# hypothesis: these symbols may have real data at coarser resolutions
# (1h, 1d) but genuinely no 1-MINUTE candle data at all - commodities-
# style synthetic products may simply update too infrequently for
# CoinDCX to generate 1m aggregates, even while daily/hourly data is
# completely normal. Testing directly instead of guessing again.
print("\n" + "=" * 70)
print("INTERVAL TEST: does 1m specifically fail while coarser intervals work?")
print("=" * 70)
from coindcx_fetcher import fetch_coindcx_klines
from datetime import datetime, timedelta, timezone

now = datetime.now(timezone.utc)
start_30d = (now - timedelta(days=30)).date().isoformat()

for target in TARGETS:
    print(f"\n--- {target} ---")
    for interval in ["1m", "5m", "15m", "1h", "1d"]:
        try:
            candles = fetch_coindcx_klines(target, interval, start_30d, now.isoformat(), stagger_delay=False)
            print(f"  {interval}: SUCCESS - {len(candles)} candles returned")
        except Exception as e:
            print(f"  {interval}: FAILED - {e}")

