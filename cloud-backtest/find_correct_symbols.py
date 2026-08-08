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
