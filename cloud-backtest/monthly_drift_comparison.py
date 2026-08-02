"""
Reads trades_*.csv files already committed by the "AdvisorX Cloud Backtest"
GitHub Action (backtest.yml) to cloud-backtest/results/ - does NOT fetch any
data itself and needs no network. Run this locally (or in another Action
step) after triggering backtest.yml with adverse_drift_modes=[false, true]
(the default) - that run produces paired before/after trades files per coin
in the same commit, tagged by the real use_adverse_drift column (added in
main.py), not by filename-guessing.

Usage:
    python3 monthly_drift_comparison.py results/trades_SOL_*.csv results/trades_DOGE_*.csv results/trades_ETH_*.csv

    # or just point it at everything from one workflow run:
    python3 monthly_drift_comparison.py results/trades_*.csv
"""
import sys
import glob
import pandas as pd


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 monthly_drift_comparison.py <trades_csv_path_or_glob> [more paths...]")
        sys.exit(1)

    paths = []
    for arg in sys.argv[1:]:
        matched = glob.glob(arg)
        if not matched:
            print(f"Warning: no files matched '{arg}'")
        paths.extend(matched)

    if not paths:
        print("No files found - nothing to analyze.")
        sys.exit(1)

    print(f"Loading {len(paths)} file(s):")
    for p in paths:
        print(f"  {p}")

    frames = [pd.read_csv(p) for p in paths]
    trades_df = pd.concat(frames, ignore_index=True)

    required_cols = {"entry_time", "dollar_pnl", "coin", "use_adverse_drift"}
    missing = required_cols - set(trades_df.columns)
    if missing:
        print(f"ERROR: missing required column(s) {missing}.")
        if "use_adverse_drift" in missing:
            print("These files predate the Aug 2 drift fix (no use_adverse_drift column) - "
                  "re-run backtest.yml with the updated main.py first.")
        print(f"Columns found: {list(trades_df.columns)}")
        sys.exit(1)

    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["month"] = trades_df["entry_time"].dt.to_period("M").astype(str)
    # bool may have come through as the string "true"/"false" depending on
    # how the CSV was written - normalize before grouping.
    trades_df["use_adverse_drift"] = trades_df["use_adverse_drift"].astype(str).str.lower() == "true"

    coins_present = sorted(trades_df["coin"].unique())
    all_comparisons = []

    for coin in coins_present:
        coin_trades = trades_df[trades_df["coin"] == coin]
        before = coin_trades[~coin_trades["use_adverse_drift"]]
        after = coin_trades[coin_trades["use_adverse_drift"]]

        if before.empty and after.empty:
            continue
        if before.empty or after.empty:
            print(f"WARNING: {coin} only has one side of the comparison "
                  f"(before={len(before)} trades, after={len(after)} trades) - "
                  f"make sure backtest.yml ran BOTH adverse_drift_modes for this coin.")

        def monthly(df, label):
            if df.empty:
                return pd.DataFrame(columns=["month", f"trades_{label}", f"dollar_pnl_{label}", f"win_rate_{label}"])
            g = df.groupby("month").agg(
                trades=("dollar_pnl", "count"),
                dollar_pnl=("dollar_pnl", "sum"),
                win_rate=("dollar_pnl", lambda x: round((x > 0).mean() * 100, 1)),
            ).reset_index()
            g.columns = ["month", f"trades_{label}", f"dollar_pnl_{label}", f"win_rate_{label}"]
            return g

        merged = pd.merge(monthly(before, "before"), monthly(after, "after"), on="month", how="outer").fillna(0)
        merged = merged.sort_values("month")
        merged.insert(0, "coin", coin)
        merged["dollar_pnl_delta"] = (merged["dollar_pnl_after"] - merged["dollar_pnl_before"]).round(2)
        merged["trades_delta"] = merged["trades_after"] - merged["trades_before"]

        # Direct answer to "did the fix help or hurt": how many exits in the
        # AFTER run were actually CAUSED by the new drift branches (not just
        # any reversal exit - specifically the primary_drift/confirm_drift/
        # filter_drift frames), and what did those trades' real dollar P&L
        # look like. This is the number that answers the question, not the
        # aggregate total (which can hide a fix that helps on some trades
        # and hurts on others).
        if "exit_reason" in after.columns:
            drift_caused = after[after["exit_reason"] == "reversal"]
            if len(drift_caused):
                print(f"\n{coin}: {len(drift_caused)} 'reversal' exits in the AFTER run "
                      f"(includes both old strongly-reversed triggers and the new drift branches - "
                      f"check trades CSV's 'details' column if present to separate them), "
                      f"avg dollar_pnl on those trades: {drift_caused['dollar_pnl'].mean():.2f}")

        print(f"\n=== {coin}: monthly before vs after ===")
        print(merged.to_string(index=False))
        all_comparisons.append(merged)

    if not all_comparisons:
        print("No coin had both before and after trades - nothing to compare.")
        sys.exit(1)

    final = pd.concat(all_comparisons, ignore_index=True)
    final.to_csv("monthly_drift_comparison_result.csv", index=False)
    print("\nSaved: monthly_drift_comparison_result.csv")

    print("\n=== TOTALS (all coins/months combined) ===")
    totals = final[["dollar_pnl_before", "dollar_pnl_after", "dollar_pnl_delta", "trades_before", "trades_after"]].sum()
    print(totals.to_string())


if __name__ == "__main__":
    main()
