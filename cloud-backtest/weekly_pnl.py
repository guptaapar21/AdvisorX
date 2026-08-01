"""
Weekly P&L breakdown.

Reads one or more trades_*.csv files already produced by backtest_engine.py
(dollar_pnl is already computed correctly per trade - $500 capital, 5% fixed
risk, post-fee - this script only groups that existing, already-correct
number by week, it does not recompute any fee/dollar math itself) and
reports real dollar profit/loss week by week across the full backtest
window, instead of only one aggregate total for the whole year.

Accepts multiple files at once specifically so results from different
coins - each using ITS OWN correct final settings (different score
threshold, different max-hold-time) - can be combined into one unified
weekly view, since the current sweep tooling can't express "coin A at
setting X, coin B at setting Y" within a single run.

Usage:
    python3 weekly_pnl.py results/trades_SOL_score80.0_hold18.0h_*.csv \
                          results/trades_DOGE_score79.0_hold48.0h_*.csv \
                          results/trades_ETH_score81.0_hold60.0h_*.csv
"""

import sys
import glob
import pandas as pd


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 weekly_pnl.py <trades_csv_path_or_glob> [more paths...]")
        sys.exit(1)

    paths = []
    for arg in sys.argv[1:]:
        matched = glob.glob(arg)
        if not matched:
            print(f"Warning: no files matched '{arg}'")
        paths.extend(matched)

    if not paths:
        print("No trade files found - nothing to analyze.")
        sys.exit(1)

    print(f"Loading {len(paths)} file(s):")
    for p in paths:
        print(f"  {p}")

    frames = [pd.read_csv(p) for p in paths]
    trades_df = pd.concat(frames, ignore_index=True)

    required_cols = {"entry_time", "dollar_pnl", "symbol"}
    missing = required_cols - set(trades_df.columns)
    if missing:
        print(f"ERROR: missing required column(s) {missing} - these files may predate the dollar_pnl fix, or aren't trades CSVs.")
        print(f"Columns found: {list(trades_df.columns)}")
        sys.exit(1)

    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)

    # ISO calendar week - Monday-start, consistent year-to-year, avoids the
    # ambiguity of "which week is this" near year boundaries that a plain
    # day-of-year/7 calculation would introduce.
    iso = trades_df["entry_time"].dt.isocalendar()
    trades_df["iso_year"] = iso["year"]
    trades_df["iso_week"] = iso["week"]
    trades_df["week_label"] = trades_df["iso_year"].astype(str) + "-W" + trades_df["iso_week"].astype(str).str.zfill(2)
    # Real calendar date for the Monday of that week, so the output is
    # human-readable, not just an ISO week number nobody can place in time.
    trades_df["week_start"] = trades_df["entry_time"].dt.to_period("W-SUN").apply(lambda p: p.start_time.date())

    weekly = trades_df.groupby(["week_label", "week_start"], as_index=False).agg(
        trades=("dollar_pnl", "count"),
        dollar_pnl=("dollar_pnl", "sum"),
        wins=("dollar_pnl", lambda s: (s > 0).sum()),
    )
    weekly = weekly.sort_values("week_start").reset_index(drop=True)
    weekly["win_rate_pct"] = (weekly["wins"] / weekly["trades"] * 100).round(1)
    weekly["cumulative_dollar_pnl"] = weekly["dollar_pnl"].cumsum().round(2)
    weekly["dollar_pnl"] = weekly["dollar_pnl"].round(2)
    weekly = weekly.drop(columns=["wins"])

    # Fill in any WEEK WITH ZERO TRADES so the series doesn't silently skip
    # quiet weeks - a missing row would misleadingly look like the week
    # never happened, rather than "nothing qualified that week".
    full_range = pd.date_range(weekly["week_start"].min(), weekly["week_start"].max(), freq="W-MON")
    full_index = pd.DataFrame({"week_start": [d.date() for d in full_range]})
    weekly = full_index.merge(weekly, on="week_start", how="left")
    weekly["trades"] = weekly["trades"].fillna(0).astype(int)
    weekly["dollar_pnl"] = weekly["dollar_pnl"].fillna(0.0)
    weekly["win_rate_pct"] = weekly["win_rate_pct"].fillna(0.0)
    weekly["cumulative_dollar_pnl"] = weekly["dollar_pnl"].cumsum().round(2)
    weekly["week_label"] = weekly["week_label"].fillna("(no trades)")

    print()
    print(f"Total trades across all files: {len(trades_df)}")
    print(f"Weeks covered: {len(weekly)} ({weekly['week_start'].min()} to {weekly['week_start'].max()})")
    print(f"Total $ P&L (sum of all weeks): {weekly['dollar_pnl'].sum():.2f}")
    print()
    print(weekly.to_string(index=False))

    weekly.to_csv("weekly_pnl_result.csv", index=False)
    print()
    print("Saved to weekly_pnl_result.csv")

    # Quick per-symbol breakdown too, since a combined multi-coin total can
    # hide one coin quietly underperforming another.
    print()
    print("--- Per-symbol totals (for context, not weekly) ---")
    by_symbol = trades_df.groupby("symbol", as_index=False).agg(
        trades=("dollar_pnl", "count"),
        total_dollar_pnl=("dollar_pnl", "sum"),
    )
    by_symbol["total_dollar_pnl"] = by_symbol["total_dollar_pnl"].round(2)
    print(by_symbol.to_string(index=False))


if __name__ == "__main__":
    main()
