"""
Session-based time-gating analysis.

Reads one or more trades_*.csv files already produced by backtest_engine.py
(entry_time is already recorded on every trade - no new backtest run needed)
and checks whether performance differs meaningfully by which trading session
each trade's entry fell into.

Standard session hours (UTC):
  Asian:            00:00-08:00
  London:           08:00-16:00
  New York:         13:00-21:00
  London/NY overlap: 13:00-16:00 (the highest-liquidity window, checked separately)
  Dead zone:        21:00-00:00 (after NY close, before Asian open)
  Weekend:          Saturday/Sunday, any hour (checked separately, overrides the above)

A trade can fall into more than one labeled window (e.g. 14:00 UTC on a
Tuesday is both "london" and "new_york" and "london_ny_overlap") - reported
separately per session, not mutually exclusive, since the question is
"does this window help", not "assign one single label per trade".

Usage:
    python3 session_analysis.py results/trades_SOL_conservative_*.csv
    python3 session_analysis.py results/trades_*.csv   (all trades combined)
"""

import sys
import glob
import pandas as pd


def tag_sessions(entry_time):
    """Returns a list of session labels this entry_time belongs to."""
    hour = entry_time.hour
    weekday = entry_time.weekday()  # 0=Monday ... 5=Saturday, 6=Sunday

    labels = []
    if weekday >= 5:
        labels.append("weekend")
    if 0 <= hour < 8:
        labels.append("asian")
    if 8 <= hour < 16:
        labels.append("london")
    if 13 <= hour < 21:
        labels.append("new_york")
    if 13 <= hour < 16:
        labels.append("london_ny_overlap")
    if hour >= 21 or hour < 0:
        labels.append("dead_zone")
    return labels


def analyze(trades_df):
    trades_df = trades_df.copy()
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["r_achieved"] = pd.to_numeric(trades_df["r_achieved"], errors="coerce")

    all_sessions = ["asian", "london", "new_york", "london_ny_overlap", "dead_zone", "weekend"]
    rows = []
    for session in all_sessions:
        mask = trades_df["entry_time"].apply(lambda t: session in tag_sessions(t))
        subset = trades_df[mask]
        if len(subset) == 0:
            rows.append({"session": session, "trades": 0, "avg_r": None, "win_rate": None})
            continue
        avg_r = subset["r_achieved"].mean()
        win_rate = (subset["r_achieved"] > 0).mean() * 100
        rows.append({
            "session": session,
            "trades": len(subset),
            "avg_r": round(avg_r, 3),
            "win_rate": round(win_rate, 1),
        })

    overall_avg_r = trades_df["r_achieved"].mean()
    overall_win_rate = (trades_df["r_achieved"] > 0).mean() * 100
    rows.append({"session": "ALL (baseline, for comparison)", "trades": len(trades_df),
                 "avg_r": round(overall_avg_r, 3), "win_rate": round(overall_win_rate, 1)})

    return pd.DataFrame(rows)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 session_analysis.py <trades_csv_path_or_glob> [more paths...]")
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

    print(f"Loading {len(paths)} file(s): {paths}")
    frames = [pd.read_csv(p) for p in paths]
    trades_df = pd.concat(frames, ignore_index=True)

    if "entry_time" not in trades_df.columns or "r_achieved" not in trades_df.columns:
        print("ERROR: expected columns 'entry_time' and 'r_achieved' not found in the provided file(s).")
        print(f"Columns found: {list(trades_df.columns)}")
        sys.exit(1)

    print(f"Total trades loaded: {len(trades_df)}")
    print()

    result = analyze(trades_df)
    print(result.to_string(index=False))

    result.to_csv("session_analysis_result.csv", index=False)
    print()
    print("Saved to session_analysis_result.csv")


if __name__ == "__main__":
    main()
