"""Main entry point for the Polymarket backtesting system.

Usage:
    python main.py                          # Full analysis with defaults
    python main.py --fetch-only             # Only fetch and cache data
    python main.py --quick                  # Quick run with fewer combos
    python main.py --strategies flat kelly  # Specific strategies
    python main.py --thresholds 0.05 0.10   # Specific thresholds
    python main.py --bet-sizes 1 10 50      # Specific bet sizes
    python main.py --windows 30 90 365      # Specific window sizes
    python main.py --bankroll 5000          # Custom starting bankroll
    python main.py --max-markets 1000       # Limit markets fetched
    python main.py --no-cache               # Force re-fetch all data
"""

import argparse
import json
import os
import sys

import pandas as pd
from tqdm import tqdm

import config
from fetcher import fetch_resolved_markets, fetch_price_at_entry
from backtester import prepare_market_data, run_full_analysis, run_sliding_window_backtest
from visualizer import (
    plot_roi_by_threshold,
    plot_roi_by_window_size,
    plot_win_rate_vs_roi,
    plot_heatmap_strategy_threshold,
    plot_bet_size_impact,
    plot_sliding_window_timeline,
    generate_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Polymarket Low-Probability Betting Backtester")

    parser.add_argument("--fetch-only", action="store_true",
                        help="Only fetch and cache market data, don't run backtest")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with reduced parameter combinations")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-fetch all data (ignore cache)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")

    parser.add_argument("--strategies", nargs="+", default=None,
                        help=f"Strategies to test. Available: {list(config.STRATEGIES.keys())}")
    parser.add_argument("--thresholds", nargs="+", type=float, default=None,
                        help="Probability thresholds to test (e.g., 0.05 0.10 0.20)")
    parser.add_argument("--bet-sizes", nargs="+", type=float, default=None,
                        help="Base bet sizes to test (e.g., 1 10 50)")
    parser.add_argument("--windows", nargs="+", type=int, default=None,
                        help="Window sizes in days (e.g., 30 90 365)")
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Starting bankroll (default: 1000)")
    parser.add_argument("--max-markets", type=int, default=None,
                        help="Max number of markets to fetch")
    parser.add_argument("--slippage", type=int, default=None,
                        help="Slippage in basis points (default: 50 = 0.5%%)")
    parser.add_argument("--fees", type=int, default=None,
                        help="Transaction fee in basis points (default: 0)")
    parser.add_argument("--min-volume", type=float, default=None,
                        help="Minimum market volume to include (default: 500)")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.max_markets:
        config.MAX_MARKETS = args.max_markets
    if args.slippage is not None:
        config.SLIPPAGE_BPS = args.slippage
    if args.fees is not None:
        config.TRANSACTION_FEE_BPS = args.fees
    if args.min_volume is not None:
        config.MIN_VOLUME = args.min_volume

    use_cache = not args.no_cache

    # --- Step 1: Fetch resolved markets ---
    print("\n[1/4] Fetching resolved markets...")
    markets = fetch_resolved_markets(use_cache=use_cache)
    print(f"  Total resolved markets: {len(markets)}")
    print(f"  Resolved YES: {sum(1 for m in markets if m['resolved_yes'])}")
    print(f"  Resolved NO: {sum(1 for m in markets if not m['resolved_yes'])}")

    if not markets:
        print("ERROR: No markets fetched. Check your internet connection.")
        sys.exit(1)

    # --- Step 2: Fetch price histories ---
    print("\n[2/4] Fetching price histories...")
    price_data_map = {}
    failed = 0

    for market in tqdm(markets, desc="Fetching prices"):
        prices = fetch_price_at_entry(market, use_cache=use_cache)
        if prices:
            price_data_map[market["id"]] = prices
        else:
            failed += 1

    print(f"  Markets with price data: {len(price_data_map)}")
    print(f"  Markets without price data: {failed}")

    if args.fetch_only:
        print("\n--fetch-only specified. Data cached. Exiting.")
        return

    # --- Step 3: Prepare and run backtests ---
    print("\n[3/4] Running backtests...")
    entries = prepare_market_data(markets, price_data_map)
    print(f"  Prepared {len(entries)} market entries for backtesting")

    if not entries:
        print("ERROR: No entries with valid price data. Try fetching more markets.")
        sys.exit(1)

    # Set parameters
    if args.quick:
        strategies = args.strategies or ["flat", "kelly", "proportional"]
        thresholds = args.thresholds or [0.05, 0.10, 0.20]
        bet_sizes = args.bet_sizes or [1, 10, 50]
        window_sizes = args.windows or [90, 365]
    else:
        strategies = args.strategies
        thresholds = args.thresholds
        bet_sizes = args.bet_sizes
        window_sizes = args.windows

    results_df = run_full_analysis(
        entries,
        strategies=strategies,
        thresholds=thresholds,
        bet_sizes=bet_sizes,
        bankroll=args.bankroll,
        window_sizes=window_sizes,
    )

    # Save raw results
    os.makedirs("output", exist_ok=True)
    results_df.to_csv("output/backtest_results.csv", index=False)
    print(f"  Results saved to output/backtest_results.csv ({len(results_df)} rows)")

    # --- Step 4: Generate report and plots ---
    print("\n[4/4] Generating report...")
    generate_report(results_df)

    if not args.no_plots:
        print("\nGenerating plots...")
        try:
            plot_roi_by_threshold(results_df)
            plot_roi_by_window_size(results_df)
            plot_win_rate_vs_roi(results_df)
            plot_heatmap_strategy_threshold(results_df, metric="roi")
            plot_heatmap_strategy_threshold(results_df, metric="sharpe_ratio")
            plot_bet_size_impact(results_df)

            # Generate sliding window timeline for best config
            agg = results_df[results_df["window_start"] == "AGGREGATE"]
            valid = agg[agg["total_bets"] >= 10]
            if not valid.empty:
                best = valid.loc[valid["roi"].idxmax()]
                best_windows = run_sliding_window_backtest(
                    entries, best["strategy"], best["threshold"],
                    best["base_bet"], args.bankroll, int(best["window_days"])
                )
                if best_windows:
                    plot_sliding_window_timeline(best_windows)

            print(f"Plots saved to output/")
        except Exception as e:
            print(f"Warning: Could not generate some plots: {e}")

    print("\nDone! Check output/ directory for results.")


if __name__ == "__main__":
    main()
