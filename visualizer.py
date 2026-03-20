"""Visualization for backtest results."""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = "output"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_roi_by_threshold(df, save=True):
    """Plot ROI for each threshold, grouped by strategy."""
    ensure_output_dir()
    agg = df[df["window_start"] == "AGGREGATE"]
    if agg.empty:
        print("No aggregate data to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for strat in agg["strategy"].unique():
        strat_data = agg[agg["strategy"] == strat]
        # Average across bet sizes for cleaner view
        grouped = strat_data.groupby("threshold")["roi"].mean()
        ax.plot(grouped.index, grouped.values, marker="o", label=strat, linewidth=2)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Probability Threshold", fontsize=12)
    ax.set_ylabel("ROI (Return on Investment)", fontsize=12)
    ax.set_title("ROI by Probability Threshold & Strategy", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_DIR}/roi_by_threshold.png", dpi=150)
    plt.show()


def plot_roi_by_window_size(df, save=True):
    """Plot how ROI varies with window size."""
    ensure_output_dir()
    agg = df[df["window_start"] == "AGGREGATE"]
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for strat in agg["strategy"].unique():
        strat_data = agg[agg["strategy"] == strat]
        grouped = strat_data.groupby("window_days")["roi"].mean()
        ax.plot(grouped.index, grouped.values, marker="s", label=strat, linewidth=2)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Window Size (days)", fontsize=12)
    ax.set_ylabel("Average ROI", fontsize=12)
    ax.set_title("ROI by Window Size & Strategy", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_DIR}/roi_by_window_size.png", dpi=150)
    plt.show()


def plot_win_rate_vs_roi(df, save=True):
    """Scatter plot of win rate vs ROI for all configurations."""
    ensure_output_dir()
    agg = df[df["window_start"] == "AGGREGATE"]
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    strategies = agg["strategy"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

    for strat, color in zip(strategies, colors):
        strat_data = agg[agg["strategy"] == strat]
        scatter = ax.scatter(
            strat_data["win_rate"], strat_data["roi"],
            c=[color], label=strat, alpha=0.7,
            s=strat_data["total_bets"].clip(upper=500) * 0.5 + 20,
            edgecolors="black", linewidth=0.5,
        )

    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Break even")
    ax.set_xlabel("Win Rate", fontsize=12)
    ax.set_ylabel("ROI", fontsize=12)
    ax.set_title("Win Rate vs ROI (size = number of bets)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_DIR}/win_rate_vs_roi.png", dpi=150)
    plt.show()


def plot_cumulative_pnl(backtest_results, title="Cumulative P&L", save=True):
    """Plot cumulative P&L over time for a list of BacktestResult objects."""
    ensure_output_dir()
    fig, ax = plt.subplots(figsize=(14, 6))

    for result in backtest_results:
        if not result.bets:
            continue
        dates = [b.entry_date for b in result.bets]
        cum_pnl = np.cumsum([b.profit for b in result.bets])
        label = f"{result.strategy_name} t={result.threshold} b=${result.base_bet}"
        ax.plot(dates, cum_pnl, label=label, linewidth=1.5)

    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Profit ($)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_DIR}/cumulative_pnl.png", dpi=150)
    plt.show()


def plot_heatmap_strategy_threshold(df, metric="roi", save=True):
    """Heatmap of a metric across strategies and thresholds."""
    ensure_output_dir()
    agg = df[df["window_start"] == "AGGREGATE"]
    if agg.empty:
        return

    # Average across bet sizes and window sizes
    pivot = agg.groupby(["strategy", "threshold"])[metric].mean().unstack()

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{t:.0%}" for t in pivot.columns], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    # Add value annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not math.isnan(val):
                text = f"{val:.1%}" if metric == "roi" else f"{val:.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=9,
                        color="black" if abs(val) < 0.3 else "white")

    plt.colorbar(im, ax=ax, label=metric.upper())
    ax.set_xlabel("Probability Threshold", fontsize=12)
    ax.set_ylabel("Strategy", fontsize=12)
    ax.set_title(f"{metric.upper()} Heatmap: Strategy × Threshold", fontsize=14)

    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_DIR}/heatmap_{metric}.png", dpi=150)
    plt.show()


def plot_bet_size_impact(df, save=True):
    """Show how different bet sizes affect ROI and risk."""
    ensure_output_dir()
    agg = df[df["window_start"] == "AGGREGATE"]
    if agg.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ROI by bet size
    ax = axes[0]
    for strat in agg["strategy"].unique():
        strat_data = agg[agg["strategy"] == strat]
        grouped = strat_data.groupby("base_bet")["roi"].mean()
        ax.plot(grouped.index, grouped.values, marker="o", label=strat, linewidth=2)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Base Bet Size ($)", fontsize=12)
    ax.set_ylabel("Average ROI", fontsize=12)
    ax.set_title("ROI by Bet Size", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Total profit by bet size
    ax = axes[1]
    for strat in agg["strategy"].unique():
        strat_data = agg[agg["strategy"] == strat]
        grouped = strat_data.groupby("base_bet")["total_profit"].mean()
        ax.plot(grouped.index, grouped.values, marker="s", label=strat, linewidth=2)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Base Bet Size ($)", fontsize=12)
    ax.set_ylabel("Average Total Profit ($)", fontsize=12)
    ax.set_title("Total Profit by Bet Size", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_DIR}/bet_size_impact.png", dpi=150)
    plt.show()


def plot_sliding_window_timeline(window_results, save=True):
    """Show ROI across sliding windows over time."""
    ensure_output_dir()
    if not window_results:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    dates = [r.window_start for r in window_results]
    rois = [r.roi for r in window_results]
    bets_count = [r.total_bets for r in window_results]

    # ROI timeline
    ax = axes[0]
    colors = ["green" if r >= 0 else "red" for r in rois]
    ax.bar(range(len(dates)), rois, color=colors, alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("ROI", fontsize=12)
    ax.set_title(f"Sliding Window ROI Timeline ({window_results[0].strategy_name}, "
                 f"threshold={window_results[0].threshold})", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    # Bets count timeline
    ax = axes[1]
    ax.bar(range(len(dates)), bets_count, color="steelblue", alpha=0.7)
    ax.set_xlabel("Window Index", fontsize=12)
    ax.set_ylabel("Number of Bets", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_DIR}/sliding_window_timeline.png", dpi=150)
    plt.show()


def generate_report(df):
    """Print a text summary report of the best configurations."""
    agg = df[df["window_start"] == "AGGREGATE"]
    if agg.empty:
        print("No aggregate results to report.")
        return

    print("\n" + "=" * 80)
    print("POLYMARKET LOW-PROBABILITY BETTING BACKTEST REPORT")
    print("=" * 80)

    # Best overall ROI
    valid = agg[agg["total_bets"] >= 10]
    if valid.empty:
        print("\nNot enough data (need at least 10 bets per config)")
        return

    print(f"\nTotal configurations tested: {len(agg)}")
    print(f"Configurations with 10+ bets: {len(valid)}")

    print("\n--- TOP 10 BY ROI (min 10 bets) ---")
    top_roi = valid.nlargest(10, "roi")
    for _, row in top_roi.iterrows():
        print(f"  {row['strategy']:15s} | threshold={row['threshold']:.0%} | "
              f"bet=${row['base_bet']:>3.0f} | window={row['window_days']:>3.0f}d | "
              f"ROI={row['roi']:>+7.1%} | bets={row['total_bets']:>4.0f} | "
              f"profit=${row['total_profit']:>+8.2f}")

    print("\n--- TOP 10 BY PROFIT (min 10 bets) ---")
    top_profit = valid.nlargest(10, "total_profit")
    for _, row in top_profit.iterrows():
        print(f"  {row['strategy']:15s} | threshold={row['threshold']:.0%} | "
              f"bet=${row['base_bet']:>3.0f} | window={row['window_days']:>3.0f}d | "
              f"profit=${row['total_profit']:>+8.2f} | ROI={row['roi']:>+7.1%} | "
              f"bets={row['total_bets']:>4.0f}")

    print("\n--- TOP 10 BY SHARPE RATIO (min 10 bets) ---")
    top_sharpe = valid.nlargest(10, "sharpe_ratio")
    for _, row in top_sharpe.iterrows():
        print(f"  {row['strategy']:15s} | threshold={row['threshold']:.0%} | "
              f"bet=${row['base_bet']:>3.0f} | window={row['window_days']:>3.0f}d | "
              f"sharpe={row['sharpe_ratio']:>+6.3f} | ROI={row['roi']:>+7.1%}")

    print("\n--- WORST 5 BY ROI (biggest losers) ---")
    worst_roi = valid.nsmallest(5, "roi")
    for _, row in worst_roi.iterrows():
        print(f"  {row['strategy']:15s} | threshold={row['threshold']:.0%} | "
              f"bet=${row['base_bet']:>3.0f} | ROI={row['roi']:>+7.1%} | "
              f"loss=${row['total_profit']:>+8.2f}")

    # Strategy comparison
    print("\n--- STRATEGY COMPARISON (averaged across all configs) ---")
    strat_summary = valid.groupby("strategy").agg({
        "roi": "mean",
        "win_rate": "mean",
        "total_profit": "sum",
        "total_bets": "sum",
        "sharpe_ratio": "mean",
    }).round(4)
    print(strat_summary.to_string())

    # Threshold comparison
    print("\n--- THRESHOLD COMPARISON (averaged across all configs) ---")
    thresh_summary = valid.groupby("threshold").agg({
        "roi": "mean",
        "win_rate": "mean",
        "total_bets": "sum",
        "total_profit": "sum",
    }).round(4)
    print(thresh_summary.to_string())

    print("\n" + "=" * 80)
