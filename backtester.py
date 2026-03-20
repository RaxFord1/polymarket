"""Backtesting engine with sliding window support."""

import math
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

from strategies import create_strategy


def _ts_to_date(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).date()


def _parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        return None


class Bet:
    """Represents a single simulated bet."""

    __slots__ = ("market_id", "question", "entry_price", "bet_size",
                 "resolved_yes", "payout", "profit", "roi", "entry_date", "resolve_date")

    def __init__(self, market_id, question, entry_price, bet_size,
                 resolved_yes, entry_date, resolve_date):
        self.market_id = market_id
        self.question = question
        self.entry_price = entry_price
        self.bet_size = bet_size
        self.resolved_yes = resolved_yes
        self.entry_date = entry_date
        self.resolve_date = resolve_date

        if resolved_yes:
            self.payout = bet_size / entry_price  # YES share pays $1
            self.profit = self.payout - bet_size
        else:
            self.payout = 0
            self.profit = -bet_size

        self.roi = self.profit / bet_size if bet_size > 0 else 0


class BacktestResult:
    """Stores results for a single backtest configuration."""

    def __init__(self, strategy_name, threshold, base_bet, window_days, window_start, window_end):
        self.strategy_name = strategy_name
        self.threshold = threshold
        self.base_bet = base_bet
        self.window_days = window_days
        self.window_start = window_start
        self.window_end = window_end
        self.bets = []

    def add_bet(self, bet):
        self.bets.append(bet)

    @property
    def total_bets(self):
        return len(self.bets)

    @property
    def wins(self):
        return sum(1 for b in self.bets if b.resolved_yes)

    @property
    def losses(self):
        return self.total_bets - self.wins

    @property
    def win_rate(self):
        return self.wins / self.total_bets if self.total_bets > 0 else 0

    @property
    def total_wagered(self):
        return sum(b.bet_size for b in self.bets)

    @property
    def total_profit(self):
        return sum(b.profit for b in self.bets)

    @property
    def total_payout(self):
        return sum(b.payout for b in self.bets)

    @property
    def roi(self):
        return self.total_profit / self.total_wagered if self.total_wagered > 0 else 0

    @property
    def avg_entry_price(self):
        return np.mean([b.entry_price for b in self.bets]) if self.bets else 0

    @property
    def max_drawdown(self):
        if not self.bets:
            return 0
        cumulative = np.cumsum([b.profit for b in self.bets])
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        return float(np.max(drawdown)) if len(drawdown) > 0 else 0

    @property
    def sharpe_ratio(self):
        if len(self.bets) < 2:
            return 0
        returns = [b.roi for b in self.bets]
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        return float(mean_ret / std_ret) if std_ret > 0 else 0

    @property
    def profit_factor(self):
        gross_profit = sum(b.profit for b in self.bets if b.profit > 0)
        gross_loss = abs(sum(b.profit for b in self.bets if b.profit < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    def summary(self):
        return {
            "strategy": self.strategy_name,
            "threshold": self.threshold,
            "base_bet": self.base_bet,
            "window_days": self.window_days,
            "window_start": str(self.window_start),
            "window_end": str(self.window_end),
            "total_bets": self.total_bets,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "total_wagered": round(self.total_wagered, 2),
            "total_profit": round(self.total_profit, 2),
            "roi": round(self.roi, 4),
            "avg_entry_price": round(self.avg_entry_price, 4),
            "max_drawdown": round(self.max_drawdown, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "profit_factor": round(self.profit_factor, 4),
        }


def prepare_market_data(markets, price_data_map):
    """Prepare markets with their entry prices into a sorted list.

    For each market, we take the median price from available history
    as the "entry price" (simulating buying at a typical price point).
    """
    entries = []

    for market in markets:
        mid = market["id"]
        prices = price_data_map.get(mid)

        if not prices or len(prices) < 2:
            continue

        resolve_date = _parse_date(market.get("end_date") or market.get("closed_time"))
        if not resolve_date:
            continue

        # Calculate entry price as median of all historical prices
        price_values = [p[1] for p in prices if 0 < p[1] < 1]
        if not price_values:
            continue

        # Use multiple entry price points for more realistic simulation
        # We'll use the price at 25%, 50%, 75% of the market's lifetime
        n = len(price_values)
        entry_candidates = [
            price_values[n // 4],       # early
            price_values[n // 2],       # mid
            price_values[3 * n // 4],   # late
        ]

        # Use earliest available timestamp as approximate start
        start_ts = prices[0][0]
        entry_date = _ts_to_date(start_ts)

        entries.append({
            "market": market,
            "entry_prices": entry_candidates,
            "median_price": float(np.median(price_values)),
            "min_price": min(price_values),
            "entry_date": entry_date,
            "resolve_date": resolve_date,
        })

    entries.sort(key=lambda x: x["entry_date"])
    return entries


def run_backtest(entries, strategy_name, threshold, base_bet, bankroll,
                 window_start, window_end, window_days):
    """Run a single backtest with given parameters."""
    strategy = create_strategy(strategy_name, base_bet, bankroll)
    result = BacktestResult(strategy_name, threshold, base_bet,
                            window_days, window_start, window_end)

    for entry in entries:
        if entry["entry_date"] < window_start or entry["entry_date"] > window_end:
            continue

        # Use median price as entry point
        entry_price = entry["median_price"]

        # Only bet if price is below threshold (low probability event)
        if entry_price >= threshold:
            continue

        bet_size = strategy.get_bet_size(entry_price, threshold)

        if not strategy.can_bet(bet_size):
            continue

        market = entry["market"]
        bet = Bet(
            market_id=market["id"],
            question=market["question"],
            entry_price=entry_price,
            bet_size=bet_size,
            resolved_yes=market["resolved_yes"],
            entry_date=entry["entry_date"],
            resolve_date=entry["resolve_date"],
        )

        result.add_bet(bet)
        strategy.bankroll += bet.profit
        strategy.record_outcome(bet.resolved_yes)

        # Bust check
        if strategy.bankroll <= 0:
            break

    return result


def run_sliding_window_backtest(entries, strategy_name, threshold, base_bet,
                                 bankroll, window_days, slide_step_days=None):
    """Run backtest across sliding windows.

    Args:
        entries: Prepared market data from prepare_market_data()
        strategy_name: Name of strategy to use
        threshold: Max price to enter a bet
        base_bet: Base bet size
        bankroll: Starting bankroll per window
        window_days: Size of each window in days
        slide_step_days: Days to slide the window (default: window_days // 4)

    Returns:
        List of BacktestResult objects, one per window position
    """
    if not entries:
        return []

    if slide_step_days is None:
        slide_step_days = max(window_days // 4, 7)

    all_dates = [e["entry_date"] for e in entries]
    min_date = min(all_dates)
    max_date = max(all_dates)

    results = []
    current_start = min_date

    while True:
        current_end = current_start + pd.Timedelta(days=window_days)
        current_end_date = current_end.date() if hasattr(current_end, 'date') else current_end

        # Convert to date objects for comparison
        from datetime import timedelta
        if isinstance(current_start, pd.Timestamp):
            ws = current_start.date()
        else:
            ws = current_start
        we = ws + timedelta(days=window_days)

        if ws > max_date:
            break

        result = run_backtest(entries, strategy_name, threshold, base_bet,
                              bankroll, ws, we, window_days)

        if result.total_bets > 0:
            results.append(result)

        from datetime import timedelta
        if isinstance(current_start, pd.Timestamp):
            current_start = (current_start + pd.Timedelta(days=slide_step_days)).date()
        else:
            current_start = current_start + timedelta(days=slide_step_days)

    return results


def run_full_analysis(entries, strategies=None, thresholds=None, bet_sizes=None,
                      bankroll=1000.0, window_sizes=None):
    """Run comprehensive analysis across all parameter combinations.

    Returns a DataFrame with results for every combination.
    """
    from config import DEFAULT_THRESHOLDS, DEFAULT_BET_SIZES, DEFAULT_WINDOW_SIZES, STRATEGIES

    if strategies is None:
        strategies = list(STRATEGIES.keys())
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    if bet_sizes is None:
        bet_sizes = DEFAULT_BET_SIZES
    if window_sizes is None:
        window_sizes = DEFAULT_WINDOW_SIZES

    all_results = []
    total_combos = len(strategies) * len(thresholds) * len(bet_sizes) * len(window_sizes)

    with tqdm(total=total_combos, desc="Running backtests") as pbar:
        for strat_name in strategies:
            for threshold in thresholds:
                for bet_size in bet_sizes:
                    for window_days in window_sizes:
                        window_results = run_sliding_window_backtest(
                            entries, strat_name, threshold, bet_size,
                            bankroll, window_days
                        )

                        for wr in window_results:
                            all_results.append(wr.summary())

                        # Also add aggregate across all windows
                        if window_results:
                            agg = _aggregate_window_results(window_results, strat_name,
                                                            threshold, bet_size, window_days)
                            all_results.append(agg)

                        pbar.update(1)

    return pd.DataFrame(all_results)


def _aggregate_window_results(results, strategy_name, threshold, base_bet, window_days):
    """Aggregate results across all windows for a given config."""
    total_bets = sum(r.total_bets for r in results)
    total_wins = sum(r.wins for r in results)
    total_wagered = sum(r.total_wagered for r in results)
    total_profit = sum(r.total_profit for r in results)

    return {
        "strategy": strategy_name,
        "threshold": threshold,
        "base_bet": base_bet,
        "window_days": window_days,
        "window_start": "AGGREGATE",
        "window_end": "AGGREGATE",
        "total_bets": total_bets,
        "wins": total_wins,
        "losses": total_bets - total_wins,
        "win_rate": round(total_wins / total_bets, 4) if total_bets > 0 else 0,
        "total_wagered": round(total_wagered, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(total_profit / total_wagered, 4) if total_wagered > 0 else 0,
        "avg_entry_price": round(
            np.mean([r.avg_entry_price for r in results if r.total_bets > 0]), 4
        ),
        "max_drawdown": round(max(r.max_drawdown for r in results), 2),
        "sharpe_ratio": round(
            np.mean([r.sharpe_ratio for r in results if r.total_bets > 0]), 4
        ),
        "profit_factor": round(
            np.mean([r.profit_factor for r in results
                     if r.total_bets > 0 and not math.isinf(r.profit_factor)]), 4
        ) if any(not math.isinf(r.profit_factor) for r in results if r.total_bets > 0) else 0,
    }
