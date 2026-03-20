"""Generate realistic synthetic market data for testing the backtest pipeline.

Simulates Polymarket-like resolved markets with realistic properties:
- Low-probability events resolve YES ~5-15% of the time (depending on price range)
- Price histories follow random walks with mean-reversion
- Market lifetimes vary from days to months
- Volume follows a power-law distribution
"""

import json
import math
import os
import random
from datetime import datetime, timedelta, timezone

import numpy as np

import config


def generate_price_history(start_ts, end_ts, initial_price, final_outcome,
                           num_points=50, surprise=False):
    """Generate a realistic price history for a market.

    Prices follow a mean-reverting random walk with drift toward resolution.
    If surprise=True, the price stays low most of the time then spikes only
    at the very end (simulating a surprise YES outcome).
    """
    if num_points < 3:
        num_points = 3

    timestamps = np.linspace(start_ts, end_ts, num_points).astype(int)
    prices = [initial_price]

    if surprise and final_outcome:
        # Surprise outcome: price stays near initial_price until last 10-20%
        spike_point = random.uniform(0.8, 0.95)
        for i in range(1, num_points):
            progress = i / num_points
            if progress < spike_point:
                # Hover around initial price with noise
                volatility = initial_price * 0.3
                shock = random.gauss(0, volatility)
                mean_rev = 0.15 * (initial_price - prices[-1])
                new_price = prices[-1] + mean_rev + shock
            else:
                # Spike toward 1.0
                spike_progress = (progress - spike_point) / (1 - spike_point)
                target = initial_price + (0.95 - initial_price) * spike_progress**0.5
                new_price = prices[-1] + 0.2 * (target - prices[-1]) + random.gauss(0, 0.02)
            prices.append(max(0.001, min(0.999, new_price)))
    else:
        final_price = 0.95 if final_outcome else 0.02
        for i in range(1, num_points):
            progress = i / num_points
            target = initial_price * (1 - progress**2) + final_price * progress**2
            volatility = 0.03 * (1 - progress * 0.7)
            shock = random.gauss(0, volatility)
            mean_reversion = 0.1 * (target - prices[-1])
            new_price = prices[-1] + mean_reversion + shock
            prices.append(max(0.001, min(0.999, new_price)))

    return [[int(ts), round(p, 4)] for ts, p in zip(timestamps, prices)]


def generate_markets(n_markets=1500, seed=42):
    """Generate n_markets synthetic resolved markets."""
    random.seed(seed)
    np.random.seed(seed)

    markets = []

    # Categories of questions for realism
    categories = [
        ("Will {person} win the {year} {office} election?", "politics"),
        ("Will {crypto} reach ${price} by {date}?", "crypto"),
        ("Will {team} win {event}?", "sports"),
        ("Will {metric} exceed {value} in {period}?", "economics"),
        ("Will {event} happen before {date}?", "world"),
    ]

    persons = ["Biden", "Trump", "DeSantis", "Haley", "Newsom", "Harris", "Ramaswamy"]
    offices = ["presidential", "senate", "gubernatorial"]
    cryptos = ["BTC", "ETH", "SOL", "DOGE", "XRP", "ADA", "AVAX"]
    teams = ["Lakers", "Yankees", "Chiefs", "Real Madrid", "Liverpool", "Celtics"]
    events_sport = ["the NBA Finals", "the Super Bowl", "the World Series", "Champions League"]
    metrics = ["US GDP growth", "inflation", "unemployment", "S&P 500", "Fed rate"]

    base_date = datetime(2023, 1, 1, tzinfo=timezone.utc)

    for i in range(n_markets):
        # Market lifetime: 7 to 365 days
        lifetime_days = random.choice([
            random.randint(7, 30),    # short-term (40%)
            random.randint(30, 90),   # medium-term (35%)
            random.randint(90, 365),  # long-term (25%)
        ])

        # Start date spread over 2 years
        start_offset = random.randint(0, 730)
        start_date = base_date + timedelta(days=start_offset)
        end_date = start_date + timedelta(days=lifetime_days)

        # Initial price (probability) - biased toward low probabilities for our use case
        # Mix of very low, low, medium, and high probability markets
        price_bucket = random.random()
        if price_bucket < 0.25:
            initial_price = random.uniform(0.01, 0.05)   # very low prob
        elif price_bucket < 0.50:
            initial_price = random.uniform(0.05, 0.15)   # low prob
        elif price_bucket < 0.75:
            initial_price = random.uniform(0.15, 0.40)   # medium prob
        else:
            initial_price = random.uniform(0.40, 0.90)   # high prob

        # Resolution: lower price = less likely to resolve YES
        # Add alpha for low-prob events (the hypothesis being tested)
        if initial_price < 0.05:
            resolve_prob = initial_price * 1.5   # 50% edge on very low prob
        elif initial_price < 0.10:
            resolve_prob = initial_price * 1.3   # 30% edge
        elif initial_price < 0.20:
            resolve_prob = initial_price * 1.15  # 15% edge
        else:
            resolve_prob = initial_price * 0.95  # slight negative edge on high prob

        resolved_yes = random.random() < resolve_prob
        # For low-prob YES outcomes, most are surprises (price stays low until end)
        surprise = resolved_yes and initial_price < 0.20 and random.random() < 0.7

        # Generate question
        cat = random.choice(categories)
        question = f"Synthetic market #{i}: {cat[1]} event (initial p={initial_price:.1%})"

        # Volume follows power law
        volume = round(random.paretovariate(1.5) * 1000, 2)

        market = {
            "id": f"synth_{i:05d}",
            "question": question,
            "slug": f"synth-market-{i}",
            "outcomes": ["Yes", "No"],
            "outcome_prices": [1.0, 0.0] if resolved_yes else [0.0, 1.0],
            "resolved_yes": resolved_yes,
            "clob_token_ids": f"token_{i}_yes,token_{i}_no",
            "condition_id": f"condition_{i}",
            "volume": volume,
            "end_date": end_date.isoformat(),
            "closed_time": end_date.isoformat(),
            "start_date": start_date.isoformat(),
        }
        markets.append(market)

        # Generate price history
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        num_points = max(10, lifetime_days * 2)  # ~2 price points per day

        price_history = generate_price_history(
            start_ts, end_ts, initial_price, resolved_yes, num_points,
            surprise=surprise
        )

        # Save price history cache
        os.makedirs(config.PRICES_CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(config.PRICES_CACHE_DIR, f"{market['id']}.json")
        with open(cache_file, "w") as f:
            json.dump(price_history, f)

    # Save markets cache
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(config.MARKETS_CACHE, "w") as f:
        json.dump(markets, f)

    # Print stats
    yes_count = sum(1 for m in markets if m["resolved_yes"])
    low_prob = [m for m in markets if m["outcome_prices"] == [1.0, 0.0] or True]

    print(f"Generated {len(markets)} synthetic markets")
    print(f"  Resolved YES: {yes_count} ({yes_count/len(markets):.1%})")
    print(f"  Resolved NO: {len(markets) - yes_count} ({(len(markets)-yes_count)/len(markets):.1%})")

    # Price distribution
    prices = []
    for i in range(len(markets)):
        cache_file = os.path.join(config.PRICES_CACHE_DIR, f"synth_{i:05d}.json")
        with open(cache_file) as f:
            hist = json.load(f)
        median_p = np.median([p[1] for p in hist])
        prices.append(median_p)

    print(f"\n  Median price distribution:")
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.30]:
        count = sum(1 for p in prices if p < threshold)
        yes_in_range = sum(1 for m, p in zip(markets, prices)
                          if p < threshold and m["resolved_yes"])
        rate = yes_in_range / count if count > 0 else 0
        print(f"    Below {threshold:.0%}: {count} markets, {rate:.1%} win rate")

    return markets


if __name__ == "__main__":
    generate_markets()
