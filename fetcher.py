"""Fetch resolved markets and historical prices from Polymarket APIs."""

import json
import os
import time
from datetime import datetime, timezone

import requests
from tqdm import tqdm

import config


def fetch_resolved_markets(use_cache=True):
    """Fetch all resolved/closed markets from Gamma API."""
    if use_cache and os.path.exists(config.MARKETS_CACHE):
        print(f"Loading cached markets from {config.MARKETS_CACHE}")
        with open(config.MARKETS_CACHE, "r") as f:
            return json.load(f)

    print("Fetching resolved markets from Gamma API...")
    markets = []
    offset = 0

    with tqdm(desc="Fetching markets") as pbar:
        while offset < config.MAX_MARKETS:
            resp = requests.get(
                f"{config.GAMMA_API}/markets",
                params={
                    "closed": "true",
                    "limit": config.FETCH_LIMIT,
                    "offset": offset,
                    "order": "endDate",
                    "ascending": "false",
                },
                timeout=30,
            )
            resp.raise_for_status()
            batch = resp.json()

            if not batch:
                break

            for m in batch:
                outcome_prices = m.get("outcomePrices", "")
                if not outcome_prices:
                    continue
                try:
                    prices = json.loads(outcome_prices)
                except (json.JSONDecodeError, TypeError):
                    continue

                # Only keep markets with clear resolution (1/0)
                if len(prices) >= 2 and (prices[0] in ("1", "0")):
                    markets.append({
                        "id": m.get("id"),
                        "question": m.get("question", ""),
                        "slug": m.get("slug", ""),
                        "outcomes": json.loads(m.get("outcomes", "[]")),
                        "outcome_prices": [float(p) for p in prices],
                        "resolved_yes": prices[0] == "1",
                        "clob_token_ids": m.get("clobTokenIds", m.get("clobTokenIDs", "")),
                        "condition_id": m.get("conditionId", ""),
                        "volume": float(m.get("volumeNum", 0) or 0),
                        "end_date": m.get("endDate", ""),
                        "closed_time": m.get("closedTime", ""),
                        "start_date": m.get("startDate", ""),
                    })

            offset += config.FETCH_LIMIT
            pbar.update(len(batch))
            time.sleep(config.REQUEST_DELAY)

            if len(batch) < config.FETCH_LIMIT:
                break

    print(f"Fetched {len(markets)} resolved markets")

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(config.MARKETS_CACHE, "w") as f:
        json.dump(markets, f)

    return markets


def _parse_clob_token_ids(raw):
    """Parse clobTokenIds which can be a JSON string or comma-separated."""
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return [t.strip() for t in raw.split(",") if t.strip()]


def fetch_price_at_entry(market, use_cache=True):
    """Fetch the YES token price history for a market.

    Returns a list of (timestamp, price) tuples, or None if unavailable.
    """
    os.makedirs(config.PRICES_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(config.PRICES_CACHE_DIR, f"{market['id']}.json")

    if use_cache and os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    token_ids = _parse_clob_token_ids(market.get("clob_token_ids", ""))
    if not token_ids:
        return None

    # First token is typically YES
    yes_token = token_ids[0]

    # Try prices-history with explicit timestamps
    history = _fetch_price_history_clob(yes_token, market)

    # Fallback to trades data
    if not history:
        history = _fetch_price_from_trades(market)

    if history:
        with open(cache_file, "w") as f:
            json.dump(history, f)

    return history


def _fetch_price_history_clob(token_id, market):
    """Fetch price history from CLOB API using time chunks."""
    start_date = market.get("start_date") or market.get("end_date")
    end_date = market.get("end_date") or market.get("closed_time")

    if not start_date or not end_date:
        return None

    try:
        start_ts = int(datetime.fromisoformat(start_date.replace("Z", "+00:00")).timestamp())
        end_ts = int(datetime.fromisoformat(end_date.replace("Z", "+00:00")).timestamp())
    except (ValueError, TypeError):
        return None

    all_points = []
    chunk_seconds = config.PRICE_HISTORY_CHUNK_DAYS * 86400
    current_start = start_ts

    while current_start < end_ts:
        current_end = min(current_start + chunk_seconds, end_ts)
        try:
            resp = requests.get(
                f"{config.CLOB_API}/prices-history",
                params={
                    "market": token_id,
                    "startTs": current_start,
                    "endTs": current_end,
                    "fidelity": config.PRICE_HISTORY_FIDELITY,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            history = data.get("history", [])
            for point in history:
                all_points.append([int(point["t"]), float(point["p"])])
        except (requests.RequestException, KeyError, ValueError):
            pass

        current_start = current_end
        time.sleep(config.REQUEST_DELAY)

    return all_points if all_points else None


def _fetch_price_from_trades(market):
    """Fallback: reconstruct price history from trade data."""
    condition_id = market.get("condition_id")
    if not condition_id:
        return None

    all_trades = []
    offset = 0

    for _ in range(10):  # max 5000 trades
        try:
            resp = requests.get(
                f"{config.DATA_API}/trades",
                params={
                    "market": condition_id,
                    "limit": 500,
                    "offset": offset,
                },
                timeout=30,
            )
            resp.raise_for_status()
            trades = resp.json()

            if not trades:
                break

            for t in trades:
                try:
                    ts = int(datetime.fromisoformat(
                        t["timestamp"].replace("Z", "+00:00")
                    ).timestamp()) if isinstance(t.get("timestamp"), str) else int(t.get("timestamp", 0))
                    price = float(t.get("price", 0))
                    if price > 0:
                        all_trades.append([ts, price])
                except (ValueError, KeyError):
                    continue

            offset += 500
            time.sleep(config.REQUEST_DELAY)

            if len(trades) < 500:
                break
        except requests.RequestException:
            break

    if all_trades:
        all_trades.sort(key=lambda x: x[0])
        return all_trades

    return None


if __name__ == "__main__":
    markets = fetch_resolved_markets(use_cache=False)
    print(f"\nTotal resolved markets: {len(markets)}")
    print(f"Resolved YES: {sum(1 for m in markets if m['resolved_yes'])}")
    print(f"Resolved NO: {sum(1 for m in markets if not m['resolved_yes'])}")

    # Fetch prices for first 5 as test
    for m in markets[:5]:
        print(f"\n{m['question'][:80]}")
        prices = fetch_price_at_entry(m, use_cache=False)
        if prices:
            print(f"  Price points: {len(prices)}, range: {prices[0][1]:.2f} - {prices[-1][1]:.2f}")
        else:
            print("  No price data available")
