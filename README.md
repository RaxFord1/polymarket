# Polymarket Low-Probability Betting Backtester

Backtest whether betting on low-probability events on Polymarket is profitable. Analyzes historical resolved markets with sliding window analysis, multiple strategies, variable bet sizes, and configurable probability thresholds.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Quick test run (fewer parameter combinations)
python main.py --quick

# Full analysis (all strategies × thresholds × bet sizes × windows)
python main.py

# Only fetch and cache data (no backtest)
python main.py --fetch-only
```

## How It Works

### Data Pipeline
1. **Fetch resolved markets** from Polymarket Gamma API (`gamma-api.polymarket.com`)
2. **Fetch historical prices** from CLOB API (`clob.polymarket.com/prices-history`)
3. Falls back to **trade data** from Data API if price history is unavailable
4. All data is cached locally in `cache/` to avoid re-fetching

### Backtesting Logic
- For each resolved market, the system takes the **median historical price** as the simulated entry price
- If that price is **below the threshold** (e.g., < 10%), a bet is placed
- The strategy determines **how much** to bet
- If the event resolved YES → payout = `bet_size / entry_price` (e.g., buy at $0.05 → pays $1 = 20x)
- If the event resolved NO → lose the bet

### Sliding Window
The backtest runs across **overlapping time windows** (e.g., 90-day windows sliding by 22 days). This shows how the strategy performs in different market conditions over time, not just as one aggregate number.

## Strategies

| Strategy | Description |
|----------|-------------|
| `flat` | Fixed bet size on every qualifying event |
| `kelly` | Fractional Kelly criterion (25%) — sizes bets based on estimated edge |
| `proportional` | Bigger bet when the gap between price and threshold is larger |
| `inverse` | Lower price = bigger bet (chasing higher payouts) |
| `martingale` | Doubles bet after each loss, resets after a win (capped at 64x) |

## CLI Options

```bash
python main.py \
  --strategies flat kelly proportional   # which strategies to test
  --thresholds 0.05 0.10 0.20           # probability thresholds
  --bet-sizes 1 10 50                    # base bet amounts ($)
  --windows 30 90 365                    # sliding window sizes (days)
  --bankroll 5000                        # starting bankroll per window
  --max-markets 2000                     # limit number of markets fetched
  --no-cache                             # force re-fetch all data
  --no-plots                             # skip chart generation
```

## Output

Results go to `output/`:
- `backtest_results.csv` — raw results for every parameter combination and window
- `roi_by_threshold.png` — ROI curves per strategy across thresholds
- `roi_by_window_size.png` — how window size affects returns
- `win_rate_vs_roi.png` — scatter plot (bubble size = number of bets)
- `heatmap_roi.png` — strategy × threshold heatmap
- `heatmap_sharpe_ratio.png` — risk-adjusted return heatmap
- `bet_size_impact.png` — how bet sizing affects ROI and total profit
- `sliding_window_timeline.png` — ROI over time for the best config
- Console prints a **text report** with top configs by ROI, profit, and Sharpe ratio

## Known Issue

Polymarket APIs are behind Cloudflare. If you get SSL/TLS errors, your network or VPN may be blocking the connection. The code works on unrestricted networks.

## What Needs To Be Done

### Phase 1: Validate Data (Priority: High)
- [ ] Run `python main.py --fetch-only` on a machine with Polymarket access
- [ ] Verify fetched market count (expect 1000+ resolved markets)
- [ ] Check price data coverage — how many markets have usable price histories
- [ ] Inspect `cache/resolved_markets.json` for data quality (missing fields, weird values)

### Phase 2: Run Backtests & Analyze (Priority: High)
- [ ] Run `python main.py --quick` first to sanity-check results
- [ ] Run full `python main.py` and review the report + charts
- [ ] Key questions to answer:
  - Is there any threshold where ROI is consistently positive?
  - Which strategy performs best risk-adjusted (Sharpe)?
  - Does window size matter (are results stable over time or just lucky periods)?
  - Is there enough volume in low-% events to make this practical?

### Phase 3: Improve Realism (Priority: Medium)
- [ ] **Slippage model** — currently assumes you buy at median price; real fills may be worse
- [ ] **Liquidity filter** — skip markets with low volume (can't actually fill orders)
- [ ] **Time-to-resolution weighting** — a 5% bet that takes 1 day vs 6 months has very different annualized returns
- [ ] **Multiple entry points** — instead of one median price, simulate entering at different times
- [ ] **Transaction fees** — Polymarket has no trading fees currently, but include a toggle

### Phase 4: Advanced Strategies (Priority: Medium)
- [ ] **Portfolio-level Kelly** — size across multiple simultaneous bets, not just one at a time
- [ ] **Category filtering** — do low-% bets work better in politics vs crypto vs sports?
- [ ] **Momentum/contrarian** — bet on events where price is dropping (contrarian) or rising (momentum)
- [ ] **Exit strategy** — sell YES shares before resolution if price rises enough (take profit)
- [ ] **Correlation analysis** — avoid betting on correlated events (e.g., multiple elections)

### Phase 5: Live Trading (Priority: Low — only if backtests are promising)
- [ ] Connect to CLOB API with authentication (needs Polygon wallet + API keys)
- [ ] Implement order placement via `POST /order`
- [ ] Add real-time monitoring of open positions
- [ ] Risk limits (max daily loss, max open bets, max per-market exposure)
- [ ] Alert system (Telegram/Discord) for placed bets and outcomes

## Project Structure

```
├── config.py          # All configurable parameters and constants
├── fetcher.py         # API calls to Gamma/CLOB/Data APIs + caching
├── strategies.py      # 5 betting strategies with shared interface
├── backtester.py      # Sliding window engine + full parameter sweep
├── visualizer.py      # Charts (matplotlib) + text report generation
├── main.py            # CLI entry point with argparse
├── requirements.txt   # Python dependencies
├── cache/             # (gitignored) cached API responses
└── output/            # (gitignored) backtest results + charts
```
