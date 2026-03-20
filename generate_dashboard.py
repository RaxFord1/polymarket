"""Generate a self-contained HTML dashboard from pre-computed backtest results.

Usage:
    python generate_dashboard.py              # Generate output/dashboard.html
    python generate_dashboard.py --offline    # Embed Chart.js inline (no CDN)
"""

import argparse
import json
import os
import sys
from datetime import timedelta

import numpy as np

import config
from backtester import prepare_market_data, run_backtest, _parse_date, _ts_to_date

# Dashboard parameter grid
THRESHOLDS = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
BANKROLLS = [100, 250, 500, 1000, 2500, 5000, 10000, 25000]
STRATEGIES = ["flat", "kelly", "proportional", "inverse", "martingale"]
BET_FRACTION = 0.01  # base_bet = 1% of bankroll


def load_cached_data():
    """Load markets and price data from cache."""
    if not os.path.exists(config.MARKETS_CACHE):
        print(f"ERROR: No cached markets at {config.MARKETS_CACHE}")
        print("Run 'python main.py --fetch-only' or 'python generate_test_data.py' first.")
        sys.exit(1)

    with open(config.MARKETS_CACHE) as f:
        markets = json.load(f)

    price_data_map = {}
    for market in markets:
        cache_file = os.path.join(config.PRICES_CACHE_DIR, f"{market['id']}.json")
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                price_data_map[market["id"]] = json.load(f)

    return markets, price_data_map


def compute_dashboard_data(markets, price_data_map):
    """Run backtests for all parameter combinations and collect results."""
    entries = prepare_market_data(markets, price_data_map)
    if not entries:
        print("ERROR: No valid entries. Check your cached data.")
        sys.exit(1)

    print(f"Prepared {len(entries)} market entries")

    # Find date range for full-window backtest
    all_dates = [e["entry_date"] for e in entries]
    min_date = min(all_dates)
    max_date = max(all_dates)
    total_days = (max_date - min_date).days

    results = {}
    cumulative_pnl = {}
    total = len(BANKROLLS) * len(THRESHOLDS) * len(STRATEGIES)
    done = 0

    for bankroll in BANKROLLS:
        base_bet = bankroll * BET_FRACTION
        for threshold in THRESHOLDS:
            for strategy in STRATEGIES:
                key = f"{bankroll}_{threshold}_{strategy}"

                result = run_backtest(
                    entries, strategy, threshold, base_bet, bankroll,
                    min_date, max_date, total_days
                )

                summary = result.summary()
                results[key] = summary

                # Collect cumulative P&L (downsampled to max 200 points)
                if result.bets:
                    pnl_points = []
                    cum = 0
                    for bet in result.bets:
                        cum += bet.profit
                        pnl_points.append({
                            "d": str(bet.entry_date),
                            "p": round(cum, 2),
                        })
                    # Downsample if too many points
                    if len(pnl_points) > 200:
                        step = len(pnl_points) / 200
                        pnl_points = [pnl_points[int(i * step)]
                                      for i in range(200)]
                    cumulative_pnl[key] = pnl_points

                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{total} combinations computed...")

    print(f"  {done}/{total} combinations computed.")

    # Compute threshold summary (averaged across bankrolls and strategies)
    threshold_summary = {}
    for threshold in THRESHOLDS:
        roi_vals = []
        wr_vals = []
        bets_vals = []
        for bankroll in BANKROLLS:
            for strategy in STRATEGIES:
                key = f"{bankroll}_{threshold}_{strategy}"
                r = results.get(key, {})
                if r.get("total_bets", 0) > 0:
                    roi_vals.append(r["roi"])
                    wr_vals.append(r["win_rate"])
                    bets_vals.append(r["total_bets"])
        threshold_summary[str(threshold)] = {
            "avg_roi": round(float(np.mean(roi_vals)), 4) if roi_vals else 0,
            "avg_win_rate": round(float(np.mean(wr_vals)), 4) if wr_vals else 0,
            "avg_bets": round(float(np.mean(bets_vals)), 1) if bets_vals else 0,
        }

    return {
        "meta": {
            "total_markets": len(markets),
            "total_entries": len(entries),
            "date_range": [str(min_date), str(max_date)],
            "thresholds": THRESHOLDS,
            "bankrolls": BANKROLLS,
            "strategies": STRATEGIES,
            "bet_fraction": BET_FRACTION,
            "slippage_bps": config.SLIPPAGE_BPS,
            "fee_bps": config.TRANSACTION_FEE_BPS,
            "min_volume": config.MIN_VOLUME,
        },
        "strategy_info": {k: v for k, v in config.STRATEGIES.items()},
        "results": results,
        "cumulative_pnl": cumulative_pnl,
        "threshold_summary": threshold_summary,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Polymarket Backtester Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text-muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --orange: #d29922; --purple: #bc8cff;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
    padding: 20px; max-width: 1400px; margin: 0 auto;
  }
  h1 { font-size: 1.8em; margin-bottom: 4px; }
  h2 { font-size: 1.2em; color: var(--text-muted); margin-bottom: 20px; font-weight: normal; }
  h3 { font-size: 1.1em; margin-bottom: 12px; color: var(--accent); }

  .controls {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 20px; margin-bottom: 24px; display: flex; gap: 24px; align-items: flex-end;
    flex-wrap: wrap;
  }
  .control-group { display: flex; flex-direction: column; gap: 6px; }
  .control-group label { font-size: 0.85em; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
  .control-group input, .control-group select {
    background: var(--bg); border: 1px solid var(--border); color: var(--text);
    padding: 8px 12px; border-radius: 6px; font-size: 1em; min-width: 160px;
  }
  .control-group input:focus, .control-group select:focus {
    outline: none; border-color: var(--accent);
  }

  .summary-cards {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin-bottom: 24px;
  }
  .card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px; text-align: center;
  }
  .card .value { font-size: 1.8em; font-weight: 700; }
  .card .label { font-size: 0.8em; color: var(--text-muted); margin-top: 4px; }
  .positive { color: var(--green); }
  .negative { color: var(--red); }

  .charts-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px;
  }
  .chart-container {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px;
  }
  .chart-container.full-width { grid-column: 1 / -1; }
  canvas { max-height: 350px; }

  table {
    width: 100%; border-collapse: collapse; background: var(--surface);
    border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
  }
  th, td { padding: 10px 14px; text-align: right; border-bottom: 1px solid var(--border); }
  th { background: var(--bg); color: var(--text-muted); font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.5px; }
  td:first-child, th:first-child { text-align: left; }
  tr:hover { background: rgba(88, 166, 255, 0.05); }
  .strat-badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; font-weight: 600;
  }

  .meta-info {
    color: var(--text-muted); font-size: 0.8em; margin-top: 20px;
    padding-top: 16px; border-top: 1px solid var(--border);
  }

  @media (max-width: 800px) {
    .charts-grid { grid-template-columns: 1fr; }
    .controls { flex-direction: column; }
  }
</style>
</head>
<body>

<h1>Polymarket Low-Probability Betting Backtester</h1>
<h2>Interactive dashboard — adjust deposit and threshold to explore strategies</h2>

<div class="controls">
  <div class="control-group">
    <label>Deposit ($)</label>
    <input type="number" id="depositInput" value="1000" min="50" max="100000" step="100">
  </div>
  <div class="control-group">
    <label>Probability Threshold</label>
    <select id="thresholdSelect"></select>
  </div>
  <div class="control-group">
    <label>Bet Size</label>
    <span id="betSizeDisplay" style="padding: 8px 0; color: var(--accent); font-weight: 600;"></span>
  </div>
</div>

<div class="summary-cards" id="summaryCards"></div>

<div class="charts-grid">
  <div class="chart-container">
    <h3>ROI by Strategy</h3>
    <canvas id="roiBarChart"></canvas>
  </div>
  <div class="chart-container">
    <h3>Profit by Strategy ($)</h3>
    <canvas id="profitBarChart"></canvas>
  </div>
  <div class="chart-container full-width">
    <h3>Cumulative P&L Over Time</h3>
    <canvas id="cumulativePnlChart"></canvas>
  </div>
  <div class="chart-container">
    <h3>ROI Across All Thresholds</h3>
    <canvas id="roiByThresholdChart"></canvas>
  </div>
  <div class="chart-container">
    <h3>Win Rate vs ROI</h3>
    <canvas id="winRateRoiChart"></canvas>
  </div>
</div>

<h3>Strategy Comparison Table</h3>
<table id="strategyTable">
  <thead>
    <tr>
      <th>Strategy</th><th>Total Bets</th><th>Win Rate</th><th>Total Profit</th>
      <th>ROI</th><th>Max Drawdown</th><th>Sharpe</th><th>Profit Factor</th><th>Avg Days Held</th>
    </tr>
  </thead>
  <tbody id="strategyTableBody"></tbody>
</table>

<div class="meta-info" id="metaInfo"></div>

<script>
const DATA = __JSON_DATA__;

const COLORS = {
  flat: '#58a6ff', kelly: '#3fb950', proportional: '#d29922',
  inverse: '#bc8cff', martingale: '#f85149'
};
const STRATEGY_LABELS = {
  flat: 'Flat', kelly: 'Kelly', proportional: 'Proportional',
  inverse: 'Inverse', martingale: 'Martingale'
};

let charts = {};

function init() {
  // Populate threshold dropdown
  const sel = document.getElementById('thresholdSelect');
  DATA.meta.thresholds.forEach(t => {
    const opt = document.createElement('option');
    opt.value = t;
    opt.textContent = (t * 100).toFixed(0) + '%';
    if (t === 0.10) opt.selected = true;
    sel.appendChild(opt);
  });

  // Meta info
  document.getElementById('metaInfo').innerHTML =
    `Data: ${DATA.meta.total_markets} markets, ${DATA.meta.total_entries} with price data ` +
    `(${DATA.meta.date_range[0]} to ${DATA.meta.date_range[1]}). ` +
    `Slippage: ${DATA.meta.slippage_bps}bps. Fees: ${DATA.meta.fee_bps}bps. Min volume: $${DATA.meta.min_volume}.`;

  // Create charts
  createCharts();

  // Event listeners
  document.getElementById('depositInput').addEventListener('input', update);
  document.getElementById('thresholdSelect').addEventListener('change', update);

  update();
}

function findNearestBankroll(deposit) {
  const brs = DATA.meta.bankrolls;
  let nearest = brs[0];
  let minDist = Math.abs(deposit - brs[0]);
  for (const b of brs) {
    if (Math.abs(deposit - b) < minDist) {
      minDist = Math.abs(deposit - b);
      nearest = b;
    }
  }
  return nearest;
}

function getResults(bankroll, threshold) {
  const out = {};
  DATA.meta.strategies.forEach(s => {
    const key = `${bankroll}_${threshold}_${s}`;
    out[s] = DATA.results[key] || null;
  });
  return out;
}

function scaleResult(result, scaleFactor) {
  if (!result) return null;
  return {
    ...result,
    total_profit: +(result.total_profit * scaleFactor).toFixed(2),
    total_wagered: +(result.total_wagered * scaleFactor).toFixed(2),
    max_drawdown: +(result.max_drawdown * scaleFactor).toFixed(2),
    base_bet: +(result.base_bet * scaleFactor).toFixed(2),
  };
}

function update() {
  const deposit = parseFloat(document.getElementById('depositInput').value) || 1000;
  const threshold = parseFloat(document.getElementById('thresholdSelect').value);
  const nearestBankroll = findNearestBankroll(deposit);
  const scaleFactor = deposit / nearestBankroll;

  document.getElementById('betSizeDisplay').textContent =
    `$${(deposit * DATA.meta.bet_fraction).toFixed(2)} per bet (${(DATA.meta.bet_fraction*100)}% of deposit)`;

  const raw = getResults(nearestBankroll, threshold);
  const results = {};
  for (const s of DATA.meta.strategies) {
    results[s] = scaleResult(raw[s], scaleFactor);
  }

  updateSummaryCards(results, deposit);
  updateTable(results);
  updateRoiBar(results);
  updateProfitBar(results);
  updateCumulativePnl(nearestBankroll, threshold, scaleFactor);
  updateRoiByThreshold(nearestBankroll);
  updateWinRateRoi(nearestBankroll);
}

function updateSummaryCards(results, deposit) {
  // Find best strategy
  let bestStrat = null, bestRoi = -Infinity;
  let totalBets = 0, totalProfit = 0;
  for (const s of DATA.meta.strategies) {
    const r = results[s];
    if (!r) continue;
    totalBets += r.total_bets;
    totalProfit += r.total_profit;
    if (r.roi > bestRoi && r.total_bets >= 5) { bestRoi = r.roi; bestStrat = s; }
  }
  const avgWinRate = DATA.meta.strategies.reduce((sum, s) => {
    return sum + (results[s]?.win_rate || 0);
  }, 0) / DATA.meta.strategies.length;

  const html = `
    <div class="card"><div class="value ${bestRoi >= 0 ? 'positive' : 'negative'}">${STRATEGY_LABELS[bestStrat] || '—'}</div><div class="label">Best Strategy</div></div>
    <div class="card"><div class="value ${bestRoi >= 0 ? 'positive' : 'negative'}">${(bestRoi * 100).toFixed(1)}%</div><div class="label">Best ROI</div></div>
    <div class="card"><div class="value">${totalBets}</div><div class="label">Total Bets (all strategies)</div></div>
    <div class="card"><div class="value ${totalProfit >= 0 ? 'positive' : 'negative'}">$${totalProfit.toFixed(0)}</div><div class="label">Total Profit (all strategies)</div></div>
    <div class="card"><div class="value">${(avgWinRate * 100).toFixed(1)}%</div><div class="label">Avg Win Rate</div></div>
  `;
  document.getElementById('summaryCards').innerHTML = html;
}

function updateTable(results) {
  const tbody = document.getElementById('strategyTableBody');
  let html = '';
  const sorted = DATA.meta.strategies.slice().sort((a, b) => (results[b]?.roi || 0) - (results[a]?.roi || 0));
  for (const s of sorted) {
    const r = results[s];
    if (!r) continue;
    const roiClass = r.roi >= 0 ? 'positive' : 'negative';
    const profitClass = r.total_profit >= 0 ? 'positive' : 'negative';
    html += `<tr>
      <td><span class="strat-badge" style="background:${COLORS[s]}22;color:${COLORS[s]}">${STRATEGY_LABELS[s]}</span></td>
      <td>${r.total_bets}</td>
      <td>${(r.win_rate * 100).toFixed(1)}%</td>
      <td class="${profitClass}">$${r.total_profit.toFixed(2)}</td>
      <td class="${roiClass}">${(r.roi * 100).toFixed(1)}%</td>
      <td>$${r.max_drawdown.toFixed(2)}</td>
      <td>${r.sharpe_ratio.toFixed(3)}</td>
      <td>${r.profit_factor === Infinity ? '∞' : r.profit_factor.toFixed(2)}</td>
      <td>${r.avg_days_held.toFixed(0)}d</td>
    </tr>`;
  }
  tbody.innerHTML = html;
}

function createCharts() {
  const darkGrid = { color: 'rgba(255,255,255,0.06)' };
  const darkTick = { color: '#8b949e' };

  charts.roiBar = new Chart(document.getElementById('roiBarChart'), {
    type: 'bar',
    data: { labels: [], datasets: [{ data: [], backgroundColor: [] }] },
    options: {
      responsive: true, plugins: { legend: { display: false } },
      scales: { y: { grid: darkGrid, ticks: { ...darkTick, callback: v => (v*100)+'%' } }, x: { grid: darkGrid, ticks: darkTick } }
    }
  });

  charts.profitBar = new Chart(document.getElementById('profitBarChart'), {
    type: 'bar',
    data: { labels: [], datasets: [{ data: [], backgroundColor: [] }] },
    options: {
      responsive: true, plugins: { legend: { display: false } },
      scales: { y: { grid: darkGrid, ticks: { ...darkTick, callback: v => '$'+v } }, x: { grid: darkGrid, ticks: darkTick } }
    }
  });

  charts.cumPnl = new Chart(document.getElementById('cumulativePnlChart'), {
    type: 'line',
    data: { datasets: [] },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#8b949e' } } },
      scales: {
        x: { type: 'category', grid: darkGrid, ticks: { ...darkTick, maxTicksLimit: 12 } },
        y: { grid: darkGrid, ticks: { ...darkTick, callback: v => '$'+v } }
      }
    }
  });

  charts.roiByThreshold = new Chart(document.getElementById('roiByThresholdChart'), {
    type: 'line',
    data: { datasets: [] },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#8b949e' } } },
      scales: {
        x: { grid: darkGrid, ticks: { ...darkTick, callback: v => (v*100)+'%' } },
        y: { grid: darkGrid, ticks: { ...darkTick, callback: v => (v*100)+'%' } }
      }
    }
  });

  charts.winRateRoi = new Chart(document.getElementById('winRateRoiChart'), {
    type: 'scatter',
    data: { datasets: [] },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#8b949e' } } },
      scales: {
        x: { grid: darkGrid, ticks: { ...darkTick, callback: v => (v*100)+'%' }, title: { display: true, text: 'Win Rate', color: '#8b949e' } },
        y: { grid: darkGrid, ticks: { ...darkTick, callback: v => (v*100)+'%' }, title: { display: true, text: 'ROI', color: '#8b949e' } }
      }
    }
  });
}

function updateRoiBar(results) {
  const strats = DATA.meta.strategies;
  charts.roiBar.data.labels = strats.map(s => STRATEGY_LABELS[s]);
  charts.roiBar.data.datasets[0].data = strats.map(s => results[s]?.roi || 0);
  charts.roiBar.data.datasets[0].backgroundColor = strats.map(s =>
    (results[s]?.roi || 0) >= 0 ? COLORS[s] : '#f8514966'
  );
  charts.roiBar.update();
}

function updateProfitBar(results) {
  const strats = DATA.meta.strategies;
  charts.profitBar.data.labels = strats.map(s => STRATEGY_LABELS[s]);
  charts.profitBar.data.datasets[0].data = strats.map(s => results[s]?.total_profit || 0);
  charts.profitBar.data.datasets[0].backgroundColor = strats.map(s =>
    (results[s]?.total_profit || 0) >= 0 ? COLORS[s] : '#f8514966'
  );
  charts.profitBar.update();
}

function updateCumulativePnl(bankroll, threshold, scaleFactor) {
  const datasets = [];
  for (const s of DATA.meta.strategies) {
    const key = `${bankroll}_${threshold}_${s}`;
    const pnl = DATA.cumulative_pnl[key];
    if (!pnl || pnl.length === 0) continue;
    datasets.push({
      label: STRATEGY_LABELS[s],
      data: pnl.map(p => p.p * scaleFactor),
      borderColor: COLORS[s],
      backgroundColor: COLORS[s] + '22',
      borderWidth: 2, pointRadius: 0, tension: 0.3, fill: false,
    });
  }
  // Use dates from the longest series
  let longestKey = null, longestLen = 0;
  for (const s of DATA.meta.strategies) {
    const key = `${bankroll}_${threshold}_${s}`;
    const pnl = DATA.cumulative_pnl[key];
    if (pnl && pnl.length > longestLen) { longestLen = pnl.length; longestKey = key; }
  }
  const labels = longestKey ? DATA.cumulative_pnl[longestKey].map(p => p.d) : [];

  charts.cumPnl.data.labels = labels;
  charts.cumPnl.data.datasets = datasets;
  charts.cumPnl.update();
}

function updateRoiByThreshold(bankroll) {
  const datasets = [];
  for (const s of DATA.meta.strategies) {
    const rois = DATA.meta.thresholds.map(t => {
      const key = `${bankroll}_${t}_${s}`;
      return DATA.results[key]?.roi || 0;
    });
    datasets.push({
      label: STRATEGY_LABELS[s],
      data: rois,
      borderColor: COLORS[s],
      backgroundColor: COLORS[s] + '22',
      borderWidth: 2, pointRadius: 4, tension: 0.3,
    });
  }
  charts.roiByThreshold.data.labels = DATA.meta.thresholds;
  charts.roiByThreshold.data.datasets = datasets;
  charts.roiByThreshold.update();
}

function updateWinRateRoi(bankroll) {
  const datasets = [];
  for (const s of DATA.meta.strategies) {
    const points = [];
    for (const t of DATA.meta.thresholds) {
      const key = `${bankroll}_${t}_${s}`;
      const r = DATA.results[key];
      if (r && r.total_bets > 0) {
        points.push({ x: r.win_rate, y: r.roi, r: Math.min(Math.max(r.total_bets / 10, 3), 15) });
      }
    }
    datasets.push({
      label: STRATEGY_LABELS[s],
      data: points,
      backgroundColor: COLORS[s] + 'aa',
      borderColor: COLORS[s],
      pointRadius: points.map(p => p.r),
    });
  }
  charts.winRateRoi.data.datasets = datasets;
  charts.winRateRoi.update();
}

document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>"""


def generate_html(dashboard_data, offline=False):
    """Generate the HTML file with embedded data."""
    json_str = json.dumps(dashboard_data, separators=(',', ':'))
    html = HTML_TEMPLATE.replace('__JSON_DATA__', json_str)
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate backtester dashboard")
    parser.add_argument("--offline", action="store_true", help="Embed Chart.js inline")
    args = parser.parse_args()

    print("Loading cached data...")
    markets, price_data_map = load_cached_data()
    print(f"  {len(markets)} markets, {len(price_data_map)} with price data")

    print("Computing backtest results...")
    dashboard_data = compute_dashboard_data(markets, price_data_map)

    print("Generating HTML...")
    html = generate_html(dashboard_data, offline=args.offline)

    os.makedirs("output", exist_ok=True)
    output_path = "output/dashboard.html"
    with open(output_path, "w") as f:
        f.write(html)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Dashboard saved to {output_path} ({size_mb:.1f} MB)")
    print("Open in any browser to use.")


if __name__ == "__main__":
    main()
