"""Configuration for the Polymarket backtesting system."""

# API endpoints
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"

# Data fetching
FETCH_LIMIT = 100  # markets per page
MAX_MARKETS = 5000  # max markets to fetch
REQUEST_DELAY = 0.15  # seconds between requests to avoid rate limiting
PRICE_HISTORY_CHUNK_DAYS = 15  # chunk size for price history requests
PRICE_HISTORY_FIDELITY = 60  # minutes between price points

# Cache
CACHE_DIR = "cache"
MARKETS_CACHE = f"{CACHE_DIR}/resolved_markets.json"
PRICES_CACHE_DIR = f"{CACHE_DIR}/prices"

# Backtesting defaults
DEFAULT_THRESHOLDS = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
DEFAULT_BET_SIZES = [1, 5, 10, 25, 50, 100]
DEFAULT_WINDOW_SIZES = [30, 60, 90, 180, 365]  # days

# Realism parameters
SLIPPAGE_BPS = 50  # basis points of slippage per trade (0.5%)
MIN_VOLUME = 500  # minimum market volume to include ($)
TRANSACTION_FEE_BPS = 0  # Polymarket currently has 0 fees, but toggle-able

# Strategy types
STRATEGIES = {
    "flat": "Fixed bet size on all qualifying events",
    "kelly": "Kelly criterion sizing based on estimated edge",
    "proportional": "Bet proportional to (threshold - price), bigger edge = bigger bet",
    "inverse": "Bet inversely proportional to price (lower odds = bigger bet)",
    "martingale": "Double bet after each loss, reset after win",
}
