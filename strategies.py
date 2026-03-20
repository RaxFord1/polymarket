"""Betting strategies for backtesting."""

import math


class BaseStrategy:
    """Base class for betting strategies."""

    def __init__(self, base_bet, bankroll=1000.0):
        self.base_bet = base_bet
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.consecutive_losses = 0

    def reset(self, bankroll=None):
        self.bankroll = bankroll if bankroll is not None else self.initial_bankroll
        self.consecutive_losses = 0

    def get_bet_size(self, price, threshold):
        """Return bet size given current price and threshold. Override in subclasses."""
        raise NotImplementedError

    def record_outcome(self, won):
        if won:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

    def can_bet(self, bet_size):
        return self.bankroll >= bet_size > 0


class FlatStrategy(BaseStrategy):
    """Fixed bet size on all qualifying events."""

    name = "flat"

    def get_bet_size(self, price, threshold):
        return self.base_bet


class KellyStrategy(BaseStrategy):
    """Kelly criterion: f* = (bp - q) / b where b=odds, p=estimated prob, q=1-p.

    Uses a fractional Kelly (25%) to reduce variance.
    We estimate the true probability as slightly higher than market price
    (the edge assumption that makes low-prob bets worthwhile).
    """

    name = "kelly"

    def __init__(self, base_bet, bankroll=1000.0, kelly_fraction=0.25, edge_factor=1.5):
        super().__init__(base_bet, bankroll)
        self.kelly_fraction = kelly_fraction
        self.edge_factor = edge_factor

    def get_bet_size(self, price, threshold):
        if price <= 0 or price >= 1:
            return 0

        # Assume true probability is edge_factor * market price
        estimated_prob = min(price * self.edge_factor, threshold)
        b = (1.0 / price) - 1  # decimal odds minus 1
        q = 1 - estimated_prob

        kelly_f = (b * estimated_prob - q) / b
        if kelly_f <= 0:
            return 0

        bet = self.bankroll * kelly_f * self.kelly_fraction
        return min(bet, self.base_bet * 5)  # cap at 5x base


class ProportionalStrategy(BaseStrategy):
    """Bet proportional to the edge (threshold - price).
    Bigger perceived edge = bigger bet.
    """

    name = "proportional"

    def get_bet_size(self, price, threshold):
        if price >= threshold:
            return 0
        edge = (threshold - price) / threshold  # normalized 0-1
        return self.base_bet * (0.5 + edge * 2)  # 0.5x to 2.5x base


class InverseStrategy(BaseStrategy):
    """Bet inversely proportional to price.
    Lower odds = bigger bet (higher potential payout).
    """

    name = "inverse"

    def get_bet_size(self, price, threshold):
        if price <= 0.001:
            return self.base_bet * 3
        # At price=0.01, multiplier=3x. At price=0.20, multiplier~0.7x
        multiplier = min(0.1 / price, 3.0)
        return self.base_bet * max(multiplier, 0.5)


class MartingaleStrategy(BaseStrategy):
    """Double bet after each loss, reset to base after a win.
    Capped at 6 doublings (64x base) to prevent blowup.
    """

    name = "martingale"

    def get_bet_size(self, price, threshold):
        doublings = min(self.consecutive_losses, 6)
        return self.base_bet * (2 ** doublings)


STRATEGY_CLASSES = {
    "flat": FlatStrategy,
    "kelly": KellyStrategy,
    "proportional": ProportionalStrategy,
    "inverse": InverseStrategy,
    "martingale": MartingaleStrategy,
}


def create_strategy(name, base_bet, bankroll=1000.0):
    """Factory function to create a strategy by name."""
    cls = STRATEGY_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_CLASSES.keys())}")
    return cls(base_bet=base_bet, bankroll=bankroll)
