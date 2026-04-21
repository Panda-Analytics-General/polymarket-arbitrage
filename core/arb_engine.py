"""
Arbitrage Engine Module
========================

Detects trading opportunities including:
1. Bundle mispricing (YES + NO != 1.0)
2. Market-making spread capture
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from polymarket_client.models import (
    MarketState,
    Opportunity,
    OpportunityType,
    OrderBook,
    OrderSide,
    Signal,
    TokenType,
)


logger = logging.getLogger(__name__)


@dataclass
class ArbConfig:
    """Configuration for the arbitrage engine."""
    # Bundle arbitrage
    min_edge: float = 0.01  # Minimum edge required (1%)
    bundle_arb_enabled: bool = True
    
    # Market-making
    min_spread: float = 0.05  # Minimum spread to MM (5c)
    mm_enabled: bool = True
    tick_size: float = 0.01
    
    # Sizing
    default_order_size: float = 50.0
    min_order_size: float = 5.0
    max_order_size: float = 200.0
    
    # Signal expiry
    signal_expiry_seconds: float = 5.0
    
    # Fees (in basis points - 100 bps = 1%)
    # Polymarket taker fees are price-dependent (feeRate * p * (1-p)).
    # 100 bps (~1%) approximates the average near p=0.5 for most categories.
    # Sell orders pay no taker fee (see _check_bundle_arbitrage).
    maker_fee_bps: float = 0        # Limit orders adding liquidity
    taker_fee_bps: float = 100      # Taking liquidity
    gas_cost_per_order: float = 0.02  # ~$0.02 on Polygon


@dataclass
class OpportunityTiming:
    """Tracks timing of a specific opportunity."""
    opportunity_id: str
    market_id: str
    opportunity_type: str
    detected_at: datetime
    edge: float
    expired_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    was_executed: bool = False
    
    def mark_expired(self, executed: bool = False) -> None:
        """Mark this opportunity as expired."""
        self.expired_at = datetime.utcnow()
        self.duration_ms = (self.expired_at - self.detected_at).total_seconds() * 1000
        self.was_executed = executed


@dataclass
class ArbStats:
    """Statistics for the arbitrage engine."""
    bundle_opportunities_detected: int = 0
    mm_opportunities_detected: int = 0
    signals_generated: int = 0
    last_opportunity_time: Optional[datetime] = None
    
    # Opportunity duration tracking
    total_opportunities_tracked: int = 0
    avg_opportunity_duration_ms: float = 0.0
    min_opportunity_duration_ms: float = float('inf')
    max_opportunity_duration_ms: float = 0.0
    opportunities_under_100ms: int = 0
    opportunities_under_500ms: int = 0
    opportunities_under_1s: int = 0
    opportunities_over_1s: int = 0


class ArbEngine:
    """
    Arbitrage and market-making opportunity detection engine.
    
    Analyzes market states from the DataFeed and generates trading
    signals for the ExecutionEngine.
    """
    
    def __init__(self, config: ArbConfig):
        self.config = config
        self.stats = ArbStats()
        
        # Track recent opportunities to avoid duplicates
        self._recent_opportunities: dict[str, Opportunity] = {}
        self._opportunity_cooldown: dict[str, datetime] = {}
        
        # Track active opportunities for duration measurement
        self._active_opportunities: dict[str, OpportunityTiming] = {}
        self._opportunity_history: list[OpportunityTiming] = []
        
        logger.info(f"ArbEngine initialized with min_edge={config.min_edge}, min_spread={config.min_spread}")
    
    def analyze(self, market_state: MarketState) -> list[Signal]:
        """
        Analyze a market state and generate trading signals.
        
        Returns a list of signals (may be empty if no opportunities).
        """
        signals: list[Signal] = []
        
        order_book = market_state.order_book
        market_id = market_state.market.market_id
        
        # Check if previously tracked opportunities have expired
        self._check_expired_opportunities(market_id, order_book)
        
        # Check for bundle arbitrage
        if self.config.bundle_arb_enabled:
            bundle_signal = self._check_bundle_arbitrage(market_id, order_book)
            if bundle_signal:
                signals.append(bundle_signal)
        
        # Check for market-making opportunities
        if self.config.mm_enabled:
            mm_signals = self._check_market_making(market_id, order_book)
            signals.extend(mm_signals)
        
        return signals
    
    def _check_expired_opportunities(self, market_id: str, order_book: OrderBook) -> None:
        """Check if any tracked opportunities have expired (prices moved away)."""
        now = datetime.utcnow()
        expired_keys = []
        
        for key, timing in self._active_opportunities.items():
            if timing.market_id != market_id:
                continue
            
            # Check if opportunity still exists
            still_valid = False
            
            if "bundle_long" in timing.opportunity_type:
                # Check if total ask is still < 1 - min_edge
                if order_book.best_ask_yes and order_book.best_ask_no:
                    total_ask = order_book.best_ask_yes + order_book.best_ask_no
                    if 1.0 - total_ask >= self.config.min_edge * 0.5:  # Use lower threshold
                        still_valid = True
                        
            elif "bundle_short" in timing.opportunity_type:
                # Check if total bid is still > 1 + min_edge
                if order_book.best_bid_yes and order_book.best_bid_no:
                    total_bid = order_book.best_bid_yes + order_book.best_bid_no
                    if total_bid - 1.0 >= self.config.min_edge * 0.5:
                        still_valid = True
            
            # Also expire if too old (10 seconds max)
            age_seconds = (now - timing.detected_at).total_seconds()
            if age_seconds > 10:
                still_valid = False
            
            if not still_valid:
                timing.mark_expired(executed=False)
                self._record_opportunity_duration(timing)
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._active_opportunities[key]
    
    def _record_opportunity_duration(self, timing: OpportunityTiming) -> None:
        """Record the duration of an expired opportunity and update stats."""
        if timing.duration_ms is None:
            return
        
        self._opportunity_history.append(timing)
        
        # Keep only last 1000 opportunities
        if len(self._opportunity_history) > 1000:
            self._opportunity_history = self._opportunity_history[-500:]
        
        # Update stats
        self.stats.total_opportunities_tracked += 1
        
        # Update min/max
        if timing.duration_ms < self.stats.min_opportunity_duration_ms:
            self.stats.min_opportunity_duration_ms = timing.duration_ms
        if timing.duration_ms > self.stats.max_opportunity_duration_ms:
            self.stats.max_opportunity_duration_ms = timing.duration_ms
        
        # Update running average
        n = self.stats.total_opportunities_tracked
        old_avg = self.stats.avg_opportunity_duration_ms
        self.stats.avg_opportunity_duration_ms = old_avg + (timing.duration_ms - old_avg) / n
        
        # Update duration buckets
        if timing.duration_ms < 100:
            self.stats.opportunities_under_100ms += 1
        elif timing.duration_ms < 500:
            self.stats.opportunities_under_500ms += 1
        elif timing.duration_ms < 1000:
            self.stats.opportunities_under_1s += 1
        else:
            self.stats.opportunities_over_1s += 1
        
        logger.info(
            f"Opportunity EXPIRED: {timing.opportunity_type} | "
            f"duration={timing.duration_ms:.0f}ms | edge={timing.edge:.4f} | "
            f"market={timing.market_id}"
        )
    
    def _start_tracking_opportunity(self, opportunity: Opportunity) -> None:
        """Start tracking an opportunity for duration measurement."""
        key = f"{opportunity.market_id}_{opportunity.opportunity_type.value}"
        
        # Don't double-track
        if key in self._active_opportunities:
            return
        
        timing = OpportunityTiming(
            opportunity_id=opportunity.opportunity_id,
            market_id=opportunity.market_id,
            opportunity_type=opportunity.opportunity_type.value,
            detected_at=datetime.utcnow(),
            edge=opportunity.edge,
        )
        self._active_opportunities[key] = timing
    
    def mark_opportunity_executed(self, market_id: str, opportunity_type: str) -> None:
        """Mark an opportunity as executed (for accurate tracking)."""
        key = f"{market_id}_{opportunity_type}"
        if key in self._active_opportunities:
            timing = self._active_opportunities[key]
            timing.mark_expired(executed=True)
            self._record_opportunity_duration(timing)
            del self._active_opportunities[key]
    
    def get_timing_stats(self) -> dict:
        """Get opportunity timing statistics for dashboard."""
        recent_history = self._opportunity_history[-100:] if self._opportunity_history else []
        
        return {
            "total_tracked": self.stats.total_opportunities_tracked,
            "avg_duration_ms": round(self.stats.avg_opportunity_duration_ms, 1),
            "min_duration_ms": round(self.stats.min_opportunity_duration_ms, 1) if self.stats.min_opportunity_duration_ms != float('inf') else None,
            "max_duration_ms": round(self.stats.max_opportunity_duration_ms, 1),
            "under_100ms": self.stats.opportunities_under_100ms,
            "under_500ms": self.stats.opportunities_under_500ms,
            "under_1s": self.stats.opportunities_under_1s,
            "over_1s": self.stats.opportunities_over_1s,
            "active_opportunities": len(self._active_opportunities),
            "recent_durations": [
                {
                    "type": t.opportunity_type,
                    "duration_ms": round(t.duration_ms, 1) if t.duration_ms else 0,
                    "edge": round(t.edge, 4),
                    "executed": t.was_executed,
                    "time": t.detected_at.isoformat(),
                }
                for t in recent_history[-20:]
            ]
        }
    
    def _check_bundle_arbitrage(self, market_id: str, order_book: OrderBook) -> Optional[Signal]:
        """
        Check for bundle mispricing opportunities.
        
        Bundle Long: Buy YES + NO when total_ask < 1 - min_edge - fees
        Bundle Short: Sell YES + NO when total_bid > 1 + min_edge + fees
        
        Fees are factored in to ensure net profitability!
        """
        # Get prices
        best_ask_yes = order_book.best_ask_yes
        best_ask_no = order_book.best_ask_no
        best_bid_yes = order_book.best_bid_yes
        best_bid_no = order_book.best_bid_no
        
        # Need all prices to evaluate
        if None in (best_ask_yes, best_ask_no, best_bid_yes, best_bid_no):
            return None
        
        total_ask = best_ask_yes + best_ask_no
        total_bid = best_bid_yes + best_bid_no
        
        # Calculate total fees for 2 orders (buy YES + buy NO, or sell both)
        # Fee is percentage of notional, applied to each leg
        taker_fee_pct = self.config.taker_fee_bps / 10000  # Convert bps to decimal
        gas_cost = self.config.gas_cost_per_order * 2  # 2 orders
        
        # For bundle long: we buy both, pay taker fees on each
        # Fee cost = taker_fee_pct * (ask_yes + ask_no) = taker_fee_pct * total_ask
        fee_cost_long = taker_fee_pct * total_ask
        
        # For bundle short: we sell both. Polymarket does NOT charge taker fees
        # on sell orders, so there is no per-notional fee to subtract.
        fee_cost_short = 0.0
        
        opportunity: Optional[Opportunity] = None
        
        # Check for bundle long opportunity (buy both for < $1)
        # Must be profitable AFTER fees: 1.0 - total_ask - fees > min_edge
        gross_edge_long = 1.0 - total_ask
        net_edge_long = gross_edge_long - fee_cost_long - gas_cost
        
        if net_edge_long >= self.config.min_edge:
            edge = net_edge_long  # Use NET edge (after fees)
            
            # Calculate max size based on liquidity
            yes_ask_size = order_book.yes.best_ask_size or 0
            no_ask_size = order_book.no.best_ask_size or 0
            max_size = min(yes_ask_size, no_ask_size)
            
            suggested_size = min(
                self.config.default_order_size / max(best_ask_yes, best_ask_no),
                max_size
            )
            suggested_size = max(self.config.min_order_size, suggested_size)
            
            opportunity = Opportunity(
                opportunity_id=f"bundle_long_{uuid.uuid4().hex[:8]}",
                opportunity_type=OpportunityType.BUNDLE_LONG,
                market_id=market_id,
                edge=edge,
                best_bid_yes=best_bid_yes,
                best_ask_yes=best_ask_yes,
                best_bid_no=best_bid_no,
                best_ask_no=best_ask_no,
                suggested_size=suggested_size,
                max_size=max_size,
                expires_at=datetime.utcnow() + timedelta(seconds=self.config.signal_expiry_seconds),
            )
            
            self.stats.bundle_opportunities_detected += 1
            logger.info(
                f"Bundle LONG opportunity: {market_id} | "
                f"total_ask={total_ask:.4f} | gross={gross_edge_long:.4f} | "
                f"fees={fee_cost_long:.4f} | NET edge={edge:.4f} | size={suggested_size:.2f}"
            )
        
        # Check for bundle short opportunity (sell both for > $1)
        # Must be profitable AFTER fees: total_bid - 1.0 - fees > min_edge
        gross_edge_short = total_bid - 1.0
        net_edge_short = gross_edge_short - fee_cost_short - gas_cost
        
        if opportunity is None and net_edge_short >= self.config.min_edge:
            edge = net_edge_short  # Use NET edge (after fees)
            
            # Calculate max size based on liquidity
            yes_bid_size = order_book.yes.best_bid_size or 0
            no_bid_size = order_book.no.best_bid_size or 0
            max_size = min(yes_bid_size, no_bid_size)
            
            suggested_size = min(
                self.config.default_order_size / max(best_bid_yes, best_bid_no),
                max_size
            )
            suggested_size = max(self.config.min_order_size, suggested_size)
            
            opportunity = Opportunity(
                opportunity_id=f"bundle_short_{uuid.uuid4().hex[:8]}",
                opportunity_type=OpportunityType.BUNDLE_SHORT,
                market_id=market_id,
                edge=edge,
                best_bid_yes=best_bid_yes,
                best_ask_yes=best_ask_yes,
                best_bid_no=best_bid_no,
                best_ask_no=best_ask_no,
                suggested_size=suggested_size,
                max_size=max_size,
                expires_at=datetime.utcnow() + timedelta(seconds=self.config.signal_expiry_seconds),
            )
            
            self.stats.bundle_opportunities_detected += 1
            logger.info(
                f"Bundle SHORT opportunity: {market_id} | "
                f"total_bid={total_bid:.4f} | gross={gross_edge_short:.4f} | "
                f"fees={fee_cost_short:.4f} | NET edge={edge:.4f} | size={suggested_size:.2f}"
                f"total_bid={total_bid:.4f} | edge={edge:.4f} | size={suggested_size:.2f}"
            )
        
        if not opportunity:
            return None
        
        # Check cooldown to avoid spam
        cooldown_key = f"{market_id}_{opportunity.opportunity_type.value}"
        if cooldown_key in self._opportunity_cooldown:
            if datetime.utcnow() < self._opportunity_cooldown[cooldown_key]:
                return None
        
        self._opportunity_cooldown[cooldown_key] = datetime.utcnow() + timedelta(seconds=2)
        self._recent_opportunities[opportunity.opportunity_id] = opportunity
        self.stats.last_opportunity_time = datetime.utcnow()
        
        # Start tracking for duration measurement
        self._start_tracking_opportunity(opportunity)
        
        # Generate signal
        return self._create_bundle_signal(opportunity)
    
    def _create_bundle_signal(self, opportunity: Opportunity) -> Signal:
        """Create a trading signal for a bundle arbitrage opportunity."""
        orders = []
        
        if opportunity.opportunity_type == OpportunityType.BUNDLE_LONG:
            # Buy both YES and NO at ask prices
            orders = [
                {
                    "token_type": TokenType.YES,
                    "side": OrderSide.BUY,
                    "price": opportunity.best_ask_yes,
                    "size": opportunity.suggested_size,
                    "strategy_tag": "bundle_arb",
                },
                {
                    "token_type": TokenType.NO,
                    "side": OrderSide.BUY,
                    "price": opportunity.best_ask_no,
                    "size": opportunity.suggested_size,
                    "strategy_tag": "bundle_arb",
                },
            ]
        else:
            # Sell both YES and NO at bid prices
            orders = [
                {
                    "token_type": TokenType.YES,
                    "side": OrderSide.SELL,
                    "price": opportunity.best_bid_yes,
                    "size": opportunity.suggested_size,
                    "strategy_tag": "bundle_arb",
                },
                {
                    "token_type": TokenType.NO,
                    "side": OrderSide.SELL,
                    "price": opportunity.best_bid_no,
                    "size": opportunity.suggested_size,
                    "strategy_tag": "bundle_arb",
                },
            ]
        
        signal = Signal(
            signal_id=f"sig_{uuid.uuid4().hex[:12]}",
            action="place_orders",
            market_id=opportunity.market_id,
            opportunity=opportunity,
            orders=orders,
            priority=10,  # High priority for arb
        )
        
        self.stats.signals_generated += 1
        return signal
    
    def _check_market_making(self, market_id: str, order_book: OrderBook) -> list[Signal]:
        """
        Check for market-making opportunities on YES and NO tokens.
        
        Place limit orders inside the spread to capture the bid-ask spread.
        """
        signals = []
        
        # Check YES token
        yes_signal = self._check_mm_token(market_id, order_book.yes, TokenType.YES)
        if yes_signal:
            signals.append(yes_signal)
        
        # Check NO token
        no_signal = self._check_mm_token(market_id, order_book.no, TokenType.NO)
        if no_signal:
            signals.append(no_signal)
        
        return signals
    
    def _check_mm_token(
        self,
        market_id: str,
        token_book,
        token_type: TokenType
    ) -> Optional[Signal]:
        """Check market-making opportunity for a single token."""
        best_bid = token_book.best_bid
        best_ask = token_book.best_ask
        spread = token_book.spread
        
        if spread is None or best_bid is None or best_ask is None:
            return None
        
        # Check if spread is wide enough
        if spread < self.config.min_spread:
            return None
        
        # Check cooldown
        cooldown_key = f"mm_{market_id}_{token_type.value}"
        if cooldown_key in self._opportunity_cooldown:
            if datetime.utcnow() < self._opportunity_cooldown[cooldown_key]:
                return None
        
        self._opportunity_cooldown[cooldown_key] = datetime.utcnow() + timedelta(seconds=5)
        
        # Calculate our prices (inside the spread)
        our_bid = best_bid + self.config.tick_size
        our_ask = best_ask - self.config.tick_size
        
        # Make sure we still have positive edge
        if our_ask <= our_bid:
            return None
        
        our_spread = our_ask - our_bid
        if our_spread < self.config.tick_size * 2:
            return None
        
        # Calculate size
        order_size = self.config.default_order_size / ((our_bid + our_ask) / 2)
        order_size = min(order_size, self.config.max_order_size)
        order_size = max(order_size, self.config.min_order_size)
        
        # Create opportunity for logging
        opportunity = Opportunity(
            opportunity_id=f"mm_{token_type.value}_{uuid.uuid4().hex[:8]}",
            opportunity_type=OpportunityType.MM_BID if token_type == TokenType.YES else OpportunityType.MM_ASK,
            market_id=market_id,
            edge=our_spread / 2,  # Expected edge per side
            suggested_size=order_size,
            max_size=order_size * 2,
        )
        
        self.stats.mm_opportunities_detected += 1
        self.stats.last_opportunity_time = datetime.utcnow()
        
        logger.info(
            f"MM opportunity: {market_id}/{token_type.value} | "
            f"spread={spread:.4f} | our_spread={our_spread:.4f} | size={order_size:.2f}"
        )
        
        # Generate signal with both bid and ask orders
        orders = [
            {
                "token_type": token_type,
                "side": OrderSide.BUY,
                "price": our_bid,
                "size": order_size,
                "strategy_tag": "market_making",
            },
            {
                "token_type": token_type,
                "side": OrderSide.SELL,
                "price": our_ask,
                "size": order_size,
                "strategy_tag": "market_making",
            },
        ]
        
        signal = Signal(
            signal_id=f"sig_{uuid.uuid4().hex[:12]}",
            action="place_orders",
            market_id=market_id,
            opportunity=opportunity,
            orders=orders,
            priority=5,  # Lower priority than arb
        )
        
        self.stats.signals_generated += 1
        return signal
    
    def get_recent_opportunities(self, max_age_seconds: float = 60.0) -> list[Opportunity]:
        """Get recently detected opportunities."""
        cutoff = datetime.utcnow() - timedelta(seconds=max_age_seconds)
        return [
            opp for opp in self._recent_opportunities.values()
            if opp.detected_at > cutoff
        ]
    
    def clear_expired_opportunities(self) -> int:
        """Remove expired opportunities from cache."""
        now = datetime.utcnow()
        expired = [
            opp_id for opp_id, opp in self._recent_opportunities.items()
            if opp.expires_at and opp.expires_at < now
        ]
        for opp_id in expired:
            del self._recent_opportunities[opp_id]
        return len(expired)
    
    def get_stats(self) -> ArbStats:
        """Get engine statistics."""
        return self.stats

