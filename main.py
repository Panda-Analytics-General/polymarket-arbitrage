#!/usr/bin/env python3
"""
Polymarket Arbitrage Trading Bot
=================================

Main entry point for the trading bot.

Usage:
    python main.py                      # Run in dry-run mode (default)
    python main.py --live               # Run in live mode
    python main.py --backtest           # Run backtest
    python main.py --config my.yaml     # Use custom config file
"""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Optional

from polymarket_client import PolymarketClient
from core.data_feed import DataFeed
from core.arb_engine import ArbEngine, ArbConfig
from core.execution import ExecutionEngine, ExecutionConfig
from core.risk_manager import RiskManager, RiskConfig
from core.portfolio import Portfolio
from utils.config_loader import load_config, BotConfig
from utils.logging_utils import setup_logging, performance_logger


logger = logging.getLogger(__name__)


class TradingBot:
    """
    Main trading bot orchestrator.
    
    Coordinates all components and manages the trading lifecycle.
    """
    
    def __init__(self, config: BotConfig):
        self.config = config
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Components (initialized in start())
        self.client: Optional[PolymarketClient] = None
        self.data_feed: Optional[DataFeed] = None
        self.arb_engine: Optional[ArbEngine] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.risk_manager: Optional[RiskManager] = None
        self.portfolio: Optional[Portfolio] = None
        
        # Statistics
        self._start_time: Optional[datetime] = None
        self._update_count = 0
        self._signal_count = 0
    
    async def start(self) -> None:
        """Initialize and start all components."""
        logger.info("=" * 60)
        logger.info("Polymarket Arbitrage Bot Starting")
        logger.info("=" * 60)
        logger.info(f"Mode: {'DRY RUN' if self.config.is_dry_run else 'LIVE'}")
        logger.info(f"Markets: {self.config.trading.markets or 'Auto-discover'}")
        
        self._start_time = datetime.utcnow()
        self._running = True
        
        # Initialize API client
        self.client = PolymarketClient(
            rest_url=self.config.api.polymarket_rest_url,
            ws_url=self.config.api.polymarket_ws_url,
            gamma_url=self.config.api.gamma_api_url,
            api_key=self.config.api.api_key,
            api_secret=self.config.api.api_secret,
            private_key=self.config.api.private_key,
            timeout=self.config.api.timeout_seconds,
            max_retries=self.config.api.max_retries,
            retry_delay=self.config.api.retry_delay_seconds,
            dry_run=self.config.is_dry_run,
        )
        await self.client.connect()
        
        # Initialize portfolio
        initial_balance = (
            self.config.mode.dry_run_initial_balance 
            if self.config.is_dry_run 
            else 0.0
        )
        self.portfolio = Portfolio(initial_balance=initial_balance)
        
        # Initialize risk manager
        self.risk_manager = RiskManager(RiskConfig(
            max_position_per_market=self.config.risk.max_position_per_market,
            max_global_exposure=self.config.risk.max_global_exposure,
            max_daily_loss=self.config.risk.max_daily_loss,
            max_drawdown_pct=self.config.risk.max_drawdown_pct,
            trade_only_high_volume=self.config.risk.trade_only_high_volume,
            min_24h_volume=self.config.risk.min_24h_volume,
            whitelist=self.config.risk.whitelist,
            blacklist=self.config.risk.blacklist,
            kill_switch_enabled=self.config.risk.kill_switch_enabled,
            auto_unwind_on_breach=self.config.risk.auto_unwind_on_breach,
        ))
        
        # Initialize execution engine
        self.execution_engine = ExecutionEngine(
            client=self.client,
            risk_manager=self.risk_manager,
            portfolio=self.portfolio,
            config=ExecutionConfig(
                slippage_tolerance=self.config.trading.slippage_tolerance,
                order_timeout_seconds=self.config.trading.order_timeout_seconds,
                dry_run=self.config.is_dry_run,
            ),
        )
        await self.execution_engine.start()
        
        # Initialize arbitrage engine
        self.arb_engine = ArbEngine(ArbConfig(
            min_edge=self.config.trading.min_edge,
            bundle_arb_enabled=self.config.trading.bundle_arb_enabled,
            min_spread=self.config.trading.min_spread,
            mm_enabled=self.config.trading.mm_enabled,
            tick_size=self.config.trading.tick_size,
            default_order_size=self.config.trading.default_order_size,
            min_order_size=self.config.trading.min_order_size,
            max_order_size=self.config.trading.max_order_size,
            maker_fee_bps=self.config.trading.maker_fee_bps,
            taker_fee_bps=self.config.trading.taker_fee_bps,
            gas_cost_per_order=self.config.trading.estimated_gas_per_order,
        ))
        
        # Initialize data feed
        market_ids = self.config.trading.markets.copy()
        self.data_feed = DataFeed(
            client=self.client,
            market_ids=market_ids,
            position_refresh_interval=5.0,
            on_update=self._on_market_update,
            config=self.config,
        )
        await self.data_feed.start()
        
        # Wait for initial data
        logger.info("Waiting for market data...")
        if not await self.data_feed.wait_for_data(timeout=30.0):
            logger.warning("Timeout waiting for initial data, proceeding anyway")
        
        logger.info("Bot started successfully!")
        logger.info("-" * 60)
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
        
        # Start fill simulation for dry run
        if self.config.is_dry_run and self.config.mode.simulate_fills:
            asyncio.create_task(self._simulate_fills())
    
    def _on_market_update(self, market_id: str, market_state) -> None:
        """Callback for market state updates."""
        self._update_count += 1
        
        # Check risk limits
        if not self.risk_manager.within_global_limits():
            logger.warning("Risk limits exceeded, skipping analysis")
            return
        
        # Analyze for opportunities
        signals = self.arb_engine.analyze(market_state)
        
        for signal in signals:
            self._signal_count += 1
            # Submit signal asynchronously
            asyncio.create_task(self.execution_engine.submit_signal(signal))
    
    async def _monitoring_loop(self) -> None:
        """Periodic monitoring and logging."""
        interval = self.config.monitoring.snapshot_interval
        
        while self._running:
            try:
                await asyncio.sleep(interval)
                
                # Log portfolio snapshot
                pnl = self.portfolio.get_pnl()
                exposure = self.portfolio.get_total_exposure()
                positions = len(self.portfolio.get_all_positions())
                open_orders = self.execution_engine.open_order_count
                
                performance_logger.log_snapshot(pnl, exposure, positions, open_orders)
                
                # Update risk manager
                self.risk_manager.update_pnl(
                    pnl["realized_pnl"],
                    pnl["unrealized_pnl"]
                )
                
                # Log statistics
                arb_stats = self.arb_engine.get_stats()
                exec_stats = self.execution_engine.get_stats()
                risk_summary = self.risk_manager.get_summary()
                
                logger.info(
                    f"Stats | Updates: {self._update_count} | "
                    f"Signals: {self._signal_count} | "
                    f"Orders: {exec_stats.orders_placed} placed, {exec_stats.orders_filled} filled | "
                    f"PnL: ${pnl['total_pnl']:.2f}"
                )
                
                if risk_summary["kill_switch_triggered"]:
                    logger.critical("KILL SWITCH ACTIVE - Trading halted")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    async def _simulate_fills(self) -> None:
        """Simulate order fills in dry run mode."""
        import random
        
        while self._running:
            try:
                await asyncio.sleep(2.0)  # Check every 2 seconds
                
                # Get open orders
                orders = self.execution_engine.get_open_orders()
                
                for order in orders:
                    # Random chance of fill
                    if random.random() < self.config.mode.fill_probability:
                        trade = self.client.simulate_fill(order.order_id)
                        if trade:
                            self.execution_engine.handle_fill(trade)
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fill simulation error: {e}")
    
    async def stop(self) -> None:
        """Stop all components gracefully."""
        logger.info("Shutting down...")
        self._running = False
        
        if self.data_feed:
            await self.data_feed.stop()
        
        if self.execution_engine:
            await self.execution_engine.stop()
        
        if self.client:
            await self.client.disconnect()
        
        # Final summary
        if self.portfolio:
            summary = self.portfolio.get_summary()
            logger.info("=" * 60)
            logger.info("Final Portfolio Summary")
            logger.info("=" * 60)
            logger.info(f"Total PnL: ${summary['pnl']['total_pnl']:.2f}")
            logger.info(f"  Realized: ${summary['pnl']['realized_pnl']:.2f}")
            logger.info(f"  Unrealized: ${summary['pnl']['unrealized_pnl']:.2f}")
            logger.info(f"Total Trades: {summary['total_trades']}")
            logger.info(f"Win Rate: {summary['win_rate']:.1%}")
            logger.info(f"Total Volume: ${summary['total_volume']:.2f}")
        
        if self.arb_engine:
            stats = self.arb_engine.get_stats()
            logger.info("-" * 60)
            logger.info(f"Bundle Opportunities: {stats.bundle_opportunities_detected}")
            logger.info(f"MM Opportunities: {stats.mm_opportunities_detected}")
            logger.info(f"Signals Generated: {stats.signals_generated}")
        
        logger.info("=" * 60)
        logger.info("Bot stopped")
        
        self._shutdown_event.set()
    
    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()


async def run_backtest(config: BotConfig, duration: float = 300.0) -> None:
    """Run a backtest simulation."""
    from utils.backtest import BacktestConfig, BacktestEngine, run_backtest as _run_backtest
    
    logger.info("Starting backtest mode...")
    
    # Create components
    portfolio = Portfolio(initial_balance=config.mode.dry_run_initial_balance)
    
    risk_manager = RiskManager(RiskConfig(
        max_position_per_market=config.risk.max_position_per_market,
        max_global_exposure=config.risk.max_global_exposure,
        max_daily_loss=config.risk.max_daily_loss,
        max_drawdown_pct=config.risk.max_drawdown_pct,
    ))
    
    arb_engine = ArbEngine(ArbConfig(
        min_edge=config.trading.min_edge,
        bundle_arb_enabled=config.trading.bundle_arb_enabled,
        min_spread=config.trading.min_spread,
        mm_enabled=config.trading.mm_enabled,
        tick_size=config.trading.tick_size,
        default_order_size=config.trading.default_order_size,
        maker_fee_bps=config.trading.maker_fee_bps,
        taker_fee_bps=config.trading.taker_fee_bps,
        gas_cost_per_order=config.trading.estimated_gas_per_order,
    ))
    
    # Use placeholder client for execution
    client = PolymarketClient(dry_run=True)
    await client.connect()
    
    execution_engine = ExecutionEngine(
        client=client,
        risk_manager=risk_manager,
        portfolio=portfolio,
        config=ExecutionConfig(dry_run=True),
    )
    await execution_engine.start()
    
    # Run backtest
    backtest_config = BacktestConfig(
        initial_balance=config.mode.dry_run_initial_balance,
        simulate_fills=True,
        fill_probability=config.mode.fill_probability,
    )
    
    # Generate market IDs
    market_ids = config.trading.markets or [f"market_{i}" for i in range(3)]
    
    result = await _run_backtest(
        config=backtest_config,
        market_ids=market_ids,
        arb_engine=arb_engine,
        execution_engine=execution_engine,
        risk_manager=risk_manager,
        portfolio=portfolio,
        duration_seconds=duration,
    )
    
    await execution_engine.stop()
    await client.disconnect()
    
    return result


async def main_async(args: argparse.Namespace) -> None:
    """Async main function."""
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Override mode from command line
    if args.live:
        config.mode.trading_mode = "live"
    elif args.dry_run:
        config.mode.trading_mode = "dry_run"
    
    # Run backtest if requested
    if args.backtest:
        await run_backtest(config, duration=args.backtest_duration)
        return
    
    # Create and run the bot
    bot = TradingBot(config)
    
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(bot.stop())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass
    
    try:
        await bot.start()
        await bot.wait_for_shutdown()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        await bot.stop()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        await bot.stop()
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Polymarket Arbitrage Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Run in dry-run mode
  python main.py --live             Run in live trading mode
  python main.py --backtest         Run backtest simulation
  python main.py -c custom.yaml     Use custom config file
        """
    )
    
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live trading mode"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Run in dry-run mode (default)"
    )
    
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest simulation"
    )
    
    parser.add_argument(
        "--backtest-duration",
        type=float,
        default=300.0,
        help="Backtest duration in simulated seconds (default: 300)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(console_level=log_level)
    
    # Run the async main
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nShutdown complete.")


if __name__ == "__main__":
    main()

