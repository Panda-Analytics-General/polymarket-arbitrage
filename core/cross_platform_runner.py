"""
Cross-Platform-Only Runner
==========================

Dedicated runner that detects arbitrage between Polymarket and Kalshi.

Flow:
    1. Load markets from both platforms.
    2. Match them once using MarketMatcher (strict mode for secure matching).
    3. Poll both order books for each matched pair on a rotation.
    4. Evaluate each pair with CrossPlatformArbEngine.check_arbitrage.
    5. Log opportunities. Execution is dry-run only (no live trading here).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from polymarket_client import PolymarketClient
from polymarket_client.models import TokenType
from kalshi_client import KalshiClient
from core.cross_platform_arb import CrossPlatformArbEngine, MarketPair
from core.llm_verifier import OllamaVerifier
from utils.config_loader import BotConfig


logger = logging.getLogger(__name__)


@dataclass
class CrossPlatformRunnerStats:
    pairs_matched: int = 0
    pairs_polled: int = 0
    opportunities_found: int = 0
    last_snapshot: Optional[datetime] = None


class CrossPlatformRunner:
    """Cross-platform-only arbitrage loop."""

    def __init__(
        self,
        config: BotConfig,
        min_similarity: float = 0.85,
        strict_matching: bool = True,
        poll_interval: float = 5.0,
        pair_batch_size: int = 25,
        use_llm_verifier: bool = False,
        llm_model: str = "llama3.1:8b",
        llm_base_url: str = "http://localhost:11434",
        llm_min_confidence: float = 0.6,
        llm_concurrency: int = 4,
    ):
        """
        Args:
            config: Loaded BotConfig.
            min_similarity: Minimum similarity score the matcher must reach.
            strict_matching: Enable strict mode in MarketMatcher.
            poll_interval: Delay (seconds) between batches when polling pairs.
            pair_batch_size: Number of pairs to poll concurrently per batch.
            use_llm_verifier: If True, run each lexically matched pair through
                a local Ollama LLM to verify resolution criteria match.
            llm_model: Ollama model name (must be pulled locally).
            llm_base_url: Ollama server URL.
            llm_min_confidence: Minimum LLM confidence to accept
                'equivalent' verdict.
        """
        self.config = config
        self.poll_interval = poll_interval
        self.pair_batch_size = pair_batch_size
        self.use_llm_verifier = use_llm_verifier
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_min_confidence = llm_min_confidence
        self.llm_concurrency = llm_concurrency
        self._running = False
        self._start_time: Optional[datetime] = None
        self.stats = CrossPlatformRunnerStats()

        self.polymarket_client: Optional[PolymarketClient] = None
        self.kalshi_client: Optional[KalshiClient] = None

        self.engine = CrossPlatformArbEngine(
            min_edge=config.trading.min_edge,
            polymarket_taker_fee=config.trading.taker_fee_bps / 10000,
            kalshi_taker_fee=0.01,  # Kalshi doesn't publish bps; ~1% avg
            gas_cost=config.trading.estimated_gas_per_order,
            strict_matching=strict_matching,
            min_similarity=min_similarity,
        )

        # Cache token ids and full market objects for matched Polymarket markets
        self._poly_token_ids: dict[str, tuple[str, str]] = {}
        self._poly_markets_by_id: dict[str, object] = {}
        self._kalshi_markets_by_ticker: dict[str, object] = {}
        self._matched_pairs: list[MarketPair] = []
        # pair_id -> (best_net_edge_seen, description_str)
        self._near_miss: dict[str, tuple[float, str]] = {}

    async def start(self) -> None:
        logger.info("=" * 60)
        logger.info("Cross-Platform Arbitrage Runner")
        logger.info("=" * 60)
        logger.info(f"Mode: {'DRY RUN' if self.config.is_dry_run else 'LIVE'}")
        logger.info(f"Strict matching: {self.engine.matcher.strict}")
        logger.info(f"Min similarity: {self.engine.matcher.min_similarity}")
        logger.info(f"Min edge (net): {self.engine.min_edge}")
        logger.info("=" * 60)

        if not self.config.is_dry_run:
            logger.warning(
                "Cross-platform live execution is not implemented. "
                "Forcing dry-run behavior (opportunities will only be logged)."
            )

        self._running = True
        self._start_time = datetime.utcnow()

        # Polymarket client (read-only use here)
        self.polymarket_client = PolymarketClient(
            rest_url=self.config.api.polymarket_rest_url,
            ws_url=self.config.api.polymarket_ws_url,
            gamma_url=self.config.api.gamma_api_url,
            timeout=self.config.api.timeout_seconds,
            max_retries=self.config.api.max_retries,
            retry_delay=self.config.api.retry_delay_seconds,
            dry_run=True,
        )
        await self.polymarket_client.connect()

        # Kalshi client uses async context manager
        self.kalshi_client = KalshiClient(
            timeout=self.config.api.timeout_seconds,
            max_retries=self.config.api.max_retries,
            dry_run=True,
        )
        await self.kalshi_client.__aenter__()

        # Load markets and match
        await self._load_and_match()

        if not self._matched_pairs:
            logger.warning("No matched pairs found - nothing to monitor.")
            return

        # Start polling loop
        asyncio.create_task(self._poll_loop(), name="xplat_poll_loop")
        asyncio.create_task(self._snapshot_loop(), name="xplat_snapshot_loop")

    async def stop(self) -> None:
        logger.info("Shutting down cross-platform runner...")
        self._running = False

        if self.kalshi_client:
            try:
                await self.kalshi_client.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing Kalshi client: {e}")

        if self.polymarket_client:
            await self.polymarket_client.disconnect()

        self._log_final_summary()

    async def _load_and_match(self) -> None:
        """Load markets from both platforms and run the matcher."""
        logger.info("Fetching Polymarket active markets...")
        poly_markets = await self.polymarket_client.list_markets({"active": True})
        logger.info(f"  -> {len(poly_markets)} Polymarket markets")

        logger.info("Fetching Kalshi open markets (via /events, excludes parlays)...")
        kalshi_markets = await self.kalshi_client.list_events_with_markets(
            status="open",
            max_events=10000,
        )
        logger.info(f"  -> {len(kalshi_markets)} Kalshi single-outcome markets")

        # Index token ids for later orderbook fetches
        for m in poly_markets:
            if m.yes_token_id and m.no_token_id:
                self._poly_token_ids[m.market_id] = (m.yes_token_id, m.no_token_id)
            self._poly_markets_by_id[m.market_id] = m
        for km in kalshi_markets:
            self._kalshi_markets_by_ticker[km.ticker] = km

        # Run matcher
        logger.info("Matching markets (strict=%s, min_sim=%.2f)...",
                    self.engine.matcher.strict, self.engine.matcher.min_similarity)
        self._matched_pairs = await self.engine.matcher.find_matches(
            poly_markets, kalshi_markets
        )
        # Keep only pairs for which we have Polymarket token IDs (others can't be priced)
        self._matched_pairs = [
            p for p in self._matched_pairs
            if p.polymarket_id in self._poly_token_ids
        ]
        logger.info(
            f"=== LEXICALLY MATCHED PAIRS: {len(self._matched_pairs)} "
            f"(with orderbook-capable Polymarket tokens) ==="
        )

        # Optional LLM verification of resolution criteria
        if self.use_llm_verifier and self._matched_pairs:
            self._matched_pairs = await self._verify_pairs_with_llm(self._matched_pairs)

        logger.info(f"=== MATCHED PAIRS READY: {len(self._matched_pairs)} ===")
        self.stats.pairs_matched = len(self._matched_pairs)
        for p in self._matched_pairs[:20]:
            logger.info(
                f"  [{p.category}] sim={p.similarity_score:.2f} | "
                f"P: {p.polymarket_question[:60]} | K: {p.kalshi_title[:60]}"
            )
        if len(self._matched_pairs) > 20:
            logger.info(f"  ... and {len(self._matched_pairs) - 20} more")

    async def _verify_pairs_with_llm(
        self, pairs: list[MarketPair]
    ) -> list[MarketPair]:
        """Run each lexically matched pair through Ollama; keep only equivalents."""
        logger.info(
            f"Running LLM verification on {len(pairs)} pairs "
            f"(model={self.llm_model}, min_conf={self.llm_min_confidence})..."
        )
        kept: list[MarketPair] = []
        async with OllamaVerifier(
            base_url=self.llm_base_url,
            model=self.llm_model,
            max_concurrency=self.llm_concurrency,
        ) as verifier:
            if not await verifier.health_check():
                logger.error(
                    "Ollama server unreachable or model not pulled. "
                    "Skipping LLM verification and KEEPING all lexical matches."
                )
                return pairs

            async def _verify_one(idx: int, pair: MarketPair):
                poly_m = self._poly_markets_by_id.get(pair.polymarket_id)
                kalshi_m = self._kalshi_markets_by_ticker.get(pair.kalshi_ticker)
                poly_rules = getattr(poly_m, "description", "") if poly_m else ""
                kalshi_rules = "\n\n".join(
                    s for s in (
                        getattr(kalshi_m, "rules_primary", "") if kalshi_m else "",
                        getattr(kalshi_m, "rules_secondary", "") if kalshi_m else "",
                    ) if s
                )
                decision = await verifier.verify_pair(
                    pair.polymarket_question,
                    poly_rules,
                    pair.kalshi_title,
                    kalshi_rules,
                )
                keep = (
                    decision.is_equivalent
                    and decision.confidence >= self.llm_min_confidence
                )
                logger.info(
                    f"  [{idx}/{len(pairs)}] {'KEEP' if keep else 'DROP'} "
                    f"verdict={decision.verdict} conf={decision.confidence:.2f} | "
                    f"{pair.polymarket_question[:50]} <> {pair.kalshi_title[:50]} "
                    f"| {decision.reasoning[:120]}"
                )
                return pair if keep else None

            results = await asyncio.gather(*[
                _verify_one(i, p) for i, p in enumerate(pairs, 1)
            ], return_exceptions=False)
            kept = [p for p in results if p is not None]

        logger.info(f"LLM verification: {len(kept)}/{len(pairs)} pairs accepted.")
        return kept

    async def _poll_loop(self) -> None:
        """Continuously poll order books for all matched pairs."""
        while self._running:
            try:
                for i in range(0, len(self._matched_pairs), self.pair_batch_size):
                    if not self._running:
                        return
                    batch = self._matched_pairs[i:i + self.pair_batch_size]
                    await asyncio.gather(
                        *(self._check_pair(p) for p in batch),
                        return_exceptions=True,
                    )
                    await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Poll loop error: {e}")
                await asyncio.sleep(1.0)

    async def _check_pair(self, pair: MarketPair) -> None:
        """Fetch orderbooks for a single matched pair and check for arbitrage."""
        try:
            yes_token, no_token = self._poly_token_ids[pair.polymarket_id]

            # Fetch both platforms in parallel
            poly_yes_task = self.polymarket_client._fetch_token_orderbook(
                yes_token, TokenType.YES
            )
            poly_no_task = self.polymarket_client._fetch_token_orderbook(
                no_token, TokenType.NO
            )
            kalshi_task = self.kalshi_client.get_orderbook_unified(pair.kalshi_ticker)

            poly_yes, poly_no, kalshi_ob = await asyncio.gather(
                poly_yes_task, poly_no_task, kalshi_task,
                return_exceptions=True,
            )

            if isinstance(poly_yes, Exception) or isinstance(poly_no, Exception):
                return
            if isinstance(kalshi_ob, Exception) or kalshi_ob is None:
                return

            from polymarket_client.models import OrderBook
            poly_ob = OrderBook(
                market_id=pair.polymarket_id,
                yes=poly_yes,
                no=poly_no,
                timestamp=datetime.utcnow(),
            )

            self.stats.pairs_polled += 1
            opp = self.engine.check_arbitrage(pair, poly_ob, kalshi_ob)
            if opp:
                self.stats.opportunities_found += 1
                logger.info(
                    "ARB! %s | net=%.4f (%.2f%%) | buy %s @%.3f sell @%.3f | "
                    "liq buy=%.0f sell=%.0f | P:%s <> K:%s",
                    opp.token,
                    opp.net_edge,
                    opp.edge_pct * 100,
                    opp.buy_platform,
                    opp.buy_price,
                    opp.sell_price,
                    opp.buy_liquidity,
                    opp.sell_liquidity,
                    pair.polymarket_question[:50],
                    pair.kalshi_title[:50],
                )
            else:
                # Track top near-misses for diagnostics
                self._record_near_miss(pair, poly_ob, kalshi_ob)
        except Exception as e:
            logger.debug(f"Pair check failed for {pair.pair_id}: {e}")

    def _record_near_miss(self, pair, poly_ob, kalshi_ob) -> None:
        """Compute best possible net edge (even if negative) and track it."""
        try:
            py_ask = poly_ob.best_ask_yes or 0
            py_bid = poly_ob.best_bid_yes or 0
            pn_ask = poly_ob.best_ask_no or 0
            pn_bid = poly_ob.best_bid_no or 0
            ky_ask = kalshi_ob.best_ask_yes or 0
            ky_bid = kalshi_ob.best_bid_yes or 0
            kn_ask = kalshi_ob.best_ask_no or 0
            kn_bid = kalshi_ob.best_bid_no or 0

            pt_fee = self.engine.polymarket_taker_fee
            kt_fee = self.engine.kalshi_taker_fee
            gas = self.engine.gas_cost

            candidates = []
            if py_ask and ky_bid:
                candidates.append(("YES poly->kalshi", ky_bid - py_ask
                                   - py_ask*pt_fee - ky_bid*kt_fee - 2*gas,
                                   py_ask, ky_bid))
            if ky_ask and py_bid:
                candidates.append(("YES kalshi->poly", py_bid - ky_ask
                                   - ky_ask*kt_fee - py_bid*pt_fee - 2*gas,
                                   ky_ask, py_bid))
            if pn_ask and kn_bid:
                candidates.append(("NO poly->kalshi", kn_bid - pn_ask
                                   - pn_ask*pt_fee - kn_bid*kt_fee - 2*gas,
                                   pn_ask, kn_bid))
            if kn_ask and pn_bid:
                candidates.append(("NO kalshi->poly", pn_bid - kn_ask
                                   - kn_ask*kt_fee - pn_bid*pt_fee - 2*gas,
                                   kn_ask, pn_bid))
            if not candidates:
                return
            label, net, bp, sp = max(candidates, key=lambda x: x[1])
            desc = (
                f"{label} buy@{bp:.3f} sell@{sp:.3f} | "
                f"{pair.polymarket_question[:40]} <> {pair.kalshi_title[:40]}"
            )
            prev = self._near_miss.get(pair.pair_id)
            if prev is None or net > prev[0]:
                self._near_miss[pair.pair_id] = (net, desc)
        except Exception:
            pass

    async def _snapshot_loop(self) -> None:
        """Periodic log of runtime stats."""
        interval = max(30.0, self.config.monitoring.snapshot_interval)
        while self._running:
            try:
                await asyncio.sleep(interval)
                self.stats.last_snapshot = datetime.utcnow()
                logger.info(
                    "Stats | pairs=%d | polls=%d | opportunities=%d",
                    self.stats.pairs_matched,
                    self.stats.pairs_polled,
                    self.stats.opportunities_found,
                )
                # Log top near-misses (pairs closest to arbitrage)
                top = sorted(
                    self._near_miss.values(), key=lambda x: x[0], reverse=True
                )[:5]
                if top:
                    logger.info("Top 5 near-miss spreads (net edge, negative = loss):")
                    for edge, desc in top:
                        logger.info(f"  net={edge:+.4f} | {desc}")
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Snapshot loop error: {e}")

    def _log_final_summary(self) -> None:
        engine_stats = self.engine.get_stats()
        logger.info("=" * 60)
        logger.info("Cross-Platform Runner Summary")
        logger.info("=" * 60)
        logger.info(f"Matched pairs: {self.stats.pairs_matched}")
        logger.info(f"Pair polls: {self.stats.pairs_polled}")
        logger.info(f"Opportunities detected: {self.stats.opportunities_found}")
        logger.info(f"Avg net edge: {engine_stats['avg_edge']:.4f}")

        # Log top opportunities
        recent = self.engine.get_recent_opportunities(limit=10)
        if recent:
            logger.info("-" * 60)
            logger.info("Recent opportunities:")
            for opp in recent:
                logger.info(f"  {opp}")
        logger.info("=" * 60)
