"""
Top-Volume Arbitrage Scan
=========================

Quick one-shot scan:
  1. Fetch top-N Polymarket and Kalshi markets by volume
  2. Lexically match them (strict)
  3. Pull orderbooks once per matched pair
  4. Print any opportunities + top near-misses

Run:
  python scan_top_volume.py [--top 100] [--min-similarity 0.85] [--min-edge 0.005]
"""

import argparse
import asyncio
import logging
from datetime import datetime

from utils.config_loader import load_config
from polymarket_client.api import PolymarketClient
from polymarket_client.models import OrderBook, TokenType
from kalshi_client.api import KalshiClient
from core.cross_platform_arb import CrossPlatformArbEngine, MarketMatcher

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
# Quiet noisy libs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("polymarket_client.api").setLevel(logging.WARNING)
logging.getLogger("kalshi_client.api").setLevel(logging.WARNING)
logging.getLogger("core.cross_platform_arb").setLevel(logging.WARNING)
log = logging.getLogger("scan")


async def main(top_n: int, min_sim: float, min_edge: float):
    cfg = load_config("config.yaml")

    poly = PolymarketClient(
        rest_url=cfg.api.polymarket_rest_url,
        ws_url=cfg.api.polymarket_ws_url,
        gamma_url=cfg.api.gamma_api_url,
        timeout=cfg.api.timeout_seconds,
        max_retries=cfg.api.max_retries,
        retry_delay=cfg.api.retry_delay_seconds,
        dry_run=True,
    )
    await poly.connect()
    kalshi = KalshiClient(
        timeout=cfg.api.timeout_seconds,
        max_retries=cfg.api.max_retries,
        dry_run=True,
    )
    await kalshi.__aenter__()

    try:
        log.info("Fetching markets...")
        # Polymarket already returns sorted by volume24hr desc
        poly_all = await poly.list_markets()
        kalshi_all = await kalshi.list_events_with_markets()
        log.info(f"Polymarket: {len(poly_all)} markets, Kalshi: {len(kalshi_all)} markets")

        poly_top = sorted(
            poly_all, key=lambda m: getattr(m, "volume_24h", 0), reverse=True
        )[:top_n]
        kalshi_top = sorted(
            kalshi_all, key=lambda m: getattr(m, "volume", 0), reverse=True
        )[:top_n]

        log.info("=" * 72)
        log.info(f"Top {top_n} Polymarket by 24h volume:")
        for i, m in enumerate(poly_top[:10], 1):
            log.info(f"  {i:2d}. ${m.volume_24h:>12,.0f}  {m.question[:70]}")
        log.info(f"Top {top_n} Kalshi by total volume:")
        for i, m in enumerate(kalshi_top[:10], 1):
            log.info(f"  {i:2d}. {m.volume:>12,.0f}  {m.title[:70]}")
        log.info("=" * 72)

        log.info(f"Matching (strict, min_sim={min_sim})...")
        matcher = MarketMatcher(min_similarity=min_sim, strict=True)
        pairs = await matcher.find_matches(poly_top, kalshi_top)
        log.info(f"=== {len(pairs)} matched pairs ===")
        for p in pairs[:30]:
            log.info(f"  sim={p.similarity_score:.2f} [{p.category}] "
                     f"P:{p.polymarket_question[:55]} <> K:{p.kalshi_title[:55]}")
        if not pairs:
            log.info("No lexical matches in top-volume slice.")
            return
        # Index for orderbook fetch
        poly_by_id = {m.market_id: m for m in poly_top}

        engine = CrossPlatformArbEngine(
            min_edge=min_edge,
            polymarket_taker_fee=cfg.trading.taker_fee_bps / 10000,
            kalshi_taker_fee=0.01,
            gas_cost=cfg.trading.estimated_gas_per_order,
            strict_matching=True,
            min_similarity=min_sim,
        )

        opportunities = []
        near_miss = []  # (net_edge, label)

        async def check_one(pair, idx, total):
            poly_m = poly_by_id.get(pair.polymarket_id)
            if not poly_m or not poly_m.yes_token_id:
                return
            try:
                py_task = poly._fetch_token_orderbook(poly_m.yes_token_id, TokenType.YES)
                pn_task = poly._fetch_token_orderbook(poly_m.no_token_id, TokenType.NO)
                k_task = kalshi.get_orderbook_unified(pair.kalshi_ticker)
                py, pn, kob = await asyncio.gather(
                    py_task, pn_task, k_task, return_exceptions=True
                )
                if any(isinstance(x, Exception) or x is None for x in (py, pn, kob)):
                    return
                poly_ob = OrderBook(
                    market_id=pair.polymarket_id, yes=py, no=pn,
                    timestamp=datetime.utcnow(),
                )
                opp = engine.check_arbitrage(pair, poly_ob, kob)
                if opp:
                    opportunities.append((pair, opp))
                else:
                    # compute best near-miss net edge across 4 directions
                    best = best_label = None
                    pt, kt, gas = engine.polymarket_taker_fee, engine.kalshi_taker_fee, engine.gas_cost
                    for label, buy, sell in [
                        ("YES poly->kalshi", poly_ob.best_ask_yes, kob.best_bid_yes),
                        ("YES kalshi->poly", kob.best_ask_yes, poly_ob.best_bid_yes),
                        ("NO  poly->kalshi", poly_ob.best_ask_no, kob.best_bid_no),
                        ("NO  kalshi->poly", kob.best_ask_no, poly_ob.best_bid_no),
                    ]:
                        if not buy or not sell:
                            continue
                        # apply fees on whichever side - approximate
                        gross = sell - buy
                        fees = buy * (pt if "poly->" in label else kt) \
                             + sell * (kt if "poly->" in label else pt) \
                             + 2 * gas
                        net = gross - fees
                        if best is None or net > best:
                            best, best_label = net, f"{label} buy@{buy:.3f} sell@{sell:.3f}"
                    if best is not None:
                        near_miss.append((
                            best,
                            f"{best_label} | P:{pair.polymarket_question[:45]} <> K:{pair.kalshi_title[:45]}",
                        ))
            except Exception as e:
                log.debug(f"check failed: {e}")

        # Run in batches of 8
        BATCH = 8
        for i in range(0, len(pairs), BATCH):
            batch = pairs[i:i+BATCH]
            await asyncio.gather(*[check_one(p, i+j, len(pairs)) for j, p in enumerate(batch)])
            log.info(f"  scanned {min(i+BATCH, len(pairs))}/{len(pairs)} pairs...")

        log.info("=" * 72)
        log.info(f"OPPORTUNITIES FOUND: {len(opportunities)}")
        for pair, opp in opportunities:
            log.info(f"  {opp}")
            log.info(f"    P: {pair.polymarket_question}")
            log.info(f"    K: {pair.kalshi_title}")

        log.info("=" * 72)
        log.info("Top 15 near-miss spreads (net edge AFTER fees, negative = loss):")
        for net, label in sorted(near_miss, reverse=True)[:15]:
            log.info(f"  net={net:+.4f}  {label}")
        log.info("=" * 72)

    finally:
        await poly.disconnect()
        await kalshi.__aexit__(None, None, None)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--min-similarity", type=float, default=0.85)
    p.add_argument("--min-edge", type=float, default=0.005)
    args = p.parse_args()
    asyncio.run(main(args.top, args.min_similarity, args.min_edge))
