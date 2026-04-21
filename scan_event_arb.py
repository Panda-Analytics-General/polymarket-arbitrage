"""
Event-Level Cross-Platform Arbitrage Scan
=========================================

Strategy:
  1. Pull events (with nested markets) from both platforms, ranked by volume.
  2. Take top-N events per side.
  3. Match events by title similarity.
  4. Inside each matched event, join markets by candidate-name similarity
     (Polymarket `groupItemTitle` <-> Kalshi `yes_sub_title`).
  5. Fetch orderbooks once per joined market and report any net-positive arb.

Run:
  python scan_event_arb.py [--top 500] [--event-sim 0.7] [--candidate-sim 0.8]
"""

import argparse
import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional

import httpx

from utils.config_loader import load_config
from polymarket_client.api import PolymarketClient
from polymarket_client.models import OrderBook, TokenType
from kalshi_client.api import KalshiClient

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
for noisy in ("httpx", "httpcore", "polymarket_client.api", "kalshi_client.api"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("evtarb")


# ---------------------------------------------------------------------------
# Data classes (lightweight)
# ---------------------------------------------------------------------------

@dataclass
class PolyMarketLite:
    market_id: str
    question: str
    candidate: str          # groupItemTitle or fallback
    yes_token_id: str
    no_token_id: str
    volume_24h: float


@dataclass
class PolyEvent:
    event_id: str
    title: str
    volume_24h: float
    markets: list[PolyMarketLite]


@dataclass
class KalshiMarketLite:
    ticker: str
    title: str
    candidate: str          # yes_sub_title / subtitle
    volume: float


@dataclass
class KalshiEvent:
    event_ticker: str
    title: str
    volume: float
    markets: list[KalshiMarketLite]


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]")
_STOP = {
    "the", "a", "an", "of", "in", "to", "for", "on", "at", "by", "be",
    "will", "is", "are", "was", "were", "and", "or",
}


def normalize_title(s: str) -> str:
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [t for t in s.split() if t not in _STOP]
    return " ".join(tokens)


def normalize_name(s: str) -> str:
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()


def name_similarity(a: str, b: str) -> float:
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    # token-set fast path: same set of words → match
    if set(na.split()) == set(nb.split()):
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

GAMMA = "https://gamma-api.polymarket.com"


async def fetch_polymarket_events(client: httpx.AsyncClient, top_n: int) -> list[PolyEvent]:
    out: list[PolyEvent] = []
    offset = 0
    page = 100
    while len(out) < top_n:
        r = await client.get(
            f"{GAMMA}/events",
            params={
                "closed": "false",
                "active": "true",
                "limit": page,
                "offset": offset,
                "order": "volume24hr",
                "ascending": "false",
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        for e in data:
            ev_id = str(e.get("id", ""))
            title = e.get("title", "") or ""
            vol = float(e.get("volume24hr") or 0)
            markets: list[PolyMarketLite] = []
            for m in e.get("markets") or []:
                if m.get("closed") or not m.get("active"):
                    continue
                clob_raw = m.get("clobTokenIds") or ""
                yes_id = no_id = ""
                if clob_raw:
                    try:
                        ids = json.loads(clob_raw)
                        if isinstance(ids, list) and len(ids) >= 2:
                            yes_id, no_id = str(ids[0]), str(ids[1])
                    except Exception:
                        pass
                if not yes_id or not no_id:
                    continue
                cand = (m.get("groupItemTitle") or "").strip()
                if not cand:
                    # Derive from question by stripping common templates
                    q = m.get("question", "")
                    cand = re.sub(
                        r"^will\s+|\s+win\s+the.*$|\s+be\s+the.*$",
                        "", q, flags=re.IGNORECASE,
                    ).strip()
                markets.append(PolyMarketLite(
                    market_id=str(m.get("id", "")),
                    question=m.get("question", "") or "",
                    candidate=cand,
                    yes_token_id=yes_id,
                    no_token_id=no_id,
                    volume_24h=float(m.get("volume24hr") or 0),
                ))
            if markets:
                out.append(PolyEvent(ev_id, title, vol, markets))
        if len(data) < page:
            break
        offset += page
        await asyncio.sleep(0.1)
    out.sort(key=lambda e: e.volume_24h, reverse=True)
    return out[:top_n]


async def fetch_kalshi_events(kalshi: KalshiClient, top_n: int) -> list[KalshiEvent]:
    """Reuse list_events_with_markets and regroup by event_ticker.
    Event title is recovered from the composite KalshiMarket.title which is
    formatted as '<event_title>: <subtitle>'."""
    flat = await kalshi.list_events_with_markets(status="open", max_events=10000)
    by_ev: dict[str, dict] = {}
    for m in flat:
        ev_title = (m.event_title or "").strip()
        if not ev_title:
            # Fallback: strip ': <subtitle>' from composite
            ev_title = m.title
            if m.subtitle and ev_title.endswith(": " + m.subtitle):
                ev_title = ev_title[: -(len(m.subtitle) + 2)]
        ev = by_ev.setdefault(
            m.event_ticker, {"title": ev_title, "markets": []}
        )
        if not ev["title"] and ev_title:
            ev["title"] = ev_title
        cand = (m.subtitle or "").strip() or m.title
        ev["markets"].append(KalshiMarketLite(
            ticker=m.ticker,
            title=m.title,
            candidate=cand,
            volume=float(m.volume or 0),
        ))

    events: list[KalshiEvent] = []
    for ticker, info in by_ev.items():
        title = info["title"] or (info["markets"][0].title if info["markets"] else "")
        total_vol = sum(m.volume for m in info["markets"])
        events.append(KalshiEvent(ticker, title, total_vol, info["markets"]))
    events.sort(key=lambda e: e.volume, reverse=True)
    return events[:top_n]


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_events(
    poly: list[PolyEvent],
    kalshi: list[KalshiEvent],
    min_sim: float,
) -> list[tuple[PolyEvent, KalshiEvent, float]]:
    """Greedy best-match between event titles."""
    used_k: set[int] = set()
    pairs = []
    for pe in poly:
        best: Optional[tuple[int, KalshiEvent, float]] = None
        for i, ke in enumerate(kalshi):
            if i in used_k:
                continue
            s = title_similarity(pe.title, ke.title)
            if best is None or s > best[2]:
                best = (i, ke, s)
        if best and best[2] >= min_sim:
            used_k.add(best[0])
            pairs.append((pe, best[1], best[2]))
    return pairs


def join_candidates(
    pe: PolyEvent,
    ke: KalshiEvent,
    min_sim: float,
) -> list[tuple[PolyMarketLite, KalshiMarketLite, float]]:
    used_k: set[int] = set()
    joined = []
    for pm in pe.markets:
        best = None
        for i, km in enumerate(ke.markets):
            if i in used_k:
                continue
            s = name_similarity(pm.candidate, km.candidate)
            if best is None or s > best[2]:
                best = (i, km, s)
        if best and best[2] >= min_sim:
            used_k.add(best[0])
            joined.append((pm, best[1], best[2]))
    return joined


# ---------------------------------------------------------------------------
# Arbitrage detection
# ---------------------------------------------------------------------------

def calc_arb(poly_ob, kalshi_ob, fee_poly: float, fee_kalshi: float, gas: float):
    """Return best (label, gross, net, buy, sell, buy_size, sell_size) or None."""
    py_ask = poly_ob.best_ask_yes or 0
    py_bid = poly_ob.best_bid_yes or 0
    pn_ask = poly_ob.best_ask_no or 0
    pn_bid = poly_ob.best_bid_no or 0
    ky_ask = kalshi_ob.best_ask_yes or 0
    ky_bid = kalshi_ob.best_bid_yes or 0
    kn_ask = kalshi_ob.best_ask_no or 0
    kn_bid = kalshi_ob.best_bid_no or 0

    candidates = []
    if py_ask and ky_bid:
        candidates.append(("YES poly->kalshi", py_ask, ky_bid, fee_poly, fee_kalshi))
    if ky_ask and py_bid:
        candidates.append(("YES kalshi->poly", ky_ask, py_bid, fee_kalshi, fee_poly))
    if pn_ask and kn_bid:
        candidates.append(("NO  poly->kalshi", pn_ask, kn_bid, fee_poly, fee_kalshi))
    if kn_ask and pn_bid:
        candidates.append(("NO  kalshi->poly", kn_ask, pn_bid, fee_kalshi, fee_poly))

    best = None
    for label, buy, sell, fb, fs in candidates:
        gross = sell - buy
        net = gross - buy * fb - sell * fs - 2 * gas
        if best is None or net > best[2]:
            best = (label, gross, net, buy, sell)
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(top_n: int, evt_sim: float, cand_sim: float, min_edge: float):
    cfg = load_config("config.yaml")
    fee_poly = cfg.trading.taker_fee_bps / 10000
    fee_kalshi = 0.01
    gas = cfg.trading.estimated_gas_per_order

    log.info("Fetching events from both platforms (top by volume)...")
    async with httpx.AsyncClient() as http:
        poly_events_task = fetch_polymarket_events(http, top_n)
        kalshi = KalshiClient(timeout=cfg.api.timeout_seconds, dry_run=True)
        await kalshi.__aenter__()
        try:
            kalshi_events_task = fetch_kalshi_events(kalshi, top_n)
            poly_events, kalshi_events = await asyncio.gather(
                poly_events_task, kalshi_events_task
            )
        finally:
            pass  # keep kalshi open for orderbook fetches

        log.info(f"Polymarket events: {len(poly_events)} (top {top_n} by vol24h)")
        log.info(f"Kalshi events:     {len(kalshi_events)} (top {top_n} by vol)")

        log.info("Top 5 Polymarket events:")
        for e in poly_events[:5]:
            log.info(f"  ${e.volume_24h:>12,.0f}  {len(e.markets):3d} mkts  {e.title[:60]}")
        log.info("Top 5 Kalshi events:")
        for e in kalshi_events[:5]:
            log.info(f"  {e.volume:>12,.0f}  {len(e.markets):3d} mkts  {e.title[:60]}")

        log.info(f"Matching events (min title sim={evt_sim})...")
        ev_pairs = match_events(poly_events, kalshi_events, evt_sim)
        log.info(f"=== EVENT MATCHES: {len(ev_pairs)} ===")
        for pe, ke, s in ev_pairs[:20]:
            log.info(f"  sim={s:.2f}  P:{pe.title[:55]}  <>  K:{ke.title[:55]}")
        if len(ev_pairs) > 20:
            log.info(f"  ... and {len(ev_pairs)-20} more")

        # Join candidates within each event
        joined: list[tuple[PolyEvent, KalshiEvent, PolyMarketLite, KalshiMarketLite, float]] = []
        for pe, ke, _ in ev_pairs:
            for pm, km, cs in join_candidates(pe, ke, cand_sim):
                joined.append((pe, ke, pm, km, cs))
        log.info(f"=== CANDIDATE PAIRS JOINED: {len(joined)} ===")

        if not joined:
            log.info("No candidate-level joins — try lowering --candidate-sim.")
            await kalshi.__aexit__(None, None, None)
            return

        # Now check arbitrage
        log.info(f"Fetching orderbooks for {len(joined)} pairs...")
        poly_client = PolymarketClient(
            rest_url=cfg.api.polymarket_rest_url,
            ws_url=cfg.api.polymarket_ws_url,
            gamma_url=cfg.api.gamma_api_url,
            timeout=cfg.api.timeout_seconds,
            dry_run=True,
        )
        await poly_client.connect()

        opportunities = []
        near_miss = []
        fetch_fail = [0]

        async def check(idx: int, pe, ke, pm, km, cs):
            try:
                py = await poly_client._fetch_token_orderbook(pm.yes_token_id, TokenType.YES)
                pn = await poly_client._fetch_token_orderbook(pm.no_token_id, TokenType.NO)
                kob = await kalshi.get_orderbook_unified(km.ticker)
                if not (py and pn and kob):
                    fetch_fail[0] += 1
                    return
                poly_ob = OrderBook(
                    market_id=pm.market_id, yes=py, no=pn,
                    timestamp=datetime.utcnow(),
                )
                best = calc_arb(poly_ob, kob, fee_poly, fee_kalshi, gas)
                if best is None:
                    fetch_fail[0] += 1
                    return
                label, gross, net, buy, sell = best
                desc = (
                    f"{pe.title[:35]} | {pm.candidate[:18]} <> {km.candidate[:18]} | "
                    f"{label} buy@{buy:.3f} sell@{sell:.3f}"
                )
                if net >= min_edge:
                    opportunities.append((net, desc, pm, km))
                else:
                    near_miss.append((net, desc))
            except Exception as e:
                fetch_fail[0] += 1
                log.debug(f"check {idx} failed: {e}")

        # Run with bounded concurrency
        sem = asyncio.Semaphore(2)
        done = [0]

        async def limited(args):
            async with sem:
                await check(*args)
                done[0] += 1
                if done[0] % 25 == 0:
                    log.info(f"  ...checked {done[0]}/{len(joined)}")

        await asyncio.gather(*[
            limited((i, *t)) for i, t in enumerate(joined)
        ])

        log.info("=" * 72)
        log.info(f"OPPORTUNITIES (net edge >= {min_edge:.4f}): {len(opportunities)}")
        for net, desc, pm, km in sorted(opportunities, reverse=True):
            log.info(f"  net={net:+.4f}  {desc}")

        log.info("-" * 72)
        log.info(f"Top 15 near-misses (net < {min_edge:.4f}):")
        for net, desc in sorted(near_miss, reverse=True)[:15]:
            log.info(f"  net={net:+.4f}  {desc}")
        log.info("=" * 72)

        log.info("SUMMARY:")
        log.info(f"  events scanned (each side): {top_n}")
        log.info(f"  matched event pairs:        {len(ev_pairs)}")
        log.info(f"  joined candidate pairs:     {len(joined)}")
        log.info(f"  orderbook fetches failed:   {fetch_fail[0]}")
        log.info(f"  arbitrage opportunities:    {len(opportunities)}")

        await poly_client.disconnect()
        await kalshi.__aexit__(None, None, None)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=500)
    p.add_argument("--event-sim", type=float, default=0.7)
    p.add_argument("--candidate-sim", type=float, default=0.8)
    p.add_argument("--min-edge", type=float, default=0.005)
    args = p.parse_args()
    asyncio.run(main(args.top, args.event_sim, args.candidate_sim, args.min_edge))
