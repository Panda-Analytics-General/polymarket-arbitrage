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
    description: str = ""
    end_date: str = ""


@dataclass
class PolyEvent:
    event_id: str
    title: str
    volume_24h: float
    markets: list[PolyMarketLite]
    slug: str = ""


@dataclass
class KalshiMarketLite:
    ticker: str
    title: str
    candidate: str          # yes_sub_title / subtitle
    volume: float
    rules_primary: str = ""
    rules_secondary: str = ""
    close_time: str = ""


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
                    description=m.get("description", "") or "",
                    end_date=m.get("endDate") or m.get("end_date_iso") or "",
                ))
            if markets:
                out.append(PolyEvent(ev_id, title, vol, markets, slug=e.get("slug", "") or ""))
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
            rules_primary=m.rules_primary or "",
            rules_secondary=m.rules_secondary or "",
            close_time=m.close_time.isoformat() if m.close_time else "",
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
# ---------------------------------------------------------------------------
# Arbitrage detection (depth-aware)
# ---------------------------------------------------------------------------

# Fee models (per $1 notional traded):
#   Polymarket: published formula 4 * feeRate * p * (1-p), with feeRate ~ 0.01
#               => max 1% at p=0.5, ~0% at extremes. Charged on both buy & sell.
#   Kalshi:     published formula 0.07 * p * (1-p) per contract ($1 notional)
#               => max 1.75% at p=0.5. Charged on both buy & sell.
#   Gas:        only Polymarket leg incurs gas (~$0.02 per order). Kalshi free.

def _poly_fee(price: float) -> float:
    """Polymarket taker fee per $1 notional at given price."""
    return 0.04 * price * (1.0 - price)

def _kalshi_fee(price: float) -> float:
    """Kalshi taker fee per $1 notional at given price."""
    return 0.07 * price * (1.0 - price)


def _walk_books(buy_levels, sell_levels, buy_is_poly: bool, gas: float):
    """Walk buy asks (ascending) vs sell bids (descending) and accumulate
    profitable fills. Each level is a PriceLevel(price, size) where size is
    in contracts ($1 notional each).

    Returns (gross_profit, net_profit, total_size, best_buy_px, best_sell_px).
    Returns None if no profitable level exists.
    """
    if not buy_levels or not sell_levels:
        return None

    # Copy as mutable
    asks = [(l.price, l.size) for l in buy_levels]
    bids = [(l.price, l.size) for l in sell_levels]

    # Verify sort order
    asks.sort(key=lambda x: x[0])           # cheapest first
    bids.sort(key=lambda x: x[0], reverse=True)  # highest first

    best_buy = asks[0][0]
    best_sell = bids[0][0]

    gross = 0.0
    net = 0.0
    size = 0.0
    ai = bi = 0
    a_remain = asks[0][1] if asks else 0
    b_remain = bids[0][1] if bids else 0
    gas_charged = False

    while ai < len(asks) and bi < len(bids):
        buy_px, _ = asks[ai]
        sell_px, _ = bids[bi]

        if buy_is_poly:
            fb = _poly_fee(buy_px)
            fs = _kalshi_fee(sell_px)
        else:
            fb = _kalshi_fee(buy_px)
            fs = _poly_fee(sell_px)

        per_unit = sell_px - buy_px - fb - fs
        if per_unit <= 0:
            break

        fill = min(a_remain, b_remain)
        if fill <= 0:
            break

        gross += (sell_px - buy_px) * fill
        net += per_unit * fill
        size += fill

        if not gas_charged:
            net -= gas  # one Polygon tx fee per arb (size-independent)
            gas_charged = True

        a_remain -= fill
        b_remain -= fill
        if a_remain <= 1e-9:
            ai += 1
            if ai < len(asks):
                a_remain = asks[ai][1]
        if b_remain <= 1e-9:
            bi += 1
            if bi < len(bids):
                b_remain = bids[bi][1]

    if size == 0 or net <= 0:
        return None
    return (gross, net, size, best_buy, best_sell)


def calc_arb_depth(poly_ob, kalshi_ob, gas: float):
    """Test all 4 directions and return best fill plan.
    Returns dict with: label, net_profit, gross_profit, size, buy_px, sell_px.
    """
    directions = [
        # (label, buy_levels, sell_levels, buy_is_poly)
        ("YES poly->kalshi", poly_ob.yes.asks.levels,  kalshi_ob.yes.bids.levels, True),
        ("YES kalshi->poly", kalshi_ob.yes.asks.levels, poly_ob.yes.bids.levels,  False),
        ("NO  poly->kalshi", poly_ob.no.asks.levels,   kalshi_ob.no.bids.levels,  True),
        ("NO  kalshi->poly", kalshi_ob.no.asks.levels,  poly_ob.no.bids.levels,   False),
    ]
    best = None
    for label, buys, sells, buy_is_poly in directions:
        r = _walk_books(buys, sells, buy_is_poly, gas)
        if r is None:
            continue
        gross, net, size, b_px, s_px = r
        if best is None or net > best["net_profit"]:
            best = {
                "label": label,
                "gross_profit": gross,
                "net_profit": net,
                "size": size,
                "buy_px": b_px,
                "sell_px": s_px,
            }
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(top_n: int, evt_sim: float, cand_sim: float, min_edge: float,
               llm_model: str = "", llm_concurrency: int = 4):
    cfg = load_config("config.yaml")
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
                best = calc_arb_depth(poly_ob, kob, gas)
                if best is None:
                    return  # no profitable direction (still a successful fetch)
                edge_per_unit = best["sell_px"] - best["buy_px"]
                desc = (
                    f"{pe.title[:35]} | {pm.candidate[:18]} <> {km.candidate[:18]} | "
                    f"{best['label']} buy@{best['buy_px']:.3f} sell@{best['sell_px']:.3f} "
                    f"size={best['size']:.0f} net=${best['net_profit']:.2f}"
                )
                if edge_per_unit >= min_edge:
                    opportunities.append((best["net_profit"], desc, best, pe, pm, km, ke))
                else:
                    near_miss.append((best["net_profit"], desc))
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
        log.info(f"OPPORTUNITIES (per-unit edge >= {min_edge:.4f}): {len(opportunities)}")
        opps_sorted = sorted(opportunities, key=lambda x: x[0], reverse=True)
        for net_profit, desc, _b, _pe, _pm, _km, _ke in opps_sorted:
            log.info(f"  ${net_profit:>8.2f}  {desc}")

        # ----- LLM verification (optional) ---------------------------------
        verified: list = []
        rejected: list = []
        if llm_model and opps_sorted:
            from core.llm_verifier import OllamaVerifier
            log.info("-" * 72)
            log.info(f"LLM verification: {len(opps_sorted)} opportunities -> {llm_model}")
            async with OllamaVerifier(model=llm_model, max_concurrency=llm_concurrency) as verifier:
                ok = await verifier.health_check()
                if not ok:
                    log.warning(f"Ollama model {llm_model} not available; skipping verification")
                else:
                    async def verify_one(item):
                        net_profit, desc, best, pe, pm, km, ke = item
                        kalshi_rules = (km.rules_primary or "").strip()
                        if km.rules_secondary:
                            kalshi_rules += "\n\nAdditional: " + km.rules_secondary
                        if km.close_time:
                            kalshi_rules += f"\n\nClose time: {km.close_time}"
                        poly_rules = (pm.description or "").strip()
                        if pm.end_date:
                            poly_rules += f"\n\nEnd date: {pm.end_date}"
                        decision = await verifier.verify_pair(
                            poly_question=pm.question or pe.title,
                            poly_rules=poly_rules,
                            kalshi_question=km.title,
                            kalshi_rules=kalshi_rules,
                        )
                        return item, decision

                    results = await asyncio.gather(*[verify_one(o) for o in opps_sorted])
                    for item, dec in results:
                        net_profit, desc, *_ = item
                        tag = f"{dec.verdict:>11s} c={dec.confidence:.2f}"
                        if dec.is_equivalent:
                            verified.append((item, dec))
                            log.info(f"  ✅ {tag}  ${net_profit:>7.2f}  {desc[:90]}")
                        else:
                            rejected.append((item, dec))
                            log.info(f"  ❌ {tag}  ${net_profit:>7.2f}  {desc[:90]}")
                            log.info(f"        reason: {dec.reasoning[:200]}")

        total_profit = sum(o[0] for o in opportunities)
        verified_profit = sum(v[0][0] for v in verified)

        # Top-10 verified with URLs
        if verified:
            log.info("-" * 72)
            log.info("TOP 10 VERIFIED OPPORTUNITIES WITH URLS:")
            verified_sorted = sorted(verified, key=lambda x: x[0][0], reverse=True)
            for i, (item, dec) in enumerate(verified_sorted[:10], 1):
                net_profit, desc, best, pe, pm, km, ke = item
                poly_url = f"https://polymarket.com/event/{pe.slug}" if pe.slug else f"(no slug, event_id={pe.event_id})"
                kalshi_url = f"https://kalshi.com/markets/{ke.event_ticker.lower()}"
                log.info(f"  {i:2d}. ${net_profit:>8.2f}  {pm.question[:60]}")
                log.info(f"       candidate: {pm.candidate} <> {km.candidate}")
                log.info(f"       poly:   {poly_url}")
                log.info(f"       kalshi: {kalshi_url}")
        log.info("-" * 72)
        log.info(f"Top 15 near-misses (per-unit edge < {min_edge:.4f}):")
        for net, desc in sorted(near_miss, reverse=True)[:15]:
            log.info(f"  ${net:>+8.2f}  {desc}")
        log.info("=" * 72)

        log.info("SUMMARY:")
        log.info(f"  events scanned (each side):     {top_n}")
        log.info(f"  matched event pairs:            {len(ev_pairs)}")
        log.info(f"  joined candidate pairs:         {len(joined)}")
        log.info(f"  orderbook fetches failed:       {fetch_fail[0]}")
        log.info(f"  arbitrage opportunities:        {len(opportunities)}")
        log.info(f"  total potential net profit:    ${total_profit:.2f}")
        if llm_model and (verified or rejected):
            log.info(f"  LLM-verified (equivalent):      {len(verified)}")
            log.info(f"  LLM-rejected (different/unc):   {len(rejected)}")
            log.info(f"  verified net profit:           ${verified_profit:.2f}")
        if opportunities:
            top5 = opps_sorted[:5]
            top5_sum = sum(o[0] for o in top5)
            log.info(f"  top-5 opportunities profit:    ${top5_sum:.2f}")
            top10 = opps_sorted[:10]
            top10_sum = sum(o[0] for o in top10)
            log.info(f"  top-10 opportunities profit:   ${top10_sum:.2f}")

        await poly_client.disconnect()
        await kalshi.__aexit__(None, None, None)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=500)
    p.add_argument("--event-sim", type=float, default=0.7)
    p.add_argument("--candidate-sim", type=float, default=0.8)
    p.add_argument("--min-edge", type=float, default=0.005)
    p.add_argument("--llm-model", type=str, default="",
                   help="e.g. qwen3.5:9b. Empty disables LLM verification.")
    p.add_argument("--llm-concurrency", type=int, default=4)
    args = p.parse_args()
    asyncio.run(main(args.top, args.event_sim, args.candidate_sim,
                     args.min_edge, args.llm_model, args.llm_concurrency))
