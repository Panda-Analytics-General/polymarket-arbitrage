"""
Polymarket API Client
======================

Abstracted client for Polymarket REST and WebSocket APIs.
Designed to be easily pluggable with real API implementations.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, AsyncIterator, Optional

import httpx
import websockets
from websockets.exceptions import ConnectionClosed

from polymarket_client.models import (
    Market,
    Order,
    OrderBook,
    OrderBookSide,
    OrderSide,
    OrderStatus,
    Position,
    PriceLevel,
    TokenOrderBook,
    TokenType,
    Trade,
)


logger = logging.getLogger(__name__)


class BasePolymarketClient(ABC):
    """Abstract base class for Polymarket client implementations."""
    
    @abstractmethod
    async def list_markets(self, filters: Optional[dict] = None) -> list[Market]:
        """Fetch list of available markets."""
        pass
    
    @abstractmethod
    async def get_market(self, market_id: str) -> Market:
        """Get details for a specific market."""
        pass
    
    @abstractmethod
    async def get_orderbook(self, market_id: str) -> OrderBook:
        """Fetch current order book for a market."""
        pass
    
    @abstractmethod
    async def stream_orderbook(self, market_ids: list[str]) -> AsyncIterator[tuple[str, OrderBook]]:
        """Stream order book updates for multiple markets."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> dict[str, dict[TokenType, Position]]:
        """Get all current positions."""
        pass
    
    @abstractmethod
    async def place_order(
        self,
        market_id: str,
        token_type: TokenType,
        side: OrderSide,
        price: float,
        size: float,
        strategy_tag: str = ""
    ) -> Order:
        """Place a limit order."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> None:
        """Cancel an open order."""
        pass
    
    @abstractmethod
    async def get_open_orders(self, market_id: Optional[str] = None) -> list[Order]:
        """Get all open orders, optionally filtered by market."""
        pass
    
    @abstractmethod
    async def get_trades(self, market_id: Optional[str] = None, limit: int = 100) -> list[Trade]:
        """Get recent trades."""
        pass


class PolymarketClient(BasePolymarketClient):
    """
    Polymarket API client implementation.
    
    This implementation provides the structure for real API integration.
    Currently uses placeholder implementations that can be replaced with
    actual Polymarket CLOB API calls.
    """
    
    def __init__(
        self,
        rest_url: str = "https://clob.polymarket.com",
        ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        gamma_url: str = "https://gamma-api.polymarket.com",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        passphrase: Optional[str] = None,
        private_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        dry_run: bool = True,
    ):
        self.rest_url = rest_url.rstrip("/")
        self.ws_url = ws_url
        self.gamma_url = gamma_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.private_key = private_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.dry_run = dry_run
        
        # HTTP client
        self._http_client: Optional[httpx.AsyncClient] = None
        
        # WebSocket connection
        self._ws_connection = None
        self._ws_subscriptions: set[str] = set()
        
        # Simulated state for dry run
        self._simulated_orders: dict[str, Order] = {}
        self._simulated_positions: dict[str, dict[TokenType, Position]] = {}
        self._simulated_trades: list[Trade] = []
        
        # Cache for market data (avoids re-fetching)
        self._markets_cache: dict[str, Market] = {}
        
    async def __aenter__(self) -> "PolymarketClient":
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()
    
    async def connect(self) -> None:
        """Initialize HTTP client."""
        self._http_client = httpx.AsyncClient(
            timeout=self.timeout,
            headers=self._get_headers(),
        )
        logger.info(f"Polymarket client connected (dry_run={self.dry_run})")
    
    async def disconnect(self) -> None:
        """Close connections."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        if self._ws_connection:
            await self._ws_connection.close()
            self._ws_connection = None
        logger.info("Polymarket client disconnected")
    
    def _get_headers(self) -> dict[str, str]:
        """Get authentication headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            # TODO: Implement proper CLOB API authentication
            # Polymarket uses L1/L2 authentication with signatures
            headers["POLY_API_KEY"] = self.api_key
        return headers
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        json_data: Optional[dict] = None,
        base_url: Optional[str] = None,
    ) -> Any:
        """Make an HTTP request with retry logic."""
        if not self._http_client:
            await self.connect()
        
        url = f"{base_url or self.rest_url}{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                response = await self._http_client.request(
                    method,
                    url,
                    params=params,
                    json=json_data,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP error {e.response.status_code} on {url}: {e}")
                if e.response.status_code >= 500:
                    # Retry on server errors
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue
                raise
            except httpx.RequestError as e:
                logger.warning(f"Request error on {url}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise
    
    async def list_markets(self, filters: Optional[dict] = None) -> list[Market]:
        """
        Fetch list of available markets from Gamma API.
        
        Endpoint: GET https://gamma-api.polymarket.com/markets
        
        Uses pagination to get ALL active markets across all categories!
        """
        try:
            params = filters.copy() if filters else {}
            params.setdefault("closed", "false")
            params.setdefault("order", "volume24hr")
            params.setdefault("ascending", "false")
            
            all_markets = []
            offset = 0
            limit = 100  # Gamma API max per request
            max_markets = 5000  # Get up to 5000 markets!
            
            logger.info("Fetching ALL available markets from Polymarket...")
            
            # Paginate to get all markets
            while True:
                params["limit"] = limit
                params["offset"] = offset
                
                data = await self._request(
                    "GET", 
                    "/markets",
                    params=params,
                    base_url=self.gamma_url,
                )
                
                if not data:
                    break
                
                batch_valid = 0
                for item in data:
                    market = self._parse_market(item)
                    if market and market.yes_token_id and market.no_token_id:
                        all_markets.append(market)
                        # Cache the market for later use
                        self._markets_cache[market.market_id] = market
                        batch_valid += 1
                
                logger.info(f"Fetched batch: offset={offset}, got {len(data)} markets ({batch_valid} valid)")
                
                if len(data) < limit:
                    # No more pages
                    break
                    
                offset += limit
                
                # Rate limiting - don't hammer the API
                await asyncio.sleep(0.15)
                
                # Safety cap
                if len(all_markets) >= max_markets:
                    logger.info(f"Reached {max_markets} market cap")
                    break
            
            logger.info(f"=== TOTAL: {len(all_markets)} active markets with valid tokens ===")
            return all_markets
            
        except Exception as e:
            logger.error(f"Failed to fetch markets from API: {e}")
            raise
    
    async def list_events(self, filters: Optional[dict] = None) -> list[dict]:
        """
        Fetch events (which contain markets) from Gamma API.
        
        Endpoint: GET https://gamma-api.polymarket.com/events
        
        Events are useful for getting grouped markets.
        """
        try:
            params = filters.copy() if filters else {}
            params.setdefault("closed", "false")
            params.setdefault("limit", 50)
            params.setdefault("order", "id")
            params.setdefault("ascending", "false")
            
            data = await self._request(
                "GET",
                "/events",
                params=params,
                base_url=self.gamma_url,
            )
            
            return data
            
        except Exception as e:
            logger.warning(f"Failed to fetch events: {e}")
            return []
    
    def _parse_market(self, data: dict) -> Optional[Market]:
        """Parse market data from Gamma API response."""
        try:
            market_id = str(data.get("id", ""))
            condition_id = data.get("conditionId", "")
            
            if not market_id:
                return None
            
            # Parse clobTokenIds - JSON string like '["tokenId1","tokenId2"]'
            clob_token_ids_raw = data.get("clobTokenIds", "")
            yes_token_id = ""
            no_token_id = ""
            
            if clob_token_ids_raw:
                try:
                    # It's a JSON array string
                    token_ids = json.loads(clob_token_ids_raw)
                    if isinstance(token_ids, list):
                        yes_token_id = str(token_ids[0]).strip() if len(token_ids) > 0 else ""
                        no_token_id = str(token_ids[1]).strip() if len(token_ids) > 1 else ""
                except (json.JSONDecodeError, TypeError):
                    # Fallback: try comma-separated
                    token_ids = clob_token_ids_raw.split(",")
                    yes_token_id = token_ids[0].strip() if len(token_ids) > 0 else ""
                    no_token_id = token_ids[1].strip() if len(token_ids) > 1 else ""
            
            # Parse outcomes - JSON string like '["Yes", "No"]'
            outcomes_str = data.get("outcomes", "")
            # Parse outcome prices - JSON string like '[0.65, 0.35]'
            outcome_prices_str = data.get("outcomePrices", "")
            
            return Market(
                market_id=market_id,
                condition_id=condition_id,
                question=data.get("question", "") or "",
                description=data.get("description", "") or "",
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                active=bool(data.get("active", True)),
                closed=bool(data.get("closed", False)),
                resolved=data.get("umaResolutionStatus") == "resolved",
                volume_24h=float(data.get("volume24hr") or data.get("volume24hrClob") or 0),
                liquidity=float(data.get("liquidityNum") or data.get("liquidityClob") or 0),
                category=data.get("category", "") or "",
            )
        except Exception as e:
            logger.warning(f"Failed to parse market: {e}")
            return None
    
    def _get_placeholder_markets(self) -> list[Market]:
        """Get placeholder markets for testing."""
        return [
            Market(
                market_id="placeholder_market_1",
                condition_id="0x1234",
                question="Will BTC be above $100k by end of 2025?",
                description="Resolves YES if Bitcoin price exceeds $100,000",
                yes_token_id="yes_token_1",
                no_token_id="no_token_1",
                active=True,
                volume_24h=50000.0,
                liquidity=100000.0,
            ),
            Market(
                market_id="placeholder_market_2",
                condition_id="0x5678",
                question="Will ETH 2.0 be fully deployed by Q2 2025?",
                description="Resolves YES if Ethereum completes all upgrades",
                yes_token_id="yes_token_2",
                no_token_id="no_token_2",
                active=True,
                volume_24h=25000.0,
                liquidity=50000.0,
            ),
        ]
    
    async def get_market(self, market_id: str) -> Market:
        """
        Get details for a specific market.
        
        Can fetch by ID or by slug:
        - GET /markets/{id} - by numeric ID
        - GET /markets/slug/{slug} - by slug
        """
        try:
            # Try fetching by ID first
            data = await self._request(
                "GET",
                f"/markets/{market_id}",
                base_url=self.gamma_url,
            )
            market = self._parse_market(data)
            if market:
                return market
            raise ValueError("Failed to parse market")
        except Exception as e:
            logger.warning(f"Failed to fetch market {market_id}: {e}")
            if self.dry_run:
                return Market(
                    market_id=market_id,
                    condition_id=market_id,
                    question=f"Market {market_id}",
                    active=True,
                )
            raise
    
    async def get_market_by_slug(self, slug: str) -> Market:
        """
        Get market by its slug.
        
        Endpoint: GET /markets/slug/{slug}
        
        The slug can be extracted from Polymarket URLs:
        https://polymarket.com/event/some-event-slug
        """
        try:
            data = await self._request(
                "GET",
                f"/markets/slug/{slug}",
                base_url=self.gamma_url,
            )
            market = self._parse_market(data)
            if market:
                return market
            raise ValueError("Failed to parse market")
        except Exception as e:
            logger.error(f"Failed to fetch market by slug {slug}: {e}")
            raise
    
    async def get_event_by_slug(self, slug: str) -> dict:
        """
        Get event by its slug.
        
        Endpoint: GET /events/slug/{slug}
        
        Events contain multiple related markets.
        """
        try:
            data = await self._request(
                "GET",
                f"/events/slug/{slug}",
                base_url=self.gamma_url,
            )
            return data
        except Exception as e:
            logger.error(f"Failed to fetch event by slug {slug}: {e}")
            raise
    
    async def get_orderbook(self, market_id: str) -> OrderBook:
        """
        Fetch current order book for a market.
        
        Uses Polymarket CLOB API:
        GET https://clob.polymarket.com/book?token_id={token_id}
        """
        # Get market to find token IDs
        market = await self.get_market(market_id)
        
        if not market.yes_token_id or not market.no_token_id:
            logger.warning(f"No token IDs for market {market_id}")
            return OrderBook(market_id=market_id, timestamp=datetime.utcnow())
        
        # Fetch REAL order books from CLOB API
        yes_book = await self._fetch_token_orderbook(market.yes_token_id, TokenType.YES)
        no_book = await self._fetch_token_orderbook(market.no_token_id, TokenType.NO)
        
        return OrderBook(
            market_id=market_id,
            yes=yes_book,
            no=no_book,
            timestamp=datetime.utcnow(),
        )
    
    async def _fetch_token_orderbook(self, token_id: str, token_type: TokenType) -> TokenOrderBook:
        """Fetch order book for a single token from CLOB API."""
        try:
            data = await self._request(
                "GET",
                "/book",
                params={"token_id": token_id},
                base_url=self.rest_url,
            )
            
            # Parse bids and asks.
            # Polymarket /book returns bids sorted ascending (worst->best)
            # and asks sorted descending (worst->best), so we reverse both
            # to get best-first ordering expected by the rest of the code.
            raw_bids = list(data.get("bids", []))
            raw_asks = list(data.get("asks", []))

            bids_sorted = sorted(
                raw_bids, key=lambda x: float(x.get("price", 0)), reverse=True
            )
            asks_sorted = sorted(
                raw_asks, key=lambda x: float(x.get("price", 0))
            )

            bids = [
                PriceLevel(
                    price=float(b.get("price", 0)),
                    size=float(b.get("size", 0)),
                )
                for b in bids_sorted[:10]
            ]
            asks = [
                PriceLevel(
                    price=float(a.get("price", 0)),
                    size=float(a.get("size", 0)),
                )
                for a in asks_sorted[:10]
            ]
            
            return TokenOrderBook(
                token_type=token_type,
                bids=OrderBookSide(levels=bids),
                asks=OrderBookSide(levels=asks),
            )
            
        except Exception as e:
            logger.warning(f"Failed to fetch orderbook for token {token_id}: {e}")
            # Return empty book
            return TokenOrderBook(
                token_type=token_type,
                bids=OrderBookSide(levels=[]),
                asks=OrderBookSide(levels=[]),
            )
    
    def _generate_simulated_orderbook(self, market_id: str) -> OrderBook:
        """Generate a simulated order book for testing."""
        import random
        
        # Simulate realistic prices with occasional mispricings
        yes_mid = 0.50 + random.uniform(-0.30, 0.30)
        
        # 20% chance of significant mispricing (arb opportunity!)
        if random.random() < 0.20:
            inefficiency = random.uniform(-0.08, 0.08)  # Bigger mispricing
        else:
            inefficiency = random.uniform(-0.02, 0.02)  # Normal slight inefficiency
        
        no_mid = 1.0 - yes_mid + inefficiency
        
        spread = random.uniform(0.02, 0.06)
        
        def generate_levels(mid: float, is_bid: bool, count: int = 5) -> list[PriceLevel]:
            levels = []
            for i in range(count):
                offset = (i + 1) * 0.01
                if is_bid:
                    price = max(0.01, mid - spread/2 - offset)
                else:
                    price = min(0.99, mid + spread/2 + offset)
                size = random.uniform(100, 1000)
                levels.append(PriceLevel(price=round(price, 2), size=round(size, 2)))
            return levels
        
        yes_book = TokenOrderBook(
            token_type=TokenType.YES,
            bids=OrderBookSide(levels=generate_levels(yes_mid, is_bid=True)),
            asks=OrderBookSide(levels=generate_levels(yes_mid, is_bid=False)),
        )
        
        no_book = TokenOrderBook(
            token_type=TokenType.NO,
            bids=OrderBookSide(levels=generate_levels(no_mid, is_bid=True)),
            asks=OrderBookSide(levels=generate_levels(no_mid, is_bid=False)),
        )
        
        return OrderBook(
            market_id=market_id,
            yes=yes_book,
            no=no_book,
            timestamp=datetime.utcnow(),
        )
    
    async def stream_orderbook(self, market_ids: list[str], use_simulation: bool = False) -> AsyncIterator[tuple[str, OrderBook]]:
        """
        Stream order book updates.
        
        If use_simulation=True, generates simulated data with opportunities.
        Otherwise fetches REAL data from Polymarket CLOB API.
        """
        if use_simulation:
            async for item in self._stream_simulated_orderbooks(market_ids):
                yield item
            return
        
        logger.info(f"Starting REAL orderbook stream for {len(market_ids)} markets")
        
        # We already have token IDs in the cached markets - use them directly!
        # Build token map from cached market data (no extra API calls needed)
        market_tokens: dict[str, tuple[str, str]] = {}
        
        for market_id in market_ids:
            if market_id in self._markets_cache:
                market = self._markets_cache[market_id]
                if market.yes_token_id and market.no_token_id:
                    market_tokens[market_id] = (market.yes_token_id, market.no_token_id)
        
        logger.info(f"Have token IDs for {len(market_tokens)} markets (from cache)")
        
        if not market_tokens:
            logger.warning("No markets with valid token IDs found!")
            return
        
        # Settings for processing large market counts
        active_batch_size = 500  # Process 500 markets per rotation
        markets_per_request_batch = 20  # Fetch 20 at a time within the active batch
        request_delay = 0.05  # 50ms between API calls
        batch_delay = 0.3  # 300ms between request batches
        rotation_delay = 2.0  # 2 seconds before rotating to next 500
        
        market_list = list(market_tokens.keys())
        total_markets = len(market_list)
        current_offset = 0
        
        logger.info(f"Will rotate through {total_markets} markets, {active_batch_size} at a time")
        
        try:
            while True:
                # Get current batch of 500 markets
                end_offset = min(current_offset + active_batch_size, total_markets)
                active_markets = market_list[current_offset:end_offset]
                
                logger.info(f"Processing markets {current_offset+1}-{end_offset} of {total_markets}")
                
                # Process this batch
                for i in range(0, len(active_markets), markets_per_request_batch):
                    request_batch = active_markets[i:i + markets_per_request_batch]
                    
                    for market_id in request_batch:
                        try:
                            yes_token, no_token = market_tokens[market_id]
                            
                            # Fetch REAL order books from CLOB API
                            yes_book = await self._fetch_token_orderbook(yes_token, TokenType.YES)
                            no_book = await self._fetch_token_orderbook(no_token, TokenType.NO)
                            
                            orderbook = OrderBook(
                                market_id=market_id,
                                yes=yes_book,
                                no=no_book,
                                timestamp=datetime.utcnow(),
                            )
                            
                            yield (market_id, orderbook)
                            await asyncio.sleep(request_delay)
                            
                        except Exception as e:
                            # Silently skip errors - don't spam logs
                            continue
                    
                    await asyncio.sleep(batch_delay)
                
                # Move to next batch of 500
                current_offset = end_offset
                if current_offset >= total_markets:
                    current_offset = 0  # Start over from beginning
                    logger.info("Completed full market cycle, starting over...")
                
                await asyncio.sleep(rotation_delay)
                
        except asyncio.CancelledError:
            logger.info("Orderbook stream cancelled")
            raise
        except Exception as e:
            logger.error(f"Orderbook stream error: {e}")
            raise
    
    async def _stream_simulated_orderbooks(self, market_ids: list[str]) -> AsyncIterator[tuple[str, OrderBook]]:
        """Generate simulated order books with occasional arbitrage opportunities."""
        import random
        
        logger.info(f"Starting SIMULATED orderbook stream for {len(market_ids)} markets")
        
        # Use subset for faster updates
        active_markets = market_ids[:100] if len(market_ids) > 100 else market_ids
        
        try:
            while True:
                # Update 10-20 random markets per cycle
                batch = random.sample(active_markets, min(15, len(active_markets)))
                
                for market_id in batch:
                    orderbook = self._generate_simulated_orderbook(market_id)
                    yield (market_id, orderbook)
                    await asyncio.sleep(0.02)  # Fast updates
                
                await asyncio.sleep(0.5)  # Brief pause between cycles
                
        except asyncio.CancelledError:
            logger.info("Simulated orderbook stream cancelled")
            raise

    async def _connect_websocket(self, market_ids: list[str]) -> None:
        """
        Connect to Polymarket WebSocket.
        
        TODO: Implement actual WebSocket connection and subscription.
        """
        try:
            self._ws_connection = await websockets.connect(
                self.ws_url,
                ping_interval=30,
                ping_timeout=10,
            )
            
            # Subscribe to markets
            for market_id in market_ids:
                subscribe_msg = json.dumps({
                    "type": "subscribe",
                    "market": market_id,
                    "channel": "book",
                })
                await self._ws_connection.send(subscribe_msg)
                self._ws_subscriptions.add(market_id)
            
            logger.info(f"WebSocket connected, subscribed to {len(market_ids)} markets")
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
    
    async def get_positions(self) -> dict[str, dict[TokenType, Position]]:
        """
        Get all current positions.
        
        TODO: Implement with actual Polymarket API.
        """
        if self.dry_run:
            return self._simulated_positions.copy()
        
        try:
            # Real API call would go here
            data = await self._request("GET", "/positions")
            positions = {}
            for item in data:
                market_id = item["market_id"]
                token_type = TokenType.YES if item["outcome"] == "Yes" else TokenType.NO
                positions.setdefault(market_id, {})[token_type] = Position(
                    market_id=market_id,
                    token_type=token_type,
                    size=float(item["size"]),
                    avg_entry_price=float(item.get("avg_price", 0)),
                    realized_pnl=float(item.get("realized_pnl", 0)),
                )
            return positions
        except Exception as e:
            logger.warning(f"Failed to fetch positions: {e}")
            return {}
    
    async def place_order(
        self,
        market_id: str,
        token_type: TokenType,
        side: OrderSide,
        price: float,
        size: float,
        strategy_tag: str = ""
    ) -> Order:
        """
        Place a limit order.
        
        TODO: Implement with actual Polymarket CLOB API:
        POST https://clob.polymarket.com/order
        """
        order_id = f"order_{uuid.uuid4().hex[:12]}"
        order = Order(
            order_id=order_id,
            market_id=market_id,
            token_type=token_type,
            side=side,
            price=price,
            size=size,
            status=OrderStatus.OPEN,
            strategy_tag=strategy_tag,
        )
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Placing order: {order}")
            self._simulated_orders[order_id] = order
            return order
        
        try:
            # TODO: Implement actual order placement
            # Would need to:
            # 1. Build order with proper token IDs
            # 2. Sign with private key
            # 3. Submit to CLOB
            payload = {
                "market_id": market_id,
                "token_id": "",  # TODO: Map token_type to actual token ID
                "side": side.value,
                "price": str(price),
                "size": str(size),
            }
            
            data = await self._request("POST", "/order", json_data=payload)
            order.order_id = data.get("order_id", order_id)
            order.status = OrderStatus.OPEN
            
            logger.info(f"Order placed: {order.order_id}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            order.status = OrderStatus.REJECTED
            raise
    
    async def cancel_order(self, order_id: str) -> None:
        """
        Cancel an open order.
        
        TODO: Implement with actual Polymarket CLOB API:
        DELETE https://clob.polymarket.com/order/{order_id}
        """
        if self.dry_run:
            if order_id in self._simulated_orders:
                self._simulated_orders[order_id].status = OrderStatus.CANCELLED
                logger.info(f"[DRY RUN] Cancelled order: {order_id}")
            return
        
        try:
            await self._request("DELETE", f"/order/{order_id}")
            logger.info(f"Order cancelled: {order_id}")
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise
    
    async def cancel_all_orders(self, market_id: Optional[str] = None) -> int:
        """Cancel all open orders, optionally for a specific market."""
        orders = await self.get_open_orders(market_id)
        cancelled = 0
        
        for order in orders:
            try:
                await self.cancel_order(order.order_id)
                cancelled += 1
            except Exception as e:
                logger.warning(f"Failed to cancel order {order.order_id}: {e}")
        
        return cancelled
    
    async def get_open_orders(self, market_id: Optional[str] = None) -> list[Order]:
        """Get all open orders."""
        if self.dry_run:
            orders = [
                o for o in self._simulated_orders.values()
                if o.is_open and (market_id is None or o.market_id == market_id)
            ]
            return orders
        
        try:
            params = {"market_id": market_id} if market_id else None
            data = await self._request("GET", "/orders", params=params)
            
            orders = []
            for item in data:
                orders.append(Order(
                    order_id=item["order_id"],
                    market_id=item["market_id"],
                    token_type=TokenType.YES if item["outcome"] == "Yes" else TokenType.NO,
                    side=OrderSide(item["side"]),
                    price=float(item["price"]),
                    size=float(item["size"]),
                    filled_size=float(item.get("filled_size", 0)),
                    status=OrderStatus(item["status"]),
                ))
            return orders
        except Exception as e:
            logger.warning(f"Failed to fetch open orders: {e}")
            return []
    
    async def get_trades(self, market_id: Optional[str] = None, limit: int = 100) -> list[Trade]:
        """Get recent trades."""
        if self.dry_run:
            trades = self._simulated_trades[-limit:]
            if market_id:
                trades = [t for t in trades if t.market_id == market_id]
            return trades
        
        try:
            params = {"limit": limit}
            if market_id:
                params["market_id"] = market_id
            
            data = await self._request("GET", "/trades", params=params)
            
            trades = []
            for item in data:
                trades.append(Trade(
                    trade_id=item["trade_id"],
                    order_id=item["order_id"],
                    market_id=item["market_id"],
                    token_type=TokenType.YES if item["outcome"] == "Yes" else TokenType.NO,
                    side=OrderSide(item["side"]),
                    price=float(item["price"]),
                    size=float(item["size"]),
                    fee=float(item.get("fee", 0)),
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                ))
            return trades
        except Exception as e:
            logger.warning(f"Failed to fetch trades: {e}")
            return []
    
    def simulate_fill(self, order_id: str, fill_size: Optional[float] = None) -> Optional[Trade]:
        """
        Simulate an order fill (for dry run mode).
        Returns the generated trade if successful.
        """
        if order_id not in self._simulated_orders:
            return None
        
        order = self._simulated_orders[order_id]
        if not order.is_open:
            return None
        
        fill_size = fill_size or order.remaining_size
        fill_size = min(fill_size, order.remaining_size)
        
        # Create trade with realistic Polymarket fees
        # Taker fee is ~1.5% (150 bps), maker is 0%
        # Assume taker for simulation (conservative)
        notional = fill_size * order.price
        fee_rate = 0.015  # 1.5% taker fee
        fee = notional * fee_rate
        
        trade = Trade(
            trade_id=f"trade_{uuid.uuid4().hex[:12]}",
            order_id=order_id,
            market_id=order.market_id,
            token_type=order.token_type,
            side=order.side,
            price=order.price,
            size=fill_size,
            fee=fee,  # Realistic 1.5% fee
        )
        
        # Update order
        order.filled_size += fill_size
        order.updated_at = datetime.utcnow()
        if order.remaining_size <= 0:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        # Update position
        self._update_simulated_position(trade)
        self._simulated_trades.append(trade)
        
        logger.info(f"[DRY RUN] Simulated fill: {trade}")
        return trade
    
    def _update_simulated_position(self, trade: Trade) -> None:
        """Update simulated position after a trade."""
        market_id = trade.market_id
        token_type = trade.token_type
        
        if market_id not in self._simulated_positions:
            self._simulated_positions[market_id] = {}
        
        if token_type not in self._simulated_positions[market_id]:
            self._simulated_positions[market_id][token_type] = Position(
                market_id=market_id,
                token_type=token_type,
                size=0,
                avg_entry_price=0,
            )
        
        pos = self._simulated_positions[market_id][token_type]
        
        # Update position
        if trade.side == OrderSide.BUY:
            new_size = pos.size + trade.size
            if new_size > 0:
                pos.avg_entry_price = (
                    (pos.avg_entry_price * pos.size + trade.price * trade.size) / new_size
                )
            pos.size = new_size
        else:
            # SELL reduces position
            if pos.size > 0:
                realized = (trade.price - pos.avg_entry_price) * trade.size
                pos.realized_pnl += realized
            pos.size -= trade.size

