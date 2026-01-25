import asyncio
import time
from decimal import Decimal
from typing import Any, Dict, Optional

from hummingbot.core.data_type.common import TradeType
from hummingbot.core.gateway.gateway_http_client import GatewayHttpClient
from hummingbot.core.network_iterator import NetworkStatus
from hummingbot.data_feed.candles_feed.candles_base import CandlesBase


class GatewayPriceCandles(CandlesBase):
    def __init__(self, connector: str, trading_pair: str, interval: str = "1m", max_records: int = 150):
        self._connector = connector
        self._chain: Optional[str] = None
        self._network: Optional[str] = None
        self._sample_task: Optional[asyncio.Task] = None
        self._logged_chain_error = False
        super().__init__(trading_pair=trading_pair, interval=interval, max_records=max_records)

    @property
    def name(self):
        return f"gateway_{self._connector.replace('/', '_')}_{self._trading_pair}"

    @property
    def rest_url(self):
        return ""

    @property
    def health_check_url(self):
        return ""

    @property
    def candles_url(self):
        return ""

    @property
    def candles_endpoint(self):
        return ""

    @property
    def candles_max_result_per_rest_request(self):
        return self.max_records

    @property
    def wss_url(self):
        return ""

    @property
    def rate_limits(self):
        return []

    @property
    def intervals(self):
        return self.interval_to_seconds

    def get_exchange_trading_pair(self, trading_pair):
        return trading_pair

    async def check_network(self) -> NetworkStatus:
        if await GatewayHttpClient.get_instance().ping_gateway():
            return NetworkStatus.CONNECTED
        return NetworkStatus.NOT_CONNECTED

    async def start_network(self):
        if self._sample_task is None or self._sample_task.done():
            self._sample_task = asyncio.create_task(self._sample_loop())

    async def stop_network(self):
        if self._sample_task is not None:
            self._sample_task.cancel()
            self._sample_task = None

    async def _sample_loop(self):
        while True:
            try:
                price = await self._fetch_price()
                if price is not None and price > 0:
                    self._append_price(price)
                await self._sleep(self.interval_in_seconds)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger().exception("Unexpected error sampling gateway price candles.")
                await self._sleep(1.0)

    def _append_price(self, price: Decimal) -> None:
        now = int(time.time())
        timestamp = self._round_timestamp_to_interval_multiple(now)
        row = self._build_row(timestamp, price)

        if not self._candles:
            self._bootstrap_candles(timestamp, price)
            return

        last_ts = int(self._candles[-1][0])
        if timestamp == last_ts:
            self._update_last_row(price)
            return
        if timestamp < last_ts:
            return
        self._candles.append(row)
        self._ws_candle_available.set()

    def _bootstrap_candles(self, timestamp: int, price: Decimal) -> None:
        start_ts = timestamp - (self.max_records - 1) * self.interval_in_seconds
        for i in range(self.max_records):
            ts = int(start_ts + i * self.interval_in_seconds)
            self._candles.append(self._build_row(ts, price))
        self._ws_candle_available.set()

    def _update_last_row(self, price: Decimal) -> None:
        last = list(self._candles[-1])
        last[2] = max(last[2], float(price))
        last[3] = min(last[3], float(price))
        last[4] = float(price)
        self._candles[-1] = last

    def _build_row(self, timestamp: int, price: Decimal):
        price_f = float(price)
        return [
            float(timestamp),
            price_f,
            price_f,
            price_f,
            price_f,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    async def _fetch_price(self) -> Optional[Decimal]:
        if self._chain is None or self._network is None:
            await self._ensure_chain_network()
        if self._chain is None or self._network is None:
            return None
        base, quote = self._split_pair()
        if not base or not quote:
            return None
        gateway = GatewayHttpClient.get_instance()
        resp: Dict[str, Any] = await gateway.get_price(
            chain=self._chain,
            network=self._network,
            connector=self._connector,
            base_asset=base,
            quote_asset=quote,
            amount=Decimal("1"),
            side=TradeType.SELL,
            fail_silently=True,
        )
        price = resp.get("price") if isinstance(resp, dict) else None
        if price is None:
            return None
        return Decimal(str(price))

    async def _ensure_chain_network(self) -> None:
        gateway = GatewayHttpClient.get_instance()
        chain, network, error = await gateway.get_connector_chain_network(self._connector)
        if chain and network:
            self._chain = chain
            self._network = network
            self._logged_chain_error = False
            return
        if not self._logged_chain_error:
            self.logger().warning("Gateway connector info unavailable for %s: %s", self._connector, error)
            self._logged_chain_error = True

    def _split_pair(self) -> tuple:
        parts = self._trading_pair.split("-")
        if len(parts) != 2:
            return "", ""
        return parts[0], parts[1]
