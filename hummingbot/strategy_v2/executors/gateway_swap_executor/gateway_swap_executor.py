import logging
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple, Union

from hummingbot.core.data_type.common import TradeType
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    BuyOrderCreatedEvent,
    MarketOrderFailureEvent,
    OrderFilledEvent,
    SellOrderCompletedEvent,
    SellOrderCreatedEvent,
)
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.strategy_v2.budget.budget_coordinator import BudgetCoordinatorRegistry
from hummingbot.strategy_v2.executors.executor_base import ExecutorBase
from hummingbot.strategy_v2.executors.gateway_swap_executor.data_types import (
    GatewaySwapExecutorConfig,
    GatewaySwapExecutorStatus,
)
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import CloseType, TrackedOrder

QuoteResponse = Dict[str, Any]
OrderCreatedEvent = Union[BuyOrderCreatedEvent, SellOrderCreatedEvent]
OrderCompletedEvent = Union[BuyOrderCompletedEvent, SellOrderCompletedEvent]


class GatewaySwapExecutor(ExecutorBase):
    """Execute a single gateway swap using a strict quoteId flow and delayed retries."""
    _logger: Optional[HummingbotLogger] = None

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    def __init__(
        self,
        strategy: ScriptStrategyBase,
        config: GatewaySwapExecutorConfig,
        update_interval: float = 1.0,
        max_retries: int = 10,
    ):
        super().__init__(
            strategy=strategy,
            config=config,
            connectors=[config.connector_name],
            update_interval=update_interval,
        )
        self.config: GatewaySwapExecutorConfig = config
        self._order: Optional[TrackedOrder] = None
        self._order_created_ts: Optional[float] = None
        self._order_timeout_ts: Optional[float] = None
        self._order_not_found_count = 0
        self._next_retry_ts: Optional[float] = None
        self._current_retries = 0
        self._max_retries = max_retries
        self._last_error: Optional[str] = None
        self._executed_amount_base: Optional[Decimal] = None
        self._executed_amount_quote: Optional[Decimal] = None
        self._executed_amount_in: Optional[Decimal] = None
        self._executed_amount_out: Optional[Decimal] = None
        self._token_in: Optional[str] = None
        self._token_out: Optional[str] = None
        self._budget_coordinator = (
            BudgetCoordinatorRegistry.get(config.budget_key) if config.budget_key else None
        )
        if config.poll_interval_sec is not None and config.poll_interval_sec > 0:
            self.update_interval = float(config.poll_interval_sec)

    async def control_task(self):
        if self.status == RunnableStatus.RUNNING:
            if self._order is None:
                await self._place_swap_order()
            else:
                await self._check_order_status()
            return
        if self.status == RunnableStatus.SHUTTING_DOWN:
            await self._check_order_status()

    async def _place_swap_order(self):
        """Fetch a quote and place the swap order, respecting retry delays."""
        if not self._ready_for_retry():
            return
        self._next_retry_ts = None

        connector = self._get_connector(fail_on_missing=True)
        if connector is None:
            return

        swap_params = self._resolve_swap_params()
        if swap_params is None:
            self._mark_failed()
            return
        is_buy, trading_pair, amount = swap_params

        if not hasattr(connector, "get_quote"):
            self._set_last_error(f"Connector {self.config.connector_name} does not support get_quote")
            self._mark_failed()
            return

        quote = await self._fetch_quote(connector, trading_pair, is_buy, amount)
        if quote:
            self.logger().info(
                f"Gateway swap quote: connector={self.config.connector_name} pair={trading_pair} "
                f"side={'BUY' if is_buy else 'SELL'} amount={amount} slippage={self.config.slippage_pct} "
                f"pool={self.config.pool_address} quote_id={quote.get('quoteId')} price={quote.get('price')} "
                f"min_out={quote.get('minAmountOut')} max_in={quote.get('maxAmountIn')} "
                f"price_impact={quote.get('priceImpactPct')}"
            )
        quote_price = self._extract_quote_price(quote)
        if quote_price is None or quote_price <= 0:
            self._handle_retryable_error("quote_price_unavailable")
            return

        quote_id = quote.get("quoteId") if quote else None
        if not quote_id:
            self._handle_retryable_error("quote_id_unavailable")
            return
        pool_address = quote.get("poolAddress") if quote else None
        if not pool_address:
            pool_address = self.config.pool_address

        try:
            self.logger().info(
                f"Gateway swap submit: connector={self.config.connector_name} pair={trading_pair} "
                f"side={'BUY' if is_buy else 'SELL'} amount={amount} price={quote_price} "
                f"slippage={self.config.slippage_pct} pool={pool_address} quote_id={quote_id}"
            )
            order_id = await self._place_order(
                connector=connector,
                is_buy=is_buy,
                trading_pair=trading_pair,
                amount=amount,
                price=quote_price,
                pool_address=pool_address,
                quote_id=quote_id,
                quote=quote,
            )
        except Exception as e:
            self._handle_retryable_error(f"place_order_failed:{e}")
            return

        self._order = TrackedOrder(order_id=order_id)
        self._order_created_ts = self._now()
        self._order_timeout_ts = None
        self._order_not_found_count = 0
        self._last_error = None

    async def _check_order_status(self):
        connector = self._get_connector()
        if connector is None or self._order is None:
            return

        order_id = self._order.order_id
        tracked_order = connector.get_order(order_id)
        if tracked_order is None:
            self._order_not_found_count += 1
            if self._order_not_found_count >= 3:
                self._handle_retryable_error("order_not_found")
            return
        self._order_not_found_count = 0
        self._order.order = tracked_order

        try:
            await connector.update_order_status([tracked_order])
        except Exception as e:
            self.logger().debug(f"Swap order status update failed: {e}")

        if tracked_order.is_done:
            if tracked_order.is_failure or tracked_order.is_cancelled:
                self._handle_retryable_error("order_failed")
                return
            self._complete_order(tracked_order)
            return

        if self._is_timed_out():
            if self._order_timeout_ts is None:
                self._order_timeout_ts = self._now()
                self.logger().warning(
                    "Swap order timed out, waiting for confirmation: %s",
                    order_id,
                )
                return
            if (self._now() - self._order_timeout_ts) < self.config.timeout_sec:
                return
            self._handle_retryable_error("order_timeout")

    def _capture_execution(self, tracked_order):
        executed_base = getattr(tracked_order, "executed_amount_base", Decimal("0"))
        executed_quote = getattr(tracked_order, "executed_amount_quote", Decimal("0"))
        self._executed_amount_base = Decimal(str(executed_base or 0))
        self._executed_amount_quote = Decimal(str(executed_quote or 0))

        base_token, quote_token = self.config.trading_pair.split("-")
        if self.config.amount_in_is_quote:
            self._token_in = quote_token
            self._token_out = base_token
            self._executed_amount_in = self._executed_amount_base
            self._executed_amount_out = self._executed_amount_quote
            return

        if self.config.side == TradeType.BUY:
            self._token_in = quote_token
            self._token_out = base_token
            self._executed_amount_in = self._executed_amount_quote
            self._executed_amount_out = self._executed_amount_base
            return

        self._token_in = base_token
        self._token_out = quote_token
        self._executed_amount_in = self._executed_amount_base
        self._executed_amount_out = self._executed_amount_quote
        return

    def _complete_order(self, tracked_order):
        self._capture_execution(tracked_order)
        self.close_type = CloseType.COMPLETED
        self.stop()

    def _resolve_swap_params(self) -> Optional[Tuple[bool, str, Decimal]]:
        trading_pair = self.config.trading_pair
        amount = self.config.amount
        is_buy = self.config.side == TradeType.BUY

        if amount <= 0:
            self._set_last_error("Swap amount must be > 0")
            return None

        if self.config.amount_in_is_quote:
            if self.config.side != TradeType.BUY:
                self._set_last_error("amount_in_is_quote requires side=BUY")
                return None
            base, quote = trading_pair.split("-")
            return False, f"{quote}-{base}", amount

        return is_buy, trading_pair, amount

    async def _fetch_quote(
        self,
        connector,
        trading_pair: str,
        is_buy: bool,
        amount: Decimal,
    ) -> Optional[QuoteResponse]:
        try:
            if not hasattr(connector, "get_quote"):
                return None
            return await connector.get_quote(
                trading_pair=trading_pair,
                is_buy=is_buy,
                amount=amount,
                slippage_pct=self.config.slippage_pct,
                pool_address=self.config.pool_address,
            )
        except Exception as e:
            self.logger().warning(f"Swap quote failed: {e}")
            return None

    @staticmethod
    def _to_decimal(value: Any) -> Optional[Decimal]:
        if value is None:
            return None
        return Decimal(str(value))

    @staticmethod
    def _extract_quote_price(quote: Optional[QuoteResponse]) -> Optional[Decimal]:
        if not quote:
            return None
        return GatewaySwapExecutor._to_decimal(quote.get("price"))

    async def _estimate_input_amount(
        self,
        connector,
        trading_pair: str,
        is_buy: bool,
        amount: Decimal,
    ) -> Tuple[Optional[str], Decimal]:
        base_token, quote_token = trading_pair.split("-")
        if not is_buy:
            return base_token, amount

        quote = await self._fetch_quote(connector, trading_pair, is_buy, amount)
        max_amount_in = self._to_decimal(quote.get("maxAmountIn") if quote else None)
        if max_amount_in is not None:
            return quote_token, max_amount_in
        amount_in = self._to_decimal(quote.get("amountIn") if quote else None)
        if amount_in is not None:
            return quote_token, amount_in

        price = self._extract_quote_price(quote)
        if price is None:
            return None, Decimal("0")
        return quote_token, amount * price

    def _is_timed_out(self) -> bool:
        if self._order_created_ts is None:
            return False
        return (self._now() - self._order_created_ts) > self.config.timeout_sec

    def _now(self) -> float:
        return self._strategy.current_timestamp

    def _handle_retryable_error(self, reason: str):
        self._current_retries += 1
        self._last_error = reason
        max_retries = self._max_retries if self.config.max_retries is None else self.config.max_retries
        if self._current_retries > max_retries:
            self._mark_failed()
            return
        self._order = None
        self._order_created_ts = None
        self._order_timeout_ts = None
        self._order_not_found_count = 0
        self._next_retry_ts = self._now() + float(self.config.retry_delay_sec)

    def _mark_failed(self):
        self.close_type = CloseType.FAILED
        self._status = RunnableStatus.SHUTTING_DOWN
        self.stop()

    def _set_last_error(self, message: str, *, log: bool = True):
        self._last_error = message
        if log:
            self.logger().error(message)

    def _ready_for_retry(self) -> bool:
        return self._next_retry_ts is None or self._now() >= self._next_retry_ts

    def _get_connector(self, *, fail_on_missing: bool = False):
        connector = self.connectors.get(self.config.connector_name)
        if connector is None and fail_on_missing:
            self._set_last_error(f"Connector {self.config.connector_name} not found")
            self._mark_failed()
        return connector

    async def _place_order(
        self,
        connector,
        is_buy: bool,
        trading_pair: str,
        amount: Decimal,
        price: Decimal,
        pool_address: Optional[str],
        quote_id: str,
        quote: QuoteResponse,
    ) -> str:
        if self._budget_coordinator:
            async with self._budget_coordinator.action_lock:
                return connector.place_order(
                    is_buy=is_buy,
                    trading_pair=trading_pair,
                    amount=amount,
                    price=price,
                    slippage_pct=self.config.slippage_pct,
                    pool_address=pool_address,
                    quote_id=quote_id,
                    quote_response=quote,
                )
        return connector.place_order(
            is_buy=is_buy,
            trading_pair=trading_pair,
            amount=amount,
            price=price,
            slippage_pct=self.config.slippage_pct,
            pool_address=pool_address,
            quote_id=quote_id,
            quote_response=quote,
        )

    def early_stop(self, keep_position: bool = False):
        self._status = RunnableStatus.SHUTTING_DOWN

    async def validate_sufficient_balance(self):
        """Pre-flight check for available balance of the input token."""
        connector = self._get_connector()
        if connector is None:
            self._set_last_error(f"Connector {self.config.connector_name} not found")
            self.close_type = CloseType.FAILED
            self.stop()
            return

        swap_params = self._resolve_swap_params()
        if swap_params is None:
            self.close_type = CloseType.FAILED
            self.stop()
            return
        is_buy, trading_pair, amount = swap_params

        token_in, amount_in = await self._estimate_input_amount(connector, trading_pair, is_buy, amount)
        if token_in is None or amount_in <= 0:
            return

        available = Decimal(str(connector.get_available_balance(token_in) or 0))
        if available < amount_in:
            self._set_last_error("insufficient_balance", log=False)
            self.close_type = CloseType.INSUFFICIENT_BALANCE
            self.stop()

    def get_net_pnl_quote(self) -> Decimal:
        return Decimal("0")

    def get_net_pnl_pct(self) -> Decimal:
        return Decimal("0")

    def get_cum_fees_quote(self) -> Decimal:
        return self._order.cum_fees_quote if self._order else Decimal("0")

    @property
    def filled_amount_quote(self) -> Decimal:
        if self.config.amount_in_is_quote:
            return self._executed_amount_in or Decimal("0")
        if self._executed_amount_quote is not None:
            return self._executed_amount_quote
        return Decimal("0")

    def get_custom_info(self) -> Dict:
        status = GatewaySwapExecutorStatus(
            state=self.status.name,
            order_id=self._order.order_id if self._order else None,
            last_error=self._last_error,
            trading_pair=self.config.trading_pair,
            side=self.config.side.name if self.config.side else None,
            amount_in_is_quote=self.config.amount_in_is_quote,
            token_in=self._token_in,
            token_out=self._token_out,
            amount_in=self._executed_amount_in,
            amount_out=self._executed_amount_out,
            executed_amount_base=self._executed_amount_base,
            executed_amount_quote=self._executed_amount_quote,
        )
        return status.model_dump()

    def _update_tracked_order(self, order_id: str):
        if not self._order or self._order.order_id != order_id:
            return
        in_flight_order = self.get_in_flight_order(self.config.connector_name, order_id)
        if in_flight_order is not None:
            self._order.order = in_flight_order

    def process_order_created_event(self, _, market, event: OrderCreatedEvent):
        self._update_tracked_order(event.order_id)

    def process_order_filled_event(self, _, market, event: OrderFilledEvent):
        self._update_tracked_order(event.order_id)

    def process_order_completed_event(self, _, market, event: OrderCompletedEvent):
        self._update_tracked_order(event.order_id)
        if self._order and self._order.order_id == event.order_id and self._order.order:
            self._complete_order(self._order.order)

    def process_order_failed_event(self, _, market, event: MarketOrderFailureEvent):
        if self._order and event.order_id == self._order.order_id:
            self._handle_retryable_error("order_failed")
