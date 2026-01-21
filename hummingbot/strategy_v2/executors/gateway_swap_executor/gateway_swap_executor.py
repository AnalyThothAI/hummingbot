import asyncio
import logging
import time
from decimal import Decimal
from typing import Dict, Optional, Tuple

from hummingbot.core.data_type.common import TradeType
from hummingbot.core.data_type.in_flight_order import OrderState
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


class GatewaySwapExecutor(ExecutorBase):
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
        super().__init__(strategy=strategy, config=config, connectors=[config.connector_name], update_interval=update_interval)
        self.config: GatewaySwapExecutorConfig = config
        self._order: Optional[TrackedOrder] = None
        self._order_created_ts: Optional[float] = None
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

    async def control_task(self):
        if self.status == RunnableStatus.RUNNING:
            if self._order is None:
                await self._place_swap_order()
            else:
                await self._check_order_status()
        elif self.status == RunnableStatus.SHUTTING_DOWN:
            await self._check_order_status()

    async def _place_swap_order(self):
        connector = self.connectors.get(self.config.connector_name)
        if connector is None:
            self._last_error = f"Connector {self.config.connector_name} not found"
            self.logger().error(self._last_error)
            self._mark_failed()
            return

        is_buy, trading_pair, amount = self._resolve_swap_params()
        if trading_pair is None:
            self._mark_failed()
            return

        quote_price = await self._get_quote_price(connector, trading_pair, is_buy, amount)
        if quote_price is None or quote_price <= 0:
            self._handle_retryable_error("quote_price_unavailable")
            return

        try:
            if self._budget_coordinator:
                async with self._budget_coordinator.action_lock:
                    order_id = connector.place_order(
                        is_buy=is_buy,
                        trading_pair=trading_pair,
                        amount=amount,
                        price=quote_price,
                        slippage_pct=self.config.slippage_pct,
                        pool_address=self.config.pool_address,
                    )
            else:
                order_id = connector.place_order(
                    is_buy=is_buy,
                    trading_pair=trading_pair,
                    amount=amount,
                    price=quote_price,
                    slippage_pct=self.config.slippage_pct,
                    pool_address=self.config.pool_address,
                )
        except Exception as e:
            self._handle_retryable_error(f"place_order_failed:{e}")
            return

        self._order = TrackedOrder(order_id=order_id)
        self._order_created_ts = time.time()
        self._last_error = None

    async def _check_order_status(self):
        connector = self.connectors.get(self.config.connector_name)
        if connector is None or self._order is None:
            return

        order_id = self._order.order_id
        tracked_order = connector.get_order(order_id)
        if tracked_order is None:
            return

        try:
            await connector.update_order_status([tracked_order])
        except Exception as e:
            self.logger().debug(f"Swap order status update failed: {e}")

        if tracked_order.is_done:
            if tracked_order.is_failure or tracked_order.is_cancelled:
                self._handle_retryable_error("order_failed")
                return
            self._capture_execution(tracked_order)
            self.close_type = CloseType.COMPLETED
            self.stop()
            return

        if self._is_timed_out():
            self._handle_retryable_error("order_timeout")

    def _resolve_swap_params(self) -> Tuple[bool, Optional[str], Decimal]:
        trading_pair = self.config.trading_pair
        is_buy = self.config.side == TradeType.BUY
        amount = self.config.amount

        if amount <= 0:
            self._last_error = "Swap amount must be > 0"
            self.logger().error(self._last_error)
            return is_buy, None, amount

        if self.config.amount_in_is_quote:
            if self.config.side != TradeType.BUY:
                self._last_error = "amount_in_is_quote requires side=BUY"
                self.logger().error(self._last_error)
                return is_buy, None, amount
            base, quote = trading_pair.split("-")
            trading_pair = f"{quote}-{base}"
            is_buy = False

        return is_buy, trading_pair, amount

    async def _get_quote_price(self, connector, trading_pair: str, is_buy: bool, amount: Decimal) -> Optional[Decimal]:
        try:
            price = await connector.get_quote_price(
                trading_pair=trading_pair,
                is_buy=is_buy,
                amount=amount,
                slippage_pct=self.config.slippage_pct,
                pool_address=self.config.pool_address,
            )
        except Exception as e:
            self.logger().warning(f"Swap quote failed: {e}")
            return None
        return Decimal(str(price)) if price is not None else None

    def _is_timed_out(self) -> bool:
        if self._order_created_ts is None:
            return False
        return (time.time() - self._order_created_ts) > self.config.timeout_sec

    def _handle_retryable_error(self, reason: str):
        self._current_retries += 1
        self._last_error = reason
        if self._current_retries > max(self.config.max_retries, self._max_retries):
            self._mark_failed()
            return
        self._order = None
        self._order_created_ts = None
        asyncio.create_task(self._sleep(float(self.config.retry_delay_sec)))

    async def _sleep(self, delay: float):
        await asyncio.sleep(delay)

    def _mark_failed(self):
        self.close_type = CloseType.FAILED
        self._status = RunnableStatus.SHUTTING_DOWN
        self.stop()

    def early_stop(self, keep_position: bool = False):
        self._status = RunnableStatus.SHUTTING_DOWN

    async def validate_sufficient_balance(self):
        pass

    def get_net_pnl_quote(self) -> Decimal:
        return Decimal("0")

    def get_net_pnl_pct(self) -> Decimal:
        return Decimal("0")

    def get_cum_fees_quote(self) -> Decimal:
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
