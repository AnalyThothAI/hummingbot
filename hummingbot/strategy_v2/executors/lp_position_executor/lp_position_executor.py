import asyncio
import logging
from decimal import Decimal
from typing import Dict, Optional

from hummingbot.core.event.event_forwarder import SourceInfoEventForwarder
from hummingbot.core.event.events import (
    MarketEvent,
    RangePositionLiquidityAddedEvent,
    RangePositionLiquidityRemovedEvent,
    RangePositionUpdateFailureEvent,
)
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.strategy_v2.executors.executor_base import ExecutorBase
from hummingbot.strategy_v2.executors.lp_position_executor.data_types import (
    LPPositionExecutorConfig,
    LPPositionState,
    LPPositionStates,
)
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import CloseType, TrackedOrder


class LPPositionExecutor(ExecutorBase):
    """
    Executor for a single LP position lifecycle.

    - Opens position on start
    - Monitors and reports state (IN_RANGE, OUT_OF_RANGE)
    - Tracks out_of_range_since timestamp for rebalancing decisions
    - Closes position when stopped (unless keep_position=True)

    Rebalancing is handled by Controller (stops this executor, creates new one).

    Note: update_interval is passed by ExecutorOrchestrator (default 1.0s).
    max_retries is also passed by orchestrator.
    """
    _logger: Optional[HummingbotLogger] = None

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    def __init__(
        self,
        strategy: ScriptStrategyBase,
        config: LPPositionExecutorConfig,
        update_interval: float = 1.0,
        max_retries: int = 10
    ):
        # Extract connector names from config for ExecutorBase
        connectors = [config.connector_name]
        super().__init__(strategy, connectors, config, update_interval)
        self.config: LPPositionExecutorConfig = config
        self.lp_position_state = LPPositionState()
        self._current_retries = 0
        self._max_retries = max_retries
        self._setup_lp_event_forwarders()

    def _setup_lp_event_forwarders(self):
        """Setup LP event forwarders (same pattern as ExecutorBase)"""
        self._lp_add_forwarder = SourceInfoEventForwarder(self.process_lp_added_event)
        self._lp_remove_forwarder = SourceInfoEventForwarder(self.process_lp_removed_event)
        self._lp_failure_forwarder = SourceInfoEventForwarder(self.process_lp_failure_event)
        self._lp_event_pairs = [
            (MarketEvent.RangePositionLiquidityAdded, self._lp_add_forwarder),
            (MarketEvent.RangePositionLiquidityRemoved, self._lp_remove_forwarder),
            (MarketEvent.RangePositionUpdateFailure, self._lp_failure_forwarder),
        ]

    async def on_start(self):
        """Start executor - will create position in first control_task"""
        await super().on_start()

    async def control_task(self):
        """Main control loop - simple state machine"""
        current_time = self._strategy.current_timestamp

        # Fetch position info when position exists to get current amounts
        if self.lp_position_state.position_address:
            await self._update_position_info()

        current_price = self._get_current_price()
        self.lp_position_state.update_state(current_price, current_time)

        match self.lp_position_state.state:
            case LPPositionStates.NOT_ACTIVE:
                # Create position
                await self._create_position()

            case LPPositionStates.OPENING | LPPositionStates.CLOSING:
                # Wait for events
                pass

            case LPPositionStates.IN_RANGE | LPPositionStates.OUT_OF_RANGE:
                # Position active - just monitor (controller handles rebalance decision)
                # Executor tracks out_of_range_since, controller reads it to decide when to rebalance
                pass

            case LPPositionStates.COMPLETE:
                # Position closed - close_type already set by early_stop()
                self.stop()

            case LPPositionStates.RETRIES_EXCEEDED:
                # Already shutting down from failure handler
                pass

    async def _update_position_info(self):
        """Fetch current position info from connector to update amounts and fees"""
        if not self.lp_position_state.position_address:
            return

        connector = self.connectors.get(self.config.connector_name)
        if connector is None:
            return

        try:
            position_info = await connector.get_position_info(
                trading_pair=self.config.trading_pair,
                position_address=self.lp_position_state.position_address
            )

            if position_info:
                # Update amounts and fees from live position data
                self.lp_position_state.base_amount = Decimal(str(position_info.base_token_amount))
                self.lp_position_state.quote_amount = Decimal(str(position_info.quote_token_amount))
                self.lp_position_state.base_fee = Decimal(str(position_info.base_fee_amount))
                self.lp_position_state.quote_fee = Decimal(str(position_info.quote_fee_amount))
                # Update price bounds from actual position (may differ slightly from config)
                self.lp_position_state.lower_price = Decimal(str(position_info.lower_price))
                self.lp_position_state.upper_price = Decimal(str(position_info.upper_price))
        except Exception as e:
            self.logger().debug(f"Error fetching position info: {e}")

    async def _create_position(self):
        """
        Create position based on config.
        Calls connector.add_liquidity() which maps to gateway open_position endpoint.
        """
        connector = self.connectors.get(self.config.connector_name)
        if connector is None:
            self.logger().error(f"Connector {self.config.connector_name} not found")
            return

        # Calculate mid price for add_liquidity call
        mid_price = (self.config.lower_price + self.config.upper_price) / Decimal("2")

        order_id = connector.add_liquidity(
            trading_pair=self.config.trading_pair,
            price=float(mid_price),
            lower_price=float(self.config.lower_price),
            upper_price=float(self.config.upper_price),
            base_token_amount=float(self.config.base_amount),
            quote_token_amount=float(self.config.quote_amount),
            pool_address=self.config.pool_address,
            extra_params=self.config.extra_params,
        )
        self.lp_position_state.active_open_order = TrackedOrder(order_id=order_id)
        self.lp_position_state.state = LPPositionStates.OPENING

    async def _close_position(self):
        """
        Close position (removes all liquidity and closes position).
        Calls connector.remove_liquidity() which maps to gateway close_position endpoint.
        """
        connector = self.connectors.get(self.config.connector_name)
        if connector is None:
            self.logger().error(f"Connector {self.config.connector_name} not found")
            return

        order_id = connector.remove_liquidity(
            trading_pair=self.config.trading_pair,
            position_address=self.lp_position_state.position_address,
        )
        self.lp_position_state.active_close_order = TrackedOrder(order_id=order_id)
        self.lp_position_state.state = LPPositionStates.CLOSING

    def register_events(self):
        """Register for LP events on connector"""
        super().register_events()
        for connector in self.connectors.values():
            for event_pair in self._lp_event_pairs:
                connector.add_listener(event_pair[0], event_pair[1])

    def unregister_events(self):
        """Unregister LP events"""
        super().unregister_events()
        for connector in self.connectors.values():
            for event_pair in self._lp_event_pairs:
                connector.remove_listener(event_pair[0], event_pair[1])

    def process_lp_added_event(
        self,
        event_tag: int,
        market: any,
        event: RangePositionLiquidityAddedEvent
    ):
        """
        Handle successful liquidity add.
        Extracts position data directly from event - no need to fetch positions separately.
        """
        if self.lp_position_state.active_open_order and \
           event.order_id == self.lp_position_state.active_open_order.order_id:
            # Extract position data directly from event
            self.lp_position_state.position_address = event.position_address
            # Track rent paid to create position
            self.lp_position_state.position_rent = event.position_rent or Decimal("0")
            self.lp_position_state.base_amount = event.base_amount or Decimal("0")
            self.lp_position_state.quote_amount = event.quote_amount or Decimal("0")
            self.lp_position_state.lower_price = event.lower_price
            self.lp_position_state.upper_price = event.upper_price

            # Clear active_open_order to indicate opening is complete
            self.lp_position_state.active_open_order = None

            self.logger().info(
                f"Position created: {event.position_address}, "
                f"rent: {event.position_rent} SOL, "
                f"base: {event.base_amount}, quote: {event.quote_amount}"
            )

            # Reset retry counter on success
            self._current_retries = 0

    def process_lp_removed_event(
        self,
        event_tag: int,
        market: any,
        event: RangePositionLiquidityRemovedEvent
    ):
        """
        Handle successful liquidity remove (close position).
        """
        if self.lp_position_state.active_close_order and \
           event.order_id == self.lp_position_state.active_close_order.order_id:
            # Mark position as closed by clearing position_address
            # Note: We don't set is_filled - it's a read-only property.
            self.lp_position_state.state = LPPositionStates.COMPLETE

            # Track rent refunded on close
            self.lp_position_state.position_rent_refunded = event.position_rent_refunded or Decimal("0")

            # Update final amounts (tokens returned from position)
            self.lp_position_state.base_amount = event.base_amount or Decimal("0")
            self.lp_position_state.quote_amount = event.quote_amount or Decimal("0")

            # Track fees collected in this close operation
            self.lp_position_state.base_fee = event.base_fee or Decimal("0")
            self.lp_position_state.quote_fee = event.quote_fee or Decimal("0")

            self.logger().info(
                f"Position closed: {self.lp_position_state.position_address}, "
                f"rent refunded: {event.position_rent_refunded} SOL, "
                f"base: {event.base_amount}, quote: {event.quote_amount}, "
                f"fees: {event.base_fee} base / {event.quote_fee} quote"
            )

            # Clear active_close_order and position_address
            self.lp_position_state.active_close_order = None
            self.lp_position_state.position_address = None

            # Reset retry counter on success
            self._current_retries = 0

    def process_lp_failure_event(
        self,
        event_tag: int,
        market: any,
        event: RangePositionUpdateFailureEvent
    ):
        """
        Handle LP operation failure (timeout/error) with retry logic.
        """
        # Check if this failure is for our pending operation
        is_open_failure = (
            self.lp_position_state.active_open_order and
            event.order_id == self.lp_position_state.active_open_order.order_id
        )
        is_close_failure = (
            self.lp_position_state.active_close_order and
            event.order_id == self.lp_position_state.active_close_order.order_id
        )

        if not is_open_failure and not is_close_failure:
            return  # Not our order

        operation_type = "open" if is_open_failure else "close"
        self._current_retries += 1

        if self._current_retries >= self._max_retries:
            # Max retries exceeded - transition to failed state
            self.logger().error(
                f"LP {operation_type} failed after {self._max_retries} retries. "
                "Blockchain may be down or severely congested."
            )
            self.lp_position_state.state = LPPositionStates.RETRIES_EXCEEDED
            # Stop executor - controller will see RETRIES_EXCEEDED and handle appropriately
            self.close_type = CloseType.FAILED
            self._status = RunnableStatus.SHUTTING_DOWN
            return

        # Log and retry
        self.logger().warning(
            f"LP {operation_type} failed (retry {self._current_retries}/{self._max_retries}). "
            "Chain may be congested. Retrying..."
        )

        # Clear failed order to trigger retry in next control_task cycle
        if is_open_failure:
            self.lp_position_state.active_open_order = None
            # State will be NOT_ACTIVE, control_task will retry _create_position()
        elif is_close_failure:
            self.lp_position_state.active_close_order = None
            # State stays at current (IN_RANGE/OUT_OF_RANGE), early_stop will retry _close_position()

    def early_stop(self, keep_position: bool = False):
        """Stop executor (like GridExecutor.early_stop)"""
        self._status = RunnableStatus.SHUTTING_DOWN
        self.close_type = CloseType.POSITION_HOLD if keep_position or self.config.keep_position else CloseType.EARLY_STOP

        # Close position if not keeping it
        if not keep_position and not self.config.keep_position:
            if self.lp_position_state.state in [LPPositionStates.IN_RANGE, LPPositionStates.OUT_OF_RANGE]:
                asyncio.create_task(self._close_position())

    @property
    def filled_amount_quote(self) -> Decimal:
        """Returns total position value in quote currency (tokens + fees)"""
        current_price = self._get_current_price()
        if current_price is None or current_price == Decimal("0"):
            return Decimal("0")

        # Total value = (base * price + quote) + (base_fee * price + quote_fee)
        token_value = (
            self.lp_position_state.base_amount * current_price +
            self.lp_position_state.quote_amount
        )
        fee_value = (
            self.lp_position_state.base_fee * current_price +
            self.lp_position_state.quote_fee
        )
        return token_value + fee_value

    def get_custom_info(self) -> Dict:
        """Report state to controller"""
        current_price = self._get_current_price()
        # Convert side int to display string (side is set by controller in config)
        side_map = {0: "BOTH", 1: "BUY", 2: "SELL"}
        side_str = side_map.get(self.config.side, "")
        return {
            # Side: 0=BOTH (both-sided), 1=BUY (quote only), 2=SELL (base only)
            "side": side_str,
            "state": self.lp_position_state.state.value,
            "position_address": self.lp_position_state.position_address,
            "current_price": float(current_price) if current_price else None,
            "lower_price": float(self.lp_position_state.lower_price),
            "upper_price": float(self.lp_position_state.upper_price),
            "base_amount": float(self.lp_position_state.base_amount),
            "quote_amount": float(self.lp_position_state.quote_amount),
            "base_fee": float(self.lp_position_state.base_fee),
            "quote_fee": float(self.lp_position_state.quote_fee),
            # Rent tracking
            "position_rent": float(self.lp_position_state.position_rent),
            "position_rent_refunded": float(self.lp_position_state.position_rent_refunded),
            # Timer tracking - executor tracks when it went out of bounds
            "out_of_range_since": self.lp_position_state.out_of_range_since,
        }

    def get_lp_position_summary(self):
        """Create LPPositionSummary for reporting."""
        from hummingbot.strategy_v2.executors.data_types import LPPositionSummary

        current_price = self._get_current_price() or Decimal("0")
        total_value = (
            self.lp_position_state.base_amount * current_price +
            self.lp_position_state.quote_amount
        )

        # Map side: 0=BOTH, 1=BUY, 2=SELL
        side_map = {0: "BOTH", 1: "BUY", 2: "SELL"}
        side = side_map.get(self.config.side, "BOTH")

        return LPPositionSummary(
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            position_address=self.lp_position_state.position_address or "",
            side=side,
            state=self.lp_position_state.state.value,
            current_price=current_price,
            lower_price=self.lp_position_state.lower_price,
            upper_price=self.lp_position_state.upper_price,
            base_amount=self.lp_position_state.base_amount,
            quote_amount=self.lp_position_state.quote_amount,
            base_token=self.config.base_token,
            quote_token=self.config.quote_token,
            base_fee=self.lp_position_state.base_fee,
            quote_fee=self.lp_position_state.quote_fee,
            total_value_quote=total_value,
            unrealized_pnl_quote=self.get_net_pnl_quote(),
            out_of_range_since=self.lp_position_state.out_of_range_since,
        )

    def to_format_status(self) -> str:
        """Format executor status for display (like XEMMExecutor)"""
        current_price = self._get_current_price()
        state = self.lp_position_state.state.value
        position_addr = self.lp_position_state.position_address or "N/A"

        # Calculate values
        base_amt = float(self.lp_position_state.base_amount)
        quote_amt = float(self.lp_position_state.quote_amount)
        base_fee = float(self.lp_position_state.base_fee)
        quote_fee = float(self.lp_position_state.quote_fee)
        lower = float(self.lp_position_state.lower_price)
        upper = float(self.lp_position_state.upper_price)
        price = float(current_price) if current_price else 0

        # Total value in quote
        total_value = base_amt * price + quote_amt + base_fee * price + quote_fee if price else 0

        # Side display
        side_map = {0: "BOTH", 1: "BUY", 2: "SELL"}
        side = side_map.get(self.config.side, "")

        return f"""
LP Position: {position_addr[:16]}... | State: {state} | Side: {side}
-----------------------------------------------------------------------------------------------------------------------
    - Range: [{lower:.6f} - {upper:.6f}] | Price: {price:.6f}
    - Tokens: {base_amt:.6f} base / {quote_amt:.6f} quote | Fees: {base_fee:.6f} / {quote_fee:.6f}
    - Total Value: {total_value:.6f} quote | PnL: {float(self.get_net_pnl_quote()):.6f} ({float(self.get_net_pnl_pct()):.2f}%)
-----------------------------------------------------------------------------------------------------------------------
"""

    # Required abstract methods from ExecutorBase
    async def validate_sufficient_balance(self):
        """Validate sufficient balance for LP position. ExecutorBase calls this in on_start()."""
        # LP connector handles balance validation during add_liquidity
        pass

    def get_net_pnl_quote(self) -> Decimal:
        """
        Returns net P&L in quote currency.

        P&L = (current_position_value + fees_earned) - initial_value
        """
        if not self.lp_position_state.position_address:
            return Decimal("0")

        current_price = self._get_current_price()
        if current_price is None or current_price == Decimal("0"):
            return Decimal("0")

        # Initial value (from config)
        initial_value = (
            self.config.base_amount * current_price +
            self.config.quote_amount
        )

        # Current position value (tokens in position)
        current_value = (
            self.lp_position_state.base_amount * current_price +
            self.lp_position_state.quote_amount
        )

        # Fees earned (LP swap fees, not transaction costs)
        fees_earned = (
            self.lp_position_state.base_fee * current_price +
            self.lp_position_state.quote_fee
        )

        # P&L = current value + fees - initial value
        return current_value + fees_earned - initial_value

    def get_net_pnl_pct(self) -> Decimal:
        """Returns net P&L as percentage."""
        current_price = self._get_current_price()
        if current_price is None or current_price == Decimal("0"):
            return Decimal("0")

        # Initial value (from config)
        initial_value = (
            self.config.base_amount * current_price +
            self.config.quote_amount
        )

        if initial_value == Decimal("0"):
            return Decimal("0")

        pnl_quote = self.get_net_pnl_quote()
        return (pnl_quote / initial_value) * Decimal("100")

    def get_cum_fees_quote(self) -> Decimal:
        """
        Returns cumulative transaction costs in quote currency.

        NOTE: This is for transaction/gas costs, NOT LP fees earned.
        LP fees earned are included in get_net_pnl_quote() calculation.
        """
        return Decimal("0")

    def _get_current_price(self) -> Optional[Decimal]:
        """Get current price from RateOracle (LP connectors don't have get_price_by_type)"""
        try:
            price = self._strategy.market_data_provider.get_rate(self.config.trading_pair)
            return Decimal(str(price)) if price else None
        except Exception:
            return None
