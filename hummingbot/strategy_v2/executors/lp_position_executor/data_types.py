from decimal import Decimal
from enum import Enum
from typing import Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict

from hummingbot.strategy_v2.executors.data_types import ExecutorConfigBase
from hummingbot.strategy_v2.models.executors import TrackedOrder


class LPPositionStates(Enum):
    """
    State machine for LP position lifecycle.
    Price direction (above/below range) is determined from custom_info, not state.
    """
    NOT_ACTIVE = "NOT_ACTIVE"              # No position, no pending orders
    OPENING = "OPENING"                    # add_liquidity submitted, waiting
    IN_RANGE = "IN_RANGE"                  # Position active, price within bounds
    OUT_OF_RANGE = "OUT_OF_RANGE"          # Position active, price outside bounds
    CLOSING = "CLOSING"                    # remove_liquidity submitted, waiting
    COMPLETE = "COMPLETE"                  # Position closed permanently
    RETRIES_EXCEEDED = "RETRIES_EXCEEDED"  # Failed to open/close after max retries


class LPPositionExecutorConfig(ExecutorConfigBase):
    """
    Configuration for LP Position Executor.

    Initial version: Simple behavior matching current script
    - Assumes no existing positions in the pool
    - Creates position based on config
    - Closes position when executor stops (unless keep_position=True)
    """
    type: Literal["lp_position_executor"] = "lp_position_executor"

    # Pool identification
    connector_name: str  # e.g., "meteora/clmm"
    pool_address: str
    trading_pair: str  # Resolved from pool, e.g., "SOL-USDC"

    # Token info (from trading pair)
    base_token: str = ""
    quote_token: str = ""
    base_token_address: Optional[str] = None
    quote_token_address: Optional[str] = None

    # Position price bounds (calculated by controller)
    lower_price: Decimal
    upper_price: Decimal

    # Position amounts
    base_amount: Decimal = Decimal("0")  # Initial base amount
    quote_amount: Decimal = Decimal("0")  # Initial quote amount

    # Position side: 0=BOTH, 1=BUY (quote only), 2=SELL (base only)
    # Set by controller based on initial amounts
    side: int = 0

    # Connector-specific params
    extra_params: Optional[Dict] = None  # e.g., {"strategyType": 0} for Meteora

    # Early stop behavior (like PositionExecutor)
    keep_position: bool = False  # If True, don't close position on executor stop

    # Budget coordination (optional)
    budget_key: Optional[str] = None
    budget_reservation_id: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LPPositionState(BaseModel):
    """Tracks a single LP position state within executor."""
    position_address: Optional[str] = None
    lower_price: Decimal = Decimal("0")
    upper_price: Decimal = Decimal("0")
    base_amount: Decimal = Decimal("0")
    quote_amount: Decimal = Decimal("0")
    base_fee: Decimal = Decimal("0")
    quote_fee: Decimal = Decimal("0")

    # Rent tracking
    position_rent: Decimal = Decimal("0")  # SOL rent paid to create position (ADD only)
    position_rent_refunded: Decimal = Decimal("0")  # SOL rent refunded on close (REMOVE only)

    # Order tracking
    active_open_order: Optional[TrackedOrder] = None
    active_close_order: Optional[TrackedOrder] = None

    # State
    state: LPPositionStates = LPPositionStates.NOT_ACTIVE

    # Timer tracking (executor tracks when it went out of bounds)
    out_of_range_since: Optional[float] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def update_state(self, current_price: Optional[Decimal] = None, current_time: Optional[float] = None):
        """
        Update state based on position_address and price.
        Called each control_task cycle.

        Note: We don't use TrackedOrder.is_filled since it's read-only.
        Instead, we check:
        - position_address set = position was created
        - state == COMPLETE (set by event handler) = position was closed

        Args:
            current_price: Current market price
            current_time: Current timestamp (for tracking out_of_range_since)
        """
        # If already complete, stay complete
        if self.state == LPPositionStates.COMPLETE:
            return

        # If closing order is active but position still exists, we're closing
        if self.active_close_order is not None:
            self.state = LPPositionStates.CLOSING
            return

        # If open order is active but position not yet created, we're opening
        if self.active_open_order is not None and self.position_address is None:
            self.state = LPPositionStates.OPENING
            return

        # Position exists - determine state based on price location
        if self.position_address and current_price is not None:
            if current_price < self.lower_price or current_price > self.upper_price:
                self.state = LPPositionStates.OUT_OF_RANGE
            else:
                self.state = LPPositionStates.IN_RANGE
        elif self.position_address is None:
            self.state = LPPositionStates.NOT_ACTIVE

        # Track out_of_range_since timer (matches original script logic)
        if self.state == LPPositionStates.IN_RANGE:
            # Price back in range - reset timer
            self.out_of_range_since = None
        elif self.state == LPPositionStates.OUT_OF_RANGE:
            # Price out of bounds - start timer if not already started
            if self.out_of_range_since is None and current_time is not None:
                self.out_of_range_since = current_time

    def reset(self):
        """
        Reset state to initial values.
        Note: Not currently used - executors are replaced, not reused.
        Kept for potential future use (e.g., restart support).
        """
        self.position_address = None
        self.lower_price = Decimal("0")
        self.upper_price = Decimal("0")
        self.base_amount = Decimal("0")
        self.quote_amount = Decimal("0")
        self.base_fee = Decimal("0")
        self.quote_fee = Decimal("0")
        self.position_rent = Decimal("0")
        self.position_rent_refunded = Decimal("0")
        self.active_open_order = None
        self.active_close_order = None
        self.state = LPPositionStates.NOT_ACTIVE
        self.out_of_range_since = None
