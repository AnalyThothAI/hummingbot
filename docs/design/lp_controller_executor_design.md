# LP Position Manager - StrategyV2 Controller + Executor Design

## Overview

This document outlines the design for converting `lp_manage_position.py` into a StrategyV2 architecture with a dedicated Controller and Executor. The goal is to create a reusable, composable LP position management system that follows established patterns from `grid_executor` and `position_executor`.

## Current Architecture Analysis

### Current `lp_manage_position.py` Structure

The current script (~1600 lines) handles:
1. **Pool/Position Resolution**: Resolving trading pair from pool address
2. **Position Monitoring**: Tracking position bounds, checking if price is in range
3. **Rebalancing Logic**: Closing out-of-range positions, reopening single-sided
4. **Event Handling**: LP add/remove/failure events with retry logic
5. **P&L Tracking**: Recording position history, calculating P&L
6. **Status Display**: Formatting position status with visualizations

### Key State Machine

The state machine should track **position state** (like GridLevel) rather than price location.
Price location (in range, above, below) determines **rebalance direction**, not state.

```
Executor State Flow:
NOT_ACTIVE → OPENING → IN_RANGE ↔ OUT_OF_RANGE → CLOSING → COMPLETE
                ↓                                    ↓
        RETRIES_EXCEEDED                      RETRIES_EXCEEDED
```

## Proposed Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              v2_with_controllers.py (loader script)              │
│  - Lists one or more controller configs                         │
│  - Routes executor actions                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       LPController                               │
│  - Initializes connectors via update_markets()                  │
│  - Resolves pool info (trading pair, tokens)                    │
│  - Calculates position bounds from width % and price limits     │
│  - Creates/stops executors                                      │
│  - Handles rebalancing logic (timer, single-sided reopen)       │
│  - Returns CreateExecutorAction / StopExecutorAction            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LPPositionExecutor                             │
│  - Single LP position lifecycle (open → monitor → close)        │
│  - Tracks out_of_range_since (when price exits range)          │
│  - Reports state via get_custom_info()                          │
│  - Processes LP events (add, remove, failure)                   │
│  - Closes position on stop (unless keep_position=True)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Design Decisions

### Decision 1: Single Executor Per Position vs Multi-Position Executor

**Options:**
- **A) Single Executor Per Position**: Each LPPositionExecutor manages one position's lifecycle (open → monitor → close)
- **B) Multi-Position Executor**: One executor manages multiple positions simultaneously

**Recommendation: Option A - Single Executor Per Position**

**Rationale:**
- Matches existing patterns (PositionExecutor, GridExecutor manage single strategies)
- Cleaner state management (one state machine per executor)
- Easier error handling and retry logic isolation
- Controller can spawn multiple executors for multiple pools

---

### Decision 2: Executor Scope - Open+Close vs Open-Only

**Options:**
- **A) Full Lifecycle Executor**: Executor handles open, monitor, and close
- **B) Open-Only Executor**: Executor only opens position; Controller issues StopExecutorAction to close

**Recommendation: Option A - Full Lifecycle Executor**

**Rationale:**
- LP positions need atomic close+reopen for rebalancing
- Executor needs to track position state for retry logic
- Position close and reopen share context (amounts from closed position)
- Similar to how GridExecutor manages complete level lifecycle

---

### Decision 3: Rebalancing Strategy

**Options:**
- **A) Same Executor Rebalances**: Executor closes and reopens within its lifecycle
- **B) New Executor Per Rebalance**: Controller stops old executor, creates new one

**Recommendation: Option B - New Executor Per Rebalance**

**Rationale:**
- Cleaner separation: executor handles single position lifecycle only
- Controller owns rebalancing logic (timer, when to rebalance)
- Controller saves amounts from closed executor, passes to new executor config
- Simpler executor implementation (~100 lines)
- Matches pattern of other executors (PositionExecutor, GridExecutor)

---

### Decision 4: LP Event Handling

**Options:**
- **A) Events Through Executor**: Register LP event forwarders directly on connector
- **B) Events Through Strategy Callbacks**: Strategy routes events to executors

**Recommendation: Option A - Events Through Executor**

**Rationale:**
- Matches existing ExecutorBase pattern exactly (order events use same approach)
- LP events already triggered on connector via `connector.trigger_event()` in `gateway_lp.py`
- Executor self-contained: registers/unregisters its own LP event listeners
- No changes needed to StrategyV2Base or other base classes
- Simpler implementation (~20 lines vs ~50+ lines for Option B)

---

### Decision 5: State Persistence for P&L Tracking

**Options:**
- **A) Use StrategyV2 SQLite Database**: Leverage existing Executors table with `net_pnl_quote`, `custom_info` fields
- **B) Separate JSON Tracking File**: Controller manages dedicated JSON file

**Recommendation: Option A - Use StrategyV2 SQLite Database**

**Rationale:**
- Framework already persists executor data to SQLite automatically
- Executors table has `net_pnl_quote`, `net_pnl_pct`, `cum_fees_quote` fields
- `custom_info` JSON field stores LP-specific data (position_address, amounts, fees, price bounds)
- No additional tracking code needed in controller
- Controller can query closed executors from `executors_info` to calculate aggregate P&L

---

## Data Types Design

### LPPositionExecutorConfig

```python
from decimal import Decimal
from typing import Dict, Literal, Optional

from pydantic import ConfigDict

from hummingbot.strategy_v2.executors.data_types import ExecutorConfigBase


class LPPositionExecutorConfig(ExecutorConfigBase):
    """
    Configuration for LP Position Executor.

    Initial version: Simple behavior matching current script
    - Assumes no existing positions in the pool
    - Creates position based on config
    - Closes position when executor stops (unless keep_position=True)

    Future: Multi-pool version (like multi_grid_strike.py) can be added later.
    """
    type: Literal["lp_position_executor"] = "lp_position_executor"

    # Pool identification
    connector_name: str  # e.g., "meteora/clmm"
    pool_address: str
    trading_pair: str  # Resolved from pool, e.g., "SOL-USDC"

    # Token info (resolved from pool)
    base_token: str
    quote_token: str
    base_token_address: str
    quote_token_address: str

    # Position price bounds (calculated by controller)
    lower_price: Decimal
    upper_price: Decimal

    # Position amounts
    base_amount: Decimal = Decimal("0")  # Initial base amount
    quote_amount: Decimal = Decimal("0")  # Initial quote amount

    # Connector-specific params
    extra_params: Optional[Dict] = None  # e.g., {"strategyType": 0} for Meteora

    # Early stop behavior (like PositionExecutor)
    keep_position: bool = False  # If True, don't close position on executor stop

    model_config = ConfigDict(arbitrary_types_allowed=True)
```

### LPPositionExecutor State

LP-specific states that include price location (different from GridLevel):

```python
from enum import Enum
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict

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
    RETRIES_EXCEEDED = "RETRIES_EXCEEDED"  # Failed to open/close after max retries (blockchain down?)


class LPPositionState(BaseModel):
    """Tracks a single LP position state within executor."""
    position_address: Optional[str] = None
    lower_price: Decimal = Decimal("0")
    upper_price: Decimal = Decimal("0")
    base_amount: Decimal = Decimal("0")
    quote_amount: Decimal = Decimal("0")
    base_fee: Decimal = Decimal("0")
    quote_fee: Decimal = Decimal("0")

    # Rent tracking: positive on ADD (rent paid), negative on REMOVE (rent refunded)
    # This matches the DB column rent_paid in RangePositionUpdate
    rent_paid: Decimal = Decimal("0")

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
        Update state based on orders and price.
        Called each control_task cycle.

        Args:
            current_price: Current market price
            current_time: Current timestamp (for tracking out_of_range_since)
        """
        previous_state = self.state

        # Check order states first (takes priority)
        if self.active_close_order is not None:
            if self.active_close_order.is_filled:
                self.state = LPPositionStates.COMPLETE
            else:
                self.state = LPPositionStates.CLOSING
            return

        if self.active_open_order is not None:
            if not self.active_open_order.is_filled:
                self.state = LPPositionStates.OPENING
                return
            # Open order filled - position exists, check price

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
        self.rent_paid = Decimal("0")
        self.active_open_order = None
        self.active_close_order = None
        self.state = LPPositionStates.NOT_ACTIVE
        self.out_of_range_since = None
```

### LPControllerConfig

```python
from decimal import Decimal
from typing import Dict, List, Optional

from pydantic import Field

from hummingbot.core.data_type.common import MarketDict
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers import ControllerConfigBase


class LPControllerConfig(ControllerConfigBase):
    """
    Configuration for LP Controller.
    Similar to GridStrikeConfig pattern.
    """
    controller_type: str = "generic"
    controller_name: str = "lp_manager"
    candles_config: List[CandlesConfig] = []

    # Pool configuration (required)
    connector_name: str = "meteora/clmm"
    network: str = "mainnet-beta"
    pool_address: str = ""  # Required - pool address to manage

    # Position parameters
    base_amount: Decimal = Field(default=Decimal("0"), json_schema_extra={"is_updatable": True})
    quote_amount: Decimal = Field(default=Decimal("1.0"), json_schema_extra={"is_updatable": True})
    position_width_pct: Decimal = Field(default=Decimal("2.0"), json_schema_extra={"is_updatable": True})

    # Rebalancing
    rebalance_seconds: int = Field(default=60, json_schema_extra={"is_updatable": True})  # How long out of bounds before rebalancing

    # Price limits (optional - constrain position bounds)
    lower_price_limit: Optional[Decimal] = Field(default=None, json_schema_extra={"is_updatable": True})
    upper_price_limit: Optional[Decimal] = Field(default=None, json_schema_extra={"is_updatable": True})

    # Connector-specific params (optional)
    strategy_type: Optional[int] = Field(default=None, json_schema_extra={"is_updatable": True})  # Meteora only: 0=Spot, 1=Curve, 2=Bid-Ask

    def update_markets(self, markets: MarketDict) -> MarketDict:
        """Initialize connector - LP uses pool_address, not trading_pair"""
        # LP connectors don't need trading_pair for initialization
        # Pool info is resolved at runtime via get_pool_info()
        return markets.add_or_update(self.connector_name, self.pool_address)
```

### P&L Tracking

P&L is tracked automatically via the StrategyV2 framework's SQLite database:

- **Executors table** stores each executor's lifecycle with `net_pnl_quote`, `net_pnl_pct`, `cum_fees_quote`
- **`custom_info` JSON field** contains LP-specific data (position_address, amounts, fees, price bounds)
- Controller queries closed executors to calculate aggregate P&L across rebalances

No separate JSON file needed - the existing database handles everything.

---

## LPPositionExecutor Implementation

**Executor responsibility**: Single position lifecycle only (open → monitor → close)
**Controller responsibility**: Rebalancing logic (timer, stop old executor, create new one)

### Key Methods

```python
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

    def __init__(self, strategy, config: LPPositionExecutorConfig, update_interval: float = 1.0, max_retries: int = 10):
        # Extract connector names from config for ExecutorBase
        connectors = [config.connector_name]
        super().__init__(strategy, connectors, config, update_interval)
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
        current_price = self._get_current_price()
        current_time = self._strategy.current_timestamp  # For tracking out_of_range_since
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
                # Position closed
                self.stop()

            case LPPositionStates.RETRIES_EXCEEDED:
                # Already shutting down from failure handler
                pass

    async def _create_position(self):
        """
        Create position based on config.
        Calls connector.add_liquidity() which maps to gateway open_position endpoint.
        """
        connector = self.connectors.get(self.config.connector_name)
        order_id = connector.add_liquidity(
            pool_address=self.config.pool_address,
            lower_price=self.config.lower_price,
            upper_price=self.config.upper_price,
            base_amount=self.config.base_amount,
            quote_amount=self.config.quote_amount,
            extra_params=self.config.extra_params,
        )
        self.lp_position_state.active_open_order = TrackedOrder(order_id=order_id)
        self.lp_position_state.state = LPPositionStates.OPENING

    async def _close_position(self):
        """
        Close position (removes all liquidity and closes position).
        Calls connector.close_position() which maps to gateway close_position endpoint.
        """
        connector = self.connectors.get(self.config.connector_name)
        order_id = connector.close_position(
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

    # Event handlers
    def process_lp_added_event(self, event_tag, market, event: RangePositionLiquidityAddedEvent):
        """
        Handle successful liquidity add.
        Extracts position data directly from event - no need to fetch positions separately.

        Gateway response fields (from open_position):
        - positionAddress: Created position NFT address
        - positionRent: SOL deposited for position rent (~0.057 SOL)
        - baseTokenAmountAdded: Actual base token added (positive)
        - quoteTokenAmountAdded: Actual quote token added (positive)
        - fee: SOL transaction fee (separate from rent)
        """
        if self.lp_position_state.active_open_order and \
           event.order_id == self.lp_position_state.active_open_order.order_id:
            self.lp_position_state.active_open_order.is_filled = True

            # Extract position data directly from event
            self.lp_position_state.position_address = event.position_address
            # rent_paid is positive on ADD (we paid rent)
            self.lp_position_state.rent_paid = event.rent_paid
            self.lp_position_state.base_amount = event.base_amount
            self.lp_position_state.quote_amount = event.quote_amount
            self.lp_position_state.lower_price = event.lower_price
            self.lp_position_state.upper_price = event.upper_price

            self.logger().info(
                f"Position created: {event.position_address}, "
                f"rent paid: {event.rent_paid} SOL, "
                f"base: {event.base_amount}, quote: {event.quote_amount}"
            )

            # Reset retry counter on success
            self._current_retries = 0

    def process_lp_removed_event(self, event_tag, market, event: RangePositionLiquidityRemovedEvent):
        """
        Handle successful liquidity remove (close position).

        Gateway response fields (from close_position):
        - positionRentRefunded: SOL returned from position rent (~0.057 SOL)
        - baseTokenAmountRemoved: Actual base token removed (positive)
        - quoteTokenAmountRemoved: Actual quote token removed (positive)
        - baseFeeAmountCollected: LP trading fees collected in base token
        - quoteFeeAmountCollected: LP trading fees collected in quote token
        - fee: SOL transaction fee (separate from rent refund)
        """
        if self.lp_position_state.active_close_order and \
           event.order_id == self.lp_position_state.active_close_order.order_id:
            self.lp_position_state.active_close_order.is_filled = True

            # rent_paid is negative on REMOVE (we got refund)
            # This is stored as -positionRentRefunded by gateway_lp.py
            self.lp_position_state.rent_paid = event.rent_paid

            # Update final amounts (tokens returned from position)
            self.lp_position_state.base_amount = event.base_amount
            self.lp_position_state.quote_amount = event.quote_amount

            # Track fees collected in this close operation
            self.lp_position_state.base_fee = event.base_fee
            self.lp_position_state.quote_fee = event.quote_fee

            self.logger().info(
                f"Position closed: {self.lp_position_state.position_address}, "
                f"rent refunded: {abs(event.rent_paid)} SOL, "
                f"base: {event.base_amount}, quote: {event.quote_amount}, "
                f"fees: {event.base_fee} base / {event.quote_fee} quote"
            )

            # Clear position address (position no longer exists)
            self.lp_position_state.position_address = None

            # Reset retry counter on success
            self._current_retries = 0

    def process_lp_failure_event(self, event_tag, market, event: RangePositionUpdateFailureEvent):
        """
        Handle LP operation failure (timeout/error) with retry logic.

        Matches original script's did_fail_lp_update() pattern:
        - Identifies if failure is for open or close operation
        - Clears pending order state to trigger retry in next control_task
        - Tracks retry count and transitions to RETRIES_EXCEEDED if max exceeded

        This handles both:
        - Open failures (during position creation)
        - Close failures (during position closing/rebalance)
        """
        # Check if this failure is for our pending operation
        is_open_failure = (self.lp_position_state.active_open_order and
                          event.order_id == self.lp_position_state.active_open_order.order_id)
        is_close_failure = (self.lp_position_state.active_close_order and
                           event.order_id == self.lp_position_state.active_close_order.order_id)

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
            # State stays at current (IN_RANGE/OUT_OF_RANGE), control_task will retry _close_position()

    def early_stop(self, keep_position: bool = False):
        """Stop executor (like PositionExecutor.early_stop)"""
        if keep_position or self.config.keep_position:
            self.close_type = CloseType.POSITION_HOLD
        else:
            # Close position before stopping
            if self.lp_position_state.state in [LPPositionStates.IN_RANGE, LPPositionStates.OUT_OF_RANGE]:
                asyncio.create_task(self._close_position())
            self.close_type = CloseType.EARLY_STOP
        self._status = RunnableStatus.SHUTTING_DOWN

    def get_custom_info(self) -> Dict:
        """Report state to controller"""
        current_price = self._get_current_price()
        return {
            "state": self.lp_position_state.state.value,
            "position_address": self.lp_position_state.position_address,
            "current_price": float(current_price) if current_price else None,
            "lower_price": float(self.lp_position_state.lower_price),
            "upper_price": float(self.lp_position_state.upper_price),
            "base_amount": float(self.lp_position_state.base_amount),
            "quote_amount": float(self.lp_position_state.quote_amount),
            "base_fee": float(self.lp_position_state.base_fee),
            "quote_fee": float(self.lp_position_state.quote_fee),
            # Rent tracking: positive = paid (on ADD), negative = refunded (on REMOVE)
            "rent_paid": float(self.lp_position_state.rent_paid),
            # Timer tracking - executor tracks when it went out of bounds
            "out_of_range_since": self.lp_position_state.out_of_range_since,
        }

    # Required abstract methods from ExecutorBase
    async def validate_sufficient_balance(self):
        """Validate sufficient balance for LP position. ExecutorBase calls this in on_start()."""
        # LP connector handles balance validation during add_liquidity
        # Could add pre-validation here if needed
        pass

    def get_net_pnl_quote(self) -> Decimal:
        """
        Returns net P&L in quote currency.

        Calculation (matches original script's _calculate_pnl_summary):
        P&L = (current_position_value + fees_earned) - initial_value

        Where:
        - initial_value = initial_base * current_price + initial_quote
        - current_position_value = current_base * current_price + current_quote
        - fees_earned = base_fee * current_price + quote_fee

        Note: Uses current price for base token valuation (mark-to-market).
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
        """
        Returns net P&L as percentage.

        Calculation: (pnl_quote / initial_value) * 100
        """
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

        For now, returns 0 as transaction costs are not tracked at executor level.
        Future: Could track SOL spent on transactions if needed.
        """
        return Decimal("0")

    def _get_current_price(self) -> Optional[Decimal]:
        """Get current price from strategy's market data provider"""
        return self._strategy.market_data_provider.get_price_by_type(
            self.config.connector_name,
            self.config.trading_pair,
            PriceType.MidPrice
        )
```

---

## LPController Implementation

Controller handles:
- Pool info resolution
- Creating executor on start
- **Rebalancing logic** (timer, stop old executor, create new single-sided)

### Key Methods

```python
from decimal import Decimal
from typing import Dict, List, Optional

from hummingbot.strategy_v2.controllers import ControllerBase
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction, ExecutorAction
from hummingbot.strategy_v2.models.executors import ExecutorInfo
from hummingbot.core.data_type.common import PriceType


class LPController(ControllerBase):
    """Controller for LP position management with rebalancing logic"""

    def __init__(self, config: LPControllerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self._resolved = False

        # Resolved from pool once at startup (static)
        self._trading_pair: Optional[str] = None
        self._base_token: Optional[str] = None
        self._quote_token: Optional[str] = None
        self._base_token_address: Optional[str] = None
        self._quote_token_address: Optional[str] = None

        # Rebalance tracking
        # Note: out_of_range_since is tracked by executor, not controller
        self._last_executor_info: Optional[Dict] = None  # For rebalance amounts

    def initialize_rate_sources(self):
        """Initialize rate sources for price feeds (like GridStrike)"""
        if self._trading_pair:
            self.market_data_provider.initialize_rate_sources([
                ConnectorPair(
                    connector_name=self.config.connector_name,
                    trading_pair=self._trading_pair
                )
            ])

    def active_executor(self) -> Optional[ExecutorInfo]:
        """Get current active executor (should be 0 or 1)"""
        active = [e for e in self.executors_info if e.is_active]
        return active[0] if active else None

    def determine_executor_actions(self) -> List[ExecutorAction]:
        """Decide whether to create/stop executors"""
        actions = []
        executor = self.active_executor()

        # Manual kill switch
        if self.config.manual_kill_switch:
            if executor:
                actions.append(StopExecutorAction(
                    controller_id=self.config.id,
                    executor_id=executor.id
                ))
            return actions

        # No active executor
        if executor is None:
            # Wait for pool info to be resolved before creating executor
            if not self._resolved:
                return actions

            # Check if we're rebalancing (have saved info from previous executor)
            if self._last_executor_info:
                # Create new single-sided position for rebalance
                actions.append(CreateExecutorAction(
                    controller_id=self.config.id,
                    executor_config=self._create_executor_config()
                ))
            else:
                # Initial position - create executor with config amounts
                actions.append(CreateExecutorAction(
                    controller_id=self.config.id,
                    executor_config=self._create_executor_config()
                ))
            return actions

        # Check executor state
        state = executor.custom_info.get("state")

        # Don't take action while executor is in transition states
        if state in [LPPositionStates.OPENING.value, LPPositionStates.CLOSING.value]:
            return actions

        # Handle failed executor - don't auto-retry, require manual intervention
        if state == LPPositionStates.RETRIES_EXCEEDED.value:
            self.logger().error("Executor failed after max retries. Manual intervention required.")
            return actions

        # Check for rebalancing
        out_of_range_since = executor.custom_info.get("out_of_range_since")
        current_price = executor.custom_info.get("current_price")

        # Rebalancing logic: executor tracks when it went out of range,
        # controller reads it and decides when to rebalance
        if state == LPPositionStates.OUT_OF_RANGE.value:
            if out_of_range_since is not None:
                current_time = self.market_data_provider.time()
                elapsed = current_time - out_of_range_since
                if elapsed >= self.config.rebalance_seconds:
                    # Check if price is within limits before rebalancing
                    # If price is outside limits, skip rebalancing (don't chase)
                    if not self._is_price_within_limits(current_price):
                        self.logger().info(
                            f"Price {current_price} outside limits, skipping rebalance"
                        )
                        return actions  # Keep current position, don't rebalance

                    # Time to rebalance: save info for new executor, then stop current
                    self._last_executor_info = executor.custom_info.copy()
                    # Determine price direction from custom_info
                    lower_price = executor.custom_info.get("lower_price")
                    self._last_executor_info["was_below_range"] = (
                        current_price is not None and lower_price is not None and current_price < lower_price
                    )
                    actions.append(StopExecutorAction(
                        controller_id=self.config.id,
                        executor_id=executor.id
                    ))

        # Note: IN_RANGE state - timer is automatically reset by executor
        # (out_of_range_since set to None in LPPositionState.update_state())

        return actions

    def _create_executor_config(self) -> LPPositionExecutorConfig:
        """Create executor config - initial or rebalanced"""
        if self._last_executor_info:
            # Rebalancing - single-sided position
            info = self._last_executor_info
            if info["was_below_range"]:
                # Price below - use base only (position will be ABOVE current price)
                base_amt = Decimal(str(info["base_amount"])) + Decimal(str(info["base_fee"]))
                quote_amt = Decimal("0")
            else:
                # Price above - use quote only (position will be BELOW current price)
                base_amt = Decimal("0")
                quote_amt = Decimal(str(info["quote_amount"])) + Decimal(str(info["quote_fee"]))

            self._last_executor_info = None  # Clear after use
        else:
            # Initial position from config
            base_amt = self.config.base_amount
            quote_amt = self.config.quote_amount

        # Calculate price bounds from current price, width, and position type
        lower_price, upper_price = self._calculate_price_bounds(base_amt, quote_amt)

        # Build extra params (connector-specific)
        extra_params = {}
        if self.config.strategy_type is not None:
            extra_params["strategyType"] = self.config.strategy_type

        return LPPositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=self.config.connector_name,
            pool_address=self.config.pool_address,
            trading_pair=self._trading_pair,
            base_token=self._base_token,
            quote_token=self._quote_token,
            base_token_address=self._base_token_address,
            quote_token_address=self._quote_token_address,
            lower_price=lower_price,
            upper_price=upper_price,
            base_amount=base_amt,
            quote_amount=quote_amt,
            extra_params=extra_params if extra_params else None,
            keep_position=False,  # Always close on stop for rebalance
        )

    async def on_start(self):
        """Initialize controller - resolve pool info once"""
        await self._resolve_pool_info()

    async def _resolve_pool_info(self):
        """
        Resolve pool info once at startup.
        Stores static values: trading pair, token symbols, token addresses.
        """
        connector = self.connectors.get(self.config.connector_name)
        if connector:
            pool_info = await connector.get_pool_info(
                self.config.network,
                self.config.pool_address
            )
            # Store resolved values as instance variables (static, don't change)
            self._trading_pair = f"{pool_info.base_token}-{pool_info.quote_token}"
            self._base_token = pool_info.base_token
            self._quote_token = pool_info.quote_token
            self._base_token_address = pool_info.base_token_address
            self._quote_token_address = pool_info.quote_token_address

            # Initialize rate sources now that we have trading pair
            self.initialize_rate_sources()
            self._resolved = True

    async def update_processed_data(self):
        """Called every tick - no-op since we use executor's custom_info for dynamic data"""
        pass

    def _calculate_price_bounds(self, base_amt: Decimal, quote_amt: Decimal) -> tuple[Decimal, Decimal]:
        """
        Calculate position bounds from current price and width %.

        For double-sided positions (both base and quote): split width evenly
        For base-only positions: full width ABOVE price (sell base for quote)
        For quote-only positions: full width BELOW price (buy base with quote)

        This matches the original script's _compute_width_percentages() logic.
        """
        current_price = self.market_data_provider.get_price_by_type(
            self.config.connector_name, self._trading_pair, PriceType.MidPrice
        )
        total_width = self.config.position_width_pct / Decimal("100")

        if base_amt > 0 and quote_amt > 0:
            # Double-sided: split width evenly (±half)
            half_width = total_width / Decimal("2")
            lower_price = current_price * (Decimal("1") - half_width)
            upper_price = current_price * (Decimal("1") + half_width)
        elif base_amt > 0:
            # Base-only: full width ABOVE current price
            # Range: [current_price, current_price * (1 + total_width)]
            lower_price = current_price
            upper_price = current_price * (Decimal("1") + total_width)
        elif quote_amt > 0:
            # Quote-only: full width BELOW current price
            # Range: [current_price * (1 - total_width), current_price]
            lower_price = current_price * (Decimal("1") - total_width)
            upper_price = current_price
        else:
            # Default (shouldn't happen)
            half_width = total_width / Decimal("2")
            lower_price = current_price * (Decimal("1") - half_width)
            upper_price = current_price * (Decimal("1") + half_width)

        # Apply price limits if configured
        if self.config.lower_price_limit:
            lower_price = max(lower_price, self.config.lower_price_limit)
        if self.config.upper_price_limit:
            upper_price = min(upper_price, self.config.upper_price_limit)

        return lower_price, upper_price

    def _is_price_within_limits(self, price: Optional[float]) -> bool:
        """
        Check if price is within configured price limits.
        Used to skip rebalancing when price moves outside limits (don't chase).
        """
        if price is None:
            return True  # No price data, assume within limits
        price_decimal = Decimal(str(price))
        if self.config.lower_price_limit and price_decimal < self.config.lower_price_limit:
            return False
        if self.config.upper_price_limit and price_decimal > self.config.upper_price_limit:
            return False
        return True

    def to_format_status(self) -> List[str]:
        """
        Format status for display.
        Uses executor's custom_info for position data.
        Matches original script's format_status() output.
        """
        status = []
        box_width = 80

        # Header
        status.append("┌" + "─" * box_width + "┐")
        header = f"│ LP Manager: {self._trading_pair or 'Resolving...'} on {self.config.connector_name}"
        status.append(header + " " * (box_width - len(header) + 1) + "│")
        status.append("├" + "─" * box_width + "┤")

        executor = self.active_executor()

        if executor is None:
            # No active executor
            line = "│ Status: No active position"
            status.append(line + " " * (box_width - len(line) + 1) + "│")

            if self.config.base_amount > 0 or self.config.quote_amount > 0:
                line = f"│ Will create position with: {self.config.base_amount} base / {self.config.quote_amount} quote"
                status.append(line + " " * (box_width - len(line) + 1) + "│")
        else:
            # Active executor - get data from custom_info
            info = executor.custom_info
            state = info.get("state", "UNKNOWN")
            position_address = info.get("position_address", "N/A")
            current_price = info.get("current_price")
            lower_price = info.get("lower_price")
            upper_price = info.get("upper_price")
            base_amount = info.get("base_amount", 0)
            quote_amount = info.get("quote_amount", 0)
            base_fee = info.get("base_fee", 0)
            quote_fee = info.get("quote_fee", 0)
            out_of_range_since = info.get("out_of_range_since")

            # Position info
            line = f"│ Position: {position_address[:16]}..." if position_address else "│ Position: N/A"
            status.append(line + " " * (box_width - len(line) + 1) + "│")

            # State
            state_emoji = "✅" if state == "IN_RANGE" else "⚠️" if state == "OUT_OF_RANGE" else "⏳"
            line = f"│ State: {state_emoji} {state}"
            status.append(line + " " * (box_width - len(line) + 1) + "│")

            if current_price is not None:
                # Calculate total value
                token_value = base_amount * current_price + quote_amount
                fee_value = base_fee * current_price + quote_fee
                total_value = token_value + fee_value

                line = f"│ Total Value: {total_value:.6f} {self._quote_token}"
                status.append(line + " " * (box_width - len(line) + 1) + "│")

                line = f"│ Tokens: {base_amount:.6f} {self._base_token} / {quote_amount:.6f} {self._quote_token}"
                status.append(line + " " * (box_width - len(line) + 1) + "│")

                if base_fee > 0 or quote_fee > 0:
                    line = f"│ Fees: {base_fee:.6f} {self._base_token} / {quote_fee:.6f} {self._quote_token}"
                    status.append(line + " " * (box_width - len(line) + 1) + "│")

                status.append("│" + " " * box_width + "│")

                # Price range visualization
                if lower_price and upper_price:
                    status.append("│ Position Range:" + " " * (box_width - 16) + "│")
                    viz = self._create_price_range_visualization(
                        Decimal(str(lower_price)),
                        Decimal(str(current_price)),
                        Decimal(str(upper_price))
                    )
                    for viz_line in viz.split('\n'):
                        line = f"│ {viz_line}"
                        status.append(line + " " * (box_width - len(line) + 1) + "│")

                status.append("│" + " " * box_width + "│")
                line = f"│ Price: {current_price:.6f}"
                status.append(line + " " * (box_width - len(line) + 1) + "│")

                # Rebalance countdown if out of range
                if state == "OUT_OF_RANGE" and out_of_range_since:
                    elapsed = self.market_data_provider.time() - out_of_range_since
                    remaining = max(0, self.config.rebalance_seconds - elapsed)
                    line = f"│ Rebalance in: {int(remaining)}s"
                    status.append(line + " " * (box_width - len(line) + 1) + "│")

        # Price limits visualization if configured
        if self.config.lower_price_limit or self.config.upper_price_limit:
            status.append("│" + " " * box_width + "│")
            current_price = executor.custom_info.get("current_price") if executor else None
            if current_price:
                limits_viz = self._create_price_limits_visualization(Decimal(str(current_price)))
                if limits_viz:
                    for viz_line in limits_viz.split('\n'):
                        line = f"│ {viz_line}"
                        status.append(line + " " * (box_width - len(line) + 1) + "│")

        status.append("└" + "─" * box_width + "┘")
        return status

    def _create_price_range_visualization(self, lower_price: Decimal, current_price: Decimal,
                                          upper_price: Decimal) -> str:
        """Create visual representation of price range with current price marker"""
        price_range = upper_price - lower_price
        if price_range <= 0:
            return ""
        current_position = float((current_price - lower_price) / price_range)

        bar_width = 50
        current_pos = int(current_position * bar_width)

        range_bar = ['─'] * bar_width
        range_bar[0] = '├'
        range_bar[-1] = '┤'

        if current_pos < 0:
            marker_line = '● ' + ''.join(range_bar)
        elif current_pos >= bar_width:
            marker_line = ''.join(range_bar) + ' ●'
        else:
            range_bar[current_pos] = '●'
            marker_line = ''.join(range_bar)

        lower_str = f'{float(lower_price):.6f}'
        upper_str = f'{float(upper_price):.6f}'
        label_line = lower_str + ' ' * (bar_width - len(lower_str) - len(upper_str)) + upper_str

        return f"{marker_line}\n{label_line}"

    def _create_price_limits_visualization(self, current_price: Decimal) -> Optional[str]:
        """Create visualization of price limits with current price"""
        if not self.config.lower_price_limit and not self.config.upper_price_limit:
            return None

        lower_limit = self.config.lower_price_limit or Decimal("0")
        upper_limit = self.config.upper_price_limit or current_price * 2

        if not self.config.lower_price_limit:
            lower_limit = max(Decimal("0"), upper_limit * Decimal("0.5"))
        if not self.config.upper_price_limit:
            upper_limit = lower_limit * Decimal("2")

        price_range = upper_limit - lower_limit
        if price_range <= 0:
            return None

        current_position = float((current_price - lower_limit) / price_range)
        bar_width = 50
        current_pos = int(current_position * bar_width)

        limit_bar = ['─'] * bar_width
        limit_bar[0] = '['
        limit_bar[-1] = ']'

        if current_pos < 0:
            marker_line = '● ' + ''.join(limit_bar)
            limit_status = "⛔ BELOW LOWER LIMIT"
        elif current_pos >= bar_width:
            marker_line = ''.join(limit_bar) + ' ●'
            limit_status = "⛔ ABOVE UPPER LIMIT"
        else:
            limit_bar[current_pos] = '●'
            marker_line = ''.join(limit_bar)
            limit_status = "✓ Within Limits"

        lower_str = f'{float(lower_limit):.6f}' if self.config.lower_price_limit else 'None'
        upper_str = f'{float(upper_limit):.6f}' if self.config.upper_price_limit else 'None'
        label_line = lower_str + ' ' * (bar_width - len(lower_str) - len(upper_str)) + upper_str

        return f"Price Limits:\n{marker_line}\n{label_line}\nStatus: {limit_status}"
```

---

## LP Event Integration

### Event Pattern (Same as Other Executors)

LP events already exist in `MarketEvent` enum and are triggered by connectors:
- `MarketEvent.RangePositionLiquidityAdded = 300` - Used for open_position success
- `MarketEvent.RangePositionLiquidityRemoved = 301` - Used for close_position success (includes fees)
- `MarketEvent.RangePositionUpdateFailure = 303` - Used for operation failures

Note: We do NOT use these existing events:
- `RangePositionUpdate = 302` - Not needed
- `RangePositionFeeCollected = 304` - Fees are included in close_position response
- `RangePositionClosed = 305` - We use RangePositionLiquidityRemoved for close

LPPositionExecutor follows the **exact same pattern** as PositionExecutor/GridExecutor:

```python
# In LPPositionExecutor.__init__
self._lp_add_forwarder = SourceInfoEventForwarder(self.process_lp_added_event)
self._lp_remove_forwarder = SourceInfoEventForwarder(self.process_lp_removed_event)
self._lp_failure_forwarder = SourceInfoEventForwarder(self.process_lp_failure_event)

# LP event pairs (like self._event_pairs in ExecutorBase)
self._lp_event_pairs = [
    (MarketEvent.RangePositionLiquidityAdded, self._lp_add_forwarder),
    (MarketEvent.RangePositionLiquidityRemoved, self._lp_remove_forwarder),
    (MarketEvent.RangePositionUpdateFailure, self._lp_failure_forwarder),
]

def register_events(self):
    """Register for LP events on connector (same pattern as ExecutorBase)"""
    super().register_events()  # Register order events if needed
    for connector in self.connectors.values():
        for event_pair in self._lp_event_pairs:
            connector.add_listener(event_pair[0], event_pair[1])

def unregister_events(self):
    """Unregister LP events"""
    super().unregister_events()
    for connector in self.connectors.values():
        for event_pair in self._lp_event_pairs:
            connector.remove_listener(event_pair[0], event_pair[1])
```

This is the standard executor pattern - no changes needed to ExecutorBase or StrategyV2Base.

### LP Event Data Types

The existing event classes in `hummingbot/core/event/events.py` map to gateway response fields:

**Gateway Response → Event Field Mapping:**

| Gateway Field (open) | Event Field | Description |
|---------------------|-------------|-------------|
| `positionAddress` | `position_address` | Created position NFT address |
| `positionRent` | `rent_paid` (positive) | SOL deposited for position rent (~0.057 SOL) |
| `baseTokenAmountAdded` | `base_amount` | Actual base token added (positive) |
| `quoteTokenAmountAdded` | `quote_amount` | Actual quote token added (positive) |
| `fee` | `trade_fee` | SOL transaction fee (separate from rent) |

| Gateway Field (close) | Event Field | Description |
|----------------------|-------------|-------------|
| `positionRentRefunded` | `rent_paid` (negative) | SOL returned from position rent |
| `baseTokenAmountRemoved` | `base_amount` | Actual base token removed (positive) |
| `quoteTokenAmountRemoved` | `quote_amount` | Actual quote token removed (positive) |
| `baseFeeAmountCollected` | `base_fee` | LP trading fees collected in base token |
| `quoteFeeAmountCollected` | `quote_fee` | LP trading fees collected in quote token |
| `fee` | `trade_fee` | SOL transaction fee (separate from rent refund) |

**Key Design Decision: Unified `rent_paid` Field**

Instead of separate `position_rent` and `position_rent_refunded` fields, we use a single `rent_paid` field:
- **Positive on ADD**: Rent paid to create position (e.g., +0.057 SOL)
- **Negative on REMOVE**: Rent refunded when closing position (e.g., -0.057 SOL)

This matches the DB column `rent_paid` in `RangePositionUpdate` table and simplifies P&L tracking:
- Sum all `rent_paid` values to get net rent cost
- P&L calculation excludes rent (tracked separately for display)

```python
@dataclass
class RangePositionLiquidityAddedEvent:
    """
    Event triggered when liquidity is successfully added to a position.
    Maps from gateway open_position response.
    """
    timestamp: float
    order_id: str
    exchange_order_id: str  # Transaction signature
    trading_pair: str
    lower_price: Decimal
    upper_price: Decimal
    amount: Decimal
    fee_tier: str
    creation_timestamp: float
    trade_fee: TradeFeeBase
    token_id: Optional[int] = 0
    # P&L tracking fields
    position_address: Optional[str] = ""
    mid_price: Optional[Decimal] = s_decimal_0
    base_amount: Optional[Decimal] = s_decimal_0  # baseTokenAmountAdded (positive)
    quote_amount: Optional[Decimal] = s_decimal_0  # quoteTokenAmountAdded (positive)
    rent_paid: Optional[Decimal] = s_decimal_0  # positionRent (positive = paid)


@dataclass
class RangePositionLiquidityRemovedEvent:
    """
    Event triggered when liquidity is removed (partial or full close).
    Maps from gateway close_position / remove_liquidity response.
    """
    timestamp: float
    order_id: str
    exchange_order_id: str  # Transaction signature
    trading_pair: str
    token_id: str
    trade_fee: TradeFeeBase
    creation_timestamp: float
    # P&L tracking fields
    position_address: Optional[str] = ""
    lower_price: Optional[Decimal] = s_decimal_0
    upper_price: Optional[Decimal] = s_decimal_0
    mid_price: Optional[Decimal] = s_decimal_0
    base_amount: Optional[Decimal] = s_decimal_0  # baseTokenAmountRemoved (positive)
    quote_amount: Optional[Decimal] = s_decimal_0  # quoteTokenAmountRemoved (positive)
    base_fee: Optional[Decimal] = s_decimal_0  # baseFeeAmountCollected
    quote_fee: Optional[Decimal] = s_decimal_0  # quoteFeeAmountCollected
    rent_paid: Optional[Decimal] = s_decimal_0  # -positionRentRefunded (negative = refund)


@dataclass
class RangePositionUpdateFailureEvent:
    """
    Event triggered when LP operation fails (timeout, simulation error, etc).
    """
    timestamp: float
    order_id: str
    order_action: LPType  # ADD or REMOVE
```

These are the existing event classes in `hummingbot/core/event/events.py`.

### Registering with ExecutorOrchestrator

The `LPPositionExecutor` must be registered in `executor_orchestrator.py`:

```python
# In ExecutorOrchestrator._executor_mapping
_executor_mapping = {
    "position_executor": PositionExecutor,
    "grid_executor": GridExecutor,
    # ... existing executors ...
    "lp_position_executor": LPPositionExecutor,  # Add this
}
```

The orchestrator will then create executors with:
- `strategy` - the strategy object
- `config` - the executor config
- `update_interval` - from `executors_update_interval` (default 1.0s)
- `max_retries` - from `executors_max_retries` (default 10)

---

## File Structure

```
hummingbot/
├── strategy_v2/
│   ├── executors/
│   │   └── lp_position_executor/
│   │       ├── __init__.py
│   │       ├── data_types.py              # LPPositionExecutorConfig, LPPositionState
│   │       └── lp_position_executor.py    # LPPositionExecutor implementation
│   └── controllers/
│       └── lp_controller/                 # Alternative location
│
├── controllers/
│   └── generic/
│       └── lp_manager.py                  # LPController (follows grid_strike pattern)
```

---

## Migration Path

### Phase 1: Create Data Types
1. Create `lp_position_executor/data_types.py` with configs and state classes
2. Add LP event forwarders to support LP-specific events

### Phase 2: Implement LPPositionExecutor
1. Create `lp_position_executor.py` with core state machine
2. Port position open/close logic from script
3. Implement retry logic for timeouts

### Phase 3: Implement LPController
1. Create `lp_manager.py` controller
2. Port pool resolution and monitoring logic
3. Use framework's SQLite database for P&L tracking

### Phase 4: Integration
1. Test with existing LP operations
2. Validate event handling works correctly
3. Port visualization/status formatting

### Phase 5: Cleanup
1. Deprecate old script or keep as simple wrapper
2. Update documentation

---

## Resolved Design Questions

1. **State Machine** (Resolved)
   - State tracks **position/order status** (like GridLevel)
   - States: NOT_ACTIVE, OPENING, IN_RANGE, OUT_OF_RANGE, CLOSING, COMPLETE, RETRIES_EXCEEDED
   - Price direction (above/below range) determined from `custom_info` (current_price vs lower_price)

2. **Initial Position Handling** (Resolved - Simplified)
   - Initial version: **Assume no existing positions**
   - Executor creates position based on config at startup
   - Matches PositionExecutor behavior (starts fresh)
   - Future: Can add `existing_position_address` for restart support if needed

3. **Event Handling** (Resolved)
   - Same pattern as other executors - register `SourceInfoEventForwarder` on connectors
   - Uses existing `MarketEvent.RangePositionLiquidity*` events
   - No changes needed to ExecutorBase or StrategyV2Base

4. **Early Stop Behavior** (Resolved)
   - Use `early_stop(keep_position)` pattern from PositionExecutor
   - `keep_position=False` (default): Close LP position before stopping
   - `keep_position=True`: Leave position open, just stop monitoring
   - Config also has `keep_position` field for default behavior

5. **Multi-Pool Support** (Resolved - Deferred)
   - Initial version: Single pool per controller (matches current script)
   - Future: `MultiLPController` (like `multi_grid_strike.py`) can manage multiple pools
   - Reference: `controllers/generic/multi_grid_strike.py` for pattern

## Notes

**`get_user_positions(pool_address)`** in `gateway_lp.py`:
- Returns `List[CLMMPositionInfo]` - specifically for **LP positions**
- NOT for perpetual DEX positions
- Can be used in future version to discover existing positions on restart

---

## Summary

### Architecture

**LPPositionExecutor** - Single position lifecycle:
- Opens position on start
- Extracts position data directly from LP events (no separate fetch needed)
- Tracks `rent_paid` (positive on ADD = SOL deposited, negative on REMOVE = SOL refunded)
- Monitors position and reports state via `get_custom_info()`
- Tracks `out_of_range_since` timestamp (when price moves out of range)
- Reports: state, position_address, amounts, price, fees, rent_paid, out_of_range_since
- Closes position when stopped (unless `keep_position=True`)
- Runs `control_task()` every `update_interval` (default 1.0s, from ExecutorOrchestrator)

**LPController** - Orchestration and rebalancing:
- Initializes connectors via `update_markets()` (like grid_strike.py)
- Resolves pool info (trading pair, tokens) on startup
- Calculates position bounds from `position_width_pct` and price limits
- Creates executor on start with concrete price bounds
- Reads executor's `out_of_range_since` to decide when to rebalance
- When rebalance needed: stops executor, saves amounts, creates new single-sided executor
- **P&L tracking**: Uses framework's SQLite Executors table (`net_pnl_quote`, `custom_info` JSON)

**v2_with_controllers.py** - Lists controllers (no LP-specific strategy needed)

### Flow

```
Controller                          Executor
    |                                  |
    |--CreateExecutorAction----------->|
    |                                  | (opens position)
    |<--------custom_info: IN_RANGE----|
    |                                  |
    |<--custom_info: OUT_OF_RANGE,-----|  (price moves out)
    |   out_of_range_since: <ts>      |  (executor starts timer)
    |   current_price: <price>         |  (controller checks direction)
    |                                  |
    | (checks if elapsed >= rebalance_seconds)
    |--StopExecutorAction------------->|
    | (saves amounts + direction)      | (closes position)
    |                                  |
    |--CreateExecutorAction----------->|  (single-sided config)
    |                                  | (opens new position)
```

### Timing Parameters

- **update_interval** (default 1.0s): How often executor checks position status (passed by ExecutorOrchestrator)
- **rebalance_seconds** (default 60): How long price must stay out of bounds before rebalancing (controller config)

### Components

- **LPPositionStates**: NOT_ACTIVE, OPENING, IN_RANGE, OUT_OF_RANGE, CLOSING, COMPLETE, RETRIES_EXCEEDED
- **LPPositionState**: State tracking model within executor (includes `out_of_range_since`, `rent_paid`)
- **LPPositionExecutorConfig**: Position parameters (lower_price, upper_price, amounts)
- **LPPositionExecutor**: Position lifecycle (~100 lines)
- **LPControllerConfig**: Pool + policy parameters (position_width_pct, rebalance_seconds, price_limits)
- **LPController**: Rebalancing logic (~100 lines)
- **LP Event Types**: `RangePositionLiquidityAddedEvent`, `RangePositionLiquidityRemovedEvent`, `RangePositionUpdateFailureEvent`

### P&L Tracking Summary

**Database Schema** (`RangePositionUpdate` table):
- `base_amount`: Token amount added/removed (always positive)
- `quote_amount`: Token amount added/removed (always positive)
- `base_fee`: LP trading fees collected in base token (REMOVE only)
- `quote_fee`: LP trading fees collected in quote token (REMOVE only)
- `rent_paid`: Position rent - positive on ADD (paid), negative on REMOVE (refunded)

**P&L Calculation** (excludes rent):
```
P&L = (Total Close Value + Total Fees + Current Position Value) - Total Open Value
```

**Rent Tracking** (displayed separately):
```
Net Rent = sum(rent_paid for all updates)
```
- Positive: Net cost (more rent paid than refunded)
- Negative: Net refund (position closed, rent returned)
- Near zero: Position opened and closed (rent paid ≈ rent refunded)

### Future Extensions

- **MultiLPController**: Multiple pools (like `multi_grid_strike.py`)
- **Restart support**: `existing_position_address` config to resume existing positions
