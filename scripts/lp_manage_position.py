"""
lp_manage_position.py

CLMM LP position manager that automatically rebalances positions.

BEHAVIOR
--------
- Monitors existing position in the specified pool
- Monitors price vs. active position's price bounds
- When price is out-of-bounds for >= rebalance_seconds, fully closes the position and re-enters
- First position can be double-sided (if both base_amount and quote_amount provided)
- After first rebalance, all subsequent positions are SINGLE-SIDED
- Single-sided positions provide only the token needed based on where price moved

PARAMETERS
----------
- connector: CLMM connector in format 'name/type' (e.g. raydium/clmm, meteora/clmm)
- pool_address: Pool address (e.g. 2sf5NYcY4zUPXUSmG6f66mskb24t5F8S11pC1Nz5nQT3)
- base_amount: Initial base token amount (0 for quote-only position)
- quote_amount: Initial quote token amount (0 for base-only position)
  * If both are 0 and no existing position: monitoring only
  * If both provided: creates double-sided initial position
  * After rebalance: only one token provided based on price direction
- position_width_pct: TOTAL position width as percentage of mid price (e.g. 2.0 = ±1%)
- rebalance_seconds: Seconds price must stay out-of-bounds before rebalancing
- lower_price_limit: (Optional) Never provide liquidity below this price
  * When price < lower_price_limit: Skip rebalance, keep current position
  * When price > lower_price_limit: Position's lower bound won't go below this limit
- upper_price_limit: (Optional) Never provide liquidity above this price
  * When price > upper_price_limit: Skip rebalance, keep current position
  * When price < upper_price_limit: Position's upper bound won't go above this limit
- strategy_type: (Optional) Meteora-specific strategy type (0=Spot, 1=Curve, None=use default)

PRICE LIMITS USE CASES
-----------------------
- Lower limit only: Accumulation strategy (buy support, but not below X)
- Upper limit only: Profit-taking strategy (sell resistance, but not above Y)
- Both limits: Range-bound strategy (only provide liquidity in channel)

NOTES
-----
- All tick rounding and amount calculations delegated to Gateway
- After first rebalance, automatically switches to single-sided positions
- Uses actual wallet balances for rebalancing (not config amounts)
- Price limits are enforced on rebalance and new positions
"""

import asyncio
import json
import logging
import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Dict, Optional, Tuple

from pydantic import Field

from hummingbot.client.config.config_data_types import BaseClientModel
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.gateway.common_types import ConnectorType, get_connector_type
from hummingbot.connector.gateway.gateway_lp import CLMMPoolInfo, CLMMPositionInfo
from hummingbot.core.data_type.common import LPType
from hummingbot.core.event.events import (
    RangePositionLiquidityAddedEvent,
    RangePositionLiquidityRemovedEvent,
    RangePositionUpdateFailureEvent,
)
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.data_feed.market_data_provider import MarketDataProvider
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.strategy_v2.executors.data_types import ConnectorPair


class LpPositionManagerConfig(BaseClientModel):
    script_file_name: str = os.path.basename(__file__)
    config_file_name: Optional[str] = None  # Set by trading_core with the config file name
    connector: str = Field("meteora/clmm", json_schema_extra={
        "prompt": "CLMM connector in format 'name/type' (e.g. meteora/clmm, uniswap/clmm)", "prompt_on_new": True})
    pool_address: str = Field("", json_schema_extra={
        "prompt": "Pool address (e.g. 2sf5NYcY4zUPXUSmG6f66mskb24t5F8S11pC1Nz5nQT3)", "prompt_on_new": True})
    base_amount: Decimal = Field(Decimal("0"), json_schema_extra={
        "prompt": "Initial base token amount (0 for quote-only initial position)", "prompt_on_new": True})
    quote_amount: Decimal = Field(Decimal("0"), json_schema_extra={
        "prompt": "Initial quote token amount (0 for base-only initial position)", "prompt_on_new": True})
    position_width_pct: Decimal = Field(Decimal("2.0"), json_schema_extra={
        "prompt": "TOTAL position width as percentage (e.g. 2.0 for ±1% around mid price)", "prompt_on_new": True})
    rebalance_seconds: int = Field(60, json_schema_extra={
        "prompt": "Seconds price must stay out-of-bounds before rebalancing", "prompt_on_new": True})
    check_seconds: int = Field(10, json_schema_extra={
        "prompt": "Seconds between position status checks (configure based on node rate limits)", "prompt_on_new": True})
    lower_price_limit: Optional[Decimal] = Field(None, json_schema_extra={
        "prompt": "Lower price limit (optional, never provide liquidity below this price)", "prompt_on_new": False})
    upper_price_limit: Optional[Decimal] = Field(None, json_schema_extra={
        "prompt": "Upper price limit (optional, never provide liquidity above this price)", "prompt_on_new": False})
    strategy_type: Optional[int] = Field(None, json_schema_extra={
        "prompt": "Strategy type for Meteora (0=Spot, 1=Curve, None=use default)", "prompt_on_new": False})


class LpPositionManager(ScriptStrategyBase):
    """
    CLMM LP position manager that automatically rebalances when price moves out of bounds.
    """

    # Constants for configuration
    POSITION_CREATION_DELAY = 3  # seconds to wait for position to be created on-chain
    MIN_TOKEN_AMOUNT = Decimal("0.0001")  # Minimum token amount to avoid dust
    POSITION_UPDATE_INTERVAL = 10  # seconds between position info updates

    @classmethod
    def init_markets(cls, config: LpPositionManagerConfig):
        # Use placeholder for trading pair since we'll resolve it from pool_address
        cls.markets = {config.connector: {"UNKNOWN-UNKNOWN"}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: LpPositionManagerConfig):
        super().__init__(connectors)
        self.config = config
        self.exchange = config.connector
        self.connector_type = get_connector_type(config.connector)

        # Verify this is a CLMM connector
        if self.connector_type != ConnectorType.CLMM:
            raise ValueError(f"This script only supports CLMM connectors. Got: {config.connector}")

        # Pool address is the primary identifier
        self.pool_address: str = config.pool_address
        if not self.pool_address:
            raise ValueError("pool_address is required")

        # Trading pair will be resolved from pool info
        self.trading_pair: Optional[str] = None
        self.base_token: Optional[str] = None
        self.quote_token: Optional[str] = None
        self.base_token_address: Optional[str] = None
        self.quote_token_address: Optional[str] = None
        self._trading_pair_resolved = False

        # Initialize market data provider for rate oracle (required for PNL tracking)
        # We'll initialize rate sources after resolving trading_pair
        self.market_data_provider = MarketDataProvider(connectors)

        # State tracking
        self.pool_info: Optional[CLMMPoolInfo] = None
        self.position_info: Optional[CLMMPositionInfo] = None
        self.current_position_id: Optional[str] = None
        self.out_of_bounds_since: Optional[float] = None
        self.has_rebalanced_once: bool = False

        # Order tracking
        self.pending_open_order_id: Optional[str] = None
        self.pending_close_order_id: Optional[str] = None
        self.pending_operation: Optional[str] = None  # "opening", "closing"

        # Rebalance tracking
        self._closed_position_balances: Optional[Dict] = None
        self._opening_position_amounts: Optional[Dict] = None

        # Throttling for rate limiting
        self._last_check_time: float = 0
        self._last_position_update_time: float = 0
        self._last_pool_info_fetch_time: float = 0

        # P&L tracking - use config file name if available, otherwise script name
        if config.config_file_name:
            # Remove .yml extension if present
            tracking_name = config.config_file_name.replace('.yml', '').replace('.yaml', '')
        else:
            # Fallback to script name without .py extension
            tracking_name = config.script_file_name.replace('.py', '')
        self._tracking_file_path = Path("data") / f"{tracking_name}.json"
        self._tracking_data: Dict = self._load_tracking_data()

        # Log initial startup (will log more details after resolving trading pair)
        self.log_with_clock(logging.INFO,
                            f"LP Position Manager initializing for pool {self.pool_address[:8]}... on {self.exchange}")

        # Initialize position on startup (will resolve trading pair first)
        safe_ensure_future(self.initialize_position())

    async def resolve_trading_pair_from_pool(self):
        """Resolve trading pair from pool address using connector method"""
        if self._trading_pair_resolved:
            return

        try:
            connector = self.connectors[self.exchange]

            # Use connector method to resolve trading pair info
            pair_info = await connector.resolve_trading_pair_from_pool(self.pool_address)

            if not pair_info:
                raise ValueError(f"Could not resolve trading pair for pool address {self.pool_address}")

            # Extract resolved information
            self.trading_pair = pair_info["trading_pair"]
            self.base_token = pair_info["base_token"]
            self.quote_token = pair_info["quote_token"]
            self.base_token_address = pair_info["base_token_address"]
            self.quote_token_address = pair_info["quote_token_address"]
            self._trading_pair_resolved = True

            self.logger().info(f"Resolved trading pair from pool: {self.trading_pair}")

            # Initialize rate sources now that we have the trading pair
            self.market_data_provider.initialize_rate_sources([
                ConnectorPair(connector_name=self.exchange, trading_pair=self.trading_pair)
            ])

            # Validate price limits
            self._validate_price_limits()

            # Log configuration details
            config_msg = (
                f"LP Position Manager configured:\n"
                f"  Trading Pair: {self.trading_pair}\n"
                f"  Pool Address: {self.pool_address}\n"
                f"  Position width: {float(self.config.position_width_pct):.2f}% (position_width_pct)\n"
                f"  Rebalance threshold: {self.config.rebalance_seconds} seconds out-of-bounds\n"
                f"  Check interval: {self.config.check_seconds} seconds"
            )

            if self.config.lower_price_limit or self.config.upper_price_limit:
                config_msg += "\n  Price Limits:"
                if self.config.lower_price_limit:
                    config_msg += f"\n    Lower: {float(self.config.lower_price_limit):.6f}"
                if self.config.upper_price_limit:
                    config_msg += f"\n    Upper: {float(self.config.upper_price_limit):.6f}"

            self.logger().info(config_msg)

            if self.config.base_amount > 0 or self.config.quote_amount > 0:
                self.logger().info(
                    f"Initial amounts: {self.config.base_amount} {self.base_token} / "
                    f"{self.config.quote_amount} {self.quote_token}"
                )

        except Exception as e:
            self.logger().error(f"Error resolving trading pair from pool: {str(e)}", exc_info=True)
            raise

    async def initialize_position(self):
        """
        Check for existing positions or create initial position on startup.
        Orchestrates the startup flow.
        """
        await asyncio.sleep(self.POSITION_CREATION_DELAY)  # Wait for connector to initialize

        # Resolve pool and trading pair information
        if not await self._resolve_pool_info():
            return

        # Initialize SOL balance tracking now that connector is ready
        self._initialize_sol_balance_if_needed()

        # Check for existing position
        has_existing = await self.check_existing_positions()

        if has_existing:
            await self._handle_existing_position()
        else:
            await self._handle_no_existing_position()

    async def _resolve_pool_info(self) -> bool:
        """Resolve trading pair and fetch pool info. Returns True if successful."""
        # Resolve trading pair from pool address
        await self.resolve_trading_pair_from_pool()

        # Fetch pool info to get current price
        await self.fetch_pool_info()

        if not self.pool_info:
            self.logger().error(f"Pool not found for {self.trading_pair}. Please add pool via 'gateway pool' command first")
            return False

        return True

    async def _handle_existing_position(self):
        """Handle logic when an existing position is found."""
        self.logger().info(f"Found existing position {self.current_position_id}, will monitor it")

        # Check if the existing position is already out of range
        await self.check_if_position_out_of_range_on_startup()

    async def _handle_no_existing_position(self):
        """Handle logic when no existing position is found."""
        if self.config.base_amount > 0 or self.config.quote_amount > 0:
            self.logger().info("No existing position found, creating initial position...")
            await self.create_initial_position()
        else:
            self.logger().info("No existing position and no initial amounts provided - monitoring only")

    def _load_tracking_data(self) -> Dict:
        """Load P&L tracking data from JSON file"""
        if self._tracking_file_path.exists():
            try:
                with open(self._tracking_file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger().warning(f"Failed to load tracking data: {e}, starting fresh")

        # Initialize new tracking data (SOL balance will be set later when connector is ready)
        return {
            "config_file": self.config.config_file_name,
            "pool_address": self.pool_address,
            "connector": self.exchange,
            "initial_sol_balance": 0.0,
            "tracking_started": int(time.time()),
            "positions": []
        }

    def _initialize_sol_balance_if_needed(self):
        """Initialize SOL balance if not already set (called when connector is ready)"""
        if self._tracking_data.get("initial_sol_balance", 0.0) == 0.0:
            connector = self.connectors[self.config.connector]
            initial_sol = connector.get_available_balance("SOL")
            if initial_sol > 0:
                self._tracking_data["initial_sol_balance"] = float(initial_sol)
                self._save_tracking_data()
                self.logger().info(f"Initialized SOL balance tracking: {initial_sol:.4f} SOL")

    def _validate_price_limits(self):
        """Validate price limits configuration"""
        # Normalize 0 to None (treat 0 as "no limit")
        if self.config.lower_price_limit is not None and self.config.lower_price_limit == 0:
            self.config.lower_price_limit = None
        if self.config.upper_price_limit is not None and self.config.upper_price_limit == 0:
            self.config.upper_price_limit = None

        if self.config.lower_price_limit and self.config.upper_price_limit:
            if self.config.lower_price_limit >= self.config.upper_price_limit:
                raise ValueError(
                    f"Invalid price limits: lower ({float(self.config.lower_price_limit):.6f}) "
                    f"must be < upper ({float(self.config.upper_price_limit):.6f})"
                )

    def _is_price_within_limits(self, price: Decimal) -> bool:
        """Check if price is within configured price limits"""
        if self.config.lower_price_limit and price < self.config.lower_price_limit:
            return False
        if self.config.upper_price_limit and price > self.config.upper_price_limit:
            return False
        return True

    def _apply_price_limit_constraints(self, lower_bound: Decimal, upper_bound: Decimal) -> Tuple[Decimal, Decimal]:
        """Apply price limits to position bounds, returns constrained bounds"""
        if self.config.lower_price_limit:
            lower_bound = max(lower_bound, self.config.lower_price_limit)
        if self.config.upper_price_limit:
            upper_bound = min(upper_bound, self.config.upper_price_limit)

        # Validate result
        if lower_bound >= upper_bound:
            self.logger().warning(
                f"Price limit constraints resulted in invalid bounds: "
                f"lower={float(lower_bound):.6f} >= upper={float(upper_bound):.6f}"
            )

        return lower_bound, upper_bound

    def _save_tracking_data(self):
        """Save P&L tracking data to JSON file"""
        try:
            self._tracking_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._tracking_file_path, 'w') as f:
                json.dump(self._tracking_data, f, indent=2)
        except Exception as e:
            self.logger().error(f"Failed to save tracking data: {e}", exc_info=True)

    def _record_position_event(self, event_type: str, position_info: Optional[CLMMPositionInfo] = None):
        """Record position open/close event for P&L tracking"""
        if not position_info and event_type == "open":
            self.logger().warning("Cannot record open event without position info")
            return

        if not self.pool_info:
            self.logger().warning("Cannot record position event without pool info")
            return

        try:
            event_data = {
                "type": event_type,
                "timestamp": int(time.time()),
                "position_address": self.current_position_id if self.current_position_id else "unknown",
                "trading_pair": self.trading_pair,
                "lower_price": float(position_info.lower_price) if position_info else 0,
                "upper_price": float(position_info.upper_price) if position_info else 0,
                "mid_price": float(self.pool_info.price),
            }

            if event_type == "open" and position_info:
                base_amount = Decimal(str(position_info.base_token_amount))
                quote_amount = Decimal(str(position_info.quote_token_amount))
                price = Decimal(str(self.pool_info.price))

                event_data.update({
                    "base_amount": float(base_amount),
                    "quote_amount": float(quote_amount),
                    "value_in_quote": float(base_amount * price + quote_amount)
                })
            elif event_type == "close" and position_info:
                base_amount = Decimal(str(position_info.base_token_amount))
                quote_amount = Decimal(str(position_info.quote_token_amount))
                base_fees = Decimal(str(position_info.base_fee_amount))
                quote_fees = Decimal(str(position_info.quote_fee_amount))
                price = Decimal(str(self.pool_info.price))

                event_data.update({
                    "base_amount": float(base_amount),
                    "quote_amount": float(quote_amount),
                    "base_fees": float(base_fees),
                    "quote_fees": float(quote_fees),
                    "value_in_quote": float(base_amount * price + quote_amount + base_fees * price + quote_fees)
                })

            self._tracking_data["positions"].append(event_data)
            self._save_tracking_data()
            self.logger().info(f"Recorded {event_type} event for position {event_data['position_address'][:8]}...")

        except Exception as e:
            self.logger().error(f"Error recording position event: {e}", exc_info=True)

    def on_tick(self):
        """Called on each strategy tick"""
        # Throttle checks based on check_seconds (for rate limiting)
        current_time = self.current_timestamp
        if current_time - self._last_check_time < self.config.check_seconds:
            return

        self._last_check_time = current_time

        if self.pending_operation:
            # Operation in progress, connector handles timeout
            return

        if self.current_position_id:
            # Monitor existing position
            safe_ensure_future(self.monitor_and_rebalance())
        else:
            # No position yet, just update pool info
            safe_ensure_future(self.fetch_pool_info())

    async def fetch_pool_info(self):
        """
        Fetch pool information to get current price.
        Uses throttling to avoid excessive API calls.
        """
        try:
            # Throttle pool info fetches (use check_seconds)
            current_time = time.time()
            if current_time - self._last_pool_info_fetch_time < self.config.check_seconds:
                return self.pool_info  # Return cached

            # Wait for connector to be ready
            if self.exchange not in self.connectors:
                return None

            connector = self.connectors[self.exchange]

            # Use connector method to fetch pool info by address
            self.pool_info = await connector.get_pool_info_by_address(self.pool_address)

            if self.pool_info:
                self._last_pool_info_fetch_time = current_time
                return self.pool_info
            else:
                self.logger().error(f"Pool info not found for pool address {self.pool_address}")
                return None

        except Exception as e:
            self.logger().error(f"Error fetching pool info: {str(e)}", exc_info=True)
            return None

    async def check_existing_positions(self) -> bool:
        """Check if user has existing positions in this pool"""
        try:
            # Wait for connector to be ready
            if self.exchange not in self.connectors:
                return False

            connector = self.connectors[self.exchange]

            # Fetch positions for this pool (we already have pool_address)
            positions = await connector.get_user_positions(pool_address=self.pool_address)

            if positions and len(positions) > 0:
                # Use the first position found
                self.position_info = positions[0]
                self.current_position_id = self.position_info.address
                self.logger().info(f"Found existing position: {self.current_position_id}")
                return True

            return False
        except Exception:
            return False

    async def check_if_position_out_of_range_on_startup(self):
        """Check if existing position is already out of range on startup and begin rebalance countdown"""
        if not self.position_info or not self.pool_info:
            return

        try:
            current_price = Decimal(str(self.pool_info.price))
            lower_price = Decimal(str(self.position_info.lower_price))
            upper_price = Decimal(str(self.position_info.upper_price))

            # Check if price is in bounds
            in_bounds = self._price_in_bounds(current_price, lower_price, upper_price)

            if not in_bounds:
                # Position is already out of range - start the rebalance countdown
                # Use time.time() for consistency with monitor_and_rebalance
                self.out_of_bounds_since = time.time()

                # Calculate deviation using helper
                deviation, direction, bound = self._calculate_price_deviation(current_price, lower_price, upper_price)

                # Add arrow indicators (emoji for start, text for direction)
                arrow = "⬇️" if direction == "below" else "⬆️"
                text_arrow = "↓" if direction == "below" else "↑"

                msg = (f"{arrow} Out of range: {float(current_price):.6f} {text_arrow} {direction} "
                       f"{float(bound):.6f} ({deviation:.2f}%). Rebalance in {self.config.rebalance_seconds}s")
                self.logger().warning(msg)
                self.notify_hb_app_with_timestamp(msg)

                # Schedule immediate rebalance check (handles rebalance_seconds=0 case)
                safe_ensure_future(self.monitor_and_rebalance())
            else:
                self.logger().info(f"Position is in range at startup (price: {float(current_price):.6f}, "
                                   f"range: {float(lower_price):.6f} - {float(upper_price):.6f})")

        except Exception as e:
            self.logger().error(f"Error checking if position out of range on startup: {str(e)}")

    async def update_position_info(self):
        """Fetch the latest position information"""
        if not self.current_position_id or not self.trading_pair:
            return

        try:
            self.position_info = await self.connectors[self.exchange].get_position_info(
                trading_pair=self.trading_pair,
                position_address=self.current_position_id
            )

            if self.position_info:
                self.logger().info(
                    f"{self.exchange} {self.trading_pair} position: {self.current_position_id[:8]}... "
                    f"(price: {self.position_info.price:.2f}, "
                    f"range: {self.position_info.lower_price:.2f}-{self.position_info.upper_price:.2f})"
                )

        except Exception as e:
            self.logger().error(f"Error updating position info: {str(e)}")

    async def create_initial_position(self):
        """Create initial position (can be double-sided or single-sided)"""
        if self.pending_operation or not self.trading_pair:
            return

        try:
            if not self.pool_info:
                await self.fetch_pool_info()

            if not self.pool_info:
                self.logger().error("Cannot create position without pool info")
                return

            current_price = float(self.pool_info.price)
            base_amt = float(self.config.base_amount)
            quote_amt = float(self.config.quote_amount)

            # Validate balances and amounts
            if not await self._validate_position_amounts(base_amt, quote_amt):
                return

            # Compute width percentages based on position type
            lower_pct, upper_pct = self._compute_width_percentages(base_amt, quote_amt)

            # Calculate actual price bounds
            lower_price = current_price * (1 - lower_pct / 100)
            upper_price = current_price * (1 + upper_pct / 100)

            # Apply price limit constraints
            lower_price_decimal = Decimal(str(lower_price))
            upper_price_decimal = Decimal(str(upper_price))
            lower_price_decimal, upper_price_decimal = self._apply_price_limit_constraints(lower_price_decimal, upper_price_decimal)

            # Validate bounds after constraints
            if lower_price_decimal >= upper_price_decimal:
                self.logger().error("Price limit constraints resulted in invalid bounds, cannot create position")
                return

            lower_price = float(lower_price_decimal)
            upper_price = float(upper_price_decimal)

            # Log position type and range
            if base_amt > 0 and quote_amt > 0:
                self.logger().info(
                    f"Creating double-sided position at price {current_price:.6f} "
                    f"with range -{lower_pct:.2f}% to +{upper_pct:.2f}%"
                )
            elif base_amt > 0:
                self.logger().info(
                    f"Creating base-only position with {base_amt} {self.base_token} at price {current_price:.6f} "
                    f"(range: +{upper_pct:.2f}% above price)"
                )
            elif quote_amt > 0:
                self.logger().info(
                    f"Creating quote-only position with {quote_amt} {self.quote_token} at price {current_price:.6f} "
                    f"(range: -{lower_pct:.2f}% below price)"
                )
            else:
                return

            self.logger().info(
                f"Submitting position with price range: {lower_price:.6f} - {upper_price:.6f}"
            )
            self.logger().info(f"Using pool address: {self.pool_address}")

            # Store amounts being added for notification
            self._opening_position_amounts = {
                "base_amount": base_amt,
                "quote_amount": quote_amt
            }

            # Build connector-specific parameters
            extra_params = self._build_extra_params()

            order_id = self.connectors[self.exchange].add_liquidity(
                trading_pair=self.trading_pair,
                price=current_price,
                upper_width_pct=upper_pct,
                lower_width_pct=lower_pct,
                base_token_amount=base_amt,
                quote_token_amount=quote_amt,
                pool_address=self.pool_address,
                extra_params=extra_params,
            )

            self.pending_open_order_id = order_id
            self.pending_operation = "opening"
            self.logger().info(f"Initial position order submitted with ID: {order_id}")

        except Exception as e:
            self.logger().error(f"Error creating initial position: {str(e)}")
            self.pending_operation = None

    async def monitor_and_rebalance(self):
        """Monitor position and rebalance if needed"""
        if not self.current_position_id:
            return

        try:
            # Throttle position info updates to reduce unnecessary fetches
            current_time = time.time()
            if current_time - self._last_position_update_time >= self.POSITION_UPDATE_INTERVAL:
                await self.update_position_info()
                self._last_position_update_time = current_time

            # Always fetch pool info for current price
            await self.fetch_pool_info()

            if not self.pool_info or not self.position_info:
                return

            current_price = Decimal(str(self.pool_info.price))
            lower_price = Decimal(str(self.position_info.lower_price))
            upper_price = Decimal(str(self.position_info.upper_price))

            # Check if price is in bounds
            in_bounds = self._price_in_bounds(current_price, lower_price, upper_price)

            if in_bounds:
                # Price is in bounds
                if self.out_of_bounds_since is not None:
                    self.logger().info("Price moved back into position bounds, resetting timer")
                    # Notify user that position is back in range
                    msg = f"✅ {self.trading_pair} position back in range on {self.exchange}. Rebalance timer reset."
                    self.notify_hb_app_with_timestamp(msg)
                    self.out_of_bounds_since = None
            else:
                # Price is out of bounds
                current_time = time.time()

                if self.out_of_bounds_since is None:
                    self.out_of_bounds_since = current_time

                    # Calculate deviation using helper
                    deviation, direction, bound = self._calculate_price_deviation(current_price, lower_price, upper_price)
                    self.logger().info(f"Price {current_price:.6f} moved {direction} bound {bound:.6f} by {deviation:.2f}%")

                    # Add arrow indicators (emoji for start, text for direction)
                    arrow = "⬇️" if direction == "below" else "⬆️"
                    text_arrow = "↓" if direction == "below" else "↑"

                    # Notify user that position is out of range
                    msg = (f"{arrow} {self.trading_pair} out of range: {float(current_price):.6f} {text_arrow} "
                           f"{direction} {float(bound):.6f} ({deviation:.2f}%). Rebalance in {self.config.rebalance_seconds}s")
                    self.notify_hb_app_with_timestamp(msg)

                elapsed_seconds = current_time - self.out_of_bounds_since

                if elapsed_seconds >= self.config.rebalance_seconds:
                    self.logger().info(f"Price out of bounds for {elapsed_seconds:.0f} seconds (threshold: {self.config.rebalance_seconds})")

                    # Check if price is within limits before rebalancing
                    if not self._is_price_within_limits(current_price):
                        # Price outside limits - skip rebalance, keep position open
                        limit_msg = ""
                        if self.config.lower_price_limit and current_price < self.config.lower_price_limit:
                            limit_msg = f"below lower limit {float(self.config.lower_price_limit):.6f}"
                        elif self.config.upper_price_limit and current_price > self.config.upper_price_limit:
                            limit_msg = f"above upper limit {float(self.config.upper_price_limit):.6f}"

                        self.logger().info(f"Price {limit_msg}, skipping rebalance")
                    else:
                        # Normal rebalance
                        await self.rebalance_position(current_price, lower_price, upper_price)
                else:
                    self.logger().info(f"Price out of bounds for {elapsed_seconds:.0f}/{self.config.rebalance_seconds} seconds")

        except Exception as e:
            self.logger().error(f"Error in monitor_and_rebalance: {str(e)}")

    async def rebalance_position(self, current_price: Decimal, old_lower: Decimal, old_upper: Decimal):
        """Close current position and prepare to open new single-sided position"""
        if self.pending_operation or not self.trading_pair:
            return

        try:
            self.logger().info("Starting rebalance: closing current position...")

            # Calculate deviation using helper
            deviation, direction, _ = self._calculate_price_deviation(current_price, old_lower, old_upper)

            # Add arrow indicators
            text_arrow = "↓" if direction == "below" else "↑"

            # Notify user about closing position
            msg = (f"❌ Closing {self.trading_pair}: {float(current_price):.6f} {text_arrow} {direction} "
                   f"[{float(old_lower):.6f}-{float(old_upper):.6f}] ({deviation:.2f}%)")
            self.notify_hb_app_with_timestamp(msg)

            # Store amounts being removed from position (including fees)
            base_in_position = self.position_info.base_token_amount if self.position_info else 0.0
            quote_in_position = self.position_info.quote_token_amount if self.position_info else 0.0
            base_fees = self.position_info.base_fee_amount if self.position_info else 0.0
            quote_fees = self.position_info.quote_fee_amount if self.position_info else 0.0

            # Total amounts that will be returned to wallet
            total_base_removed = base_in_position + base_fees
            total_quote_removed = quote_in_position + quote_fees

            self._closed_position_balances = {
                "base_amount": base_in_position,
                "quote_amount": quote_in_position,
                "base_fee": base_fees,
                "quote_fee": quote_fees,
                "total_base_removed": total_base_removed,
                "total_quote_removed": total_quote_removed,
                "current_price": current_price,
                "old_lower": old_lower,
                "old_upper": old_upper,
            }

            # Close the current position
            order_id = self.connectors[self.exchange].remove_liquidity(
                trading_pair=self.trading_pair,
                position_address=self.current_position_id
            )

            self.pending_close_order_id = order_id
            self.pending_operation = "closing"
            self.logger().info(f"Position close order submitted with ID: {order_id}")

        except Exception as e:
            self.logger().error(f"Error starting rebalance: {str(e)}")
            self.pending_operation = None

    async def close_position_only(self):
        """Close position without rebalancing (used when price is outside limits)"""
        if self.pending_operation or not self.trading_pair or not self.current_position_id:
            return

        try:
            self.logger().info("Closing position without rebalancing (price outside limits)")

            # Don't set _closed_position_balances since we won't rebalance
            # Close the current position
            order_id = self.connectors[self.exchange].remove_liquidity(
                trading_pair=self.trading_pair,
                position_address=self.current_position_id
            )

            self.pending_close_order_id = order_id
            self.pending_operation = "closing"
            self.logger().info(f"Position close order submitted with ID: {order_id}")

        except Exception as e:
            self.logger().error(f"Error closing position: {str(e)}")
            self.pending_operation = None

    async def open_rebalanced_position(self):
        """Open new single-sided position after closing old one"""
        self.logger().info("open_rebalanced_position called")
        try:
            if not self._closed_position_balances:
                self.logger().error("No closed position balance info available")
                return

            self.logger().info(f"Closed position balances: {self._closed_position_balances}")

            if not self.trading_pair:
                self.logger().error("Trading pair not set")
                return

            info = self._closed_position_balances
            current_price = info["current_price"]
            old_lower = info["old_lower"]
            old_upper = info["old_upper"]

            # Use only the amounts removed from the closed position
            total_base_removed = info["total_base_removed"]
            total_quote_removed = info["total_quote_removed"]

            self.logger().info(
                f"Tokens received from closed position: {total_base_removed:.6f} {self.base_token}, "
                f"{total_quote_removed:.6f} {self.quote_token}"
            )

            # Determine which side to enter based on where price is relative to old range
            side = self._determine_side(current_price, old_lower, old_upper)

            # Get current pool info for latest price
            await self.fetch_pool_info()
            if not self.pool_info:
                self.logger().error("Cannot open rebalanced position without pool info")
                return

            new_mid_price = float(self.pool_info.price)

            # For single-sided position, use only amounts from closed position
            if side == "base":
                # Price is below range, provide base token only
                base_amt = total_base_removed
                quote_amt = 0.0
            else:  # quote side
                # Price is above bounds, provide quote token only
                base_amt = 0.0
                quote_amt = total_quote_removed

            if base_amt == 0 and quote_amt == 0:
                self.logger().error(
                    f"Insufficient balance to open rebalanced position! "
                    f"Received: {total_base_removed:.6f} {self.base_token}, {total_quote_removed:.6f} {self.quote_token}"
                )
                self.pending_operation = None
                self._closed_position_balances = None
                return

            # Check for dust amounts
            min_amount = float(self.MIN_TOKEN_AMOUNT)
            if base_amt > 0 and base_amt < min_amount:
                self.logger().error(
                    f"Base amount {base_amt} {self.base_token} is too small (minimum {min_amount}). "
                    f"Cannot open position."
                )
                self.pending_operation = None
                self._closed_position_balances = None
                return

            if quote_amt > 0 and quote_amt < min_amount:
                self.logger().error(
                    f"Quote amount {quote_amt} {self.quote_token} is too small (minimum {min_amount}). "
                    f"Cannot open position."
                )
                self.pending_operation = None
                self._closed_position_balances = None
                return

            # Compute width percentages based on position type
            lower_pct, upper_pct = self._compute_width_percentages(base_amt, quote_amt)

            # Calculate actual price bounds
            lower_price = new_mid_price * (1 - lower_pct / 100)
            upper_price = new_mid_price * (1 + upper_pct / 100)

            # Apply price limit constraints
            lower_price_decimal = Decimal(str(lower_price))
            upper_price_decimal = Decimal(str(upper_price))
            lower_price_decimal, upper_price_decimal = self._apply_price_limit_constraints(lower_price_decimal, upper_price_decimal)

            # Validate bounds after constraints
            if lower_price_decimal >= upper_price_decimal:
                self.logger().error("Price limit constraints resulted in invalid bounds, cannot rebalance position")
                self.pending_operation = None
                self._closed_position_balances = None
                return

            lower_price = float(lower_price_decimal)
            upper_price = float(upper_price_decimal)

            # Log rebalanced position details
            if side == "base":
                self.logger().info(
                    f"Opening base-only position with {base_amt} {self.base_token} at price {new_mid_price:.6f} "
                    f"(range: +{upper_pct:.2f}% above price, {lower_price:.6f} - {upper_price:.6f})"
                )
            else:
                self.logger().info(
                    f"Opening quote-only position with {quote_amt} {self.quote_token} at price {new_mid_price:.6f} "
                    f"(range: -{lower_pct:.2f}% below price, {lower_price:.6f} - {upper_price:.6f})"
                )

            self.logger().info(f"Using pool address: {self.pool_address}")

            # Store amounts being added for notification
            self._opening_position_amounts = {
                "base_amount": base_amt,
                "quote_amount": quote_amt
            }

            # Notify user about opening new position
            token_amt = f"{base_amt:.4f} {self.base_token}" if side == "base" else f"{quote_amt:.4f} {self.quote_token}"
            msg = f"🟢 Opening {self.trading_pair}: {token_amt} @ [{lower_price:.6f}-{upper_price:.6f}]"
            self.notify_hb_app_with_timestamp(msg)

            # Build connector-specific parameters
            extra_params = self._build_extra_params()

            order_id = self.connectors[self.exchange].add_liquidity(
                trading_pair=self.trading_pair,
                price=new_mid_price,
                upper_width_pct=upper_pct,
                lower_width_pct=lower_pct,
                base_token_amount=base_amt,
                quote_token_amount=quote_amt,
                pool_address=self.pool_address,
                extra_params=extra_params,
            )

            self.pending_open_order_id = order_id
            self.pending_operation = "opening"
            self.has_rebalanced_once = True
            self.logger().info(f"Rebalanced {side}-only position order submitted with ID: {order_id}")

            # NOTE: Don't clear _closed_position_balances here - it's needed for retry on timeout
            # It will be cleared in did_add_liquidity() after successful confirmation

        except Exception as e:
            self.logger().error(f"Error opening rebalanced position: {str(e)}")
            self.pending_operation = None
            self._closed_position_balances = None  # Clear on error since we won't retry

    def _compute_width_percentages(self, base_amt: float = 0, quote_amt: float = 0):
        """
        Compute upper and lower width percentages from total position width.

        For double-sided positions: split width evenly (±half)
        For base-only positions: full width ABOVE price (sell base for quote)
        For quote-only positions: full width BELOW price (buy base with quote)
        """
        total_width = float(self.config.position_width_pct)

        if base_amt > 0 and quote_amt > 0:
            # Double-sided: split width evenly
            half_width = total_width / 2.0
            return half_width, half_width
        elif base_amt > 0:
            # Base-only: full width ABOVE current price
            return 0, total_width
        elif quote_amt > 0:
            # Quote-only: full width BELOW current price
            return total_width, 0
        else:
            # Default (shouldn't happen)
            half_width = total_width / 2.0
            return half_width, half_width

    @staticmethod
    def _price_in_bounds(price: Decimal, lower: Decimal, upper: Decimal) -> bool:
        """Check if price is within position bounds"""
        return lower <= price <= upper

    @staticmethod
    def _determine_side(price: Decimal, lower: Decimal, upper: Decimal) -> str:
        """Determine which side to provide liquidity based on price position"""
        if price < lower:
            return "base"
        elif price > upper:
            return "quote"
        else:
            # Should not happen when rebalancing, but default to base
            return "base"

    @staticmethod
    def _calculate_price_deviation(current_price: Decimal, lower_price: Decimal, upper_price: Decimal) -> Tuple[float, str, Decimal]:
        """
        Calculate price deviation from bounds.

        :return: Tuple of (deviation_pct, direction, bound_price)
                 where direction is "below" or "above"
        """
        if float(current_price) < float(lower_price):
            deviation = abs((float(current_price) - float(lower_price)) / float(lower_price) * 100)
            direction = "below"
            bound = lower_price
        else:
            deviation = abs((float(current_price) - float(upper_price)) / float(upper_price) * 100)
            direction = "above"
            bound = upper_price

        return deviation, direction, bound

    def _build_extra_params(self) -> Optional[Dict]:
        """
        Build connector-specific extra parameters.
        Currently supports Meteora's strategyType parameter.

        :return: Dictionary of extra parameters or None
        """
        # Only add extra_params for Meteora connector
        if "meteora" in self.config.connector.lower() and self.config.strategy_type is not None:
            return {"strategyType": self.config.strategy_type}
        return None

    async def _validate_position_amounts(self, base_amt: float, quote_amt: float) -> bool:
        """
        Validate token amounts are sufficient and above dust limits.

        :return: True if valid, False otherwise
        """
        connector = self.connectors[self.exchange]

        # Fetch balances
        base_balance = float(await connector.get_balance_by_address(self.base_token_address))
        quote_balance = float(await connector.get_balance_by_address(self.quote_token_address))

        self.logger().info(f"Available balances: {base_balance} {self.base_token}, {quote_balance} {self.quote_token}")

        # Check sufficient balances
        if base_amt > 0 and base_balance < base_amt:
            self.logger().error(
                f"Insufficient {self.base_token} balance! Required: {base_amt}, Available: {base_balance}"
            )
            return False

        if quote_amt > 0 and quote_balance < quote_amt:
            self.logger().error(
                f"Insufficient {self.quote_token} balance! Required: {quote_amt}, Available: {quote_balance}"
            )
            return False

        # Check for dust amounts
        min_amount = float(self.MIN_TOKEN_AMOUNT)
        if base_amt > 0 and base_amt < min_amount:
            self.logger().error(
                f"Base amount {base_amt} {self.base_token} is too small (minimum {min_amount}). "
                f"Increase base_amount in config."
            )
            return False

        if quote_amt > 0 and quote_amt < min_amount:
            self.logger().error(
                f"Quote amount {quote_amt} {self.quote_token} is too small (minimum {min_amount}). "
                f"Increase quote_amount in config."
            )
            return False

        return True

    async def fetch_position_info_after_add(self):
        """Fetch position info after position is created"""
        try:
            await asyncio.sleep(self.POSITION_CREATION_DELAY)  # Wait for position to be created on-chain

            connector = self.connectors[self.exchange]

            # Fetch positions for this pool (pool_address is from config)
            positions = await connector.get_user_positions(pool_address=self.pool_address)

            if positions:
                # Get the most recent position
                self.position_info = positions[-1]
                self.current_position_id = self.position_info.address
                self.logger().info(f"Position info fetched: {self.current_position_id}")

                # Log actual price range from the created position
                self.logger().info(
                    f"Actual position price range: {float(self.position_info.lower_price):.6f} - "
                    f"{float(self.position_info.upper_price):.6f}"
                )

                # Record position open event for P&L tracking
                self._record_position_event("open", self.position_info)
        except Exception as e:
            self.logger().error(f"Error fetching position info after add: {str(e)}")

    # Event handlers for LP operations - These are now properly triggered by gateway_lp.py
    def did_add_liquidity(self, event: RangePositionLiquidityAddedEvent):
        """Called when liquidity is added to a position"""
        if hasattr(event, 'order_id') and event.order_id == self.pending_open_order_id:
            self.logger().info(f"Position opening order {event.order_id} confirmed!")

            # Fetch the new position info
            safe_ensure_future(self.fetch_position_info_after_add())

            # Clear pending state
            self.pending_open_order_id = None
            self.pending_operation = None
            self.out_of_bounds_since = None
            self._closed_position_balances = None  # Clear rebalance info now that open succeeded

            # Build notification message with amounts added
            if self._opening_position_amounts:
                base_amt = self._opening_position_amounts["base_amount"]
                quote_amt = self._opening_position_amounts["quote_amount"]
                msg = (f"✓ Opened {self.trading_pair}: "
                       f"Added {base_amt:.6f} {self.base_token} + {quote_amt:.6f} {self.quote_token}")
                self._opening_position_amounts = None  # Clear after use
            else:
                msg = f"✓ Opened {self.trading_pair}"

            self.notify_hb_app_with_timestamp(msg)

    def did_remove_liquidity(self, event: RangePositionLiquidityRemovedEvent):
        """Called when liquidity is removed from a position"""
        if hasattr(event, 'order_id') and event.order_id == self.pending_close_order_id:
            self.logger().info(f"Position closing order {event.order_id} confirmed!")

            # Record position close event for P&L tracking (before clearing position_info)
            if self.position_info:
                self._record_position_event("close", self.position_info)

            # Clear current position
            self.current_position_id = None
            self.position_info = None
            self.pending_close_order_id = None
            self.pending_operation = None
            self.out_of_bounds_since = None

            # Build notification message with amounts removed
            if self._closed_position_balances:
                total_base = self._closed_position_balances["total_base_removed"]
                total_quote = self._closed_position_balances["total_quote_removed"]
                msg = (f"✓ Closed {self.trading_pair}: "
                       f"Removed {total_base:.6f} {self.base_token} + {total_quote:.6f} {self.quote_token}")
            else:
                msg = f"✓ Closed {self.trading_pair}"

            self.notify_hb_app_with_timestamp(msg)

            # If this was a rebalance, open the new position
            if self._closed_position_balances:
                self.logger().info("Position closed, opening rebalanced position...")
                safe_ensure_future(self.open_rebalanced_position())
            else:
                self.logger().warning("Position closed but no rebalance info available!")

    def did_fail_lp_update(self, event: RangePositionUpdateFailureEvent):
        """Called when an LP operation fails due to transaction timeout - retry the operation"""
        # Check if this failure is for our pending operation
        is_open_failure = hasattr(event, 'order_id') and event.order_id == self.pending_open_order_id
        is_close_failure = hasattr(event, 'order_id') and event.order_id == self.pending_close_order_id

        if not is_open_failure and not is_close_failure:
            return  # Not our order

        operation_type = "open" if event.order_action == LPType.ADD else "close"
        self.logger().warning(
            f"Transaction timeout for {self.trading_pair} {operation_type} (order: {event.order_id}). "
            f"Chain may be congested. Will retry..."
        )

        # Notify user about timeout and retry
        msg = f"⏱️ {self.trading_pair} {operation_type} timed out (chain congestion), retrying..."
        self.notify_hb_app_with_timestamp(msg)

        if is_open_failure:
            # Clear pending state for open
            self.pending_open_order_id = None
            self.pending_operation = None

            # Retry opening position
            if self._closed_position_balances:
                # This was a rebalance open - retry with saved balances
                self.logger().info(f"Retrying rebalanced position open for {self.trading_pair}...")
                self.notify_hb_app_with_timestamp(f"🔄 Retrying {self.trading_pair} position open...")
                safe_ensure_future(self.open_rebalanced_position())
            elif self._opening_position_amounts:
                # This was an initial position open - retry
                self.logger().info(f"Retrying initial position open for {self.trading_pair}...")
                self.notify_hb_app_with_timestamp(f"🔄 Retrying {self.trading_pair} position open...")
                safe_ensure_future(self.create_initial_position())

        elif is_close_failure:
            # Clear pending state for close
            self.pending_close_order_id = None
            self.pending_operation = None

            # Retry closing position
            if self.current_position_id:
                self.logger().info(f"Retrying position close for {self.trading_pair}...")
                self.notify_hb_app_with_timestamp(f"🔄 Retrying {self.trading_pair} position close...")
                safe_ensure_future(self.retry_close_position())

    async def retry_close_position(self):
        """Retry closing the current position"""
        if self.pending_operation or not self.trading_pair or not self.current_position_id:
            return

        try:
            self.logger().info(f"Retrying close for position {self.current_position_id}")

            order_id = self.connectors[self.exchange].remove_liquidity(
                trading_pair=self.trading_pair,
                position_address=self.current_position_id
            )

            self.pending_close_order_id = order_id
            self.pending_operation = "closing"
            self.logger().info(f"Position close retry submitted with ID: {order_id}")

        except Exception as e:
            self.logger().error(f"Error retrying position close: {str(e)}")
            self.pending_operation = None

    def _calculate_pnl_summary(self) -> Dict:
        """Calculate P&L summary from position history"""
        positions = self._tracking_data.get("positions", [])

        opens = [p for p in positions if p["type"] == "open"]
        closes = [p for p in positions if p["type"] == "close"]

        # Calculate totals for opens
        total_open_base = sum(p.get("base_amount", 0) for p in opens)
        total_open_quote = sum(p.get("quote_amount", 0) for p in opens)
        # Base value = base_amount * mid_price at open time
        total_open_base_value = sum(p.get("base_amount", 0) * p.get("mid_price", 0) for p in opens)
        total_open_value = total_open_base_value + total_open_quote

        # Calculate totals for closes
        total_close_base = sum(p.get("base_amount", 0) for p in closes)
        total_close_quote = sum(p.get("quote_amount", 0) for p in closes)
        # Base value = base_amount * mid_price at close time
        total_close_base_value = sum(p.get("base_amount", 0) * p.get("mid_price", 0) for p in closes)
        total_close_value = total_close_base_value + total_close_quote

        # Calculate total fees collected (in quote)
        total_fees_base = sum(p.get("base_fees", 0) for p in closes)
        total_fees_quote = sum(p.get("quote_fees", 0) for p in closes)
        # Convert base fees to quote using mid_price at close time
        total_fees_base_value = sum(p.get("base_fees", 0) * p.get("mid_price", 0) for p in closes)
        total_fees_value = total_fees_base_value + total_fees_quote

        # Calculate current position value (only if script created the initial position)
        # If first record is "open", script created initial position - include current position value
        # If first record is "close", script inherited existing position - don't include current position value
        current_position_value = Decimal("0")
        current_position_base = Decimal("0")
        current_position_quote = Decimal("0")
        current_position_base_fees = Decimal("0")
        current_position_quote_fees = Decimal("0")
        first_event_is_open = positions and positions[0].get("type") == "open"

        if first_event_is_open and self.current_position_id and self.position_info and self.pool_info:
            current_price = Decimal(str(self.pool_info.price))
            current_position_base = Decimal(str(self.position_info.base_token_amount))
            current_position_quote = Decimal(str(self.position_info.quote_token_amount))
            current_position_base_fees = Decimal(str(self.position_info.base_fee_amount))
            current_position_quote_fees = Decimal(str(self.position_info.quote_fee_amount))
            # Total value = tokens + uncollected fees
            current_position_value = (
                current_position_base * current_price + current_position_quote +
                current_position_base_fees * current_price + current_position_quote_fees
            )

        # Calculate P&L: (Closed Value + Fees + Current Position Value) - Total Open Value
        total_current_value = total_close_value + total_fees_value + float(current_position_value)
        position_pnl = (total_current_value - total_open_value) if total_open_value > 0 else 0
        position_roi_pct = (position_pnl / total_open_value * 100) if total_open_value > 0 else 0

        # Get current position open timestamp if open
        current_position_open_time = None
        if self.current_position_id and opens:
            last_open = max((p for p in opens), key=lambda x: x["timestamp"])
            current_position_open_time = last_open["timestamp"]

        # SOL balance change
        connector = self.connectors[self.config.connector]
        current_sol = float(connector.get_available_balance("SOL"))
        initial_sol = self._tracking_data.get("initial_sol_balance", current_sol)
        sol_change = current_sol - initial_sol

        return {
            "opens_count": len(opens),
            "closes_count": len(closes),
            # Open totals
            "total_open_base": total_open_base,
            "total_open_base_value": total_open_base_value,
            "total_open_quote": total_open_quote,
            "total_open_value": total_open_value,
            # Close totals
            "total_close_base": total_close_base,
            "total_close_base_value": total_close_base_value,
            "total_close_quote": total_close_quote,
            "total_close_value": total_close_value,
            # Fees
            "total_fees_base": total_fees_base,
            "total_fees_base_value": total_fees_base_value,
            "total_fees_quote": total_fees_quote,
            "total_fees_value": total_fees_value,
            # Current position (if open and created by script)
            "current_position_value": float(current_position_value),
            "current_position_base": float(current_position_base),
            "current_position_quote": float(current_position_quote),
            "current_position_base_fees": float(current_position_base_fees),
            "current_position_quote_fees": float(current_position_quote_fees),
            "first_event_is_open": first_event_is_open,
            # P&L (includes current position value)
            "position_pnl": position_pnl,
            "position_roi_pct": position_roi_pct,
            "current_position_open_time": current_position_open_time,
            # SOL tracking
            "initial_sol": initial_sol,
            "current_sol": current_sol,
            "sol_change": sol_change
        }

    def _create_price_limits_visualization(self, current_price: Decimal) -> Optional[str]:
        """Create visualization of price limits with current price"""
        if not self.config.lower_price_limit and not self.config.upper_price_limit:
            return None

        lower_limit = self.config.lower_price_limit if self.config.lower_price_limit else Decimal("0")
        upper_limit = self.config.upper_price_limit if self.config.upper_price_limit else current_price * 2

        # If only one limit is set, create appropriate range
        if not self.config.lower_price_limit:
            lower_limit = max(Decimal("0"), upper_limit * Decimal("0.5"))
        if not self.config.upper_price_limit:
            upper_limit = lower_limit * Decimal("2")

        # Calculate position
        price_range = upper_limit - lower_limit
        if price_range <= 0:
            return None

        current_position = (current_price - lower_limit) / price_range

        # Create bar
        bar_width = 50
        current_pos = int(current_position * bar_width)

        # Build visualization
        limit_bar = ['─'] * bar_width
        limit_bar[0] = '['
        limit_bar[-1] = ']'

        # Place price marker
        if current_pos < 0:
            marker_line = '● ' + ''.join(limit_bar)
            status = "⛔ BELOW LOWER LIMIT"
        elif current_pos >= bar_width:
            marker_line = ''.join(limit_bar) + ' ●'
            status = "⛔ ABOVE UPPER LIMIT"
        else:
            limit_bar[current_pos] = '●'
            marker_line = ''.join(limit_bar)
            status = "✓ Within Limits"

        viz_lines = []
        viz_lines.append("Price Limits:")
        viz_lines.append(marker_line)

        # Build limit labels
        lower_str = f'{float(lower_limit):.6f}' if self.config.lower_price_limit else 'None'
        upper_str = f'{float(upper_limit):.6f}' if self.config.upper_price_limit else 'None'
        viz_lines.append(lower_str + ' ' * (bar_width - len(lower_str) - len(upper_str)) + upper_str)
        viz_lines.append(f'Status: {status}')

        return '\n'.join(viz_lines)

    def _create_price_range_visualization(self, lower_price: Decimal, current_price: Decimal,
                                          upper_price: Decimal) -> str:
        """Create visual representation of price range with current price marker"""
        # Calculate position in range (0 to 1)
        price_range = upper_price - lower_price
        current_position = (current_price - lower_price) / price_range

        # Create 50-character wide bar
        bar_width = 50
        current_pos = int(current_position * bar_width)

        # Build price range bar
        range_bar = ['─'] * bar_width
        range_bar[0] = '├'
        range_bar[-1] = '┤'

        # Place marker inside or outside range
        if current_pos < 0:
            # Price below range
            marker_line = '● ' + ''.join(range_bar)
        elif current_pos >= bar_width:
            # Price above range
            marker_line = ''.join(range_bar) + ' ●'
        else:
            # Price within range
            range_bar[current_pos] = '●'
            marker_line = ''.join(range_bar)

        viz_lines = []
        viz_lines.append(marker_line)
        lower_str = f'{float(lower_price):.6f}'
        upper_str = f'{float(upper_price):.6f}'
        viz_lines.append(lower_str + ' ' * (bar_width - len(lower_str) - len(upper_str)) + upper_str)

        return '\n'.join(viz_lines)

    def _create_distribution_visualization(self, is_base_only: bool) -> Optional[str]:
        """
        Create distribution visualization for Meteora strategy types.
        Only shows for Meteora connector with strategy_type defined.

        Args:
            is_base_only: True if base-only position (range above price), False if quote-only (range below price)

        Returns:
            Distribution visualization string or None if not applicable
        """
        # Only show for Meteora with strategy_type defined
        if "meteora" not in self.config.connector.lower() or self.config.strategy_type is None:
            return None

        bar_width = 50
        blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']  # 8 levels

        strategy_type = self.config.strategy_type

        if strategy_type == 0:
            # Spot: flat distribution
            distribution = '█' * bar_width
            label = "Spot"
        elif strategy_type == 1:
            # Curve: peak near current price
            if is_base_only:
                # Range above price, peak at left (near price)
                distribution = self._create_gradient(bar_width, blocks, descending=True)
            else:
                # Range below price, peak at right (near price)
                distribution = self._create_gradient(bar_width, blocks, descending=False)
            label = "Curve"
        elif strategy_type == 2:
            # Bid-Ask: peak far from current price
            if is_base_only:
                # Range above price, peak at right (far from price)
                distribution = self._create_gradient(bar_width, blocks, descending=False)
            else:
                # Range below price, peak at left (far from price)
                distribution = self._create_gradient(bar_width, blocks, descending=True)
            label = "Bid-Ask"
        else:
            return None

        return f"Distribution: {label}\n{distribution}"

    @staticmethod
    def _create_gradient(width: int, blocks: list, descending: bool) -> str:
        """Create a gradient bar using block characters."""
        result = []
        for i in range(width):
            # Map position to block index (0-7)
            if descending:
                block_idx = int((1 - i / width) * (len(blocks) - 1))
            else:
                block_idx = int((i / width) * (len(blocks) - 1))
            result.append(blocks[block_idx])
        return ''.join(result)

    def format_status(self) -> str:
        """Format status message for display"""
        lines = []
        pnl_summary = None  # Will be calculated when needed

        if self.pending_operation == "opening":
            lines.append(f"⏳ Opening position (order ID: {self.pending_open_order_id})")
            lines.append("Awaiting transaction confirmation...")
        elif self.pending_operation == "closing":
            lines.append(f"⏳ Closing position (order ID: {self.pending_close_order_id})")
            lines.append("Awaiting transaction confirmation...")
        elif self.current_position_id and self.position_info:
            # Calculate P&L summary early for use in display
            pnl_summary = self._calculate_pnl_summary()

            # Active position
            lines.append(f"Position: {self.current_position_id}")
            lines.append(f"Pool: {self.trading_pair}")
            lines.append(f"Connector: {self.exchange}")

            # Tokens and value section
            base_amount = Decimal(str(self.position_info.base_token_amount))
            quote_amount = Decimal(str(self.position_info.quote_token_amount))
            base_fee = Decimal(str(self.position_info.base_fee_amount))
            quote_fee = Decimal(str(self.position_info.quote_fee_amount))

            if self.pool_info:
                current_price = Decimal(str(self.pool_info.price))

                # Calculate token value
                token_value = base_amount * current_price + quote_amount

                # Calculate fee value
                fee_value = base_fee * current_price + quote_fee

                # Calculate total value (tokens + fees)
                total_value = token_value + fee_value

                lines.append(f"Total Value: {total_value:.6f} {self.quote_token}")

                # Calculate percentages
                if total_value > 0:
                    token_pct = float(token_value / total_value * 100)
                    fee_pct = float(fee_value / total_value * 100)
                else:
                    token_pct = fee_pct = 0.0

                lines.append(f"Tokens: {base_amount:.6f} {self.base_token} / {quote_amount:.6f} {self.quote_token} ({token_pct:.2f}%)")

                if base_fee > 0 or quote_fee > 0:
                    lines.append(f"Fees: {base_fee:.6f} {self.base_token} / {quote_fee:.6f} {self.quote_token} ({fee_pct:.2f}%)")

                # Show duration for current position
                if pnl_summary["current_position_open_time"] is not None:
                    from datetime import datetime
                    open_dt = datetime.fromtimestamp(pnl_summary["current_position_open_time"])
                    elapsed = int(time.time() - pnl_summary["current_position_open_time"])
                    elapsed_m = elapsed // 60
                    elapsed_s = elapsed % 60
                    lines.append(f"Duration: {open_dt.strftime('%Y-%m-%d %H:%M:%S')} ({elapsed_m}m {elapsed_s}s)")
            else:
                lines.append(f"Tokens: {base_amount:.6f} {self.base_token} / {quote_amount:.6f} {self.quote_token}")

                if base_fee > 0 or quote_fee > 0:
                    lines.append(f"Fees: {base_fee:.6f} {self.base_token} / {quote_fee:.6f} {self.quote_token}")

                # Show duration for current position
                if pnl_summary["current_position_open_time"] is not None:
                    from datetime import datetime
                    open_dt = datetime.fromtimestamp(pnl_summary["current_position_open_time"])
                    elapsed = int(time.time() - pnl_summary["current_position_open_time"])
                    elapsed_m = elapsed // 60
                    elapsed_s = elapsed % 60
                    lines.append(f"Duration: {open_dt.strftime('%Y-%m-%d %H:%M:%S')} ({elapsed_m}m {elapsed_s}s)")

            lines.append("")  # Spacer

            # Position range and width info
            lower_price = Decimal(str(self.position_info.lower_price))
            upper_price = Decimal(str(self.position_info.upper_price))

            if self.pool_info:
                current_price = Decimal(str(self.pool_info.price))

                # Price range visualization
                lines.append("Position Range:")
                lines.append(self._create_price_range_visualization(lower_price, current_price, upper_price))

                # Distribution visualization (Meteora only, only for positions created by script)
                if self.has_rebalanced_once:
                    is_base_only = base_amount > 0 and quote_amount == 0
                    dist_viz = self._create_distribution_visualization(is_base_only)
                    if dist_viz:
                        lines.append("")  # Spacer before distribution
                        lines.append(dist_viz)

                # Price and status
                lines.append("")
                lines.append(f"Price: {float(current_price):.6f}")
                if self._price_in_bounds(current_price, lower_price, upper_price):
                    lines.append("Status: ✅  In Range")
                else:
                    arrow = "⬇️" if current_price < lower_price else "⬆️"
                    lines.append(f"Status: {arrow}  Out of Range")

                lines.append("")  # Spacer after position range

                # Add price limits visualization
                limits_viz = self._create_price_limits_visualization(current_price)
                if limits_viz:
                    lines.append(limits_viz)
            else:
                lines.append(f"Position Range: {lower_price:.6f} - {upper_price:.6f}")

        else:
            lines.append(f"Monitoring {self.trading_pair} on {self.exchange}")
            lines.append("Status: ⏳ No active position")

            if self.config.base_amount > 0 or self.config.quote_amount > 0:
                lines.append(f"Will create position with: {self.config.base_amount} base / "
                             f"{self.config.quote_amount} quote tokens")

            if self.pool_info:
                lines.append(f"Current Price: {self.pool_info.price:.6f}")

                # Add price limits visualization if configured
                current_price = Decimal(str(self.pool_info.price))
                limits_viz = self._create_price_limits_visualization(current_price)
                if limits_viz:
                    lines.append("")
                    lines.append(limits_viz)

        # Add P&L Summary (always show)
        # Calculate pnl_summary if not already calculated in position display
        if pnl_summary is None:
            pnl_summary = self._calculate_pnl_summary()
        lines.append("")
        lines.append("LP Performance Summary:")

        if pnl_summary["opens_count"] > 0:
            base = self.base_token or (self.base_token_address[:8] + "..." if self.base_token_address else "BASE")
            quote = self.quote_token or (self.quote_token_address[:8] + "..." if self.quote_token_address else "QUOTE")
            lines.append("")
            lines.append(f"  Positions Opened: {pnl_summary['opens_count']}")
            lines.append(f"    {base}: {pnl_summary['total_open_base']:.6f} ({pnl_summary['total_open_base_value']:.6f} {quote})")
            lines.append(f"    {quote}: {pnl_summary['total_open_quote']:.6f}")
            lines.append(f"    Value: {pnl_summary['total_open_value']:.6f} {quote}")

            if pnl_summary["closes_count"] > 0:
                lines.append("")
                lines.append(f"  Positions Closed: {pnl_summary['closes_count']}")
                lines.append(f"    {base}: {pnl_summary['total_close_base']:.6f} ({pnl_summary['total_close_base_value']:.6f} {quote})")
                lines.append(f"    {quote}: {pnl_summary['total_close_quote']:.6f}")
                lines.append(f"    Value: {pnl_summary['total_close_value']:.6f} {quote}")
                lines.append("")
                lines.append("  Total Fees Collected:")
                lines.append(f"    {base}: {pnl_summary['total_fees_base']:.6f} ({pnl_summary['total_fees_base_value']:.6f} {quote})")
                lines.append(f"    {quote}: {pnl_summary['total_fees_quote']:.6f}")
                lines.append(f"    Value: {pnl_summary['total_fees_value']:.6f} {quote}")

            # Show current position value if script created initial position
            if pnl_summary["first_event_is_open"] and pnl_summary["current_position_value"] > 0:
                lines.append("")
                lines.append("  Current Position Value:")
                lines.append(f"    Value: {pnl_summary['current_position_value']:.6f} {quote}")

            lines.append("")
            pnl_sign = "+" if pnl_summary["position_pnl"] >= 0 else ""
            lines.append(f"  P&L: {pnl_sign}{pnl_summary['position_pnl']:.6f} {quote} ({pnl_sign}{pnl_summary['position_roi_pct']:.2f}%)")
        else:
            lines.append(f"  Positions Opened: {pnl_summary['opens_count']}")
            lines.append(f"  Positions Closed: {pnl_summary['closes_count']}")

        lines.append("")
        lines.append("Wallet SOL:")
        lines.append(f"  Initial:  {pnl_summary['initial_sol']:.4f} SOL")
        lines.append(f"  Current:  {pnl_summary['current_sol']:.4f} SOL")
        sol_change_sign = "+" if pnl_summary["sol_change"] >= 0 else ""
        lines.append(f"  Change:   {sol_change_sign}{pnl_summary['sol_change']:.4f} SOL")

        return "\n".join(lines)
