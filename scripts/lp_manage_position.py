"""
lp_manage_position.py

CLMM LP position manager that automatically rebalances positions.

BEHAVIOR
--------
- Monitors existing position in the specified pool
- Monitors price vs. active position's price bounds
- When price is out-of-bounds for >= rebalance_seconds, fully closes the position and re-enters
- First position can be double-sided (if both base_amount and quote_amount provided)
- After first rebalance, all subsequent positions are SINGLE-SIDED (more capital efficient)
- Single-sided positions provide only the token needed based on where price moved

PARAMETERS
----------
- connector: CLMM connector in format 'name/type' (e.g. raydium/clmm, meteora/clmm)
- trading_pair: Trading pair (e.g. META-SOL). Pool must be added via 'gateway pool' command first
- base_amount: Initial base token amount (0 for quote-only position)
- quote_amount: Initial quote token amount (0 for base-only position)
  * If both are 0 and no existing position: monitoring only
  * If both provided: creates double-sided initial position
  * After rebalance: only one token provided based on price direction
- position_width_pct: TOTAL position width as percentage of mid price (e.g. 2.0 = ±1%)
- rebalance_seconds: Seconds price must stay out-of-bounds before rebalancing

NOTES
-----
- All tick rounding and amount calculations delegated to Gateway
- After first rebalance, automatically switches to single-sided positions
- Uses actual wallet balances for rebalancing (not config amounts)
"""

import asyncio
import logging
import os
import time
from decimal import Decimal
from typing import Dict, Optional

from pydantic import Field

from hummingbot.client.config.config_data_types import BaseClientModel
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.gateway.common_types import ConnectorType, get_connector_type
from hummingbot.connector.gateway.gateway_lp import CLMMPoolInfo, CLMMPositionInfo
from hummingbot.core.event.events import RangePositionLiquidityAddedEvent, RangePositionLiquidityRemovedEvent
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.data_feed.market_data_provider import MarketDataProvider
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.strategy_v2.executors.data_types import ConnectorPair


class LpPositionManagerConfig(BaseClientModel):
    script_file_name: str = os.path.basename(__file__)
    connector: str = Field("meteora/clmm", json_schema_extra={
        "prompt": "CLMM connector in format 'name/type' (e.g. raydium/clmm, meteora/clmm)", "prompt_on_new": True})
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
    check_seconds: int = Field(30, json_schema_extra={
        "prompt": "Seconds between position status checks (important for rate limits)", "prompt_on_new": True})


class LpPositionManager(ScriptStrategyBase):
    """
    CLMM LP position manager that automatically rebalances when price moves out of bounds.
    """

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
        self._opening_position_amounts: Optional[Dict] = None  # Track amounts when opening position

        # Check throttling (for rate limiting)
        self._last_check_time: float = 0

        # Update throttling
        self._last_position_update_time: float = 0
        self._position_update_interval: int = 10  # Update position info every 10 seconds

        # Log initial startup (will log more details after resolving trading pair)
        self.log_with_clock(logging.INFO,
                            f"LP Position Manager initializing for pool {self.pool_address[:8]}... on {self.exchange}")

        # Initialize position on startup (will resolve trading pair first)
        safe_ensure_future(self.initialize_position())

    async def resolve_trading_pair_from_pool(self):
        """Resolve trading pair from pool address by fetching pool info"""
        if self._trading_pair_resolved:
            return

        try:
            connector = self.connectors[self.exchange]

            # Fetch pool info directly using pool address
            pool_info = await connector._get_gateway_instance().pool_info(
                connector=self.exchange,
                network=connector.network,
                pool_address=self.pool_address
            )

            if not pool_info:
                raise ValueError(f"Could not fetch pool info for pool address {self.pool_address}")

            # Get token addresses from pool info
            base_token_address = pool_info.get("baseTokenAddress")
            quote_token_address = pool_info.get("quoteTokenAddress")

            if not base_token_address or not quote_token_address:
                raise ValueError(f"Pool info missing token addresses: {pool_info}")

            # Store token addresses (needed for balance lookups)
            self.base_token_address = base_token_address
            self.quote_token_address = quote_token_address

            # Try to get token symbols
            base_token_info = connector.get_token_by_address(base_token_address)
            quote_token_info = connector.get_token_by_address(quote_token_address)

            base_symbol = base_token_info.get("symbol") if base_token_info else base_token_address
            quote_symbol = quote_token_info.get("symbol") if quote_token_info else quote_token_address

            # Set trading pair and tokens
            self.base_token = base_symbol
            self.quote_token = quote_symbol
            self.trading_pair = f"{base_symbol}-{quote_symbol}"
            self._trading_pair_resolved = True

            self.logger().info(f"Resolved trading pair from pool: {self.trading_pair}")

            # Initialize rate sources now that we have the trading pair
            self.market_data_provider.initialize_rate_sources([
                ConnectorPair(connector_name=self.exchange, trading_pair=self.trading_pair)
            ])

            # Log configuration details
            self.logger().info(
                f"LP Position Manager configured:\n"
                f"  Trading Pair: {self.trading_pair}\n"
                f"  Pool Address: {self.pool_address}\n"
                f"  Position width: ±{float(self.config.position_width_pct) / 2:.2f}% around mid price\n"
                f"  Rebalance threshold: {self.config.rebalance_seconds} seconds out-of-bounds\n"
                f"  Check interval: {self.config.check_seconds} seconds (rate limiting)"
            )

            if self.config.base_amount > 0 or self.config.quote_amount > 0:
                self.logger().info(
                    f"Initial amounts: {self.config.base_amount} {self.base_token} / "
                    f"{self.config.quote_amount} {self.quote_token}"
                )

        except Exception as e:
            self.logger().error(f"Error resolving trading pair from pool: {str(e)}", exc_info=True)
            raise

    async def initialize_position(self):
        """Check for existing positions or create initial position on startup"""
        await asyncio.sleep(3)  # Wait for connector to initialize

        # First, resolve trading pair from pool address
        await self.resolve_trading_pair_from_pool()

        # Fetch pool info to get pool address and current price
        await self.fetch_pool_info()

        if not self.pool_info:
            self.logger().error(f"Pool not found for {self.trading_pair}. Please add pool via 'gateway pool' command first")
            return

        # Check if user has existing position in this pool
        if await self.check_existing_positions():
            self.logger().info(f"Found existing position {self.current_position_id}, will monitor it")

            # Check if the existing position is already out of range
            await self.check_if_position_out_of_range_on_startup()
            return

        # No existing position - create one if user provided amounts
        if self.config.base_amount > 0 or self.config.quote_amount > 0:
            self.logger().info("No existing position found, creating initial position...")
            await self.create_initial_position()
        else:
            self.logger().info("No existing position and no initial amounts provided - monitoring only")

    def on_tick(self):
        """Called on each strategy tick"""
        # Throttle checks based on check_seconds (for rate limiting)
        current_time = self.current_timestamp
        if current_time - self._last_check_time < self.config.check_seconds:
            return

        self._last_check_time = current_time

        if self.pending_operation:
            # Operation in progress, wait for confirmation
            return

        if self.current_position_id:
            # Monitor existing position
            safe_ensure_future(self.monitor_and_rebalance())
        else:
            # No position yet, just update pool info
            safe_ensure_future(self.fetch_pool_info())

    async def fetch_pool_info(self):
        """Fetch pool information to get current price"""
        try:
            # Wait for connector to be ready
            if self.exchange not in self.connectors:
                return None

            connector = self.connectors[self.exchange]

            # Fetch pool info directly using pool address
            pool_info_resp = await connector._get_gateway_instance().pool_info(
                connector=self.exchange,
                network=connector.network,
                pool_address=self.pool_address
            )

            if pool_info_resp:
                # Parse into CLMMPoolInfo object
                self.pool_info = CLMMPoolInfo(**pool_info_resp)
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
        except Exception as e:
            self.logger().debug(f"No existing positions found or error checking: {str(e)}")
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
                self.out_of_bounds_since = self.current_timestamp

                # Calculate deviation
                if current_price < lower_price:
                    deviation = abs((float(current_price) - float(lower_price)) / float(lower_price) * 100)
                    direction = "below lower"
                    bound = lower_price
                else:
                    deviation = abs((float(current_price) - float(upper_price)) / float(upper_price) * 100)
                    direction = "above upper"
                    bound = upper_price

                msg = (f"⚠️ Position is already out of range! Price {float(current_price):.6f} is {direction} "
                       f"bound {float(bound):.6f} by {deviation:.2f}%. Rebalance countdown started "
                       f"({self.config.rebalance_seconds}s threshold)")

                self.logger().warning(msg)
                self.notify_hb_app_with_timestamp(msg)
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

            # Check balances before attempting to create position
            # Fetch balances using token addresses from pool info
            connector = self.connectors[self.exchange]
            base_balance = float(await connector.get_balance_by_address(self.base_token_address))
            quote_balance = float(await connector.get_balance_by_address(self.quote_token_address))

            self.logger().info(f"Available balances: {base_balance} {self.base_token}, {quote_balance} {self.quote_token}")

            # Validate sufficient balances
            if base_amt > 0 and base_balance < base_amt:
                self.logger().error(
                    f"Insufficient {self.base_token} balance! Required: {base_amt}, Available: {base_balance}"
                )
                return

            if quote_amt > 0 and quote_balance < quote_amt:
                self.logger().error(
                    f"Insufficient {self.quote_token} balance! Required: {quote_amt}, Available: {quote_balance}"
                )
                return

            # Check for dust amounts (less than 0.0001)
            min_amount = 0.0001
            if base_amt > 0 and base_amt < min_amount:
                self.logger().error(
                    f"Base amount {base_amt} {self.base_token} is too small (minimum {min_amount}). "
                    f"Increase base_amount in config."
                )
                return

            if quote_amt > 0 and quote_amt < min_amount:
                self.logger().error(
                    f"Quote amount {quote_amt} {self.quote_token} is too small (minimum {min_amount}). "
                    f"Increase quote_amount in config."
                )
                return

            # Compute width percentages based on position type
            lower_pct, upper_pct = self._compute_width_percentages(base_amt, quote_amt)

            # Calculate actual price bounds
            lower_price = current_price * (1 - lower_pct / 100)
            upper_price = current_price * (1 + upper_pct / 100)

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

            order_id = self.connectors[self.exchange].add_liquidity(
                trading_pair=self.trading_pair,
                price=current_price,
                upper_width_pct=upper_pct,
                lower_width_pct=lower_pct,
                base_token_amount=base_amt,
                quote_token_amount=quote_amt,
                pool_address=self.pool_address,
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
            if current_time - self._last_position_update_time >= self._position_update_interval:
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
                    if float(current_price) < float(lower_price):
                        deviation = (float(lower_price) - float(current_price)) / float(lower_price) * 100
                        direction = "below"
                        bound = lower_price
                        self.logger().info(f"Price {current_price:.6f} moved below lower bound {lower_price:.6f} by {deviation:.2f}%")
                    else:
                        deviation = (float(current_price) - float(upper_price)) / float(upper_price) * 100
                        direction = "above"
                        bound = upper_price
                        self.logger().info(f"Price {current_price:.6f} moved above upper bound {upper_price:.6f} by {deviation:.2f}%")

                    # Notify user that position is out of range
                    msg = (f"⚠️ {self.trading_pair} position out of range on {self.exchange}: "
                           f"Price {float(current_price):.6f} moved {direction} bound {float(bound):.6f} by {deviation:.2f}%. "
                           f"Will rebalance after {self.config.rebalance_seconds}s")
                    self.notify_hb_app_with_timestamp(msg)

                elapsed_seconds = current_time - self.out_of_bounds_since

                if elapsed_seconds >= self.config.rebalance_seconds:
                    self.logger().info(f"Price out of bounds for {elapsed_seconds:.0f} seconds (threshold: {self.config.rebalance_seconds})")
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

            # Determine position direction
            if float(current_price) < float(old_lower):
                direction = "below"
                deviation = (float(old_lower) - float(current_price)) / float(old_lower) * 100
            else:
                direction = "above"
                deviation = (float(current_price) - float(old_upper)) / float(old_upper) * 100

            # Notify user about rebalance trigger
            msg = (f"🔄 Rebalancing {self.trading_pair} position on {self.exchange}: "
                   f"Price {float(current_price):.6f} moved {direction} range "
                   f"[{float(old_lower):.6f}-{float(old_upper):.6f}] by {deviation:.2f}%")
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

            # Check for dust amounts (less than 0.0001)
            min_amount = 0.0001
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

            # Calculate actual price bounds for logging
            lower_price = new_mid_price * (1 - lower_pct / 100)
            upper_price = new_mid_price * (1 + upper_pct / 100)

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

            order_id = self.connectors[self.exchange].add_liquidity(
                trading_pair=self.trading_pair,
                price=new_mid_price,
                upper_width_pct=upper_pct,
                lower_width_pct=lower_pct,
                base_token_amount=base_amt,
                quote_token_amount=quote_amt,
                pool_address=self.pool_address,
            )

            self.pending_open_order_id = order_id
            self.pending_operation = "opening"
            self.has_rebalanced_once = True
            self.logger().info(f"Rebalanced {side}-only position order submitted with ID: {order_id}")

            # Clean up balance info
            self._closed_position_balances = None

        except Exception as e:
            self.logger().error(f"Error opening rebalanced position: {str(e)}")
            self.pending_operation = None

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

    async def fetch_position_info_after_add(self):
        """Fetch position info after position is created"""
        try:
            await asyncio.sleep(3)  # Wait for position to be created on-chain

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

            # Build notification message with amounts added
            if self._opening_position_amounts:
                base_amt = self._opening_position_amounts["base_amount"]
                quote_amt = self._opening_position_amounts["quote_amount"]
                msg = (f"✓ Position opened on {self.exchange}: "
                       f"Added {base_amt:.6f} {self.base_token} + {quote_amt:.6f} {self.quote_token}")
                self._opening_position_amounts = None  # Clear after use
            else:
                msg = f"✓ Position opened on {self.exchange}"

            self.notify_hb_app_with_timestamp(msg)

    def did_remove_liquidity(self, event: RangePositionLiquidityRemovedEvent):
        """Called when liquidity is removed from a position"""
        self.logger().info(f"did_remove_liquidity called with order_id: {event.order_id if hasattr(event, 'order_id') else 'NO ORDER_ID'}")
        self.logger().info(f"pending_close_order_id: {self.pending_close_order_id}")
        self.logger().info(f"_closed_position_balances set: {self._closed_position_balances is not None}")

        if hasattr(event, 'order_id') and event.order_id == self.pending_close_order_id:
            self.logger().info(f"Position closing order {event.order_id} confirmed!")

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
                msg = (f"✓ Position closed on {self.exchange}: "
                       f"Removed {total_base:.6f} {self.base_token} + {total_quote:.6f} {self.quote_token}")
            else:
                msg = f"✓ Position closed on {self.exchange}"

            self.notify_hb_app_with_timestamp(msg)

            # If this was a rebalance, open the new position
            if self._closed_position_balances:
                self.logger().info("Position closed, opening rebalanced position...")
                safe_ensure_future(self.open_rebalanced_position())
            else:
                self.logger().warning("Position closed but no rebalance info available!")
        else:
            self.logger().warning(f"Order ID mismatch or missing: event={event.order_id if hasattr(event, 'order_id') else 'None'}, pending={self.pending_close_order_id}")

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
        viz_lines.append(f'{float(lower_price):.2f}' + ' ' * (bar_width - len(f'{float(lower_price):.2f}') - len(f'{float(upper_price):.2f}')) + f'{float(upper_price):.2f}')
        viz_lines.append(f'Price: {float(current_price):.6f}')

        return '\n'.join(viz_lines)

    def format_status(self) -> str:
        """Format status message for display"""
        lines = []

        if self.pending_operation == "opening":
            lines.append(f"⏳ Opening position (order ID: {self.pending_open_order_id})")
            lines.append("Awaiting transaction confirmation...")
        elif self.pending_operation == "closing":
            lines.append(f"⏳ Closing position (order ID: {self.pending_close_order_id})")
            lines.append("Awaiting transaction confirmation...")
        elif self.current_position_id and self.position_info:
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
            else:
                lines.append(f"Tokens: {base_amount:.6f} {self.base_token} / {quote_amount:.6f} {self.quote_token}")

                if base_fee > 0 or quote_fee > 0:
                    lines.append(f"Fees: {base_fee:.6f} {self.base_token} / {quote_fee:.6f} {self.quote_token}")

            lines.append("")  # Spacer

            # Position range and width info
            lower_price = Decimal(str(self.position_info.lower_price))
            upper_price = Decimal(str(self.position_info.upper_price))

            if self.pool_info:
                current_price = Decimal(str(self.pool_info.price))

                # Price range visualization
                lines.append(self._create_price_range_visualization(lower_price, current_price, upper_price))

                if self._price_in_bounds(current_price, lower_price, upper_price):
                    lines.append("Status: ✅ In Bounds")
                else:
                    lines.append("Status: ⚠️ Out of Bounds")
            else:
                lines.append(f"Position Range: {lower_price:.6f} - {upper_price:.6f}")

            if self.out_of_bounds_since:
                elapsed = time.time() - self.out_of_bounds_since
                lines.append(f"Out of bounds for: {elapsed:.0f}/{self.config.rebalance_seconds} seconds")

        else:
            lines.append(f"Monitoring {self.trading_pair} on {self.exchange}")
            lines.append("Status: ⏳ No active position")

            if self.config.base_amount > 0 or self.config.quote_amount > 0:
                lines.append(f"Will create position with: {self.config.base_amount} base / "
                             f"{self.config.quote_amount} quote tokens")

            if self.pool_info:
                lines.append(f"Current Price: {self.pool_info.price:.6f}")

        return "\n".join(lines)
