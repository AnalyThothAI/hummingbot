import logging
from decimal import Decimal
from typing import Dict, List, Optional

from pydantic import Field

from hummingbot.core.data_type.common import MarketDict
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy_v2.controllers import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.data_types import ConnectorPair
from hummingbot.strategy_v2.executors.lp_position_executor.data_types import LPPositionExecutorConfig, LPPositionStates
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo


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
    network: str = "solana-mainnet-beta"  # Format: chain-network (e.g., ethereum-mainnet)
    trading_pair: str = ""  # e.g., "SOL-USDC"
    pool_address: str = ""  # Required - pool address to manage

    # Position parameters
    base_amount: Decimal = Field(default=Decimal("0"), json_schema_extra={"is_updatable": True})
    quote_amount: Decimal = Field(default=Decimal("1.0"), json_schema_extra={"is_updatable": True})
    position_width_pct: Decimal = Field(default=Decimal("2.0"), json_schema_extra={"is_updatable": True})

    # Rebalancing
    rebalance_seconds: int = Field(default=60, json_schema_extra={"is_updatable": True})

    # Price limits (optional - constrain position bounds)
    lower_price_limit: Optional[Decimal] = Field(default=None, json_schema_extra={"is_updatable": True})
    upper_price_limit: Optional[Decimal] = Field(default=None, json_schema_extra={"is_updatable": True})

    # Connector-specific params (optional)
    strategy_type: Optional[int] = Field(default=None, json_schema_extra={"is_updatable": True})

    def update_markets(self, markets: MarketDict) -> MarketDict:
        """Register the LP connector with trading pair"""
        return markets.add_or_update(self.connector_name, self.trading_pair)


class LPController(ControllerBase):
    """Controller for LP position management with rebalancing logic"""

    _logger: Optional[HummingbotLogger] = None

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    def __init__(self, config: LPControllerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config: LPControllerConfig = config

        # Parse token symbols from trading pair
        parts = config.trading_pair.split("-")
        self._base_token: str = parts[0] if len(parts) >= 2 else ""
        self._quote_token: str = parts[1] if len(parts) >= 2 else ""

        # Rebalance tracking
        self._last_executor_info: Optional[Dict] = None

        # Initialize rate sources
        self.market_data_provider.initialize_rate_sources([
            ConnectorPair(
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair
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

        # Manual kill switch - close position
        if self.config.manual_kill_switch:
            if executor:
                actions.append(StopExecutorAction(
                    controller_id=self.config.id,
                    executor_id=executor.id,
                    keep_position=False,  # Close position on stop
                ))
            return actions

        # No active executor - create new position
        if executor is None:
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

        # Rebalancing logic
        if state == LPPositionStates.OUT_OF_RANGE.value:
            if out_of_range_since is not None:
                current_time = self.market_data_provider.time()
                elapsed = current_time - out_of_range_since
                if elapsed >= self.config.rebalance_seconds:
                    # Check if price is within limits before rebalancing
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
                        executor_id=executor.id,
                        keep_position=False,  # Close position to rebalance
                    ))

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

        # Determine side from amounts: 0=BOTH, 1=BUY (quote only), 2=SELL (base only)
        if base_amt > 0 and quote_amt > 0:
            side = 0  # Both-sided
        elif quote_amt > 0:
            side = 1  # BUY - quote only, positioned to buy base
        else:
            side = 2  # SELL - base only, positioned to sell base

        return LPPositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=self.config.connector_name,
            pool_address=self.config.pool_address,
            trading_pair=self.config.trading_pair,
            base_token=self._base_token,
            quote_token=self._quote_token,
            lower_price=lower_price,
            upper_price=upper_price,
            base_amount=base_amt,
            quote_amount=quote_amt,
            side=side,
            extra_params=extra_params if extra_params else None,
            keep_position=False,
        )

    async def update_processed_data(self):
        """Called every tick - no-op since config provides all needed data"""
        pass

    def _calculate_price_bounds(self, base_amt: Decimal, quote_amt: Decimal) -> tuple:
        """
        Calculate position bounds from current price and width %.

        For double-sided positions (both base and quote): split width evenly
        For base-only positions: full width ABOVE price (sell base for quote)
        For quote-only positions: full width BELOW price (buy base with quote)
        """
        # Use RateOracle for price since LP connectors don't have get_price_by_type
        current_price = self.market_data_provider.get_rate(self.config.trading_pair)
        total_width = self.config.position_width_pct / Decimal("100")

        if base_amt > 0 and quote_amt > 0:
            # Double-sided: split width evenly (±half)
            half_width = total_width / Decimal("2")
            lower_price = current_price * (Decimal("1") - half_width)
            upper_price = current_price * (Decimal("1") + half_width)
        elif base_amt > 0:
            # Base-only: full width ABOVE current price
            lower_price = current_price
            upper_price = current_price * (Decimal("1") + total_width)
        elif quote_amt > 0:
            # Quote-only: full width BELOW current price
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

    def _create_price_range_visualization(self, lower_price: Decimal, current_price: Decimal,
                                          upper_price: Decimal) -> List[str]:
        """Create visual representation of price range with current price marker"""
        # Calculate position in range (0 to 1)
        price_range = upper_price - lower_price
        if price_range <= 0:
            return []
        current_position = (current_price - lower_price) / price_range

        # Create 50-character wide bar
        bar_width = 50
        current_pos = int(float(current_position) * bar_width)

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
        viz_lines.append(lower_str + ' ' * (bar_width - len(lower_str) - len(upper_str) + 2) + upper_str)

        return viz_lines

    def _create_price_limits_visualization(self, current_price: Decimal) -> List[str]:
        """Create visualization of price limits with current price"""
        if not self.config.lower_price_limit and not self.config.upper_price_limit:
            return []

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
            return []

        current_position = (current_price - lower_limit) / price_range

        # Create bar
        bar_width = 50
        current_pos = int(float(current_position) * bar_width)

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
        viz_lines.append(marker_line)

        # Build limit labels
        lower_str = f'{float(lower_limit):.6f}' if self.config.lower_price_limit else 'None'
        upper_str = f'{float(upper_limit):.6f}' if self.config.upper_price_limit else 'None'
        viz_lines.append(lower_str + ' ' * (bar_width - len(lower_str) - len(upper_str) + 2) + upper_str)
        viz_lines.append(f'Status: {status}')

        return viz_lines

    def to_format_status(self) -> List[str]:
        """
        Format status for display.
        Uses executor's custom_info for position data.
        """
        status = []
        box_width = 80

        # Header
        status.append("+" + "-" * box_width + "+")
        header = f"| LP Manager: {self.config.trading_pair or 'Resolving...'} on {self.config.connector_name}"
        status.append(header + " " * (box_width - len(header) + 1) + "|")
        status.append("+" + "-" * box_width + "+")

        executor = self.active_executor()

        if executor is None:
            # No active executor
            line = "| Status: No active position"
            status.append(line + " " * (box_width - len(line) + 1) + "|")

            if self.config.base_amount > 0 or self.config.quote_amount > 0:
                line = f"| Will create position with: {self.config.base_amount} base / {self.config.quote_amount} quote"
                status.append(line + " " * (box_width - len(line) + 1) + "|")

            # Show current price if available
            current_price = self.market_data_provider.get_rate(self.config.trading_pair)
            if current_price:
                line = f"| Current Price: {current_price:.6f}"
                status.append(line + " " * (box_width - len(line) + 1) + "|")

                # Price limits visualization
                if self.config.lower_price_limit or self.config.upper_price_limit:
                    status.append("|" + " " * box_width + "|")
                    line = "| Price Limits:"
                    status.append(line + " " * (box_width - len(line) + 1) + "|")
                    for viz_line in self._create_price_limits_visualization(Decimal(str(current_price))):
                        line = f"| {viz_line}"
                        status.append(line + " " * (box_width - len(line) + 1) + "|")
        else:
            # Active executor - get data from custom_info
            info = executor.custom_info
            state = info.get("state", "UNKNOWN")
            position_address = info.get("position_address")
            current_price = info.get("current_price")
            lower_price = info.get("lower_price")
            upper_price = info.get("upper_price")
            base_amount = info.get("base_amount", 0)
            quote_amount = info.get("quote_amount", 0)
            base_fee = info.get("base_fee", 0)
            quote_fee = info.get("quote_fee", 0)
            out_of_range_since = info.get("out_of_range_since")

            # Position info
            if position_address:
                line = f"| Position: {position_address[:16]}..."
            else:
                line = "| Position: N/A"
            status.append(line + " " * (box_width - len(line) + 1) + "|")

            # State
            state_indicator = "[OK]" if state == "IN_RANGE" else "[!]" if state == "OUT_OF_RANGE" else "[..]"
            line = f"| State: {state_indicator} {state}"
            status.append(line + " " * (box_width - len(line) + 1) + "|")

            if current_price is not None:
                # Calculate total value
                token_value = base_amount * current_price + quote_amount
                fee_value = base_fee * current_price + quote_fee
                total_value = token_value + fee_value

                line = f"| Total Value: {total_value:.6f} {self._quote_token}"
                status.append(line + " " * (box_width - len(line) + 1) + "|")

                line = f"| Tokens: {base_amount:.6f} {self._base_token} / {quote_amount:.6f} {self._quote_token}"
                status.append(line + " " * (box_width - len(line) + 1) + "|")

                line = f"| Fees: {base_fee:.6f} {self._base_token} / {quote_fee:.6f} {self._quote_token}"
                status.append(line + " " * (box_width - len(line) + 1) + "|")

                # Position range visualization
                if lower_price and upper_price:
                    status.append("|" + " " * box_width + "|")
                    line = "| Position Range:"
                    status.append(line + " " * (box_width - len(line) + 1) + "|")
                    for viz_line in self._create_price_range_visualization(
                        Decimal(str(lower_price)), Decimal(str(current_price)), Decimal(str(upper_price))
                    ):
                        line = f"| {viz_line}"
                        status.append(line + " " * (box_width - len(line) + 1) + "|")

                    line = f"| Price: {current_price:.6f}"
                    status.append(line + " " * (box_width - len(line) + 1) + "|")

                # Rebalance countdown if out of range
                if state == "OUT_OF_RANGE" and out_of_range_since:
                    elapsed = self.market_data_provider.time() - out_of_range_since
                    remaining = max(0, self.config.rebalance_seconds - elapsed)
                    line = f"| Rebalance in: {int(remaining)}s"
                    status.append(line + " " * (box_width - len(line) + 1) + "|")

                # Price limits visualization
                if self.config.lower_price_limit or self.config.upper_price_limit:
                    status.append("|" + " " * box_width + "|")
                    line = "| Price Limits:"
                    status.append(line + " " * (box_width - len(line) + 1) + "|")
                    for viz_line in self._create_price_limits_visualization(Decimal(str(current_price))):
                        line = f"| {viz_line}"
                        status.append(line + " " * (box_width - len(line) + 1) + "|")

        status.append("+" + "-" * box_width + "+")
        return status
