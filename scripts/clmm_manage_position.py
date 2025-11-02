import asyncio
import os
import time
from decimal import Decimal
from typing import Dict

from pydantic import Field

from hummingbot.client.config.config_data_types import BaseClientModel
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.gateway.gateway_http_client import GatewayHttpClient
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class CLMMPositionManagerConfig(BaseClientModel):
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    connector: str = Field("meteora/clmm", json_schema_extra={
        "prompt": "CLMM Connector (e.g. meteora/clmm, raydium/clmm)", "prompt_on_new": True})
    network: str = Field("mainnet-beta", json_schema_extra={
        "prompt": "Network (e.g. mainnet-beta, devnet)", "prompt_on_new": True})
    wallet_address: str = Field("", json_schema_extra={
        "prompt": "Wallet address (leave empty to use the default wallet for the chain)", "prompt_on_new": False})
    pool_address: str = Field("9d9mb8kooFfaD3SctgZtkxQypkshx6ezhbKio89ixyy2", json_schema_extra={
        "prompt": "Pool address (e.g. TRUMP-USDC Meteora pool)", "prompt_on_new": True})
    target_price: Decimal = Field(Decimal("10.0"), json_schema_extra={
        "prompt": "Target price to trigger position opening", "prompt_on_new": True})
    trigger_above: bool = Field(False, json_schema_extra={
        "prompt": "Trigger when price rises above target? (True for above/False for below)", "prompt_on_new": True})
    position_width_pct: Decimal = Field(Decimal("10.0"), json_schema_extra={
        "prompt": "Position width in percentage (e.g. 5.0 for ±5% around target price)", "prompt_on_new": True})
    base_token_amount: Decimal = Field(Decimal("0.1"), json_schema_extra={
        "prompt": "Base token amount to add to position (0 for quote only)", "prompt_on_new": True})
    quote_token_amount: Decimal = Field(Decimal("1.0"), json_schema_extra={
        "prompt": "Quote token amount to add to position (0 for base only)", "prompt_on_new": True})
    out_of_range_pct: Decimal = Field(Decimal("1.0"), json_schema_extra={
        "prompt": "Percentage outside range that triggers closing (e.g. 1.0 for 1%)", "prompt_on_new": True})
    out_of_range_secs: int = Field(300, json_schema_extra={
        "prompt": "Seconds price must be out of range before closing (e.g. 300 for 5 min)", "prompt_on_new": True})
    check_interval: int = Field(30, json_schema_extra={
        "prompt": "Seconds between position checks (e.g. 30)", "prompt_on_new": False})
    price_return_threshold_pct: Decimal = Field(Decimal("0.5"), json_schema_extra={
        "prompt": "Percentage price needs to return towards range to reset stop-loss timer (e.g. 0.5)", "prompt_on_new": False})


class CLMMPositionManager(ScriptStrategyBase):
    """
    This strategy monitors CLMM pool prices, opens a position when a target price is reached,
    and closes the position if the price moves out of range for a specified duration.
    """

    @classmethod
    def init_markets(cls, config: CLMMPositionManagerConfig):
        # For gateway connectors, use connector_network format
        market_name = f"{config.connector}_{config.network}"
        cls.markets = {market_name: set()}  # Empty set since we're not trading pairs directly

    def __init__(self, connectors: Dict[str, ConnectorBase], config: CLMMPositionManagerConfig):
        super().__init__(connectors)
        self.config = config
        self.exchange = f"{config.connector}_{config.network}"

        # Get the gateway LP connector from connectors
        self.gateway_lp = self.connectors.get(self.exchange)
        if not self.gateway_lp:
            self.logger().error(f"Gateway LP connector {self.exchange} not found!")
            return

        # State tracking
        self.position_opened = False
        self.position_opening = False
        self.position_closing = False
        self.position_address = None
        self.pool_info = None
        self.last_price = None
        self.position_lower_price = None
        self.position_upper_price = None
        self.out_of_range_start_time = None
        self.max_out_of_range_distance = 0.0  # Track max distance when out of range
        self.last_monitor_time = None

        # Token info - will be populated from pool_info
        self.base_token = None
        self.quote_token = None

        # Initial position value tracking for P&L
        self.initial_position_value = None

        # Log startup information
        self.logger().info("Starting CLMMPositionManager strategy")
        self.logger().info(f"Connector: {self.config.connector}")
        self.logger().info(f"Network: {self.config.network}")
        self.logger().info(f"Pool address: {self.config.pool_address}")
        self.logger().info(f"Target price: {self.config.target_price}")
        condition = "rises above" if self.config.trigger_above else "falls below"
        self.logger().info(f"Will open position when price {condition} target")
        self.logger().info(f"Position width: ±{self.config.position_width_pct}%")
        self.logger().info(f"Will close position if price is outside range by {self.config.out_of_range_pct}% for {self.config.out_of_range_secs} seconds")

        # Check Gateway status
        safe_ensure_future(self.check_gateway_and_fetch_pool_info())

    async def check_gateway_and_fetch_pool_info(self):
        """Check if Gateway server is online and fetch pool information"""
        self.logger().info("Checking Gateway server status...")
        try:
            gateway = GatewayHttpClient.get_instance()
            if await gateway.ping_gateway():
                self.logger().info("Gateway server is online!")
                # Fetch pool info to get token information
                await self.fetch_pool_info()
            else:
                self.logger().error("Gateway server is offline! Make sure Gateway is running before using this strategy.")
        except Exception as e:
            self.logger().error(f"Error connecting to Gateway server: {str(e)}")

    async def fetch_pool_info(self):
        """Fetch pool information to get tokens and current price"""
        try:
            self.logger().info(f"Fetching information for pool {self.config.pool_address}...")
            pool_info = await GatewayHttpClient.get_instance().connector_request(
                "get",
                self.config.connector,
                "pool-info",
                {"network": self.config.network, "poolAddress": self.config.pool_address}
            )

            if not pool_info:
                self.logger().error(f"Failed to get pool information for {self.config.pool_address}")
                return

            self.pool_info = pool_info

            # Extract token information (first time only)
            if self.base_token is None and "baseToken" in pool_info:
                self.base_token = pool_info["baseToken"]
            if self.quote_token is None and "quoteToken" in pool_info:
                self.quote_token = pool_info["quoteToken"]

            # Extract current price - it's at the top level of the response
            if "price" in pool_info:
                try:
                    self.last_price = Decimal(str(pool_info["price"]))
                except (ValueError, TypeError) as e:
                    self.logger().error(f"Error converting price value: {e}")
            else:
                self.logger().error("No price found in pool info response")

        except Exception as e:
            self.logger().error(f"Error fetching pool info: {str(e)}")

    async def fetch_position_info_after_fill(self):
        """Fetch position info after LP order is filled - mimics official lp_manage_position.py"""
        try:
            # Wait for the position to be fully created on-chain
            self.logger().info("开仓成功，获取仓位信息...")
            await asyncio.sleep(2)

            # Use the connector's get_user_positions method to fetch all positions for this pool
            positions = await self.gateway_lp.get_user_positions(pool_address=self.config.pool_address)

            if positions and len(positions) > 0:
                # Get the most recent position (last in the list)
                latest_position = positions[-1]
                self.position_address = latest_position.address
                self.position_lower_price = float(latest_position.lower_price)
                self.position_upper_price = float(latest_position.upper_price)

                # Calculate initial position value for P&L tracking
                if self.last_price:
                    base_amount = Decimal(str(latest_position.base_token_amount))
                    quote_amount = Decimal(str(latest_position.quote_token_amount))
                    self.initial_position_value = base_amount * self.last_price + quote_amount

                self.logger().info(f"成功获取CLMM仓位: {self.position_address}")
                self.logger().info(f"仓位价格区间: {self.position_lower_price} - {self.position_upper_price}")
                self.logger().info(f"初始仓位价值: {self.initial_position_value:.2f} {self.quote_token}")
            else:
                self.logger().warning("未发现现有仓位")

        except Exception as e:
            self.logger().error(f"获取仓位信息失败: {str(e)}")

    async def fetch_position_info(self):
        """Fetch actual position information including price bounds"""
        if not self.position_address:
            return

        try:
            self.logger().info(f"Fetching position info for {self.position_address}...")
            position_info = await GatewayHttpClient.get_instance().connector_request(
                "get",
                self.config.connector,
                "position-info",
                {
                    "network": self.config.network,
                    "positionAddress": self.position_address,
                    "walletAddress": self.gateway_lp.address  # Use the gateway connector's address
                }
            )

            if not position_info:
                self.logger().error(f"Failed to get position information for {self.position_address}")
                return

            # Extract actual position price bounds
            if "lowerPrice" in position_info and "upperPrice" in position_info:
                self.position_lower_price = float(position_info["lowerPrice"])
                self.position_upper_price = float(position_info["upperPrice"])
                self.logger().info(f"Position actual bounds: {self.position_lower_price} to {self.position_upper_price}")
            else:
                self.logger().error("Position info missing price bounds")

        except Exception as e:
            self.logger().error(f"Error fetching position info: {str(e)}")

    def on_tick(self):
        """Check price and position status on each tick with interval control"""
        current_time = time.time()

        # For positions not opened yet, check less frequently
        if not self.position_opened and not self.position_opening:
            # Use check_interval for price monitoring
            if self.last_monitor_time is None or (current_time - self.last_monitor_time) >= self.config.check_interval:
                self.last_monitor_time = current_time
                safe_ensure_future(self.check_price_and_open_position())

        # For opened positions, monitor more frequently but still with interval
        elif self.position_opened and not self.position_closing:
            # Monitor position at check_interval pace
            if self.last_monitor_time is None or (current_time - self.last_monitor_time) >= self.config.check_interval:
                self.last_monitor_time = current_time
                safe_ensure_future(self.monitor_position())

    async def check_price_and_open_position(self):
        """Check current price and open position if target is reached"""
        if self.position_opening or self.position_opened:
            return

        self.position_opening = True

        try:
            # Fetch current pool info to get the latest price
            await self.fetch_pool_info()

            if not self.last_price:
                self.logger().warning("Unable to get current price")
                self.position_opening = False
                return

            # Check if price condition is met
            condition_met = False
            if self.config.trigger_above and self.last_price > self.config.target_price:
                condition_met = True
                self.logger().info(f"Price rose above target: {self.last_price} > {self.config.target_price}")
            elif not self.config.trigger_above and self.last_price < self.config.target_price:
                condition_met = True
                self.logger().info(f"Price fell below target: {self.last_price} < {self.config.target_price}")

            if condition_met:
                self.logger().info("Price condition met! Opening position...")
                self.position_opening = False  # Reset flag so open_position can set it
                await self.open_position()
            else:
                self.logger().info(f"Current price: {self.last_price}, Target: {self.config.target_price}, "
                                   f"Condition not met yet.")
                self.position_opening = False

        except Exception as e:
            self.logger().error(f"Error in check_price_and_open_position: {str(e)}")
            self.position_opening = False

    async def open_position(self):
        """Open a concentrated liquidity position around the target price"""
        if self.position_opening or self.position_opened:
            return

        self.position_opening = True

        try:
            # Get the latest pool price before creating the position
            await self.fetch_pool_info()

            if not self.last_price:
                self.logger().error("Cannot open position: Failed to get current pool price")
                self.position_opening = False
                return

            # Use the gateway LP connector to add liquidity (new API)
            self.logger().info(f"Opening position on pool {self.config.pool_address} around price {self.last_price} with width ±{self.config.position_width_pct}%")

            # Use add_liquidity method (replaces open_position)
            # For CLMM, we need to provide upper_width_pct and lower_width_pct
            order_id = self.gateway_lp.add_liquidity(
                trading_pair="",  # Empty string since we're using pool_address directly
                price=float(self.last_price),
                upper_width_pct=float(self.config.position_width_pct),
                lower_width_pct=float(self.config.position_width_pct),
                base_token_amount=float(self.config.base_token_amount),
                quote_token_amount=float(self.config.quote_token_amount),
                pool_address=self.config.pool_address  # Pass the pool address
            )

            self.logger().info(f"Position opening order submitted: {order_id}")

            # Store order ID to track when it's filled
            self.opening_order_id = order_id

        except Exception as e:
            self.logger().error(f"Error opening position: {str(e)}")
            self.position_opening = False

    async def monitor_position(self):
        """Monitor the position with anti-fake-breakout logic"""
        if not self.position_address or self.position_closing:
            return

        try:
            # Fetch current pool info to get the latest price
            await self.fetch_pool_info()

            if not self.last_price:
                return

            current_time = time.time()
            current_price = float(self.last_price)

            # Calculate bounds with buffer
            lower_bound_with_buffer = self.position_lower_price * (1 - float(self.config.out_of_range_pct) / 100.0)
            upper_bound_with_buffer = self.position_upper_price * (1 + float(self.config.out_of_range_pct) / 100.0)

            # Determine if out of range and calculate distance
            out_of_range = False
            distance_from_range_pct = 0.0

            if current_price < lower_bound_with_buffer:
                out_of_range = True
                distance_from_range_pct = (lower_bound_with_buffer - current_price) / lower_bound_with_buffer * 100
                direction = "below"
            elif current_price > upper_bound_with_buffer:
                out_of_range = True
                distance_from_range_pct = (current_price - upper_bound_with_buffer) / upper_bound_with_buffer * 100
                direction = "above"

            # Handle out-of-range situation
            if out_of_range:
                # Track maximum distance from range for anti-fake-breakout
                if distance_from_range_pct > self.max_out_of_range_distance:
                    self.max_out_of_range_distance = distance_from_range_pct

                # Start timer if this is first time out of range
                if self.out_of_range_start_time is None:
                    self.out_of_range_start_time = current_time
                    self.logger().info(f"⚠️  价格突破范围！方向: {direction}, 距离: {distance_from_range_pct:.2f}%, 开始计时...")
                else:
                    # Check for price return (anti-fake-breakout)
                    price_return_pct = self.max_out_of_range_distance - distance_from_range_pct

                    if price_return_pct >= float(self.config.price_return_threshold_pct):
                        # Price returned significantly, might be a fake breakout - reset timer
                        self.logger().info(f"🔄 价格回归 {price_return_pct:.2f}% (从最大偏离 {self.max_out_of_range_distance:.2f}% 到 {distance_from_range_pct:.2f}%), 重置止损计时器")
                        self.out_of_range_start_time = current_time
                        self.max_out_of_range_distance = distance_from_range_pct

                # Check if enough time has passed to close
                elapsed_seconds = current_time - self.out_of_range_start_time

                if elapsed_seconds >= self.config.out_of_range_secs:
                    self.logger().info(f"⏰ 价格已超出范围 {elapsed_seconds:.0f}秒 (阈值: {self.config.out_of_range_secs}秒), 最大偏离: {self.max_out_of_range_distance:.2f}%")
                    self.logger().info("💼 执行止损，关闭仓位...")
                    await self.close_position()
                else:
                    remaining = self.config.out_of_range_secs - elapsed_seconds
                    self.logger().info(f"⏳ 价格超出范围: {direction} {distance_from_range_pct:.2f}%, 已计时 {elapsed_seconds:.0f}s, 还需 {remaining:.0f}s")

            else:
                # Price is back in range
                if self.out_of_range_start_time is not None:
                    elapsed = current_time - self.out_of_range_start_time
                    self.logger().info(f"✅ 价格回归范围内 (曾超出 {elapsed:.0f}秒), 重置计时器")
                    self.out_of_range_start_time = None
                    self.max_out_of_range_distance = 0.0

                # Calculate distance to boundaries for display
                distance_to_lower = (current_price - self.position_lower_price) / self.position_lower_price * 100
                distance_to_upper = (self.position_upper_price - current_price) / current_price * 100

                self.logger().info(f"✓ 价格在范围内: {current_price:.6f}, 距下界 +{distance_to_lower:.2f}%, 距上界 +{distance_to_upper:.2f}%")

        except Exception as e:
            self.logger().error(f"Error monitoring position: {str(e)}")

    async def close_position(self):
        """Close the concentrated liquidity position"""
        if not self.position_address or self.position_closing:
            return

        self.position_closing = True

        try:
            self.logger().info(f"Closing position {self.position_address}...")

            # Use remove_liquidity method (replaces close_position)
            order_id = self.gateway_lp.remove_liquidity(
                trading_pair="",  # Empty string since we're using position_address directly
                position_address=self.position_address,
                percentage=100.0  # Remove 100% of the liquidity
            )

            self.logger().info(f"Position closing order submitted: {order_id}")

            # Store order ID to track when it's closed
            self.closing_order_id = order_id

        except Exception as e:
            self.logger().error(f"Error closing position: {str(e)}")
            self.position_closing = False

    def did_fill_order(self, event):
        """
        Called when an order is filled.
        """
        if hasattr(self, 'opening_order_id') and event.order_id == self.opening_order_id:
            self.logger().info(f"准备开仓，当前价格: {self.last_price}")
            self.position_opened = True
            self.position_opening = False

            # Log fill details if available
            if hasattr(event, 'amount'):
                self.logger().info(f"订单已成交，数量: {event.amount}")

            # Fetch position info after the order is filled - using official approach
            safe_ensure_future(self.fetch_position_info_after_fill())

        elif hasattr(self, 'closing_order_id') and event.order_id == self.closing_order_id:
            self.logger().info(f"Position closed successfully! Order {event.order_id} filled.")
            # Reset position state
            self.position_opened = False
            self.position_closing = False
            self.position_address = None
            self.position_lower_price = None
            self.position_upper_price = None
            self.out_of_range_start_time = None
            self.max_out_of_range_distance = 0.0
            self.initial_position_value = None

    def did_fail_order(self, event):
        """
        Called when an order fails.
        """
        if hasattr(self, 'opening_order_id') and event.order_id == self.opening_order_id:
            self.logger().error(f"Failed to open position! Order {event.order_id} failed.")
            self.position_opening = False
        elif hasattr(self, 'closing_order_id') and event.order_id == self.closing_order_id:
            self.logger().error(f"Failed to close position! Order {event.order_id} failed.")
            self.position_closing = False

    def format_status(self) -> str:
        """Format status message with enhanced information"""
        lines = []
        connector_network = f"{self.config.connector}_{self.config.network}"

        # Display wallet balances if tokens are known
        if self.base_token and self.quote_token:
            lines.append("=== 钱包余额 ===")
            try:
                base_balance = self.gateway_lp.get_balance(self.base_token)
                quote_balance = self.gateway_lp.get_balance(self.quote_token)
                lines.append(f"{self.base_token}: {base_balance:.6f}")
                lines.append(f"{self.quote_token}: {quote_balance:.6f}")
            except Exception as e:
                lines.append(f"无法获取余额: {str(e)}")
            lines.append("")

        if self.position_opened:
            lines.append(f"=== 仓位状态: 已开仓 ===")
            lines.append(f"网络: {connector_network}")
            lines.append(f"仓位地址: {self.position_address[:8]}...{self.position_address[-6:]}")

            if self.position_lower_price and self.position_upper_price:
                lines.append(f"\n价格区间: {self.position_lower_price:.6f} - {self.position_upper_price:.6f}")

                # Calculate and display buffer zone
                if self.config.out_of_range_pct > 0:
                    lower_buffer = self.position_lower_price * (1 - float(self.config.out_of_range_pct) / 100.0)
                    upper_buffer = self.position_upper_price * (1 + float(self.config.out_of_range_pct) / 100.0)
                    lines.append(f"止损区间: {lower_buffer:.6f} - {upper_buffer:.6f} (±{self.config.out_of_range_pct}%)")

            if self.last_price:
                current_price = float(self.last_price)
                lines.append(f"\n当前价格: {current_price:.6f}")

                # Calculate distances to boundaries
                if self.position_lower_price and self.position_upper_price:
                    distance_to_lower = (current_price - self.position_lower_price) / self.position_lower_price * 100
                    distance_to_upper = (self.position_upper_price - current_price) / current_price * 100

                    # Determine position status
                    if current_price >= self.position_lower_price and current_price <= self.position_upper_price:
                        status_icon = "✅"
                        status_text = "在范围内"
                    else:
                        status_icon = "⚠️ "
                        status_text = "超出范围"

                    lines.append(f"状态: {status_icon} {status_text}")
                    lines.append(f"距下界: {distance_to_lower:+.2f}%")
                    lines.append(f"距上界: {distance_to_upper:+.2f}%")

                # Show P&L if we have position info
                try:
                    positions = self.gateway_lp._account_positions.get(self.config.pool_address, [])
                    if positions:
                        for pos in positions:
                            if pos.address == self.position_address:
                                base_amount = Decimal(str(pos.base_token_amount))
                                quote_amount = Decimal(str(pos.quote_token_amount))
                                current_value = base_amount * Decimal(str(current_price)) + quote_amount

                                if self.initial_position_value:
                                    pnl = current_value - self.initial_position_value
                                    pnl_pct = (pnl / self.initial_position_value) * 100
                                    pnl_icon = "📈" if pnl > 0 else "📉"
                                    lines.append(f"\n{pnl_icon} 未实现盈亏: {pnl:+.4f} {self.quote_token} ({pnl_pct:+.2f}%)")
                                    lines.append(f"初始价值: {self.initial_position_value:.4f} {self.quote_token}")
                                    lines.append(f"当前价值: {current_value:.4f} {self.quote_token}")
                                break
                except Exception as e:
                    pass  # Silently skip P&L if not available

            # Show out-of-range timer status
            if self.out_of_range_start_time:
                elapsed = time.time() - self.out_of_range_start_time
                remaining = self.config.out_of_range_secs - elapsed
                progress = elapsed / self.config.out_of_range_secs * 100

                lines.append(f"\n⏰ 止损倒计时:")
                lines.append(f"已计时: {elapsed:.0f}s / {self.config.out_of_range_secs}s ({progress:.0f}%)")
                lines.append(f"剩余: {remaining:.0f}s")
                if self.max_out_of_range_distance > 0:
                    lines.append(f"最大偏离: {self.max_out_of_range_distance:.2f}%")

                if remaining <= 60:
                    lines.append("⚠️  即将止损!")

        elif self.position_opening:
            lines.append(f"=== 仓位状态: 开仓中 ===")
            lines.append(f"网络: {connector_network}")
            lines.append("⏳ 等待交易确认...")

        elif self.position_closing:
            lines.append(f"=== 仓位状态: 平仓中 ===")
            lines.append(f"网络: {connector_network}")
            lines.append("⏳ 等待平仓交易确认...")

        else:
            lines.append(f"=== 仓位状态: 监控中 ===")
            lines.append(f"网络: {connector_network}")
            lines.append(f"池地址: {self.config.pool_address[:8]}...{self.config.pool_address[-6:]}")

            if self.last_price:
                lines.append(f"\n当前价格: {self.last_price:.6f}")
                lines.append(f"目标价格: {self.config.target_price:.6f}")

                price_diff = float(self.last_price) - float(self.config.target_price)
                price_diff_pct = (price_diff / float(self.config.target_price)) * 100

                condition = "突破上涨至" if self.config.trigger_above else "跌破下跌至"
                lines.append(f"触发条件: 价格{condition} {self.config.target_price:.6f}")
                lines.append(f"价格差距: {price_diff:+.6f} ({price_diff_pct:+.2f}%)")

                # Show if condition is met
                if (self.config.trigger_above and self.last_price > self.config.target_price) or \
                   (not self.config.trigger_above and self.last_price < self.config.target_price):
                    lines.append("✅ 条件已满足，准备开仓")
                else:
                    lines.append("⏳ 等待价格条件触发")

            # Show next check time
            if self.last_monitor_time:
                next_check = self.config.check_interval - (time.time() - self.last_monitor_time)
                if next_check > 0:
                    lines.append(f"\n下次检查: {next_check:.0f}秒后")

        lines.append(f"\n检查间隔: {self.config.check_interval}秒")

        return "\n".join(lines)