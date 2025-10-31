"""
CEX-DEX LP 套利策略

核心原理:
- 在 DEX 上被动做 LP Maker
- 在 CEX 上主动做 Taker 对冲
- 赚取价差 + LP 手续费

工作流程:
1. 监控 CEX 价格和 DEX 池子
2. 发现套利机会 → 在 DEX 开 LP 仓位
3. 等待 LP 被成交（被动）
4. LP 成交后 → 立即在 CEX 对冲
5. 计算利润并记录

参考:
- lp_manage_position.py (LP 管理)
- arbitrage_controller.py (套利逻辑)
- amm_trade_example.py (DEX 交易)
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional, Tuple, Union

from pydantic import Field

from hummingbot.client.config.config_data_types import BaseClientModel
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.gateway.common_types import ConnectorType, get_connector_type
from hummingbot.connector.gateway.gateway_lp import AMMPositionInfo, CLMMPositionInfo
from hummingbot.core.data_type.common import OrderType
from hummingbot.core.event.events import BuyOrderCompletedEvent, OrderFilledEvent, SellOrderCompletedEvent
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


# ========================================
# 配置类
# ========================================

class CexDexLpArbitrageConfig(BaseClientModel):
    """CEX-DEX LP 套利配置"""

    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))

    # ========== 交易所配置 ==========
    cex_exchange: str = Field(
        "binance",
        json_schema_extra={"prompt": "CEX 交易所名称（用于对冲）", "prompt_on_new": True}
    )

    dex_exchange: str = Field(
        "pancakeswap/clmm",
        json_schema_extra={"prompt": "DEX 交易所（格式: name/type，如 uniswap/clmm）", "prompt_on_new": True}
    )

    trading_pair: str = Field(
        "WETH-USDC",
        json_schema_extra={"prompt": "交易对", "prompt_on_new": True}
    )

    # ========== LP 配置 ==========
    lp_token_amount: Decimal = Field(
        Decimal("0.1"),
        json_schema_extra={"prompt": "LP 单边 Token 数量", "prompt_on_new": True}
    )

    lp_spread_pct: Decimal = Field(
        Decimal("0.01"),
        json_schema_extra={"prompt": "LP 价格区间宽度（小数，如 0.01 = 1%）", "prompt_on_new": True}
    )

    lp_timeout_seconds: int = Field(
        300,
        json_schema_extra={"prompt": "LP 最长持有时间（秒）", "prompt_on_new": True}
    )

    # ========== 盈利目标 ==========
    target_profitability: Decimal = Field(
        Decimal("0.02"),
        json_schema_extra={"prompt": "目标利润率（小数，如 0.02 = 2%）", "prompt_on_new": True}
    )

    min_profitability: Decimal = Field(
        Decimal("0.005"),
        json_schema_extra={"prompt": "最低利润率（小数，止损线）", "prompt_on_new": True}
    )

    # ========== 费用估算 ==========
    cex_taker_fee_pct: Decimal = Field(
        Decimal("0.001"),
        json_schema_extra={"prompt": "CEX Taker 手续费率（小数）", "prompt_on_new": False}
    )

    dex_lp_fee_pct: Decimal = Field(
        Decimal("0.003"),
        json_schema_extra={"prompt": "DEX LP 手续费率（小数，这是收入）", "prompt_on_new": False}
    )

    gas_cost_quote: Decimal = Field(
        Decimal("5"),
        json_schema_extra={"prompt": "预估 Gas 成本（Quote Token 单位）", "prompt_on_new": False}
    )

    # ========== 策略配置 ==========
    enable_sell_side: bool = Field(
        True,
        json_schema_extra={"prompt": "启用卖方套利（DEX LP 卖出，CEX 买入）", "prompt_on_new": False}
    )

    enable_buy_side: bool = Field(
        False,
        json_schema_extra={"prompt": "启用买方套利（DEX LP 买入，CEX 卖出）", "prompt_on_new": False}
    )

    check_interval_seconds: int = Field(
        10,
        json_schema_extra={"prompt": "检查间隔（秒）", "prompt_on_new": False}
    )


# ========================================
# 盈利计算器
# ========================================

class ProfitabilityCalculator:
    """盈利计算器"""

    def __init__(self, config: CexDexLpArbitrageConfig, logger):
        self.config = config
        self.logger = logger

    def estimate_total_fees_pct(self, trade_value: Decimal) -> Decimal:
        """
        估算总费用百分比

        包括:
        - CEX Taker 手续费
        - Gas 成本（转为百分比）
        - 滑点预留
        """
        # CEX 手续费
        cex_fee_pct = self.config.cex_taker_fee_pct

        # Gas 成本转为百分比
        gas_pct = self.config.gas_cost_quote / trade_value if trade_value > 0 else Decimal("0.01")

        # 滑点预留（1%）
        slippage_pct = Decimal("0.01")

        total_fees_pct = cex_fee_pct + gas_pct + slippage_pct

        return total_fees_pct

    def calculate_target_lp_price(
        self,
        cex_price: Decimal,
        is_sell_side: bool,
        trade_value: Decimal
    ) -> Decimal:
        """
        计算目标 LP 价格（开仓线）

        Args:
            cex_price: CEX 参考价格
            is_sell_side: 是否卖方套利
            trade_value: 交易价值（用于计算 gas 占比）

        Returns:
            目标 LP 价格
        """
        total_fees_pct = self.estimate_total_fees_pct(trade_value)

        # 减去 LP 手续费收入
        net_cost_pct = self.config.target_profitability + total_fees_pct - self.config.dex_lp_fee_pct

        if is_sell_side:
            # 卖方: LP 卖价 > CEX 买价 * (1 + 成本)
            target_price = cex_price * (Decimal("1") + net_cost_pct)
        else:
            # 买方: LP 买价 < CEX 卖价 * (1 - 成本)
            target_price = cex_price * (Decimal("1") - net_cost_pct)

        return target_price

    def calculate_min_lp_price(
        self,
        cex_price: Decimal,
        is_sell_side: bool,
        trade_value: Decimal
    ) -> Decimal:
        """
        计算最低 LP 价格（止损线）

        使用 min_profitability 而不是 target_profitability
        """
        total_fees_pct = self.estimate_total_fees_pct(trade_value)
        net_cost_pct = self.config.min_profitability + total_fees_pct - self.config.dex_lp_fee_pct

        if is_sell_side:
            min_price = cex_price * (Decimal("1") + net_cost_pct)
        else:
            min_price = cex_price * (Decimal("1") - net_cost_pct)

        return min_price


# ========================================
# LP 仓位管理器
# ========================================

class LpPositionManager:
    """LP 仓位管理器"""

    def __init__(
        self,
        strategy: ScriptStrategyBase,
        dex_connector: ConnectorBase,
        config: CexDexLpArbitrageConfig
    ):
        self.strategy = strategy
        self.dex_connector = dex_connector
        self.config = config
        self.connector_type = get_connector_type(config.dex_exchange)

        # 仓位信息
        self.position_info: Union[CLMMPositionInfo, AMMPositionInfo, None] = None
        self.position_opening = False
        self.position_closing = False
        self.open_order_id: Optional[str] = None
        self.close_order_id: Optional[str] = None

    def logger(self):
        return self.strategy.logger()

    async def open_lp_position(
        self,
        is_sell_side: bool,
        price_range: Tuple[Decimal, Decimal],
        token_amount: Decimal
    ) -> str:
        """
        开 LP 仓位

        Args:
            is_sell_side: True = 卖方（放入 base token），False = 买方（放入 quote token）
            price_range: (lower_bound, upper_bound)
            token_amount: Token 数量

        Returns:
            订单 ID
        """
        if self.position_opening or self.position_info:
            raise Exception("LP 仓位已存在或正在开仓")

        self.position_opening = True
        lower_bound, upper_bound = price_range
        center_price = (lower_bound + upper_bound) / 2

        try:
            # 计算 LP 参数
            if self.connector_type == ConnectorType.CLMM:
                # CLMM: 使用 width_pct
                spread_pct = float((upper_bound - center_price) / center_price)

                if is_sell_side:
                    # 卖方: 只放 base token
                    base_amount = float(token_amount)
                    quote_amount = 0.0
                else:
                    # 买方: 只放 quote token
                    base_amount = 0.0
                    quote_amount = float(token_amount)

                self.logger().info(
                    f"开 CLMM LP 仓位:\n"
                    f"   方向: {'SELL' if is_sell_side else 'BUY'}\n"
                    f"   中心价: {center_price}\n"
                    f"   价格区间: {lower_bound} - {upper_bound}\n"
                    f"   Base: {base_amount}, Quote: {quote_amount}"
                )

                order_id = self.dex_connector.add_liquidity(
                    trading_pair=self.config.trading_pair,
                    price=float(center_price),
                    upper_width_pct=spread_pct,
                    lower_width_pct=spread_pct,
                    base_token_amount=base_amount,
                    quote_token_amount=quote_amount
                )
            else:
                # AMM: 使用固定比例
                if is_sell_side:
                    base_amount = float(token_amount)
                    quote_amount = 0.0
                else:
                    base_amount = 0.0
                    quote_amount = float(token_amount)

                self.logger().info(
                    f"开 AMM LP 仓位:\n"
                    f"   方向: {'SELL' if is_sell_side else 'BUY'}\n"
                    f"   价格: {center_price}\n"
                    f"   Base: {base_amount}, Quote: {quote_amount}"
                )

                order_id = self.dex_connector.add_liquidity(
                    trading_pair=self.config.trading_pair,
                    price=float(center_price),
                    base_token_amount=base_amount,
                    quote_token_amount=quote_amount
                )

            self.open_order_id = order_id
            return order_id

        except Exception as e:
            self.logger().error(f"开 LP 仓位失败: {e}")
            self.position_opening = False
            raise

    async def close_lp_position(self) -> Optional[str]:
        """关闭 LP 仓位"""
        if not self.position_info or self.position_closing:
            return None

        self.position_closing = True

        try:
            if isinstance(self.position_info, CLMMPositionInfo):
                self.logger().info(f"关闭 CLMM LP: {self.position_info.address}")
                order_id = self.dex_connector.remove_liquidity(
                    trading_pair=self.config.trading_pair,
                    position_address=self.position_info.address
                )
            else:
                # AMM: 不需要 position_address
                self.logger().info("关闭 AMM LP")
                order_id = self.dex_connector.remove_liquidity(
                    trading_pair=self.config.trading_pair
                )

            self.close_order_id = order_id
            return order_id

        except Exception as e:
            self.logger().error(f"关闭 LP 仓位失败: {e}")
            self.position_closing = False
            raise

    async def update_position_info(self):
        """更新仓位信息"""
        if not self.position_info:
            return

        try:
            if isinstance(self.position_info, CLMMPositionInfo):
                self.position_info = await self.dex_connector.get_position_info(
                    trading_pair=self.config.trading_pair,
                    position_address=self.position_info.address
                )
            else:
                # AMM: 使用池子地址
                pool_address = await self.dex_connector.get_pool_address(self.config.trading_pair)
                if pool_address:
                    self.position_info = await self.dex_connector.get_position_info(
                        trading_pair=self.config.trading_pair,
                        position_address=pool_address
                    )
        except Exception as e:
            self.logger().error(f"更新仓位信息失败: {e}")


# ========================================
# CEX 对冲执行器
# ========================================

class CexHedgeExecutor:
    """CEX 对冲执行器"""

    def __init__(
        self,
        strategy: ScriptStrategyBase,
        cex_connector: ConnectorBase,
        config: CexDexLpArbitrageConfig
    ):
        self.strategy = strategy
        self.cex_connector = cex_connector
        self.config = config
        self.pending_order_id: Optional[str] = None

    def logger(self):
        return self.strategy.logger()

    async def hedge_lp_fill(
        self,
        is_buy: bool,
        amount: Decimal,
        price_limit: Decimal
    ) -> str:
        """
        执行 CEX 对冲

        Args:
            is_buy: True = 买入，False = 卖出
            amount: 数量
            price_limit: 限价

        Returns:
            订单 ID
        """
        try:
            # 量化参数
            amount_quantized = self.cex_connector.quantize_order_amount(
                self.config.trading_pair,
                amount
            )

            price_quantized = self.cex_connector.quantize_order_price(
                self.config.trading_pair,
                price_limit
            )

            self.logger().info(
                f"CEX 对冲:\n"
                f"   方向: {'BUY' if is_buy else 'SELL'}\n"
                f"   数量: {amount_quantized}\n"
                f"   限价: {price_quantized}"
            )

            # 下市价单（使用限价保护）
            if is_buy:
                order_id = self.strategy.buy(
                    connector_name=self.config.cex_exchange,
                    trading_pair=self.config.trading_pair,
                    amount=amount_quantized,
                    order_type=OrderType.MARKET,
                    price=price_quantized
                )
            else:
                order_id = self.strategy.sell(
                    connector_name=self.config.cex_exchange,
                    trading_pair=self.config.trading_pair,
                    amount=amount_quantized,
                    order_type=OrderType.MARKET,
                    price=price_quantized
                )

            self.pending_order_id = order_id
            return order_id

        except Exception as e:
            self.logger().error(f"CEX 对冲失败: {e}")
            raise


# ========================================
# 主策略
# ========================================

class CexDexLpArbitrageStrategy(ScriptStrategyBase):
    """CEX-DEX LP 套利策略"""

    @classmethod
    def init_markets(cls, config: CexDexLpArbitrageConfig):
        cls.markets = {
            config.cex_exchange: {config.trading_pair},
            config.dex_exchange: {config.trading_pair}
        }

    def __init__(self, connectors: Dict[str, ConnectorBase], config: CexDexLpArbitrageConfig):
        super().__init__(connectors)
        self.config = config

        # 连接器
        self.cex_connector = connectors[config.cex_exchange]
        self.dex_connector = connectors[config.dex_exchange]

        # Token 名称
        self.base_token, self.quote_token = config.trading_pair.split("-")

        # 初始化模块
        self.profit_calculator = ProfitabilityCalculator(config, self.logger())
        self.lp_manager = LpPositionManager(self, self.dex_connector, config)
        self.hedge_executor = CexHedgeExecutor(self, self.cex_connector, config)

        # 状态跟踪
        self.lp_position_opened = False
        self.lp_position_info: Optional[dict] = None  # 自定义信息
        self.last_check_time = None

        # 统计
        self.stats = {
            "total_profit": Decimal("0"),
            "completed_cycles": 0,
            "lp_open_failures": 0,
            "hedge_failures": 0,
        }

        # 启动信息
        self.log_with_clock(
            logging.INFO,
            f"CEX-DEX LP 套利策略启动:\n"
            f"   CEX: {config.cex_exchange}\n"
            f"   DEX: {config.dex_exchange}\n"
            f"   交易对: {config.trading_pair}\n"
            f"   LP 数量: {config.lp_token_amount} {self.base_token}\n"
            f"   目标利润: {config.target_profitability * 100:.2f}%\n"
            f"   最低利润: {config.min_profitability * 100:.2f}%\n"
            f"   卖方套利: {'启用' if config.enable_sell_side else '禁用'}\n"
            f"   买方套利: {'启用' if config.enable_buy_side else '禁用'}"
        )

    # ========================================
    # 主循环
    # ========================================

    def on_tick(self):
        """策略主循环"""
        current_time = datetime.now()

        # 检查间隔
        if self.last_check_time and (current_time - self.last_check_time).total_seconds() < self.config.check_interval_seconds:
            return

        self.last_check_time = current_time

        # 根据 LP 状态执行不同逻辑
        if self.lp_position_opened:
            # 持仓监控
            safe_ensure_future(self._monitor_lp_position())
        else:
            # 寻找开仓机会
            safe_ensure_future(self._check_opening_opportunity())

    # ========================================
    # 开仓逻辑
    # ========================================

    async def _check_opening_opportunity(self):
        """检查开仓机会"""
        try:
            # 获取市场数据
            cex_best_ask = await self._get_cex_best_ask()
            cex_best_bid = await self._get_cex_best_bid()
            dex_price = await self._get_dex_price()

            self.logger().info(
                f"市场数据:\n"
                f"   CEX 买价: {cex_best_ask}\n"
                f"   CEX 卖价: {cex_best_bid}\n"
                f"   DEX 价格: {dex_price}"
            )

            if not all([cex_best_ask, cex_best_bid, dex_price]):
                self.logger().warning("无法获取市场数据")
                return

            trade_value = self.config.lp_token_amount * cex_best_ask

            # 检查卖方机会
            if self.config.enable_sell_side:
                target_price = self.profit_calculator.calculate_target_lp_price(
                    cex_price=cex_best_ask,
                    is_sell_side=True,
                    trade_value=trade_value
                )

                if dex_price >= target_price:
                    self.logger().info(
                        f"发现卖方套利机会:\n"
                        f"   DEX 价格: {dex_price}\n"
                        f"   目标价格: {target_price}\n"
                        f"   CEX 买价: {cex_best_ask}"
                    )
                    await self._open_sell_side_position(target_price, cex_best_ask)
                    return

            # 检查买方机会
            if self.config.enable_buy_side:
                target_price = self.profit_calculator.calculate_target_lp_price(
                    cex_price=cex_best_bid,
                    is_sell_side=False,
                    trade_value=trade_value
                )

                if dex_price <= target_price:
                    self.logger().info(
                        f"发现买方套利机会:\n"
                        f"   DEX 价格: {dex_price}\n"
                        f"   目标价格: {target_price}\n"
                        f"   CEX 卖价: {cex_best_bid}"
                    )
                    await self._open_buy_side_position(target_price, cex_best_bid)
                    return

        except Exception as e:
            self.logger().error(f"检查开仓机会失败: {e}")

    async def _open_sell_side_position(self, target_price: Decimal, cex_price: Decimal):
        """开卖方 LP 仓位"""
        # 计算 LP 区间
        lower_bound = target_price
        upper_bound = target_price * (Decimal("1") + self.config.lp_spread_pct)

        try:
            order_id = await self.lp_manager.open_lp_position(
                is_sell_side=True,
                price_range=(lower_bound, upper_bound),
                token_amount=self.config.lp_token_amount
            )

            # 记录信息
            self.lp_position_opened = True
            self.lp_position_info = {
                "side": "SELL",
                "order_id": order_id,
                "price_range": (lower_bound, upper_bound),
                "token_amount": self.config.lp_token_amount,
                "open_time": time.time(),
                "open_cex_price": cex_price,
            }

        except Exception as e:
            self.logger().error(f"开卖方仓位失败: {e}")
            self.stats["lp_open_failures"] += 1

    async def _open_buy_side_position(self, target_price: Decimal, cex_price: Decimal):
        """开买方 LP 仓位"""
        # 计算 LP 区间
        upper_bound = target_price
        lower_bound = target_price * (Decimal("1") - self.config.lp_spread_pct)

        try:
            order_id = await self.lp_manager.open_lp_position(
                is_sell_side=False,
                price_range=(lower_bound, upper_bound),
                token_amount=self.config.lp_token_amount
            )

            self.lp_position_opened = True
            self.lp_position_info = {
                "side": "BUY",
                "order_id": order_id,
                "price_range": (lower_bound, upper_bound),
                "token_amount": self.config.lp_token_amount,
                "open_time": time.time(),
                "open_cex_price": cex_price,
            }

        except Exception as e:
            self.logger().error(f"开买方仓位失败: {e}")
            self.stats["lp_open_failures"] += 1

    # ========================================
    # 持仓监控
    # ========================================

    async def _monitor_lp_position(self):
        """监控 LP 仓位"""
        if not self.lp_position_info:
            return

        try:
            # 更新 LP 仓位信息
            await self.lp_manager.update_position_info()

            # 获取当前 CEX 价格
            side = self.lp_position_info["side"]

            if side == "SELL":
                current_cex_price = await self._get_cex_best_ask()
            else:
                current_cex_price = await self._get_cex_best_bid()

            if not current_cex_price:
                return

            # 计算止损价格
            lower_bound, upper_bound = self.lp_position_info["price_range"]
            avg_lp_price = (lower_bound + upper_bound) / 2
            trade_value = self.config.lp_token_amount * current_cex_price

            cutoff_price = self.profit_calculator.calculate_min_lp_price(
                cex_price=current_cex_price,
                is_sell_side=(side == "SELL"),
                trade_value=trade_value
            )

            # 检查止损
            if side == "SELL":
                if avg_lp_price < cutoff_price:
                    self.logger().warning(
                        f"触发止损:\n"
                        f"   LP 均价: {avg_lp_price}\n"
                        f"   止损线: {cutoff_price}\n"
                        f"   CEX 价格: {current_cex_price}"
                    )
                    await self._close_lp_position_with_reason("STOP_LOSS")
                    return
            else:
                if avg_lp_price > cutoff_price:
                    self.logger().warning(
                        f"触发止损:\n"
                        f"   LP 均价: {avg_lp_price}\n"
                        f"   止损线: {cutoff_price}\n"
                        f"   CEX 价格: {current_cex_price}"
                    )
                    await self._close_lp_position_with_reason("STOP_LOSS")
                    return

            # 检查超时
            elapsed = time.time() - self.lp_position_info["open_time"]
            if elapsed > self.config.lp_timeout_seconds:
                self.logger().info(f"LP 仓位超时 ({elapsed:.0f}秒)，关闭")
                await self._close_lp_position_with_reason("TIMEOUT")

        except Exception as e:
            self.logger().error(f"监控 LP 仓位失败: {e}")

    async def _close_lp_position_with_reason(self, reason: str):
        """关闭 LP 仓位"""
        try:
            await self.lp_manager.close_lp_position()
            self.logger().info(f"LP 仓位已关闭，原因: {reason}")
        except Exception as e:
            self.logger().error(f"关闭 LP 仓位失败: {e}")

    # ========================================
    # 事件处理
    # ========================================

    def did_fill_order(self, event: OrderFilledEvent):
        """订单成交事件"""
        # LP 开仓成交
        if hasattr(event, 'order_id') and event.order_id == self.lp_manager.open_order_id:
            self.logger().info(f"LP 开仓成交: {event.order_id}")
            self.lp_manager.position_opening = False
            # 需要异步获取 position_info
            safe_ensure_future(self._fetch_lp_position_info())

        # LP 关仓成交
        elif hasattr(event, 'order_id') and event.order_id == self.lp_manager.close_order_id:
            self.logger().info(f"LP 关仓成交: {event.order_id}")
            self.lp_manager.position_closing = False
            self.lp_manager.position_info = None
            self.lp_position_opened = False
            self.lp_position_info = None

        # CEX 对冲成交
        elif hasattr(event, 'order_id') and event.order_id == self.hedge_executor.pending_order_id:
            self.logger().info(f"CEX 对冲成交: {event.order_id}")
            safe_ensure_future(self._handle_hedge_filled(event))

    async def _fetch_lp_position_info(self):
        """获取 LP 仓位信息"""
        try:
            await asyncio.sleep(2)  # 等待链上确认

            if self.lp_manager.connector_type == ConnectorType.CLMM:
                pool_address = await self.dex_connector.get_pool_address(self.config.trading_pair)
                positions = await self.dex_connector.get_user_positions(pool_address=pool_address)
                if positions:
                    self.lp_manager.position_info = positions[-1]  # 最新的
            else:
                pool_address = await self.dex_connector.get_pool_address(self.config.trading_pair)
                if pool_address:
                    self.lp_manager.position_info = await self.dex_connector.get_position_info(
                        trading_pair=self.config.trading_pair,
                        position_address=pool_address
                    )
        except Exception as e:
            self.logger().error(f"获取 LP 仓位信息失败: {e}")

    async def _handle_hedge_filled(self, event: OrderFilledEvent):
        """处理对冲成交"""
        # 这里可以计算实际利润
        self.logger().info(
            f"对冲完成:\n"
            f"   价格: {event.price}\n"
            f"   数量: {event.amount}\n"
            f"   手续费: {event.trade_fee}"
        )

        self.stats["completed_cycles"] += 1

    # ========================================
    # 辅助方法
    # ========================================

    async def _get_cex_best_ask(self) -> Optional[Decimal]:
        """获取 CEX 最佳卖价（我们的买入价）"""
        try:
            price = await self.cex_connector.get_quote_price(
                trading_pair=self.config.trading_pair,
                is_buy=True,
                amount=self.config.lp_token_amount
            )
            return Decimal(str(price)) if price else None
        except Exception as e:
            self.logger().error(f"获取 CEX 买价失败: {e}")
            return None

    async def _get_cex_best_bid(self) -> Optional[Decimal]:
        """获取 CEX 最佳买价（我们的卖出价）"""
        try:
            price = await self.cex_connector.get_quote_price(
                trading_pair=self.config.trading_pair,
                is_buy=False,
                amount=self.config.lp_token_amount
            )
            return Decimal(str(price)) if price else None
        except Exception as e:
            self.logger().error(f"获取 CEX 卖价失败: {e}")
            return None

    async def _get_dex_pool_info(self):
        """
        获取 DEX 池子信息

        参考 lp_manage_position.py 的实现
        """
        try:
            self.logger().info(f"正在获取 {self.config.trading_pair} 池子信息...")
            self.logger().info(f"DEX Connector: {self.config.dex_exchange}")
            self.logger().info(f"Connector type: {type(self.dex_connector).__name__}")

            # 先尝试获取 pool address（用于诊断）
            try:
                if hasattr(self.dex_connector, 'get_pool_address'):
                    pool_address = await self.dex_connector.get_pool_address(
                        trading_pair=self.config.trading_pair
                    )
                    self.logger().info(f"Pool address: {pool_address}")
                else:
                    self.logger().warning("DEX connector 没有 get_pool_address 方法")
            except Exception as e:
                self.logger().warning(f"获取 pool address 失败: {e}")

            # 获取 pool info
            pool_info = await self.dex_connector.get_pool_info(
                trading_pair=self.config.trading_pair
            )

            if pool_info:
                self.logger().info(f"✅ 成功获取池子信息: price={pool_info.price}")
            else:
                self.logger().warning(f"❌ get_pool_info 返回 None - 可能的原因:")
                self.logger().warning(f"   1. 池子不存在或 trading_pair 格式错误")
                self.logger().warning(f"   2. Gateway 未正确连接")
                self.logger().warning(f"   3. 网络问题")
                self.logger().warning(f"")
                self.logger().warning(f"请检查:")
                self.logger().warning(f"   - Gateway 状态: gateway status")
                self.logger().warning(f"   - Trading pair 格式: {self.config.trading_pair}")
                self.logger().warning(f"   - DEX connector: {self.config.dex_exchange}")

            return pool_info
        except AttributeError as e:
            self.logger().error(f"DEX connector 不支持 get_pool_info 方法: {e}")
            self.logger().error(f"Connector type: {type(self.dex_connector)}")
            self.logger().error(f"Available methods: {[m for m in dir(self.dex_connector) if not m.startswith('_')]}")
            return None
        except Exception as e:
            self.logger().error(f"获取 DEX 池子信息失败: {e}", exc_info=True)
            return None

    async def _get_dex_price(self) -> Optional[Decimal]:
        """
        获取 DEX 价格（从 pool_info 中提取）

        参考 lp_manage_position.py 的实现
        """
        try:
            pool_info = await self._get_dex_pool_info()
            if pool_info and hasattr(pool_info, 'price'):
                price = Decimal(str(pool_info.price))
                self.logger().info(f"DEX 价格: {price}")
                return price
            else:
                self.logger().warning(f"无法从 pool_info 获取价格")
                return None
        except Exception as e:
            self.logger().error(f"获取 DEX 价格失败: {e}", exc_info=True)
            return None

    # ========================================
    # 状态显示
    # ========================================

    def format_status(self) -> str:
        """格式化状态显示（带详细市场数据）"""
        lines = []

        # ========== 标题 ==========
        lines.append("=" * 70)
        lines.append("CEX-DEX LP 套利策略".center(70))
        lines.append("=" * 70)
        lines.append(f"CEX: {self.config.cex_exchange:20} | DEX: {self.config.dex_exchange}")
        lines.append(f"交易对: {self.config.trading_pair:18} | LP 数量: {self.config.lp_token_amount}")
        lines.append("-" * 70)

        # ========== 实时市场数据 ==========
        lines.append("")
        lines.append("📈 实时市场数据")
        lines.append("-" * 70)

        # 异步获取最新价格（非阻塞）
        try:
            import asyncio
            loop = asyncio.get_event_loop()

            # 创建异步任务获取价格
            async def get_prices():
                cex_ask = await self._get_cex_best_ask()
                cex_bid = await self._get_cex_best_bid()
                dex_price = await self._get_dex_price()
                return cex_ask, cex_bid, dex_price

            # 如果事件循环正在运行，创建任务
            if loop.is_running():
                # 使用缓存的价格（如果有的话）
                cex_ask = getattr(self, '_cached_cex_ask', None)
                cex_bid = getattr(self, '_cached_cex_bid', None)
                dex_price = getattr(self, '_cached_dex_price', None)

                # 启动后台任务更新缓存
                safe_ensure_future(self._update_price_cache())
            else:
                # 同步获取
                cex_ask, cex_bid, dex_price = loop.run_until_complete(get_prices())

        except Exception as e:
            lines.append(f"⚠️  无法获取实时价格: {e}")
            cex_ask = None
            cex_bid = None
            dex_price = None

        # 显示 CEX 价格
        if cex_ask and cex_bid:
            cex_mid = (cex_ask + cex_bid) / 2
            cex_spread = ((cex_ask - cex_bid) / cex_mid * 100) if cex_mid > 0 else 0

            lines.append(f"CEX ({self.config.cex_exchange}):")
            lines.append(f"   买价 (Ask): {cex_ask:>12.6f} {self.quote_token}  ← 我们买入的价格")
            lines.append(f"   卖价 (Bid): {cex_bid:>12.6f} {self.quote_token}  ← 我们卖出的价格")
            lines.append(f"   中间价:     {cex_mid:>12.6f} {self.quote_token}")
            lines.append(f"   价差:       {cex_spread:>12.4f} %")
        else:
            lines.append(f"CEX: ⚠️  无法获取价格")

        lines.append("")

        # 显示 DEX 价格
        if dex_price:
            lines.append(f"DEX ({self.config.dex_exchange}):")
            lines.append(f"   报价:       {dex_price:>12.6f} {self.quote_token}")
        else:
            lines.append(f"DEX: ⚠️  无法获取价格")

        lines.append("")

        # ========== 价差分析 ==========
        if cex_ask and cex_bid and dex_price:

            lines.append("💰 套利机会分析")
            lines.append("-" * 70)

            # 卖方套利分析
            if self.config.enable_sell_side:
                lines.append("卖方套利 (DEX LP 卖出 → CEX 买入):")

                # 计算价差
                price_diff = dex_price - cex_ask
                price_diff_pct = (price_diff / cex_ask * 100) if cex_ask > 0 else 0

                # 计算目标价格
                trade_value = self.config.lp_token_amount * cex_ask
                target_price = self.profit_calculator.calculate_target_lp_price(
                    cex_price=cex_ask,
                    is_sell_side=True,
                    trade_value=trade_value
                )
                min_price = self.profit_calculator.calculate_min_lp_price(
                    cex_price=cex_ask,
                    is_sell_side=True,
                    trade_value=trade_value
                )

                # 计算费用
                total_fees_pct = self.profit_calculator.estimate_total_fees_pct(trade_value)

                lines.append(f"   DEX 价格:        {dex_price:>12.6f} {self.quote_token}")
                lines.append(f"   CEX 买价:        {cex_ask:>12.6f} {self.quote_token}")
                lines.append(f"   价差:            {price_diff:>+12.6f} {self.quote_token}  ({price_diff_pct:+.2f}%)")
                lines.append(f"   ")
                lines.append(f"   目标开仓价:      {target_price:>12.6f} {self.quote_token}  (需 {self.config.target_profitability*100:.1f}% 利润)")
                lines.append(f"   最低价格(止损):  {min_price:>12.6f} {self.quote_token}  (需 {self.config.min_profitability*100:.1f}% 利润)")
                lines.append(f"   ")
                lines.append(f"   总费用率:        {total_fees_pct*100:>12.2f} %")
                lines.append(f"     - CEX 手续费:  {self.config.cex_taker_fee_pct*100:>12.2f} %")
                lines.append(f"     - Gas 成本:    {(self.config.gas_cost_quote/trade_value*100) if trade_value > 0 else 0:>12.2f} %")
                lines.append(f"     - 滑点预留:    {1.0:>12.2f} %")
                lines.append(f"     - LP 费收入:   -{self.config.dex_lp_fee_pct*100:>12.2f} %")
                lines.append(f"   ")

                # 判断是否有机会
                if dex_price >= target_price:
                    expected_profit_pct = (dex_price - cex_ask) / cex_ask - total_fees_pct + self.config.dex_lp_fee_pct
                    expected_profit_amount = expected_profit_pct * trade_value
                    lines.append(f"   ✅ 有套利机会！")
                    lines.append(f"      预期利润:     {expected_profit_amount:>12.4f} {self.quote_token}  ({expected_profit_pct*100:+.2f}%)")
                else:
                    gap = target_price - dex_price
                    gap_pct = (gap / dex_price * 100) if dex_price > 0 else 0
                    lines.append(f"   ❌ 暂无机会")
                    lines.append(f"      需要涨幅:     {gap:>12.6f} {self.quote_token}  ({gap_pct:.2f}%)")

            lines.append("")

            # 买方套利分析
            if self.config.enable_buy_side:
                lines.append("买方套利 (DEX LP 买入 → CEX 卖出):")

                price_diff = cex_bid - dex_price
                price_diff_pct = (price_diff / dex_price * 100) if dex_price > 0 else 0

                trade_value = self.config.lp_token_amount * cex_bid
                target_price = self.profit_calculator.calculate_target_lp_price(
                    cex_price=cex_bid,
                    is_sell_side=False,
                    trade_value=trade_value
                )

                lines.append(f"   CEX 卖价:        {cex_bid:>12.6f} {self.quote_token}")
                lines.append(f"   DEX 价格:        {dex_price:>12.6f} {self.quote_token}")
                lines.append(f"   价差:            {price_diff:>+12.6f} {self.quote_token}  ({price_diff_pct:+.2f}%)")
                lines.append(f"   目标开仓价:      {target_price:>12.6f} {self.quote_token}")

                if dex_price <= target_price:
                    lines.append(f"   ✅ 有套利机会！")
                else:
                    gap = dex_price - target_price
                    gap_pct = (gap / dex_price * 100) if dex_price > 0 else 0
                    lines.append(f"   ❌ 暂无机会，需要跌幅: {gap:.6f} ({gap_pct:.2f}%)")

            lines.append("")

        # ========== LP 仓位状态 ==========
        if self.lp_position_opened and self.lp_position_info:
            lines.append("📊 LP 仓位状态")
            lines.append("-" * 70)

            side = self.lp_position_info["side"]
            lower, upper = self.lp_position_info["price_range"]
            avg_price = (lower + upper) / 2
            elapsed = time.time() - self.lp_position_info["open_time"]
            open_cex_price = self.lp_position_info["open_cex_price"]

            lines.append(f"方向:        {side}")
            lines.append(f"价格区间:    {lower:.6f} - {upper:.6f} {self.quote_token}")
            lines.append(f"均价:        {avg_price:.6f} {self.quote_token}")
            lines.append(f"数量:        {self.lp_position_info['token_amount']} {self.base_token}")
            lines.append(f"持仓时间:    {int(elapsed)}秒 / {self.config.lp_timeout_seconds}秒")
            lines.append(f"开仓CEX价:   {open_cex_price:.6f} {self.quote_token}")

            # 当前 CEX 价格变化
            if cex_ask and side == "SELL":
                current_cex = cex_ask
                price_change = current_cex - open_cex_price
                price_change_pct = (price_change / open_cex_price * 100) if open_cex_price > 0 else 0
                lines.append(f"当前CEX价:   {current_cex:.6f} {self.quote_token}  (变化: {price_change:+.6f}, {price_change_pct:+.2f}%)")

                # 预期盈亏
                if avg_price > current_cex:
                    expected_profit = (avg_price - current_cex) / current_cex * 100
                    lines.append(f"预期盈亏:    +{expected_profit:.2f}%  ✅")
                else:
                    expected_loss = (current_cex - avg_price) / avg_price * 100
                    lines.append(f"预期盈亏:    -{expected_loss:.2f}%  ⚠️")

            lines.append("")

        # ========== 统计信息 ==========
        lines.append("📊 统计信息")
        lines.append("-" * 70)
        lines.append(f"完成周期:    {self.stats['completed_cycles']}")
        lines.append(f"累计利润:    {self.stats['total_profit']:.4f} {self.quote_token}")
        if self.stats['completed_cycles'] > 0:
            avg_profit = self.stats['total_profit'] / self.stats['completed_cycles']
            lines.append(f"平均利润:    {avg_profit:.4f} {self.quote_token}")
        lines.append(f"LP开仓失败:  {self.stats['lp_open_failures']}")
        lines.append(f"对冲失败:    {self.stats['hedge_failures']}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    async def _update_price_cache(self):
        """后台更新价格缓存"""
        try:
            self._cached_cex_ask = await self._get_cex_best_ask()
            self._cached_cex_bid = await self._get_cex_best_bid()
            self._cached_dex_price = await self._get_dex_price()
        except Exception as e:
            self.logger().debug(f"更新价格缓存失败: {e}")
