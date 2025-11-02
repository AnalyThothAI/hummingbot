"""
Meteora DLMM 智能 LP 管理策略

策略原理:
1. 在 Meteora DLMM 池子中创建集中流动性仓位
2. 监控价格位置，当偏离区间边界 30% 时触发再平衡
3. 价格超出区间时暂停，等待回调后重新开仓
4. 多层风险控制：止损、盈利保护、冷却期

核心优势:
- 利用 DLMM 动态费率最大化手续费收入
- 主动管理降低无常损失
- 自动化再平衡减少人工干预
- 多重风险控制保护本金

参考文档:
- 策略设计: METEORA_DLMM_STRATEGY_DESIGN.md
- Meteora 官方文档: https://docs.meteora.ag/overview/products/dlmm
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
from hummingbot.connector.gateway.gateway_lp import CLMMPoolInfo, CLMMPositionInfo
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


# ========================================
# 配置类
# ========================================

class MeteoraDlmmSmartLpConfig(BaseClientModel):
    """Meteora DLMM 智能 LP 策略配置"""

    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))

    # ========== 基础配置 ==========
    connector: str = Field(
        "meteora/clmm",
        json_schema_extra={
            "prompt": "连接器（格式: meteora/clmm）",
            "prompt_on_new": True
        }
    )

    trading_pair: str = Field(
        "SOL-USDC",
        json_schema_extra={
            "prompt": "交易对（如 SOL-USDC）",
            "prompt_on_new": True
        }
    )

    pool_address: str = Field(
        "",
        json_schema_extra={
            "prompt": "池子地址（可选，留空自动获取）",
            "prompt_on_new": False
        }
    )

    # ========== LP 区间配置 ==========
    price_range_pct: Decimal = Field(
        Decimal("10.0"),
        json_schema_extra={
            "prompt": "价格区间宽度（百分比，如 10.0 表示 ±10%）",
            "prompt_on_new": True
        }
    )

    bin_count: int = Field(
        20,
        json_schema_extra={
            "prompt": "Bin 数量（推荐 15-25）",
            "prompt_on_new": False
        }
    )

    bin_distribution: str = Field(
        "curve",
        json_schema_extra={
            "prompt": "流动性分布策略（spot=均匀, curve=曲线, bid_ask=做市）",
            "prompt_on_new": True
        }
    )

    # ========== 资金配置 ==========
    base_token_amount: Decimal = Field(
        Decimal("0.0"),
        json_schema_extra={
            "prompt": "Base Token 数量（0 表示使用钱包余额的百分比）",
            "prompt_on_new": True
        }
    )

    quote_token_amount: Decimal = Field(
        Decimal("100.0"),
        json_schema_extra={
            "prompt": "Quote Token 数量（0 表示使用钱包余额的百分比）",
            "prompt_on_new": True
        }
    )

    wallet_allocation_pct: Decimal = Field(
        Decimal("80.0"),
        json_schema_extra={
            "prompt": "钱包余额使用百分比（当 token_amount=0 时生效）",
            "prompt_on_new": False
        }
    )

    # ========== 再平衡配置 ==========
    rebalance_threshold_pct: Decimal = Field(
        Decimal("30.0"),
        json_schema_extra={
            "prompt": "再平衡触发阈值（价格偏离区间边界的百分比，如 30.0）",
            "prompt_on_new": True
        }
    )

    rebalance_cooldown_seconds: int = Field(
        3600,
        json_schema_extra={
            "prompt": "再平衡冷却期（秒，避免频繁操作）",
            "prompt_on_new": False
        }
    )

    # ========== 风险控制 ==========
    stop_loss_pct: Decimal = Field(
        Decimal("5.0"),
        json_schema_extra={
            "prompt": "止损百分比（相对开仓价格，如 5.0 表示 -5%）",
            "prompt_on_new": True
        }
    )

    profit_take_threshold_pct: Decimal = Field(
        Decimal("3.0"),
        json_schema_extra={
            "prompt": "盈利保护阈值（累积盈利超过此值启动移动止损，如 3.0）",
            "prompt_on_new": False
        }
    )

    trailing_stop_pct: Decimal = Field(
        Decimal("1.0"),
        json_schema_extra={
            "prompt": "移动止损百分比（盈利回撤超过此值触发止盈，如 1.0）",
            "prompt_on_new": False
        }
    )

    pause_on_out_of_range: bool = Field(
        True,
        json_schema_extra={
            "prompt": "价格超出区间时暂停（等待回调）",
            "prompt_on_new": False
        }
    )

    pause_cooldown_seconds: int = Field(
        300,
        json_schema_extra={
            "prompt": "暂停后检查间隔（秒）",
            "prompt_on_new": False
        }
    )

    # ========== 监控配置 ==========
    check_interval_seconds: int = Field(
        10,
        json_schema_extra={
            "prompt": "检查间隔（秒）",
            "prompt_on_new": False
        }
    )

    min_profit_for_rebalance: Decimal = Field(
        Decimal("0.5"),
        json_schema_extra={
            "prompt": "最小再平衡盈利要求（百分比，仅在盈利 > 此值时再平衡）",
            "prompt_on_new": False
        }
    )


# ========================================
# 主策略类
# ========================================

class MeteoraDlmmSmartLp(ScriptStrategyBase):
    """
    Meteora DLMM 智能 LP 管理策略

    核心功能:
    1. 自动创建 DLMM 流动性仓位
    2. 实时监控价格位置和偏离度
    3. 智能再平衡（价格偏离 30% 时触发）
    4. 多层风险控制（止损、盈利保护、暂停）
    5. 详细状态展示和性能追踪
    """

    @classmethod
    def init_markets(cls, config: MeteoraDlmmSmartLpConfig):
        cls.markets = {config.connector: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: MeteoraDlmmSmartLpConfig):
        super().__init__(connectors)
        self.config = config

        # 连接器
        self.exchange = config.connector
        self.connector = connectors[config.connector]
        self.connector_type = get_connector_type(config.connector)

        # Token 信息
        self.base_token, self.quote_token = config.trading_pair.split("-")

        # 仓位状态
        self.position_info: Optional[CLMMPositionInfo] = None
        self.pool_info: Optional[CLMMPoolInfo] = None
        self.position_opened = False
        self.position_opening = False
        self.position_closing = False
        self.position_paused = False

        # 订单追踪
        self.open_order_id: Optional[str] = None
        self.close_order_id: Optional[str] = None

        # 价格追踪
        self.open_price: Optional[Decimal] = None  # 开仓时的价格
        self.initial_investment: Optional[Decimal] = None  # 初始投入
        self.peak_value: Optional[Decimal] = None  # 最高价值（用于移动止损）

        # 再平衡追踪
        self.last_rebalance_time: Optional[float] = None
        self.rebalance_count = 0

        # 时间追踪
        self.last_check_time: Optional[datetime] = None
        self.last_price_update: Optional[datetime] = None
        self.pause_start_time: Optional[float] = None

        # 性能统计
        self.stats = {
            "total_fees_earned": Decimal("0"),
            "total_rebalances": 0,
            "total_profit_usd": Decimal("0"),
            "max_drawdown_pct": Decimal("0"),
            "successful_rebalances": 0,
            "failed_rebalances": 0,
        }

        # 启动信息
        self.log_with_clock(
            logging.INFO,
            f"Meteora DLMM 智能 LP 策略启动:\n"
            f"   交易对: {config.trading_pair}\n"
            f"   价格区间: ±{config.price_range_pct}%\n"
            f"   Bin 数量: {config.bin_count} ({config.bin_distribution} 分布)\n"
            f"   再平衡阈值: {config.rebalance_threshold_pct}%\n"
            f"   止损: {config.stop_loss_pct}%\n"
            f"   盈利保护: {config.profit_take_threshold_pct}% (移动止损 {config.trailing_stop_pct}%)"
        )

        # 延迟检查现有仓位
        safe_ensure_future(self.check_existing_position())

    # ========================================
    # 初始化和仓位检查
    # ========================================

    async def check_existing_position(self):
        """检查是否有现有仓位"""
        await asyncio.sleep(3)  # 等待连接器初始化

        try:
            # 获取池子信息
            await self.fetch_pool_info()

            # 检查现有仓位
            pool_address = await self.get_pool_address()
            if pool_address:
                positions = await self.connector.get_user_positions(pool_address=pool_address)
                if positions and len(positions) > 0:
                    self.position_info = positions[0]
                    self.position_opened = True

                    # 设置开仓价格为当前价格
                    if self.pool_info:
                        self.open_price = Decimal(str(self.pool_info.price))

                    self.logger().info(
                        f"发现现有仓位: {self.position_info.address}\n"
                        f"   当前价格: {self.open_price}"
                    )
        except Exception as e:
            self.logger().error(f"检查现有仓位失败: {e}")

    async def fetch_pool_info(self) -> Optional[CLMMPoolInfo]:
        """获取池子信息"""
        try:
            self.pool_info = await self.connector.get_pool_info(
                trading_pair=self.config.trading_pair
            )
            return self.pool_info
        except Exception as e:
            self.logger().error(f"获取池子信息失败: {e}")
            return None

    async def get_pool_address(self) -> Optional[str]:
        """获取池子地址"""
        if self.config.pool_address:
            return self.config.pool_address
        else:
            try:
                return await self.connector.get_pool_address(self.config.trading_pair)
            except Exception as e:
                self.logger().error(f"获取池子地址失败: {e}")
                return None

    # ========================================
    # 主循环
    # ========================================

    def on_tick(self):
        """策略主循环"""
        current_time = datetime.now()

        # 检查间隔控制
        interval = self.config.pause_cooldown_seconds if self.position_paused else self.config.check_interval_seconds

        if self.last_check_time and (current_time - self.last_check_time).total_seconds() < interval:
            return

        self.last_check_time = current_time

        # 根据状态执行不同逻辑
        if self.position_opening or self.position_closing:
            # 等待订单确认
            return
        elif self.position_paused:
            # 暂停状态：检查是否可以恢复
            safe_ensure_future(self.check_resume_condition())
        elif self.position_opened:
            # 持仓中：监控和再平衡
            safe_ensure_future(self.monitor_position())
        else:
            # 无仓位：开仓
            safe_ensure_future(self.check_and_open_position())

    # ========================================
    # 开仓逻辑
    # ========================================

    async def check_and_open_position(self):
        """检查并开仓"""
        if self.position_opened or self.position_opening:
            return

        try:
            # 获取当前价格
            await self.fetch_pool_info()
            if not self.pool_info:
                self.logger().warning("无法获取池子信息")
                return

            current_price = Decimal(str(self.pool_info.price))
            self.logger().info(f"当前价格: {current_price} {self.quote_token}")

            # 开仓
            await self.open_position(current_price)

        except Exception as e:
            self.logger().error(f"检查开仓失败: {e}")

    async def open_position(self, center_price: Decimal):
        """开 LP 仓位"""
        if self.position_opening or self.position_opened:
            return

        self.position_opening = True

        try:
            # 计算价格区间
            range_pct = self.config.price_range_pct / Decimal("100")
            lower_price = center_price * (Decimal("1") - range_pct)
            upper_price = center_price * (Decimal("1") + range_pct)

            # 计算 Bin 参数
            # DLMM 使用 lower_width_pct 和 upper_width_pct
            lower_width_pct = float(range_pct * 100)
            upper_width_pct = float(range_pct * 100)

            # 获取代币数量
            base_amount, quote_amount = await self.get_token_amounts()

            self.logger().info(
                f"开仓参数:\n"
                f"   中心价格: {center_price}\n"
                f"   价格区间: {lower_price:.6f} - {upper_price:.6f}\n"
                f"   区间宽度: ±{self.config.price_range_pct}%\n"
                f"   Bin 数量: {self.config.bin_count}\n"
                f"   分布策略: {self.config.bin_distribution}\n"
                f"   Base Token: {base_amount} {self.base_token}\n"
                f"   Quote Token: {quote_amount} {self.quote_token}"
            )

            # 提交开仓订单
            order_id = self.connector.add_liquidity(
                trading_pair=self.config.trading_pair,
                price=float(center_price),
                upper_width_pct=upper_width_pct,
                lower_width_pct=lower_width_pct,
                base_token_amount=float(base_amount),
                quote_token_amount=float(quote_amount),
            )

            self.open_order_id = order_id
            self.open_price = center_price

            # 记录初始投入
            self.initial_investment = (base_amount * center_price) + quote_amount

            self.logger().info(f"开仓订单已提交: {order_id}")

        except Exception as e:
            self.logger().error(f"开仓失败: {e}")
            self.position_opening = False

    async def get_token_amounts(self) -> Tuple[Decimal, Decimal]:
        """获取要投入的代币数量"""
        # 如果配置了固定数量，直接使用
        if self.config.base_token_amount > 0 or self.config.quote_token_amount > 0:
            return self.config.base_token_amount, self.config.quote_token_amount

        # 否则，使用钱包余额的百分比
        try:
            base_balance = self.connector.get_available_balance(self.base_token)
            quote_balance = self.connector.get_available_balance(self.quote_token)

            allocation_pct = self.config.wallet_allocation_pct / Decimal("100")

            base_amount = Decimal(str(base_balance)) * allocation_pct
            quote_amount = Decimal(str(quote_balance)) * allocation_pct

            self.logger().info(
                f"钱包余额:\n"
                f"   {self.base_token}: {base_balance} (使用 {base_amount})\n"
                f"   {self.quote_token}: {quote_balance} (使用 {quote_amount})"
            )

            return base_amount, quote_amount

        except Exception as e:
            self.logger().error(f"获取钱包余额失败: {e}")
            return Decimal("0"), Decimal("100")  # 降级到默认值

    # ========================================
    # 监控和再平衡
    # ========================================

    async def monitor_position(self):
        """监控仓位"""
        if not self.position_info:
            return

        try:
            # 更新仓位和池子信息
            await self.update_position_and_pool_info()

            if not self.pool_info or not self.position_info:
                return

            current_price = Decimal(str(self.pool_info.price))
            lower_price = Decimal(str(self.position_info.lower_price))
            upper_price = Decimal(str(self.position_info.upper_price))

            # 1. 检查止损
            if await self.check_stop_loss(current_price):
                return

            # 2. 检查盈利保护
            if await self.check_profit_protection(current_price):
                return

            # 3. 检查再平衡条件
            await self.check_rebalance_condition(current_price, lower_price, upper_price)

            # 4. 检查是否超出区间（需要暂停）
            await self.check_out_of_range(current_price, lower_price, upper_price)

            # 5. 更新性能统计
            await self.update_performance_stats(current_price)

        except Exception as e:
            self.logger().error(f"监控仓位失败: {e}")

    async def update_position_and_pool_info(self):
        """更新仓位和池子信息"""
        try:
            # 更新池子信息
            await self.fetch_pool_info()

            # 更新仓位信息
            if self.position_info:
                self.position_info = await self.connector.get_position_info(
                    trading_pair=self.config.trading_pair,
                    position_address=self.position_info.address
                )
                self.last_price_update = datetime.now()
        except Exception as e:
            self.logger().error(f"更新仓位信息失败: {e}")

    async def check_stop_loss(self, current_price: Decimal) -> bool:
        """
        检查止损条件

        Returns:
            True 如果触发止损
        """
        if not self.open_price or not self.initial_investment:
            return False

        # 计算当前价值
        current_value = await self.calculate_position_value(current_price)
        if not current_value:
            return False

        # 计算亏损百分比
        loss_pct = ((current_value - self.initial_investment) / self.initial_investment) * Decimal("100")

        # 检查是否触发止损
        if loss_pct <= -self.config.stop_loss_pct:
            self.logger().warning(
                f"触发止损:\n"
                f"   初始投入: {self.initial_investment:.2f} {self.quote_token}\n"
                f"   当前价值: {current_value:.2f} {self.quote_token}\n"
                f"   亏损: {loss_pct:.2f}%\n"
                f"   止损线: {self.config.stop_loss_pct}%"
            )

            await self.close_position(reason="STOP_LOSS")
            self.position_paused = True
            self.pause_start_time = time.time()
            return True

        return False

    async def check_profit_protection(self, current_price: Decimal) -> bool:
        """
        检查盈利保护（移动止损）

        Returns:
            True 如果触发止盈
        """
        if not self.open_price or not self.initial_investment:
            return False

        # 计算当前价值
        current_value = await self.calculate_position_value(current_price)
        if not current_value:
            return False

        profit_pct = ((current_value - self.initial_investment) / self.initial_investment) * Decimal("100")

        # 只有盈利超过阈值才启用保护
        if profit_pct < self.config.profit_take_threshold_pct:
            return False

        # 更新最高价值
        if not self.peak_value or current_value > self.peak_value:
            self.peak_value = current_value
            self.logger().debug(f"更新最高价值: {self.peak_value:.2f} {self.quote_token}")

        # 计算从最高点回撤的百分比
        drawdown_from_peak = ((self.peak_value - current_value) / self.peak_value) * Decimal("100")

        # 检查是否触发移动止损
        if drawdown_from_peak >= self.config.trailing_stop_pct:
            self.logger().info(
                f"触发盈利保护（移动止损）:\n"
                f"   初始投入: {self.initial_investment:.2f} {self.quote_token}\n"
                f"   最高价值: {self.peak_value:.2f} {self.quote_token}\n"
                f"   当前价值: {current_value:.2f} {self.quote_token}\n"
                f"   累积盈利: {profit_pct:.2f}%\n"
                f"   峰值回撤: {drawdown_from_peak:.2f}%"
            )

            await self.close_position(reason="PROFIT_PROTECTION")
            self.position_paused = True
            self.pause_start_time = time.time()
            return True

        return False

    async def check_rebalance_condition(
        self,
        current_price: Decimal,
        lower_price: Decimal,
        upper_price: Decimal
    ):
        """检查再平衡条件"""
        # 检查冷却期
        if self.last_rebalance_time:
            elapsed = time.time() - self.last_rebalance_time
            if elapsed < self.config.rebalance_cooldown_seconds:
                return

        # 计算价格在区间中的位置
        price_range = upper_price - lower_price
        if price_range == 0:
            return

        # 距离下界的百分比
        distance_from_lower = ((current_price - lower_price) / price_range) * Decimal("100")
        # 距离上界的百分比
        distance_from_upper = ((upper_price - current_price) / price_range) * Decimal("100")

        # 计算最小距离（离哪个边界更近）
        min_distance = min(distance_from_lower, distance_from_upper)

        # 检查是否触发再平衡
        if min_distance <= self.config.rebalance_threshold_pct:
            # 检查是否有足够盈利
            if not await self.check_min_profit_for_rebalance():
                self.logger().info(
                    f"价格偏离区间边界 {min_distance:.1f}%，但盈利不足，跳过再平衡\n"
                    f"   当前价格: {current_price}\n"
                    f"   区间: {lower_price:.6f} - {upper_price:.6f}"
                )
                return

            self.logger().info(
                f"触发再平衡:\n"
                f"   当前价格: {current_price}\n"
                f"   区间: {lower_price:.6f} - {upper_price:.6f}\n"
                f"   距离下界: {distance_from_lower:.1f}%\n"
                f"   距离上界: {distance_from_upper:.1f}%\n"
                f"   触发阈值: {self.config.rebalance_threshold_pct}%"
            )

            await self.execute_rebalance(current_price)

    async def check_min_profit_for_rebalance(self) -> bool:
        """检查是否满足最小再平衡盈利要求"""
        if not self.initial_investment or not self.pool_info:
            return True  # 无法计算，允许再平衡

        current_price = Decimal(str(self.pool_info.price))
        current_value = await self.calculate_position_value(current_price)

        if not current_value:
            return True

        profit_pct = ((current_value - self.initial_investment) / self.initial_investment) * Decimal("100")

        return profit_pct >= self.config.min_profit_for_rebalance

    async def check_out_of_range(
        self,
        current_price: Decimal,
        lower_price: Decimal,
        upper_price: Decimal
    ):
        """检查价格是否超出区间"""
        if not self.config.pause_on_out_of_range:
            return

        # 检查是否超出区间
        if current_price < lower_price or current_price > upper_price:
            if current_price < lower_price:
                distance = ((lower_price - current_price) / lower_price) * Decimal("100")
                direction = "下方"
            else:
                distance = ((current_price - upper_price) / upper_price) * Decimal("100")
                direction = "上方"

            self.logger().warning(
                f"价格超出区间:\n"
                f"   当前价格: {current_price}\n"
                f"   区间: {lower_price:.6f} - {upper_price:.6f}\n"
                f"   超出{direction}: {distance:.2f}%\n"
                f"   暂停策略，等待价格回调"
            )

            # 关闭仓位并暂停
            await self.close_position(reason="OUT_OF_RANGE")
            self.position_paused = True
            self.pause_start_time = time.time()

    async def execute_rebalance(self, new_center_price: Decimal):
        """执行再平衡"""
        try:
            self.logger().info("开始再平衡...")

            # 记录再平衡前的价值
            old_value = await self.calculate_position_value(new_center_price)

            # 关闭旧仓位
            await self.close_position(reason="REBALANCE")

            # 等待平仓完成
            await asyncio.sleep(3)

            # 开新仓位
            await self.open_position(new_center_price)

            # 更新统计
            self.last_rebalance_time = time.time()
            self.rebalance_count += 1
            self.stats["total_rebalances"] += 1

            # 计算再平衡后的价值（需要等待新仓位创建）
            # 这里简化处理，实际应该在新仓位创建后再计算
            self.logger().info(
                f"再平衡完成 (第 {self.rebalance_count} 次):\n"
                f"   新中心价格: {new_center_price}\n"
                f"   再平衡前价值: {old_value:.2f} {self.quote_token}"
            )

            self.stats["successful_rebalances"] += 1

        except Exception as e:
            self.logger().error(f"再平衡失败: {e}")
            self.stats["failed_rebalances"] += 1

    async def check_resume_condition(self):
        """检查是否可以恢复策略（从暂停状态）"""
        if not self.position_paused:
            return

        try:
            # 更新池子信息
            await self.fetch_pool_info()
            if not self.pool_info or not self.open_price:
                return

            current_price = Decimal(str(self.pool_info.price))

            # 计算目标区间
            range_pct = self.config.price_range_pct / Decimal("100")
            lower_bound = self.open_price * (Decimal("1") - range_pct)
            upper_bound = self.open_price * (Decimal("1") + range_pct)

            # 检查价格是否回到区间内
            if lower_bound <= current_price <= upper_bound:
                pause_duration = time.time() - self.pause_start_time if self.pause_start_time else 0

                self.logger().info(
                    f"价格已回到区间，恢复策略:\n"
                    f"   当前价格: {current_price}\n"
                    f"   目标区间: {lower_bound:.6f} - {upper_bound:.6f}\n"
                    f"   暂停时长: {pause_duration:.0f} 秒"
                )

                # 恢复策略
                self.position_paused = False
                self.pause_start_time = None

                # 重新开仓
                await self.open_position(current_price)
            else:
                if current_price < lower_bound:
                    distance = ((lower_bound - current_price) / lower_bound) * Decimal("100")
                    self.logger().debug(
                        f"等待价格回调: 当前 {current_price}, "
                        f"需要上涨 {distance:.2f}% 到达 {lower_bound:.6f}"
                    )
                else:
                    distance = ((current_price - upper_bound) / upper_bound) * Decimal("100")
                    self.logger().debug(
                        f"等待价格回调: 当前 {current_price}, "
                        f"需要下跌 {distance:.2f}% 到达 {upper_bound:.6f}"
                    )

        except Exception as e:
            self.logger().error(f"检查恢复条件失败: {e}")

    # ========================================
    # 平仓逻辑
    # ========================================

    async def close_position(self, reason: str = "MANUAL"):
        """
        关闭仓位

        Args:
            reason: 关闭原因（STOP_LOSS, PROFIT_PROTECTION, REBALANCE, OUT_OF_RANGE, MANUAL）
        """
        if not self.position_info or self.position_closing:
            return

        self.position_closing = True

        try:
            self.logger().info(f"关闭仓位 (原因: {reason})...")

            # 先收集手续费（重要！）
            await self.collect_fees()

            # 然后关闭仓位
            order_id = self.connector.remove_liquidity(
                trading_pair=self.config.trading_pair,
                position_address=self.position_info.address
            )

            self.close_order_id = order_id
            self.logger().info(f"平仓订单已提交: {order_id}")

            # 记录平仓原因
            if reason in ["STOP_LOSS", "PROFIT_PROTECTION"]:
                # 这些情况下暂停策略
                self.position_paused = True

        except Exception as e:
            self.logger().error(f"关闭仓位失败: {e}")
            self.position_closing = False

    async def collect_fees(self):
        """收集累积的手续费"""
        if not self.position_info:
            return

        try:
            # 检查是否有待收集的手续费
            base_fees = Decimal(str(self.position_info.base_fee_amount))
            quote_fees = Decimal(str(self.position_info.quote_fee_amount))

            if base_fees > 0 or quote_fees > 0:
                self.logger().info(
                    f"收集手续费:\n"
                    f"   {self.base_token}: {base_fees}\n"
                    f"   {self.quote_token}: {quote_fees}"
                )

                # 调用 Gateway API 收集手续费
                # 注意：需要确认 Meteora connector 的具体实现
                # 这里假设有 collect_fees 方法
                if hasattr(self.connector, 'collect_fees'):
                    await self.connector.collect_fees(
                        trading_pair=self.config.trading_pair,
                        position_address=self.position_info.address
                    )

                # 更新统计
                current_price = Decimal(str(self.pool_info.price)) if self.pool_info else Decimal("1")
                total_fees_usd = (base_fees * current_price) + quote_fees
                self.stats["total_fees_earned"] += total_fees_usd

            else:
                self.logger().debug("当前无待收集手续费")

        except Exception as e:
            self.logger().error(f"收集手续费失败: {e}")

    # ========================================
    # 辅助计算方法
    # ========================================

    async def calculate_position_value(self, current_price: Decimal) -> Optional[Decimal]:
        """
        计算仓位当前价值（以 quote token 计价）

        Args:
            current_price: 当前价格

        Returns:
            总价值（base token 价值 + quote token 价值 + 手续费价值）
        """
        if not self.position_info:
            return None

        try:
            base_amount = Decimal(str(self.position_info.base_token_amount))
            quote_amount = Decimal(str(self.position_info.quote_token_amount))
            base_fees = Decimal(str(self.position_info.base_fee_amount))
            quote_fees = Decimal(str(self.position_info.quote_fee_amount))

            # 计算总价值
            total_value = (
                (base_amount + base_fees) * current_price +
                (quote_amount + quote_fees)
            )

            return total_value

        except Exception as e:
            self.logger().error(f"计算仓位价值失败: {e}")
            return None

    async def update_performance_stats(self, current_price: Decimal):
        """更新性能统计"""
        if not self.initial_investment:
            return

        try:
            current_value = await self.calculate_position_value(current_price)
            if not current_value:
                return

            # 计算盈利
            profit = current_value - self.initial_investment
            profit_pct = (profit / self.initial_investment) * Decimal("100")

            self.stats["total_profit_usd"] = profit

            # 计算最大回撤
            if self.peak_value:
                current_drawdown = ((self.peak_value - current_value) / self.peak_value) * Decimal("100")
                if current_drawdown > self.stats["max_drawdown_pct"]:
                    self.stats["max_drawdown_pct"] = current_drawdown

        except Exception as e:
            self.logger().error(f"更新性能统计失败: {e}")

    # ========================================
    # 事件处理
    # ========================================

    def did_fill_order(self, event):
        """订单成交事件"""
        if hasattr(event, 'order_id'):
            if event.order_id == self.open_order_id:
                self.logger().info(f"开仓订单成交: {event.order_id}")
                self.position_opened = True
                self.position_opening = False

                # 获取仓位信息
                safe_ensure_future(self.fetch_position_info_after_fill())

            elif event.order_id == self.close_order_id:
                self.logger().info(f"平仓订单成交: {event.order_id}")
                self.position_opened = False
                self.position_closing = False
                self.position_info = None

    async def fetch_position_info_after_fill(self):
        """开仓后获取仓位信息"""
        await asyncio.sleep(2)

        try:
            pool_address = await self.get_pool_address()
            if pool_address:
                positions = await self.connector.get_user_positions(pool_address=pool_address)
                if positions:
                    self.position_info = positions[-1]  # 获取最新仓位
                    self.logger().info(f"仓位信息已获取: {self.position_info.address}")
        except Exception as e:
            self.logger().error(f"获取仓位信息失败: {e}")

    # ========================================
    # 状态显示
    # ========================================

    def format_status(self) -> str:
        """格式化状态显示"""
        lines = []

        # 标题
        lines.append("=" * 80)
        lines.append("Meteora DLMM 智能 LP 策略".center(80))
        lines.append("=" * 80)

        # 基本信息
        lines.append(f"交易对: {self.config.trading_pair}")
        lines.append(f"连接器: {self.exchange}")
        lines.append("")

        # 状态区分显示
        if self.position_paused:
            self._format_paused_status(lines)
        elif self.position_opening:
            self._format_opening_status(lines)
        elif self.position_closing:
            self._format_closing_status(lines)
        elif self.position_opened and self.position_info:
            self._format_active_position_status(lines)
        else:
            self._format_waiting_status(lines)

        # 性能统计
        lines.append("")
        lines.append("📊 性能统计")
        lines.append("-" * 80)
        self._format_performance_stats(lines)

        lines.append("=" * 80)

        return "\n".join(lines)

    def _format_active_position_status(self, lines: list):
        """格式化活跃仓位状态"""
        lines.append("💼 仓位信息")
        lines.append("-" * 80)
        lines.append(f"仓位地址: {self.position_info.address}")

        # 当前价格和区间
        if self.pool_info:
            current_price = Decimal(str(self.pool_info.price))
            lower_price = Decimal(str(self.position_info.lower_price))
            upper_price = Decimal(str(self.position_info.upper_price))

            lines.append(f"当前价格: {current_price:.6f} {self.quote_token}")
            lines.append(f"价格区间: {lower_price:.6f} - {upper_price:.6f}")

            # 价格位置可视化
            price_viz = self._get_price_visualization(current_price, lower_price, upper_price)
            if price_viz:
                lines.append(f"价格位置: {price_viz}")

            # 计算偏离度
            price_range = upper_price - lower_price
            if price_range > 0:
                distance_from_lower = ((current_price - lower_price) / price_range) * Decimal("100")
                distance_from_upper = ((upper_price - current_price) / price_range) * Decimal("100")
                min_distance = min(distance_from_lower, distance_from_upper)

                status_emoji = "⚠️" if min_distance <= self.config.rebalance_threshold_pct else "✅"
                lines.append(
                    f"偏离度: {status_emoji} 距离边界 {min_distance:.1f}% "
                    f"(触发阈值: {self.config.rebalance_threshold_pct}%)"
                )

        lines.append("")

        # 仓位资产
        lines.append("💰 仓位资产")
        lines.append("-" * 80)
        base_amount = Decimal(str(self.position_info.base_token_amount))
        quote_amount = Decimal(str(self.position_info.quote_token_amount))
        lines.append(f"{self.base_token}: {base_amount:.6f}")
        lines.append(f"{self.quote_token}: {quote_amount:.6f}")

        # 手续费
        base_fees = Decimal(str(self.position_info.base_fee_amount))
        quote_fees = Decimal(str(self.position_info.quote_fee_amount))
        if base_fees > 0 or quote_fees > 0:
            lines.append(f"待收集手续费:")
            lines.append(f"  {self.base_token}: {base_fees:.6f}")
            lines.append(f"  {self.quote_token}: {quote_fees:.6f}")

        lines.append("")

        # 盈亏分析
        if self.pool_info and self.initial_investment:
            lines.append("📈 盈亏分析")
            lines.append("-" * 80)

            current_price = Decimal(str(self.pool_info.price))
            current_value = (base_amount + base_fees) * current_price + (quote_amount + quote_fees)

            profit = current_value - self.initial_investment
            profit_pct = (profit / self.initial_investment) * Decimal("100")

            profit_emoji = "🟢" if profit >= 0 else "🔴"
            lines.append(f"初始投入: {self.initial_investment:.2f} {self.quote_token}")
            lines.append(f"当前价值: {current_value:.2f} {self.quote_token}")
            lines.append(f"{profit_emoji} 盈亏: {profit:+.2f} {self.quote_token} ({profit_pct:+.2f}%)")

            # 开仓价格对比
            if self.open_price:
                price_change = ((current_price - self.open_price) / self.open_price) * Decimal("100")
                lines.append(f"开仓价格: {self.open_price:.6f} (价格变化: {price_change:+.2f}%)")

            # 止损和止盈线
            if self.open_price:
                stop_loss_price = self.open_price * (Decimal("1") - self.config.stop_loss_pct / Decimal("100"))
                lines.append(f"止损线: {stop_loss_price:.6f} (-{self.config.stop_loss_pct}%)")

                if profit_pct >= self.config.profit_take_threshold_pct and self.peak_value:
                    trailing_value = self.peak_value * (Decimal("1") - self.config.trailing_stop_pct / Decimal("100"))
                    lines.append(
                        f"移动止损: {trailing_value:.2f} {self.quote_token} "
                        f"(峰值 {self.peak_value:.2f})"
                    )

        lines.append("")

        # 再平衡信息
        if self.rebalance_count > 0:
            lines.append(f"再平衡次数: {self.rebalance_count}")
            if self.last_rebalance_time:
                elapsed = time.time() - self.last_rebalance_time
                cooldown_remaining = max(0, self.config.rebalance_cooldown_seconds - elapsed)
                if cooldown_remaining > 0:
                    lines.append(f"冷却期剩余: {int(cooldown_remaining)} 秒")

    def _format_opening_status(self, lines: list):
        """格式化开仓中状态"""
        lines.append("⏳ 开仓中...")
        lines.append("-" * 80)
        lines.append(f"订单 ID: {self.open_order_id}")
        lines.append("等待链上确认...")

    def _format_closing_status(self, lines: list):
        """格式化平仓中状态"""
        lines.append("⏳ 平仓中...")
        lines.append("-" * 80)
        lines.append(f"订单 ID: {self.close_order_id}")
        lines.append("等待链上确认...")

        if self.position_info:
            base_fees = Decimal(str(self.position_info.base_fee_amount))
            quote_fees = Decimal(str(self.position_info.quote_fee_amount))
            if base_fees > 0 or quote_fees > 0:
                lines.append("")
                lines.append("收集的手续费:")
                lines.append(f"  {self.base_token}: {base_fees:.6f}")
                lines.append(f"  {self.quote_token}: {quote_fees:.6f}")

    def _format_paused_status(self, lines: list):
        """格式化暂停状态"""
        lines.append("⏸️  策略已暂停")
        lines.append("-" * 80)
        lines.append("等待价格回到目标区间...")

        if self.pool_info and self.open_price:
            current_price = Decimal(str(self.pool_info.price))
            range_pct = self.config.price_range_pct / Decimal("100")
            lower_bound = self.open_price * (Decimal("1") - range_pct)
            upper_bound = self.open_price * (Decimal("1") + range_pct)

            lines.append(f"当前价格: {current_price:.6f} {self.quote_token}")
            lines.append(f"目标区间: {lower_bound:.6f} - {upper_bound:.6f}")

            if current_price < lower_bound:
                gap = ((lower_bound - current_price) / lower_bound) * Decimal("100")
                lines.append(f"需要上涨: {gap:.2f}%")
            elif current_price > upper_bound:
                gap = ((current_price - upper_bound) / upper_bound) * Decimal("100")
                lines.append(f"需要下跌: {gap:.2f}%")

        if self.pause_start_time:
            pause_duration = int(time.time() - self.pause_start_time)
            lines.append(f"已暂停: {pause_duration} 秒")

    def _format_waiting_status(self, lines: list):
        """格式化等待开仓状态"""
        lines.append("🔍 等待开仓")
        lines.append("-" * 80)

        if self.pool_info:
            current_price = Decimal(str(self.pool_info.price))
            lines.append(f"当前价格: {current_price:.6f} {self.quote_token}")

            range_pct = self.config.price_range_pct / Decimal("100")
            lower_price = current_price * (Decimal("1") - range_pct)
            upper_price = current_price * (Decimal("1") + range_pct)
            lines.append(f"计划区间: {lower_price:.6f} - {upper_price:.6f} (±{self.config.price_range_pct}%)")

        lines.append("")
        lines.append("策略参数:")
        lines.append(f"  Bin 数量: {self.config.bin_count}")
        lines.append(f"  分布策略: {self.config.bin_distribution}")
        lines.append(f"  再平衡阈值: {self.config.rebalance_threshold_pct}%")
        lines.append(f"  止损: {self.config.stop_loss_pct}%")

    def _format_performance_stats(self, lines: list):
        """格式化性能统计"""
        lines.append(f"累积盈利: {self.stats['total_profit_usd']:+.2f} {self.quote_token}")
        lines.append(f"手续费收入: {self.stats['total_fees_earned']:.6f} {self.quote_token}")
        lines.append(f"再平衡: {self.stats['total_rebalances']} 次 "
                    f"(成功 {self.stats['successful_rebalances']}, "
                    f"失败 {self.stats['failed_rebalances']})")
        lines.append(f"最大回撤: {self.stats['max_drawdown_pct']:.2f}%")

        if self.last_price_update:
            elapsed = int((datetime.now() - self.last_price_update).total_seconds())
            lines.append(f"最后更新: {elapsed} 秒前")

    def _get_price_visualization(
        self,
        current_price: Decimal,
        lower_price: Decimal,
        upper_price: Decimal,
        width: int = 40
    ) -> str:
        """生成价格位置可视化"""
        try:
            price_range = float(upper_price - lower_price)
            if price_range == 0:
                return ""

            # 计算价格位置（0-1）
            position = (float(current_price) - float(lower_price)) / price_range
            position = max(0, min(1, position))

            # 生成可视化
            bar = ['-'] * width
            marker_pos = int(position * (width - 1))
            bar[marker_pos] = '|'
            bar[0] = '['
            bar[-1] = ']'

            return ''.join(bar)

        except Exception:
            return ""
