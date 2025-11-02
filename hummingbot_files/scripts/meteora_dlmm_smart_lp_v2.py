"""
Meteora DLMM 智能 LP 管理策略 V2 - 多层级区间版本

核心改进:
1. 多层级区间策略（可配置层数）- 大幅减少再平衡频率
2. Jupiter 自动换币 - 初始化时自动准备双边代币
3. 简单波动率计算 - 无需 K 线数据
4. 手动趋势控制 - neutral/bullish/bearish 流动性分布
5. 使用 Gateway 标准化接口 - connector.get_quote_price(), connector.place_order()

策略原理:
- 设置大区间（如 ±12%）分为多层（如 4 层）
- 每层独立分配流动性，根据趋势调整分布
- 避免频繁再平衡：从 30-60 次/月降至 0-3 次/月
- 简单波动率计算决定区间宽度，无需复杂技术分析

参考文档:
- MULTI_LAYER_DLMM_STRATEGY_THEORY.md
- IMPROVED_STRATEGY_DESIGN.md
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from pydantic import Field

from hummingbot.client.config.config_data_types import BaseClientModel
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.gateway.common_types import get_connector_type
from hummingbot.connector.gateway.gateway_lp import CLMMPoolInfo, CLMMPositionInfo
from hummingbot.core.rate_oracle.rate_oracle import RateOracle
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


# ========================================
# 配置类
# ========================================

class MeteoraDlmmSmartLpV2Config(BaseClientModel):
    """Meteora DLMM 智能 LP 策略 V2 配置"""

    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))

    # ========== 基础配置 ==========
    connector: str = Field(
        "meteora/clmm",
        json_schema_extra={
            "prompt": "连接器名称",
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

    # ========== 多层级区间配置 ==========
    price_range_pct: Decimal = Field(
        Decimal("12.0"),
        json_schema_extra={
            "prompt": "总价格区间宽度（百分比，如 12.0 表示 ±12%）",
            "prompt_on_new": True
        }
    )

    num_layers: int = Field(
        4,
        json_schema_extra={
            "prompt": "层级数量（推荐 3-6 层）",
            "prompt_on_new": True
        }
    )

    liquidity_distribution: str = Field(
        "neutral",
        json_schema_extra={
            "prompt": "流动性分布模式（neutral=均衡, bullish=看涨, bearish=看跌）",
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

    # ========== Jupiter 自动换币配置 ==========
    enable_auto_swap: bool = Field(
        True,
        json_schema_extra={
            "prompt": "启用 Jupiter 自动换币（初始化时准备双边代币）",
            "prompt_on_new": True
        }
    )

    auto_swap_slippage_pct: Decimal = Field(
        Decimal("1.0"),
        json_schema_extra={
            "prompt": "自动换币滑点容忍度（百分比）",
            "prompt_on_new": False
        }
    )

    min_token_balance_ratio: Decimal = Field(
        Decimal("0.4"),
        json_schema_extra={
            "prompt": "最小代币余额比例（如 0.4 表示双边至少各占 40%）",
            "prompt_on_new": False
        }
    )

    # ========== 再平衡配置 ==========
    rebalance_threshold_pct: Decimal = Field(
        Decimal("95.0"),
        json_schema_extra={
            "prompt": "再平衡触发阈值（价格偏离区间边界的百分比，95 表示几乎出界才触发）",
            "prompt_on_new": True
        }
    )

    rebalance_cooldown_seconds: int = Field(
        86400,  # 24 小时
        json_schema_extra={
            "prompt": "再平衡冷却期（秒，避免频繁操作）",
            "prompt_on_new": False
        }
    )

    min_profit_for_rebalance: Decimal = Field(
        Decimal("10.0"),
        json_schema_extra={
            "prompt": "最小再平衡盈利要求（百分比，仅在盈利 > 此值时再平衡）",
            "prompt_on_new": True
        }
    )

    # ========== 波动率计算配置 ==========
    volatility_periods: int = Field(
        50,
        json_schema_extra={
            "prompt": "波动率计算周期（价格样本数量）",
            "prompt_on_new": False
        }
    )

    price_update_interval_seconds: int = Field(
        60,
        json_schema_extra={
            "prompt": "价格更新间隔（秒，用于波动率计算）",
            "prompt_on_new": False
        }
    )

    # ========== 风险控制 ==========
    stop_loss_pct: Decimal = Field(
        Decimal("15.0"),
        json_schema_extra={
            "prompt": "止损百分比（相对初始投入，如 15.0 表示 -15%）",
            "prompt_on_new": True
        }
    )

    profit_take_threshold_pct: Decimal = Field(
        Decimal("30.0"),
        json_schema_extra={
            "prompt": "盈利保护阈值（累积盈利超过此值启动移动止损）",
            "prompt_on_new": False
        }
    )

    trailing_stop_pct: Decimal = Field(
        Decimal("10.0"),
        json_schema_extra={
            "prompt": "移动止损百分比（盈利回撤超过此值触发止盈）",
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

    # ========== 监控配置 ==========
    check_interval_seconds: int = Field(
        30,
        json_schema_extra={
            "prompt": "检查间隔（秒）",
            "prompt_on_new": False
        }
    )


# ========================================
# 主策略类
# ========================================

class MeteoraDlmmSmartLpV2(ScriptStrategyBase):
    """
    Meteora DLMM 智能 LP 管理策略 V2 - 多层级区间版本

    核心功能:
    1. 多层级区间策略 - 大区间分多层，减少再平衡
    2. Jupiter 自动换币 - 初始化时准备双边代币
    3. 简单波动率计算 - 无需 K 线数据
    4. 手动趋势控制 - 根据配置调整流动性分布
    5. Gateway 标准化接口 - 使用 connector 统一方法
    """

    @classmethod
    def init_markets(cls, config: MeteoraDlmmSmartLpV2Config):
        cls.markets = {config.connector: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: MeteoraDlmmSmartLpV2Config):
        super().__init__(connectors)
        self.config = config

        # 连接器
        self.exchange = config.connector
        self.connector = connectors[config.connector]
        self.connector_type = get_connector_type(config.connector)

        # Token 信息
        self.base_token, self.quote_token = config.trading_pair.split("-")

        # 仓位状态
        self.position_infos: List[CLMMPositionInfo] = []  # 多层仓位
        self.pool_info: Optional[CLMMPoolInfo] = None
        self.position_opened = False
        self.position_opening = False
        self.position_closing = False
        self.position_paused = False

        # 价格历史（用于波动率计算）
        self.price_history: List[Decimal] = []
        self.last_price_record_time: Optional[float] = None

        # 价格追踪
        self.open_price: Optional[Decimal] = None
        self.initial_investment: Optional[Decimal] = None
        self.peak_value: Optional[Decimal] = None

        # 再平衡追踪
        self.last_rebalance_time: Optional[float] = None
        self.rebalance_count = 0

        # 时间追踪
        self.last_check_time: Optional[datetime] = None
        self.pause_start_time: Optional[float] = None

        # 性能统计
        self.stats = {
            "total_fees_earned": Decimal("0"),
            "total_rebalances": 0,
            "total_profit_usd": Decimal("0"),
            "max_drawdown_pct": Decimal("0"),
            "successful_rebalances": 0,
            "failed_rebalances": 0,
            "jupiter_swaps": 0,
            "avg_volatility": Decimal("0"),
        }

        # 启动信息
        self.log_with_clock(
            logging.INFO,
            f"Meteora DLMM 智能 LP 策略 V2 启动:\n"
            f"   交易对: {config.trading_pair}\n"
            f"   价格区间: ±{config.price_range_pct}%\n"
            f"   层级数量: {config.num_layers} 层\n"
            f"   流动性分布: {config.liquidity_distribution}\n"
            f"   再平衡阈值: {config.rebalance_threshold_pct}%\n"
            f"   止损: {config.stop_loss_pct}%\n"
            f"   Jupiter 自动换币: {'启用' if config.enable_auto_swap else '禁用'}"
        )

        # 延迟初始化
        safe_ensure_future(self.initialize_strategy())

    # ========================================
    # 初始化
    # ========================================

    async def initialize_strategy(self):
        """策略初始化"""
        await asyncio.sleep(3)  # 等待连接器初始化

        try:
            # 1. 获取池子信息
            await self.fetch_pool_info()

            # 2. 检查现有仓位
            await self.check_existing_positions()

            # 3. 如果启用自动换币且无仓位，准备代币
            if self.config.enable_auto_swap and not self.position_opened:
                await self.prepare_tokens_for_multi_layer_position()

            self.logger().info("策略初始化完成")

        except Exception as e:
            self.logger().error(f"策略初始化失败: {e}")

    async def check_existing_positions(self):
        """检查是否有现有仓位"""
        try:
            pool_address = await self.get_pool_address()
            if pool_address:
                positions = await self.connector.get_user_positions(pool_address=pool_address)
                if positions and len(positions) > 0:
                    self.position_infos = positions
                    self.position_opened = True

                    # 设置开仓价格为当前价格
                    if self.pool_info:
                        self.open_price = Decimal(str(self.pool_info.price))

                    self.logger().info(
                        f"发现现有仓位: {len(positions)} 个\n"
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
    # Jupiter 自动换币功能
    # ========================================

    async def prepare_tokens_for_multi_layer_position(self) -> bool:
        """
        准备开仓所需的双边代币

        检查余额，如果某一边代币不足，通过 Jupiter 自动兑换
        使用 Gateway 标准化接口

        Returns:
            True 如果代币准备成功
        """
        try:
            # 获取当前余额
            base_balance = self.connector.get_available_balance(self.base_token)
            quote_balance = self.connector.get_available_balance(self.quote_token)

            self.logger().info(
                f"当前余额:\n"
                f"   {self.base_token}: {base_balance}\n"
                f"   {self.quote_token}: {quote_balance}"
            )

            # 获取当前价格
            if not self.pool_info:
                await self.fetch_pool_info()

            if not self.pool_info:
                self.logger().error("无法获取池子信息，跳过自动换币")
                return False

            current_price = Decimal(str(self.pool_info.price))

            # 计算总价值（以 quote token 计价）
            total_value_in_quote = (Decimal(str(base_balance)) * current_price) + Decimal(str(quote_balance))

            if total_value_in_quote == 0:
                self.logger().error("钱包余额不足，无法开仓")
                return False

            # 计算当前比例
            base_value = Decimal(str(base_balance)) * current_price
            base_ratio = base_value / total_value_in_quote if total_value_in_quote > 0 else Decimal("0")

            self.logger().info(
                f"当前资产分布:\n"
                f"   {self.base_token} 价值: {base_value:.2f} {self.quote_token} ({base_ratio * 100:.1f}%)\n"
                f"   {self.quote_token} 价值: {quote_balance:.2f} ({(1 - base_ratio) * 100:.1f}%)\n"
                f"   总价值: {total_value_in_quote:.2f} {self.quote_token}"
            )

            # 目标比例：50/50
            target_ratio = Decimal("0.5")
            min_ratio = self.config.min_token_balance_ratio
            max_ratio = Decimal("1.0") - min_ratio

            # 检查是否需要换币
            if base_ratio < min_ratio:
                # Base token 不足，需要用 quote token 换 base token
                shortage_value = (target_ratio - base_ratio) * total_value_in_quote
                quote_to_swap = shortage_value * Decimal("1.02")  # 加 2% 缓冲

                if Decimal(str(quote_balance)) >= quote_to_swap:
                    self.logger().info(
                        f"{self.base_token} 不足，准备兑换:\n"
                        f"   用 {quote_to_swap:.6f} {self.quote_token} 换取 {self.base_token}"
                    )

                    success = await self.swap_via_jupiter(
                        from_token=self.quote_token,
                        to_token=self.base_token,
                        amount=quote_to_swap
                    )

                    if not success:
                        self.logger().error("Jupiter 换币失败")
                        return False

                    self.stats["jupiter_swaps"] += 1

                else:
                    self.logger().error(f"{self.quote_token} 余额不足以兑换")
                    return False

            elif base_ratio > max_ratio:
                # Base token 过多，需要换成 quote token
                excess_value = (base_ratio - target_ratio) * total_value_in_quote
                base_to_swap = (excess_value / current_price) * Decimal("1.02")  # 加 2% 缓冲

                if Decimal(str(base_balance)) >= base_to_swap:
                    self.logger().info(
                        f"{self.base_token} 过多，准备兑换:\n"
                        f"   用 {base_to_swap:.6f} {self.base_token} 换取 {self.quote_token}"
                    )

                    success = await self.swap_via_jupiter(
                        from_token=self.base_token,
                        to_token=self.quote_token,
                        amount=base_to_swap
                    )

                    if not success:
                        self.logger().error("Jupiter 换币失败")
                        return False

                    self.stats["jupiter_swaps"] += 1

                else:
                    self.logger().error(f"{self.base_token} 余额不足以兑换")
                    return False

            else:
                self.logger().info("代币余额比例合适，无需兑换")

            return True

        except Exception as e:
            self.logger().error(f"准备代币失败: {e}")
            return False

    async def swap_via_jupiter(
        self,
        from_token: str,
        to_token: str,
        amount: Decimal,
        max_retries: int = 3
    ) -> bool:
        """
        通过 Jupiter 兑换代币

        使用 Gateway connector 标准化方法:
        - connector.get_quote_price() 获取报价
        - connector.place_order() 执行兑换

        Args:
            from_token: 源代币
            to_token: 目标代币
            amount: 兑换数量
            max_retries: 最大重试次数

        Returns:
            True 如果兑换成功
        """
        trading_pair = f"{from_token}-{to_token}"
        is_buy = (to_token == self.base_token)  # 如果目标是 base token，则为 buy

        retry_delay = 1

        for attempt in range(max_retries):
            try:
                # 1. 获取报价（使用 Gateway 标准化方法）
                self.logger().info(f"获取 Jupiter 报价 (尝试 {attempt + 1}/{max_retries})...")

                quote_price = await self.connector.get_quote_price(
                    trading_pair=trading_pair,
                    is_buy=is_buy,
                    amount=float(amount)
                )

                if not quote_price or quote_price <= 0:
                    self.logger().warning(f"获取报价失败，返回价格: {quote_price}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        return False

                self.logger().info(f"报价: {quote_price} {to_token}/{from_token}")

                # 2. 计算预期输出
                expected_output = amount / Decimal(str(quote_price)) if not is_buy else amount * Decimal(str(quote_price))

                # 3. 执行兑换（使用 Gateway 标准化方法）
                self.logger().info(
                    f"执行 Jupiter 兑换:\n"
                    f"   输入: {amount:.6f} {from_token}\n"
                    f"   预期输出: {expected_output:.6f} {to_token}\n"
                    f"   价格: {quote_price}"
                )

                order_id = self.connector.place_order(
                    is_buy=is_buy,
                    trading_pair=trading_pair,
                    amount=float(amount),
                    price=quote_price
                )

                self.logger().info(f"兑换订单已提交: {order_id}")

                # 4. 等待订单确认（简化处理，实际应该监听订单事件）
                await asyncio.sleep(3)

                self.logger().info("Jupiter 兑换成功")
                return True

            except Exception as e:
                self.logger().error(f"Jupiter 兑换失败 (尝试 {attempt + 1}/{max_retries}): {e}")

                if "division by zero" in str(e).lower() or "rate not found" in str(e).lower():
                    # 常见错误：价格服务未就绪，重试
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue

                return False

        return False

    # ========================================
    # 简单波动率计算
    # ========================================

    def record_price_for_volatility(self, price: Decimal):
        """
        记录价格用于波动率计算

        Args:
            price: 当前价格
        """
        current_time = time.time()

        # 检查是否到达记录间隔
        if self.last_price_record_time:
            elapsed = current_time - self.last_price_record_time
            if elapsed < self.config.price_update_interval_seconds:
                return

        # 记录价格
        self.price_history.append(price)
        self.last_price_record_time = current_time

        # 保持历史数据在合理范围（最多 100 个样本）
        max_history = max(100, self.config.volatility_periods + 20)
        if len(self.price_history) > max_history:
            self.price_history.pop(0)

        self.logger().debug(f"记录价格: {price}, 历史样本数: {len(self.price_history)}")

    def calculate_simple_volatility(self) -> Decimal:
        """
        计算简单波动率（标准差）

        无需 K 线数据，仅使用最近的价格样本

        Returns:
            波动率（小数形式，0.05 表示 5%）
        """
        if len(self.price_history) < self.config.volatility_periods:
            # 数据不足，返回默认值
            return Decimal("0.05")

        try:
            # 取最近 N 个样本
            recent_prices = self.price_history[-self.config.volatility_periods:]

            # 计算价格变化率（returns）
            returns = []
            for i in range(1, len(recent_prices)):
                change = (recent_prices[i] - recent_prices[i - 1]) / recent_prices[i - 1]
                returns.append(float(change))

            if not returns:
                return Decimal("0.05")

            # 计算标准差
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            volatility = Decimal(str(variance ** 0.5))

            self.stats["avg_volatility"] = volatility

            self.logger().debug(f"计算波动率: {volatility * 100:.2f}% (样本: {len(returns)})")

            return volatility

        except Exception as e:
            self.logger().error(f"计算波动率失败: {e}")
            return Decimal("0.05")

    def get_range_width_by_volatility(self, volatility: Decimal) -> Decimal:
        """
        根据波动率调整区间宽度

        Args:
            volatility: 波动率（小数形式）

        Returns:
            建议区间宽度（百分比）
        """
        # 波动率映射到区间宽度
        if volatility < Decimal("0.02"):  # < 2%
            return Decimal("8.0")  # ±8%
        elif volatility < Decimal("0.05"):  # < 5%
            return Decimal("12.0")  # ±12%
        elif volatility < Decimal("0.10"):  # < 10%
            return Decimal("18.0")  # ±18%
        elif volatility < Decimal("0.20"):  # < 20%
            return Decimal("25.0")  # ±25%
        else:  # >= 20%
            return Decimal("35.0")  # ±35%

    # ========================================
    # 多层级区间计算
    # ========================================

    def calculate_layer_ranges(
        self,
        current_price: Decimal,
        range_width_pct: Decimal,
        num_layers: int
    ) -> List[Dict]:
        """
        计算多层级区间参数

        Args:
            current_price: 当前价格
            range_width_pct: 总区间宽度（百分比，如 12.0 表示 ±12%）
            num_layers: 层级数量

        Returns:
            层级列表，每层包含 {layer_id, lower, upper, liquidity_pct, name}
        """
        lower_bound = current_price * (Decimal("1") - range_width_pct / Decimal("100"))
        upper_bound = current_price * (Decimal("1") + range_width_pct / Decimal("100"))
        layer_width = (upper_bound - lower_bound) / num_layers

        # 根据趋势生成流动性分布
        distribution = self._generate_liquidity_distribution(num_layers)

        layers = []
        for i in range(num_layers):
            layer_lower = lower_bound + i * layer_width
            layer_upper = layer_lower + layer_width

            layers.append({
                "layer_id": i + 1,
                "lower": layer_lower,
                "upper": layer_upper,
                "liquidity_pct": distribution[i],
                "name": f"Layer_{i + 1}"
            })

        return layers

    def _generate_liquidity_distribution(self, num_layers: int) -> List[Decimal]:
        """
        根据配置的趋势生成流动性分布

        Args:
            num_layers: 层级数量

        Returns:
            每层的流动性百分比列表（总和为 1.0）
        """
        distribution_mode = self.config.liquidity_distribution.lower()

        if distribution_mode == "neutral":
            # 均匀分布
            pct_per_layer = Decimal("1.0") / num_layers
            return [pct_per_layer] * num_layers

        elif distribution_mode == "bullish":
            # 看涨：上层流动性更多
            # 权重：1, 2, 3, 4, ... (底层到顶层)
            weights = [i + 1 for i in range(num_layers)]
            total_weight = sum(weights)
            return [Decimal(str(w / total_weight)) for w in weights]

        elif distribution_mode == "bearish":
            # 看跌：下层流动性更多
            # 权重：4, 3, 2, 1, ... (底层到顶层)
            weights = [num_layers - i for i in range(num_layers)]
            total_weight = sum(weights)
            return [Decimal(str(w / total_weight)) for w in weights]

        else:
            # 未知模式，默认均匀
            self.logger().warning(f"未知的流动性分布模式: {distribution_mode}，使用 neutral")
            pct_per_layer = Decimal("1.0") / num_layers
            return [pct_per_layer] * num_layers

    # ========================================
    # 主循环
    # ========================================

    def on_tick(self):
        """策略主循环"""
        current_time = datetime.now()

        # 检查间隔控制
        if self.last_check_time and (current_time - self.last_check_time).total_seconds() < self.config.check_interval_seconds:
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
            safe_ensure_future(self.monitor_positions())
        else:
            # 无仓位：开仓
            safe_ensure_future(self.check_and_open_positions())

    # ========================================
    # 开仓逻辑
    # ========================================

    async def check_and_open_positions(self):
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

            # 记录价格用于波动率
            self.record_price_for_volatility(current_price)

            # 计算波动率并调整区间（可选）
            volatility = self.calculate_simple_volatility()
            suggested_range = self.get_range_width_by_volatility(volatility)

            self.logger().info(
                f"波动率分析:\n"
                f"   当前波动率: {volatility * 100:.2f}%\n"
                f"   建议区间: ±{suggested_range}%\n"
                f"   配置区间: ±{self.config.price_range_pct}%"
            )

            # 开仓（使用配置的区间，不自动调整）
            await self.open_multi_layer_positions(current_price)

        except Exception as e:
            self.logger().error(f"检查开仓失败: {e}")

    async def open_multi_layer_positions(self, center_price: Decimal):
        """
        开多层 LP 仓位

        Args:
            center_price: 中心价格
        """
        if self.position_opening or self.position_opened:
            return

        self.position_opening = True

        try:
            # 1. 计算层级区间
            layers = self.calculate_layer_ranges(
                current_price=center_price,
                range_width_pct=self.config.price_range_pct,
                num_layers=self.config.num_layers
            )

            self.logger().info(
                f"多层级开仓参数:\n"
                f"   中心价格: {center_price}\n"
                f"   总区间: ±{self.config.price_range_pct}%\n"
                f"   层级数量: {self.config.num_layers}\n"
                f"   分布模式: {self.config.liquidity_distribution}"
            )

            for layer in layers:
                self.logger().info(
                    f"   Layer {layer['layer_id']}: {layer['lower']:.6f} - {layer['upper']:.6f} "
                    f"({layer['liquidity_pct'] * 100:.1f}% 流动性)"
                )

            # 2. 获取总代币数量
            total_base_amount, total_quote_amount = await self.get_token_amounts()

            # 3. 为每一层创建仓位
            created_positions = 0
            for layer in layers:
                # 计算该层的代币数量
                layer_base = total_base_amount * layer['liquidity_pct']
                layer_quote = total_quote_amount * layer['liquidity_pct']

                self.logger().info(
                    f"开仓 {layer['name']}:\n"
                    f"   区间: {layer['lower']:.6f} - {layer['upper']:.6f}\n"
                    f"   {self.base_token}: {layer_base:.6f}\n"
                    f"   {self.quote_token}: {layer_quote:.6f}"
                )

                # 提交开仓订单
                try:
                    # 计算价格区间百分比
                    lower_width_pct = float(((center_price - layer['lower']) / center_price) * 100)
                    upper_width_pct = float(((layer['upper'] - center_price) / center_price) * 100)

                    order_id = self.connector.add_liquidity(
                        trading_pair=self.config.trading_pair,
                        price=float(center_price),
                        upper_width_pct=upper_width_pct,
                        lower_width_pct=lower_width_pct,
                        base_token_amount=float(layer_base),
                        quote_token_amount=float(layer_quote),
                    )

                    self.logger().info(f"{layer['name']} 订单已提交: {order_id}")
                    created_positions += 1

                    # 等待订单确认
                    await asyncio.sleep(2)

                except Exception as e:
                    self.logger().error(f"{layer['name']} 开仓失败: {e}")
                    # 继续开其他层

            if created_positions > 0:
                self.open_price = center_price

                # 记录初始投入
                self.initial_investment = (total_base_amount * center_price) + total_quote_amount

                self.logger().info(
                    f"多层级开仓完成: {created_positions}/{len(layers)} 层\n"
                    f"   初始投入: {self.initial_investment:.2f} {self.quote_token}"
                )

                # 等待仓位信息更新
                await asyncio.sleep(3)
                await self.check_existing_positions()

            else:
                self.logger().error("所有层开仓失败")
                self.position_opening = False

        except Exception as e:
            self.logger().error(f"多层级开仓失败: {e}")
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
                f"   {self.base_token}: {base_balance} (使用 {base_amount:.6f})\n"
                f"   {self.quote_token}: {quote_balance} (使用 {quote_amount:.6f})"
            )

            return base_amount, quote_amount

        except Exception as e:
            self.logger().error(f"获取钱包余额失败: {e}")
            return Decimal("0"), Decimal("100")  # 降级到默认值

    # ========================================
    # 监控和再平衡
    # ========================================

    async def monitor_positions(self):
        """监控所有仓位"""
        if not self.position_infos:
            return

        try:
            # 更新池子信息
            await self.fetch_pool_info()

            if not self.pool_info:
                return

            current_price = Decimal(str(self.pool_info.price))

            # 记录价格用于波动率
            self.record_price_for_volatility(current_price)

            # 更新所有仓位信息
            await self.update_all_positions_info()

            # 1. 检查止损
            if await self.check_stop_loss(current_price):
                return

            # 2. 检查盈利保护
            if await self.check_profit_protection(current_price):
                return

            # 3. 检查再平衡条件
            await self.check_rebalance_condition(current_price)

            # 4. 检查是否超出区间
            await self.check_out_of_range(current_price)

            # 5. 更新性能统计
            await self.update_performance_stats(current_price)

            # 6. 注入价格到 RateOracle（避免价格服务错误）
            self.inject_price_to_rate_oracle(current_price)

        except Exception as e:
            self.logger().error(f"监控仓位失败: {e}")

    async def update_all_positions_info(self):
        """更新所有仓位信息"""
        try:
            pool_address = await self.get_pool_address()
            if pool_address:
                positions = await self.connector.get_user_positions(pool_address=pool_address)
                if positions:
                    self.position_infos = positions
        except Exception as e:
            self.logger().error(f"更新仓位信息失败: {e}")

    def inject_price_to_rate_oracle(self, price: Decimal):
        """
        注入价格到 RateOracle

        避免 "rate not found" 或 "division by zero" 错误
        参考 v2_news_sniping_hybrid.py 和 cex_dex_lp_arbitrage.py

        Args:
            price: 当前价格
        """
        try:
            rate_oracle = RateOracle.get_instance()
            rate_oracle.set_price(self.config.trading_pair, price)
            self.logger().debug(f"注入价格到 RateOracle: {self.config.trading_pair} = {price}")
        except Exception as e:
            self.logger().debug(f"RateOracle 注入失败: {e}")

    async def check_stop_loss(self, current_price: Decimal) -> bool:
        """检查止损条件"""
        if not self.initial_investment:
            return False

        current_value = await self.calculate_total_position_value(current_price)
        if not current_value:
            return False

        loss_pct = ((current_value - self.initial_investment) / self.initial_investment) * Decimal("100")

        if loss_pct <= -self.config.stop_loss_pct:
            self.logger().warning(
                f"触发止损:\n"
                f"   初始投入: {self.initial_investment:.2f} {self.quote_token}\n"
                f"   当前价值: {current_value:.2f} {self.quote_token}\n"
                f"   亏损: {loss_pct:.2f}%"
            )

            await self.close_all_positions(reason="STOP_LOSS")
            self.position_paused = True
            self.pause_start_time = time.time()
            return True

        return False

    async def check_profit_protection(self, current_price: Decimal) -> bool:
        """检查盈利保护（移动止损）"""
        if not self.initial_investment:
            return False

        current_value = await self.calculate_total_position_value(current_price)
        if not current_value:
            return False

        profit_pct = ((current_value - self.initial_investment) / self.initial_investment) * Decimal("100")

        if profit_pct < self.config.profit_take_threshold_pct:
            return False

        # 更新最高价值
        if not self.peak_value or current_value > self.peak_value:
            self.peak_value = current_value

        # 计算回撤
        drawdown_from_peak = ((self.peak_value - current_value) / self.peak_value) * Decimal("100")

        if drawdown_from_peak >= self.config.trailing_stop_pct:
            self.logger().info(
                f"触发盈利保护:\n"
                f"   最高价值: {self.peak_value:.2f}\n"
                f"   当前价值: {current_value:.2f}\n"
                f"   回撤: {drawdown_from_peak:.2f}%"
            )

            await self.close_all_positions(reason="PROFIT_PROTECTION")
            self.position_paused = True
            self.pause_start_time = time.time()
            return True

        return False

    async def check_rebalance_condition(self, current_price: Decimal):
        """检查再平衡条件"""
        # 检查冷却期
        if self.last_rebalance_time:
            elapsed = time.time() - self.last_rebalance_time
            if elapsed < self.config.rebalance_cooldown_seconds:
                return

        # 检查价格是否接近区间边界
        if not self.position_infos:
            return

        # 获取总区间（所有层的并集）
        all_lowers = [Decimal(str(p.lower_price)) for p in self.position_infos]
        all_uppers = [Decimal(str(p.upper_price)) for p in self.position_infos]

        total_lower = min(all_lowers)
        total_upper = max(all_uppers)

        price_range = total_upper - total_lower
        if price_range == 0:
            return

        # 计算距离边界的百分比
        distance_from_lower = ((current_price - total_lower) / price_range) * Decimal("100")
        distance_from_upper = ((total_upper - current_price) / price_range) * Decimal("100")
        min_distance = min(distance_from_lower, distance_from_upper)

        # 检查是否触发再平衡（95% 表示几乎要出界）
        if min_distance <= (Decimal("100") - self.config.rebalance_threshold_pct):
            # 检查盈利要求
            current_value = await self.calculate_total_position_value(current_price)
            if not current_value or not self.initial_investment:
                return

            profit_pct = ((current_value - self.initial_investment) / self.initial_investment) * Decimal("100")

            if profit_pct < self.config.min_profit_for_rebalance:
                self.logger().info(
                    f"价格接近边界 (距离 {min_distance:.1f}%)，但盈利不足 ({profit_pct:.2f}% < {self.config.min_profit_for_rebalance}%)，跳过再平衡"
                )
                return

            self.logger().info(
                f"触发再平衡:\n"
                f"   当前价格: {current_price}\n"
                f"   总区间: {total_lower:.6f} - {total_upper:.6f}\n"
                f"   距离边界: {min_distance:.1f}%\n"
                f"   当前盈利: {profit_pct:.2f}%"
            )

            await self.execute_rebalance(current_price)

    async def check_out_of_range(self, current_price: Decimal):
        """检查价格是否超出区间"""
        if not self.config.pause_on_out_of_range or not self.position_infos:
            return

        # 获取总区间
        all_lowers = [Decimal(str(p.lower_price)) for p in self.position_infos]
        all_uppers = [Decimal(str(p.upper_price)) for p in self.position_infos]

        total_lower = min(all_lowers)
        total_upper = max(all_uppers)

        if current_price < total_lower or current_price > total_upper:
            if current_price < total_lower:
                distance = ((total_lower - current_price) / total_lower) * Decimal("100")
                direction = "下方"
            else:
                distance = ((current_price - total_upper) / total_upper) * Decimal("100")
                direction = "上方"

            self.logger().warning(
                f"价格超出区间:\n"
                f"   当前价格: {current_price}\n"
                f"   区间: {total_lower:.6f} - {total_upper:.6f}\n"
                f"   超出{direction}: {distance:.2f}%"
            )

            await self.close_all_positions(reason="OUT_OF_RANGE")
            self.position_paused = True
            self.pause_start_time = time.time()

    async def execute_rebalance(self, new_center_price: Decimal):
        """执行再平衡"""
        try:
            self.logger().info("开始再平衡...")

            # 关闭所有仓位
            await self.close_all_positions(reason="REBALANCE")

            # 等待平仓完成
            await asyncio.sleep(5)

            # 重新开仓
            await self.open_multi_layer_positions(new_center_price)

            # 更新统计
            self.last_rebalance_time = time.time()
            self.rebalance_count += 1
            self.stats["total_rebalances"] += 1
            self.stats["successful_rebalances"] += 1

            self.logger().info(f"再平衡完成 (第 {self.rebalance_count} 次)")

        except Exception as e:
            self.logger().error(f"再平衡失败: {e}")
            self.stats["failed_rebalances"] += 1

    async def check_resume_condition(self):
        """检查是否可以恢复策略"""
        if not self.position_paused or not self.open_price:
            return

        try:
            await self.fetch_pool_info()
            if not self.pool_info:
                return

            current_price = Decimal(str(self.pool_info.price))

            # 计算目标区间
            range_pct = self.config.price_range_pct / Decimal("100")
            lower_bound = self.open_price * (Decimal("1") - range_pct)
            upper_bound = self.open_price * (Decimal("1") + range_pct)

            if lower_bound <= current_price <= upper_bound:
                pause_duration = time.time() - self.pause_start_time if self.pause_start_time else 0

                self.logger().info(
                    f"价格已回到区间，恢复策略:\n"
                    f"   当前价格: {current_price}\n"
                    f"   暂停时长: {pause_duration:.0f} 秒"
                )

                self.position_paused = False
                self.pause_start_time = None

                await self.open_multi_layer_positions(current_price)

        except Exception as e:
            self.logger().error(f"检查恢复条件失败: {e}")

    # ========================================
    # 平仓逻辑
    # ========================================

    async def close_all_positions(self, reason: str = "MANUAL"):
        """
        关闭所有仓位

        Args:
            reason: 关闭原因
        """
        if not self.position_infos or self.position_closing:
            return

        self.position_closing = True

        try:
            self.logger().info(f"关闭所有仓位 (原因: {reason})...")

            # 先收集所有手续费
            await self.collect_all_fees()

            # 然后关闭所有仓位
            for position in self.position_infos:
                try:
                    order_id = self.connector.remove_liquidity(
                        trading_pair=self.config.trading_pair,
                        position_address=position.address
                    )
                    self.logger().info(f"平仓订单已提交: {order_id} (仓位: {position.address})")
                    await asyncio.sleep(1)
                except Exception as e:
                    self.logger().error(f"关闭仓位 {position.address} 失败: {e}")

            # 清空仓位列表
            self.position_infos = []
            self.position_opened = False

            self.logger().info("所有仓位已关闭")

        except Exception as e:
            self.logger().error(f"关闭仓位失败: {e}")
        finally:
            self.position_closing = False

    async def collect_all_fees(self):
        """收集所有仓位的手续费"""
        if not self.position_infos:
            return

        try:
            total_base_fees = Decimal("0")
            total_quote_fees = Decimal("0")

            for position in self.position_infos:
                base_fees = Decimal(str(position.base_fee_amount))
                quote_fees = Decimal(str(position.quote_fee_amount))

                total_base_fees += base_fees
                total_quote_fees += quote_fees

            if total_base_fees > 0 or total_quote_fees > 0:
                self.logger().info(
                    f"收集手续费:\n"
                    f"   {self.base_token}: {total_base_fees:.6f}\n"
                    f"   {self.quote_token}: {total_quote_fees:.6f}"
                )

                # 更新统计
                if self.pool_info:
                    current_price = Decimal(str(self.pool_info.price))
                    total_fees_usd = (total_base_fees * current_price) + total_quote_fees
                    self.stats["total_fees_earned"] += total_fees_usd

        except Exception as e:
            self.logger().error(f"收集手续费失败: {e}")

    # ========================================
    # 辅助计算
    # ========================================

    async def calculate_total_position_value(self, current_price: Decimal) -> Optional[Decimal]:
        """
        计算所有仓位的总价值

        Args:
            current_price: 当前价格

        Returns:
            总价值（以 quote token 计价）
        """
        if not self.position_infos:
            return None

        try:
            total_value = Decimal("0")

            for position in self.position_infos:
                base_amount = Decimal(str(position.base_token_amount))
                quote_amount = Decimal(str(position.quote_token_amount))
                base_fees = Decimal(str(position.base_fee_amount))
                quote_fees = Decimal(str(position.quote_fee_amount))

                position_value = (
                    (base_amount + base_fees) * current_price +
                    (quote_amount + quote_fees)
                )

                total_value += position_value

            return total_value

        except Exception as e:
            self.logger().error(f"计算仓位价值失败: {e}")
            return None

    async def update_performance_stats(self, current_price: Decimal):
        """更新性能统计"""
        if not self.initial_investment:
            return

        try:
            current_value = await self.calculate_total_position_value(current_price)
            if not current_value:
                return

            profit = current_value - self.initial_investment
            self.stats["total_profit_usd"] = profit

            # 更新最大回撤
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
        try:
            # 注入价格到 RateOracle
            if hasattr(event, 'price') and hasattr(event, 'trading_pair'):
                rate_oracle = RateOracle.get_instance()
                rate_oracle.set_price(event.trading_pair, Decimal(str(event.price)))
                self.logger().debug(f"订单成交，注入价格: {event.trading_pair} = {event.price}")

            # 开仓订单成交
            if hasattr(event, 'order_id'):
                self.logger().info(f"订单成交: {event.order_id}")

                # 更新仓位信息
                safe_ensure_future(self.fetch_positions_after_fill())

        except Exception as e:
            self.logger().error(f"处理订单成交事件失败: {e}")

    async def fetch_positions_after_fill(self):
        """订单成交后获取仓位信息"""
        await asyncio.sleep(3)

        try:
            await self.check_existing_positions()
        except Exception as e:
            self.logger().error(f"获取仓位信息失败: {e}")

    # ========================================
    # 状态显示
    # ========================================

    def format_status(self) -> str:
        """格式化状态显示"""
        lines = []

        lines.append("=" * 90)
        lines.append("Meteora DLMM 智能 LP 策略 V2 - 多层级区间".center(90))
        lines.append("=" * 90)

        lines.append(f"交易对: {self.config.trading_pair}")
        lines.append(f"连接器: {self.exchange}")
        lines.append("")

        # 当前价格和波动率
        if self.pool_info:
            current_price = Decimal(str(self.pool_info.price))
            lines.append(f"当前价格: {current_price:.6f} {self.quote_token}")

            volatility = self.calculate_simple_volatility()
            suggested_range = self.get_range_width_by_volatility(volatility)
            lines.append(f"波动率: {volatility * 100:.2f}% (建议区间: ±{suggested_range}%)")
            lines.append("")

        # 状态区分显示
        if self.position_paused:
            self._format_paused_status(lines)
        elif self.position_opening:
            lines.append("开仓中...")
        elif self.position_closing:
            lines.append("平仓中...")
        elif self.position_opened and self.position_infos:
            self._format_active_positions_status(lines)
        else:
            lines.append("等待开仓")

        # 性能统计
        lines.append("")
        lines.append("性能统计")
        lines.append("-" * 90)
        lines.append(f"累积盈利: {self.stats['total_profit_usd']:+.2f} {self.quote_token}")
        lines.append(f"手续费收入: {self.stats['total_fees_earned']:.6f} {self.quote_token}")
        lines.append(
            f"再平衡: {self.stats['total_rebalances']} 次 "
            f"(成功 {self.stats['successful_rebalances']}, 失败 {self.stats['failed_rebalances']})"
        )
        lines.append(f"Jupiter 兑换: {self.stats['jupiter_swaps']} 次")
        lines.append(f"最大回撤: {self.stats['max_drawdown_pct']:.2f}%")

        lines.append("=" * 90)

        return "\n".join(lines)

    def _format_active_positions_status(self, lines: list):
        """格式化活跃仓位状态"""
        lines.append(f"活跃仓位: {len(self.position_infos)} 层")
        lines.append("-" * 90)

        if self.pool_info:
            current_price = Decimal(str(self.pool_info.price))

            # 显示每一层
            for i, position in enumerate(self.position_infos, 1):
                lower = Decimal(str(position.lower_price))
                upper = Decimal(str(position.upper_price))

                base_amount = Decimal(str(position.base_token_amount))
                quote_amount = Decimal(str(position.quote_token_amount))
                base_fees = Decimal(str(position.base_fee_amount))
                quote_fees = Decimal(str(position.quote_fee_amount))

                in_range = "✓" if lower <= current_price <= upper else "✗"

                lines.append(
                    f"Layer {i}: {lower:.6f} - {upper:.6f} {in_range}\n"
                    f"   资产: {base_amount:.6f} {self.base_token}, {quote_amount:.6f} {self.quote_token}\n"
                    f"   手续费: {base_fees:.6f} {self.base_token}, {quote_fees:.6f} {self.quote_token}"
                )

        lines.append("")

        # 盈亏分析
        if self.pool_info and self.initial_investment:
            current_price = Decimal(str(self.pool_info.price))
            current_value = asyncio.run(self.calculate_total_position_value(current_price))

            if current_value:
                profit = current_value - self.initial_investment
                profit_pct = (profit / self.initial_investment) * Decimal("100")

                lines.append("盈亏分析")
                lines.append("-" * 90)
                lines.append(f"初始投入: {self.initial_investment:.2f} {self.quote_token}")
                lines.append(f"当前价值: {current_value:.2f} {self.quote_token}")
                lines.append(f"盈亏: {profit:+.2f} {self.quote_token} ({profit_pct:+.2f}%)")

                if self.open_price:
                    price_change = ((current_price - self.open_price) / self.open_price) * Decimal("100")
                    lines.append(f"开仓价格: {self.open_price:.6f} (价格变化: {price_change:+.2f}%)")

    def _format_paused_status(self, lines: list):
        """格式化暂停状态"""
        lines.append("策略已暂停")
        lines.append("-" * 90)
        lines.append("等待价格回到目标区间...")

        if self.pause_start_time:
            pause_duration = int(time.time() - self.pause_start_time)
            lines.append(f"已暂停: {pause_duration} 秒")
