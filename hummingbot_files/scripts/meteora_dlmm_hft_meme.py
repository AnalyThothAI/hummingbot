"""
Meteora DLMM 高频做市策略 - Meme 币专用

核心特性:
1. 高频再平衡 - 分钟级响应（60-300秒冷却）
2. 快速止损 - 60秒规则 + 5%幅度止损
3. 趋势跟随 - 区间紧跟价格，不等回调
4. 交易量监控 - 识别市场冷却，及时退出
5. 极窄区间 - 5-10%区间，最大化手续费

策略原理:
- 主动优于被动：快速调整区间跟随价格
- 60秒规则：价格超出区间 > 60秒立即再平衡
- 5%止损：下跌超过 5%立即退出
- 紧跟价格：区间始终围绕当前价格（±5-10%）

⚠️ 警告:
- 本策略风险极高，仅适用于 meme 币高波动期
- 不适合保守投资者
- 建议单次投入 < 总资金的 20%
- 务必在 devnet 测试后再用于 mainnet

参考文档: MEME_HIGH_FREQUENCY_STRATEGY.md
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

class MeteoraDlmmHftMemeConfig(BaseClientModel):
    """Meteora DLMM 高频做市策略配置（Meme 币专用）"""

    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))

    # ========== 基础配置 ==========
    connector: str = Field(
        "meteora/clmm",
        json_schema_extra={"prompt": "LP 连接器", "prompt_on_new": True}
    )

    swap_connector: str = Field(
        "jupiter/router",
        json_schema_extra={"prompt": "Swap 连接器", "prompt_on_new": True}
    )

    trading_pair: str = Field(
        "BONK-USDC",
        json_schema_extra={"prompt": "交易对（meme 币）", "prompt_on_new": True}
    )

    pool_address: str = Field(
        "",
        json_schema_extra={"prompt": "池子地址（可选）", "prompt_on_new": False}
    )

    # ========== 高频参数 ==========
    price_range_pct: Decimal = Field(
        Decimal("8.0"),
        json_schema_extra={"prompt": "价格区间宽度（建议 5-10%）", "prompt_on_new": True}
    )

    rebalance_threshold_pct: Decimal = Field(
        Decimal("75.0"),
        json_schema_extra={"prompt": "再平衡阈值（建议 70-80%）", "prompt_on_new": True}
    )

    rebalance_cooldown_seconds: int = Field(
        180,  # 3 分钟
        json_schema_extra={"prompt": "再平衡冷却期（秒，建议 60-300）", "prompt_on_new": True}
    )

    min_profit_for_rebalance: Decimal = Field(
        Decimal("2.0"),
        json_schema_extra={"prompt": "最小再平衡盈利（%，建议 1-3%）", "prompt_on_new": True}
    )

    # ========== 快速止损配置 ==========
    enable_60s_rule: bool = Field(
        True,
        json_schema_extra={"prompt": "启用 60秒规则？", "prompt_on_new": True}
    )

    out_of_range_timeout_seconds: int = Field(
        60,
        json_schema_extra={"prompt": "超出区间超时（秒）", "prompt_on_new": False}
    )

    stop_loss_pct: Decimal = Field(
        Decimal("5.0"),
        json_schema_extra={"prompt": "幅度止损（%，建议 5-8%）", "prompt_on_new": True}
    )

    enable_volume_monitoring: bool = Field(
        True,
        json_schema_extra={"prompt": "启用交易量监控？", "prompt_on_new": False}
    )

    volume_drop_threshold_pct: Decimal = Field(
        Decimal("80.0"),
        json_schema_extra={"prompt": "交易量骤降阈值（%）", "prompt_on_new": False}
    )

    # ========== 资金配置 ==========
    base_token_amount: Decimal = Field(Decimal("0.0"), json_schema_extra={"prompt_on_new": False})
    quote_token_amount: Decimal = Field(Decimal("0.0"), json_schema_extra={"prompt_on_new": False})
    wallet_allocation_pct: Decimal = Field(Decimal("80.0"), json_schema_extra={"prompt_on_new": False})

    # ========== Jupiter 自动换币 ==========
    enable_auto_swap: bool = Field(True, json_schema_extra={"prompt_on_new": False})
    auto_swap_slippage_pct: Decimal = Field(Decimal("3.0"), json_schema_extra={"prompt_on_new": False})

    # ========== 监控配置 ==========
    check_interval_seconds: int = Field(
        10,
        json_schema_extra={"prompt": "检查间隔（秒，建议 10）", "prompt_on_new": False}
    )

    # ========== 风控配置 ==========
    max_daily_loss_pct: Decimal = Field(
        Decimal("15.0"),
        json_schema_extra={"prompt": "日最大亏损（%）", "prompt_on_new": False}
    )

    max_position_hold_hours: Decimal = Field(
        Decimal("6.0"),
        json_schema_extra={"prompt": "最长持仓时间（小时）", "prompt_on_new": False}
    )


# ========================================
# 快速止损引擎
# ========================================

class FastStopLossEngine:
    """快速止损引擎"""

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config

        # 时间记录
        self.price_out_of_range_since: Optional[float] = None
        self.position_opened_at: Optional[float] = None

        # 交易量监控
        self.last_volume: Optional[Decimal] = None
        self.volume_history: List[Tuple[float, Decimal]] = []

    def reset(self):
        """重置状态"""
        self.price_out_of_range_since = None
        self.position_opened_at = time.time()

    def check_stop_loss(
        self,
        current_price: Decimal,
        open_price: Decimal,
        lower_price: Decimal,
        upper_price: Decimal,
        current_volume: Optional[Decimal] = None
    ) -> Tuple[bool, str, str]:
        """
        检查是否触发止损

        返回: (是否止损, 止损类型, 原因)
        止损类型: "HARD_STOP" (立即止损) or "SOFT_STOP" (建议止损)
        """

        now = time.time()

        # === Level 1: 幅度止损（最高优先级）===
        price_change_pct = (current_price - open_price) / open_price * Decimal("100")

        if price_change_pct <= -self.config.stop_loss_pct:
            return True, "HARD_STOP", f"下跌 {abs(price_change_pct):.2f}% 超过止损线 {self.config.stop_loss_pct}%"

        # === Level 2: 60秒规则 + 下跌 ===
        is_out_of_range = current_price < lower_price or current_price > upper_price

        if is_out_of_range:
            if self.price_out_of_range_since is None:
                self.price_out_of_range_since = now

            out_duration = now - self.price_out_of_range_since

            if self.config.enable_60s_rule and out_duration >= self.config.out_of_range_timeout_seconds:
                # 超出 60 秒，检查方向
                if current_price < lower_price and price_change_pct < -3:
                    # 下跌方向，立即止损
                    return True, "HARD_STOP", f"下跌超出区间 {out_duration:.0f}秒"
                else:
                    # 上涨方向，触发再平衡（不是止损）
                    return False, "REBALANCE", f"超出区间 {out_duration:.0f}秒，需要再平衡"
        else:
            # 价格回到区间内，重置计时
            self.price_out_of_range_since = None

        # === Level 3: 交易量骤降 ===
        if self.config.enable_volume_monitoring and current_volume is not None:
            self.volume_history.append((now, current_volume))

            # 保留最近 1 小时数据
            cutoff = now - 3600
            self.volume_history = [(t, v) for t, v in self.volume_history if t >= cutoff]

            if len(self.volume_history) >= 2:
                recent_volume = current_volume
                hour_ago_volume = self.volume_history[0][1]

                if hour_ago_volume > 0:
                    volume_change_pct = (recent_volume - hour_ago_volume) / hour_ago_volume * Decimal("100")

                    if volume_change_pct <= -self.config.volume_drop_threshold_pct:
                        return True, "SOFT_STOP", f"交易量骤降 {abs(volume_change_pct):.1f}%，市场冷却"

        # === Level 4: 持仓时长 ===
        if self.position_opened_at is not None:
            hold_hours = (now - self.position_opened_at) / 3600

            if hold_hours >= float(self.config.max_position_hold_hours):
                # 持仓过久且未盈利
                if price_change_pct < 0:
                    return True, "SOFT_STOP", f"持仓 {hold_hours:.1f}h 未盈利"

        # 无止损触发
        return False, "NONE", ""


# ========================================
# 高频再平衡决策引擎
# ========================================

class HighFrequencyRebalanceEngine:
    """高频再平衡决策引擎"""

    def __init__(self, logger):
        self.logger = logger
        self.last_rebalance_time: Optional[float] = None

    async def should_rebalance(
        self,
        current_price: Decimal,
        lower_price: Decimal,
        upper_price: Decimal,
        accumulated_fees_value: Decimal,
        position_value: Decimal,
        config,
        out_duration_seconds: float
    ) -> Tuple[bool, str]:
        """
        高频再平衡决策

        返回: (是否再平衡, 原因)
        """

        # === 因子 1: 冷却期检查 ===
        if not self._is_cooldown_passed(config.rebalance_cooldown_seconds):
            return False, f"冷却期未过（剩余 {self._remaining_cooldown(config.rebalance_cooldown_seconds):.0f}秒）"

        # === 因子 2: 超出区间检查 ===
        is_out_of_range = current_price < lower_price or current_price > upper_price

        if not is_out_of_range:
            # 在区间内，检查是否接近边界
            distance_pct = self._calculate_distance_from_edge(
                current_price, lower_price, upper_price
            )

            threshold = (100 - float(config.rebalance_threshold_pct)) / 100

            if distance_pct > Decimal(str(threshold)):
                return False, f"在区间内，距边界 {distance_pct * 100:.1f}%"

        # === 因子 3: 60秒规则（高频特有）===
        if config.enable_60s_rule and out_duration_seconds >= config.out_of_range_timeout_seconds:
            # 超出 60 秒，立即再平衡
            return True, f"超出区间 {out_duration_seconds:.0f}秒，触发 60秒规则"

        # === 因子 4: 最小盈利检查（放宽）===
        if accumulated_fees_value > 0 and position_value > 0:
            fees_pct = accumulated_fees_value / position_value * Decimal("100")

            if fees_pct >= config.min_profit_for_rebalance:
                return True, f"累积手续费 {fees_pct:.2f}% 达到阈值"

        # === 因子 5: 激进触发（距边界很近）===
        if is_out_of_range:
            # 已超出区间，即使没到 60 秒，如果超出幅度大也触发
            if current_price > upper_price:
                excess_pct = (current_price - upper_price) / upper_price * Decimal("100")
                if excess_pct > Decimal("3"):  # 超出 > 3%
                    return True, f"超出上界 {excess_pct:.2f}%，激进再平衡"
            else:
                excess_pct = (lower_price - current_price) / lower_price * Decimal("100")
                if excess_pct > Decimal("3"):  # 超出 > 3%
                    return True, f"超出下界 {excess_pct:.2f}%，激进再平衡"

        return False, "条件未满足"

    def _is_cooldown_passed(self, cooldown_seconds: int) -> bool:
        """检查冷却期"""
        if self.last_rebalance_time is None:
            return True
        return (time.time() - self.last_rebalance_time) >= cooldown_seconds

    def _remaining_cooldown(self, cooldown_seconds: int) -> float:
        """剩余冷却时间"""
        if self.last_rebalance_time is None:
            return 0
        elapsed = time.time() - self.last_rebalance_time
        return max(0, cooldown_seconds - elapsed)

    def _calculate_distance_from_edge(
        self,
        current_price: Decimal,
        lower_price: Decimal,
        upper_price: Decimal
    ) -> Decimal:
        """计算距离边界的距离（百分比）"""
        range_width = upper_price - lower_price
        distance_to_lower = current_price - lower_price
        distance_to_upper = upper_price - current_price
        min_distance = min(distance_to_lower, distance_to_upper)
        return min_distance / range_width

    def mark_rebalance_executed(self):
        """标记再平衡已执行"""
        self.last_rebalance_time = time.time()


# ========================================
# 主策略类
# ========================================

class MeteoraDlmmHftMeme(ScriptStrategyBase):
    """Meteora DLMM 高频做市策略（Meme 币专用）"""

    @classmethod
    def init_markets(cls, config: MeteoraDlmmHftMemeConfig):
        """初始化市场（Hummingbot 必需）"""
        cls.markets = {
            config.connector: {config.trading_pair},  # Meteora LP
            config.swap_connector: {config.trading_pair}  # Jupiter Swap
        }

    def __init__(self, connectors: Dict[str, ConnectorBase], config: MeteoraDlmmHftMemeConfig):
        super().__init__(connectors)
        self.config = config

        # 连接器（和 V2 版本一致，直接引用）
        self.connector = connectors[config.connector]  # Meteora LP connector
        self.swap_connector = connectors[config.swap_connector]  # Jupiter Swap connector
        self.connector_type = get_connector_type(config.connector)

        # 交易对
        self.base_token, self.quote_token = config.trading_pair.split("-")

        # 策略状态
        self.position_opened = False
        self.position_opening = False
        self.open_price: Optional[Decimal] = None
        self.initial_investment: Decimal = Decimal("0")
        self.pending_open_order_id: Optional[str] = None  # 追踪开仓订单ID
        self.tokens_prepared = False

        # 仓位信息
        self.position_id: Optional[str] = None
        self.position_info: Optional[CLMMPositionInfo] = None
        self.pool_info: Optional[CLMMPoolInfo] = None

        # 止损引擎（延迟初始化，避免logger未ready）
        self.stop_loss_engine: Optional[FastStopLossEngine] = None

        # 再平衡引擎（延迟初始化）
        self.rebalance_engine: Optional[HighFrequencyRebalanceEngine] = None

        # 统计
        self.daily_start_value: Decimal = Decimal("0")
        self.rebalance_count_today: int = 0
        self.stop_loss_count_today: int = 0

        # 时间追踪
        self.last_check_time: Optional[datetime] = None

        # 延迟初始化
        safe_ensure_future(self.initialize_strategy())

    # ========================================
    # 初始化
    # ========================================

    async def initialize_strategy(self):
        """策略初始化"""
        await asyncio.sleep(5)  # 等待连接器初始化（增加到5秒）

        try:
            # 初始化引擎
            self.stop_loss_engine = FastStopLossEngine(self.logger(), self.config)
            self.rebalance_engine = HighFrequencyRebalanceEngine(self.logger())

            # 获取池子信息
            await self.fetch_pool_info()

            # 检查现有仓位（可能失败，不影响策略启动）
            try:
                await self.check_existing_positions()
            except Exception as e:
                self.logger().warning(f"检查现有仓位失败（将在首次检查时重试）: {e}")

            self.logger().info("⚡ 高频策略初始化完成")

        except Exception as e:
            self.logger().error(f"策略初始化失败: {e}", exc_info=True)

    async def fetch_pool_info(self) -> Optional[CLMMPoolInfo]:
        """获取池子信息"""
        try:
            self.pool_info = await self.connector.get_pool_info(
                trading_pair=self.config.trading_pair
            )

            # 注入价格到 RateOracle
            if self.pool_info:
                try:
                    current_price = Decimal(str(self.pool_info.price))
                    rate_oracle = RateOracle.get_instance()
                    rate_oracle.set_price(self.config.trading_pair, current_price)
                    self.logger().debug(f"注入池子价格: {current_price}")
                except Exception as oracle_err:
                    self.logger().debug(f"RateOracle 注入失败: {oracle_err}")

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

    def on_start(self):
        """策略启动"""
        self.logger().info("=" * 60)
        self.logger().info("⚡ Meteora DLMM 高频做市策略 - Meme 币专用 ⚡")
        self.logger().info("=" * 60)

        self.logger().info(
            f"交易对: {self.config.trading_pair}\n"
            f"区间宽度: ±{self.config.price_range_pct}%\n"
            f"再平衡阈值: {self.config.rebalance_threshold_pct}%\n"
            f"冷却期: {self.config.rebalance_cooldown_seconds}秒\n"
            f"60秒规则: {'启用' if self.config.enable_60s_rule else '禁用'}\n"
            f"幅度止损: {self.config.stop_loss_pct}%\n"
            f"检查频率: {self.config.check_interval_seconds}秒"
        )

        self.logger().warning(
            "⚠️  警告: 高频策略风险极高！\n"
            "   - 仅适用于 meme 币高波动期\n"
            "   - 建议小资金测试\n"
            "   - 严格遵守止损规则"
        )

    async def on_stop(self):
        """策略停止"""
        self.logger().info("策略已停止")

    def on_tick(self):
        """策略主循环（框架每秒调用一次）"""
        # ========================================
        # 1. 连接器就绪检查（参考官方策略实现）
        # ========================================
        # 检查所有连接器是否就绪（参考 amm_arb.py:179-182）
        if not all([connector.ready for connector in self.connectors.values()]):
            self.logger().warning(
                f"{self.config.connector} 或 {self.config.swap_connector} 未就绪，等待中..."
            )
            return

        # ========================================
        # 2. 时间间隔控制
        # ========================================
        current_time = datetime.now()

        # 检查间隔控制
        if self.last_check_time and (current_time - self.last_check_time).total_seconds() < self.config.check_interval_seconds:
            return

        self.last_check_time = current_time

        # ========================================
        # 3. 状态机逻辑
        # ========================================
        if self.position_opening:
            # 等待开仓确认
            return
        elif not self.position_opened:
            # 无仓位：开仓
            safe_ensure_future(self.check_and_open_position())
        else:
            # 持仓中：高频监控
            safe_ensure_future(self.monitor_position_high_frequency())

    # ========================================
    # 开仓逻辑（同简化版）
    # ========================================

    async def check_and_open_position(self):
        """检查并开仓"""
        try:
            # 检查现有仓位（可能失败，但不影响开仓逻辑）
            try:
                await self.check_existing_positions()
            except Exception as e:
                self.logger().warning(f"检查现有仓位失败（继续开仓流程）: {e}")

            if self.position_opened:
                return

            current_price = await self.get_current_price()
            if current_price is None:
                self.logger().warning("无法获取当前价格，跳过本次开仓检查")
                return

            self.logger().info(f"准备开仓，当前价格: {current_price}")

            if self.config.enable_auto_swap and not self.tokens_prepared:
                self.logger().info("检查并准备双边代币...")
                success = await self.prepare_tokens_for_position(current_price)
                if not success:
                    self.logger().warning("代币准备失败，跳过本次开仓")
                    return
                self.tokens_prepared = True

            await self.open_position(current_price)

        except Exception as e:
            self.logger().error(f"检查开仓失败: {e}", exc_info=True)

    async def get_current_price(self) -> Optional[Decimal]:
        """
        获取当前价格（多重降级策略，参考 amm_trade_example.py 和 cex_dex_lp_arbitrage.py）

        优先级：
        1. get_pool_info() - 最完整的信息
        2. get_quote_price() - 备用方案（swap 报价）
        """
        try:
            # ========================================
            # 方法 1: get_pool_info()（推荐）
            # ========================================
            try:
                self.logger().debug(f"尝试获取池子信息: {self.config.trading_pair}")
                pool_info = await self.connector.get_pool_info(
                    trading_pair=self.config.trading_pair
                )
                if pool_info and hasattr(pool_info, 'price') and pool_info.price > 0:
                    self.pool_info = pool_info  # 更新缓存
                    price = Decimal(str(pool_info.price))
                    self.logger().debug(f"✅ 池子价格: {price} (active_bin_id: {pool_info.active_bin_id})")
                    return price
                else:
                    self.logger().warning(f"⚠️ 池子信息无效或价格为 0: {pool_info}")
            except Exception as e:
                self.logger().warning(f"⚠️ get_pool_info() 失败: {e}，尝试备用方案...")

            # ========================================
            # 方法 2: get_quote_price()（备用）
            # ========================================
            try:
                self.logger().debug(f"尝试获取报价: {self.config.trading_pair}")
                # 参考 amm_trade_example.py:86-90
                quote_price = await self.connector.get_quote_price(
                    trading_pair=self.config.trading_pair,
                    is_buy=True,  # 买入价格
                    amount=Decimal("1")  # 1 个 base token 的价格
                )
                if quote_price and quote_price > 0:
                    price = Decimal(str(quote_price))
                    self.logger().debug(f"✅ 报价价格: {price}")
                    return price
                else:
                    self.logger().warning(f"⚠️ 报价无效或价格为 0: {quote_price}")
            except Exception as e:
                self.logger().warning(f"⚠️ get_quote_price() 失败: {e}")

            # ========================================
            # 所有方法都失败
            # ========================================
            self.logger().error(
                f"❌ 无法获取 {self.config.trading_pair} 价格\n"
                f"   连接器: {self.config.connector}\n"
                f"   请检查:\n"
                f"   1. 连接器是否正常连接到 Gateway\n"
                f"   2. 交易对是否正确\n"
                f"   3. 池子是否存在"
            )
            return None

        except Exception as e:
            self.logger().error(f"❌ 获取价格时发生严重错误: {e}", exc_info=True)
            return None

    async def prepare_tokens_for_position(self, current_price: Decimal) -> bool:
        """准备代币（简化版，同之前）"""
        try:
            self.logger().info("准备双边代币...")

            await self.swap_connector.update_balances(on_interval=False)

            base_balance = self.swap_connector.get_available_balance(self.base_token)
            quote_balance = self.swap_connector.get_available_balance(self.quote_token)

            actual_base_value = Decimal(str(base_balance)) * current_price
            actual_quote_value = Decimal(str(quote_balance))
            total_value = actual_base_value + actual_quote_value

            if total_value == 0:
                self.logger().error("总余额为 0")
                return False

            target_base_amount = total_value * Decimal("0.5") / current_price
            shortage = target_base_amount - Decimal(str(base_balance))

            if abs(shortage) < Decimal("0.001"):
                return True

            if shortage > 0:
                await self.swap_via_jupiter(
                    self.quote_token, self.base_token,
                    shortage * Decimal("1.02"), "BUY"
                )
            else:
                await self.swap_via_jupiter(
                    self.base_token, self.quote_token,
                    abs(shortage) * Decimal("1.02"), "SELL"
                )

            return True

        except Exception as e:
            self.logger().error(f"准备代币失败: {e}", exc_info=True)
            return False

    async def swap_via_jupiter(self, from_token: str, to_token: str, amount: Decimal, side: str) -> bool:
        """
        Jupiter 换币（和 V2 版本一致）

        参数：
        - from_token: 输入代币
        - to_token: 输出代币
        - amount: 数量（如果是买入 base_token，这是 base_token 数量；如果是卖出，也是 base_token 数量）
        - side: "BUY" 或 "SELL"（仅用于日志，实际通过 from_token 判断）
        """
        try:
            # 1. 构造 trading_pair（始终是 BASE-QUOTE 格式）
            trading_pair = f"{self.base_token}-{self.quote_token}"

            # 2. 判断 is_buy（Gateway API 语义）
            # - is_buy=True: 买入 base_token（卖出 quote_token）
            # - is_buy=False: 卖出 base_token（买入 quote_token）
            if from_token == self.base_token:
                is_buy = False  # 卖出 base_token
            else:
                is_buy = True   # 买入 base_token

            # 3. 获取 Jupiter 报价
            self.logger().info(f"获取 Jupiter 报价...")
            quote_price = await self.swap_connector.get_quote_price(
                trading_pair=trading_pair,
                is_buy=is_buy,
                amount=amount
            )

            if not quote_price or quote_price <= 0:
                self.logger().error(f"获取报价失败，返回价格: {quote_price}")
                return False

            # 注入价格到 RateOracle
            try:
                rate_oracle = RateOracle.get_instance()
                rate_oracle.set_price(trading_pair, Decimal(str(quote_price)))
            except Exception as oracle_err:
                self.logger().debug(f"RateOracle 注入失败: {oracle_err}")

            # 4. 计算预期的 quote_token 数量
            quote_token_amount = amount * Decimal(str(quote_price))

            # 5. 打印兑换信息
            if is_buy:
                self.logger().info(
                    f"执行 Jupiter 兑换（买入 {self.base_token}）:\n"
                    f"   卖出约: {quote_token_amount:.6f} {self.quote_token}\n"
                    f"   买入: {amount:.6f} {self.base_token}\n"
                    f"   价格: {quote_price:.10f} {self.quote_token}/{self.base_token}"
                )
            else:
                self.logger().info(
                    f"执行 Jupiter 兑换（卖出 {self.base_token}）:\n"
                    f"   卖出: {amount:.6f} {self.base_token}\n"
                    f"   买入约: {quote_token_amount:.6f} {self.quote_token}\n"
                    f"   价格: {quote_price:.10f} {self.quote_token}/{self.base_token}"
                )

            # 6. 执行兑换（使用正确的参数）
            order_id = self.swap_connector.place_order(
                is_buy=is_buy,
                trading_pair=trading_pair,
                amount=amount,  # 保持 Decimal 类型
                price=quote_price
            )

            self.logger().info(f"Jupiter 兑换订单已提交: {order_id}")

            # 7. 等待订单成交（简化版，不像 V2 那样轮询）
            await asyncio.sleep(5)
            await self.swap_connector.update_balances(on_interval=False)

            return True

        except Exception as e:
            self.logger().error(f"换币失败: {e}", exc_info=True)
            return False

    async def open_position(self, center_price: Decimal):
        """开仓（紧跟价格的窄区间）"""
        if self.position_opening or self.position_opened:
            return

        self.position_opening = True

        try:
            # ========================================
            # 1. 计算价格区间
            # ========================================
            range_width_pct = self.config.price_range_pct

            # 紧跟价格：±5-10%
            lower_price = center_price * (Decimal("1") - range_width_pct / Decimal("100"))
            upper_price = center_price * (Decimal("1") + range_width_pct / Decimal("100"))

            # ========================================
            # 2. 获取代币数量
            # ========================================
            total_base, total_quote = await self.get_token_amounts()

            # 检查余额是否足够
            if total_base <= 0 and total_quote <= 0:
                self.logger().error(
                    f"❌ 开仓失败：余额不足\n"
                    f"   {self.base_token}: {total_base}\n"
                    f"   {self.quote_token}: {total_quote}"
                )
                self.position_opening = False
                return

            self.logger().info(
                f"开仓（高频模式）:\n"
                f"  价格: {center_price:.8f}\n"
                f"  区间: [{lower_price:.8f}, {upper_price:.8f}] (±{range_width_pct}%)\n"
                f"  投入: {total_base:.6f} {self.base_token} + {total_quote:.2f} {self.quote_token}"
            )

            # ========================================
            # 3. 计算 width 百分比（参考 lp_manage_position.py:461-462）
            # ========================================
            lower_width_pct = float(((center_price - lower_price) / center_price) * 100)
            upper_width_pct = float(((upper_price - center_price) / center_price) * 100)

            self.logger().debug(
                f"开仓参数:\n"
                f"  trading_pair: {self.config.trading_pair}\n"
                f"  price: {float(center_price)}\n"
                f"  upper_width_pct: {upper_width_pct:.2f}%\n"
                f"  lower_width_pct: {lower_width_pct:.2f}%\n"
                f"  base_token_amount: {float(total_base)}\n"
                f"  quote_token_amount: {float(total_quote)}"
            )

            # ========================================
            # 4. 提交开仓订单（参考 gateway_lp.py:151-171）
            # ========================================
            order_id = self.connector.add_liquidity(
                trading_pair=self.config.trading_pair,
                price=float(center_price),
                upper_width_pct=upper_width_pct,
                lower_width_pct=lower_width_pct,
                base_token_amount=float(total_base),
                quote_token_amount=float(total_quote),
            )

            self.pending_open_order_id = order_id
            self.logger().info(f"✅ 开仓订单已提交: {order_id}，等待成交确认...")

            # 暂存开仓参数，等待订单成交后使用
            self._pending_open_price = center_price
            self._pending_investment = (total_base * center_price) + total_quote

        except Exception as e:
            self.logger().error(
                f"❌ 开仓失败:\n"
                f"   错误: {e}\n"
                f"   连接器: {self.config.connector}\n"
                f"   交易对: {self.config.trading_pair}",
                exc_info=True
            )
            self.position_opening = False

    async def get_token_amounts(self) -> Tuple[Decimal, Decimal]:
        """
        获取代币数量（带详细日志）

        Returns:
            (base_amount, quote_amount)
        """
        try:
            # 如果配置了固定数量，直接使用
            if self.config.base_token_amount > 0 or self.config.quote_token_amount > 0:
                self.logger().debug(
                    f"使用配置的固定数量:\n"
                    f"  {self.base_token}: {self.config.base_token_amount}\n"
                    f"  {self.quote_token}: {self.config.quote_token_amount}"
                )
                return self.config.base_token_amount, self.config.quote_token_amount

            # 否则使用钱包余额的百分比
            base_balance = self.connector.get_available_balance(self.base_token)
            quote_balance = self.connector.get_available_balance(self.quote_token)

            self.logger().debug(
                f"当前钱包余额:\n"
                f"  {self.base_token}: {base_balance}\n"
                f"  {self.quote_token}: {quote_balance}"
            )

            allocation_pct = self.config.wallet_allocation_pct / Decimal("100")

            allocated_base = Decimal(str(base_balance)) * allocation_pct
            allocated_quote = Decimal(str(quote_balance)) * allocation_pct

            self.logger().debug(
                f"分配 {self.config.wallet_allocation_pct}% 的余额:\n"
                f"  {self.base_token}: {allocated_base}\n"
                f"  {self.quote_token}: {allocated_quote}"
            )

            return allocated_base, allocated_quote

        except Exception as e:
            self.logger().error(f"获取代币数量失败: {e}", exc_info=True)
            return Decimal("0"), Decimal("0")

    async def check_existing_positions(self):
        """检查现有仓位"""
        try:
            pool_address = await self.get_pool_address()
            if not pool_address:
                self.logger().warning("无法获取池子地址")
                return

            # 尝试获取仓位，先尝试传 pool_address 过滤
            try:
                positions = await self.connector.get_user_positions(pool_address=pool_address)
            except Exception as e:
                # 如果失败，尝试不传 pool_address（获取所有仓位）
                self.logger().warning(f"使用 pool_address 获取仓位失败，尝试获取所有仓位: {e}")
                positions = await self.connector.get_user_positions()

            if positions and len(positions) > 0:
                self.position_info = positions[0]
                self.position_id = self.position_info.address  # ✅ 修复：使用 address 字段
                self.position_opened = True

                # 设置开仓价格为当前价格
                if self.pool_info:
                    self.open_price = Decimal(str(self.pool_info.price))

                self.logger().info(f"发现现有仓位: {self.position_id}")
            else:
                self.position_opened = False
                self.position_id = None
                self.position_info = None
                self.logger().info("未发现现有仓位")
        except Exception as e:
            self.logger().error(f"检查仓位失败: {e}", exc_info=True)
            # 不重新抛出异常，让调用者自行处理

    # ========================================
    # 高频监控和止损逻辑（核心）
    # ========================================

    async def monitor_position_high_frequency(self):
        """高频监控仓位"""
        try:
            # 检查引擎是否已初始化
            if not self.stop_loss_engine or not self.rebalance_engine:
                return

            # 只在没有仓位信息时才检查（避免频繁 Gateway 调用）
            if not self.position_info:
                try:
                    await self.check_existing_positions()
                except Exception as e:
                    self.logger().warning(f"监控中检查仓位失败: {e}")
                    return

            if not self.position_opened or not self.position_info:
                return

            self.pool_info = await self.connector.get_pool_info(
                trading_pair=self.config.trading_pair
            )

            if not self.pool_info:
                return

            current_price = Decimal(str(self.pool_info.price))
            lower_price = Decimal(str(self.position_info.lower_price))
            upper_price = Decimal(str(self.position_info.upper_price))

            # 计算手续费
            base_fees = Decimal(str(self.position_info.base_fee_amount))
            quote_fees = Decimal(str(self.position_info.quote_fee_amount))
            fees_value = (base_fees * current_price) + quote_fees

            # 计算仓位价值
            position_value = await self._calculate_position_value()

            # === 优先级 1: 检查快速止损 ===
            current_volume = Decimal(str(self.pool_info.volume_24h)) if hasattr(self.pool_info, 'volume_24h') else None

            should_stop, stop_type, stop_reason = self.stop_loss_engine.check_stop_loss(
                current_price=current_price,
                open_price=self.open_price,
                lower_price=lower_price,
                upper_price=upper_price,
                current_volume=current_volume
            )

            if should_stop:
                if stop_type == "HARD_STOP":
                    self.logger().warning(f"🛑 触发硬止损: {stop_reason}")
                    await self.execute_stop_loss(stop_reason)
                    return
                elif stop_type == "SOFT_STOP":
                    self.logger().warning(f"⚠️  建议止损: {stop_reason}")
                    # 软止损：建议但不强制
                    # 可以根据累积收益决定是否退出

            # === 优先级 2: 检查高频再平衡 ===
            out_duration = (time.time() - self.stop_loss_engine.price_out_of_range_since) if self.stop_loss_engine.price_out_of_range_since else 0

            should_rebal, rebal_reason = await self.rebalance_engine.should_rebalance(
                current_price=current_price,
                lower_price=lower_price,
                upper_price=upper_price,
                accumulated_fees_value=fees_value,
                position_value=position_value,
                config=self.config,
                out_duration_seconds=out_duration
            )

            # 实时监控日志（每次都打印）
            price_change = (current_price - self.open_price) / self.open_price * Decimal("100") if self.open_price else Decimal("0")

            self.logger().info(
                f"⚡ 高频监控:\n"
                f"  价格: {current_price:.8f} ({price_change:+.2f}%)\n"
                f"  区间: [{lower_price:.8f}, {upper_price:.8f}]\n"
                f"  超出时长: {out_duration:.0f}秒\n"
                f"  手续费: {fees_value:.4f} {self.quote_token}\n"
                f"  再平衡: {rebal_reason}"
            )

            if should_rebal:
                await self.execute_high_frequency_rebalance(current_price)

        except Exception as e:
            self.logger().error(f"监控失败: {e}", exc_info=True)

    async def _calculate_position_value(self) -> Decimal:
        """计算仓位价值"""
        if not self.position_info or not self.pool_info:
            return Decimal("0")

        current_price = Decimal(str(self.pool_info.price))
        base_amount = Decimal(str(self.position_info.base_token_amount))
        quote_amount = Decimal(str(self.position_info.quote_token_amount))

        return (base_amount * current_price) + quote_amount

    async def execute_stop_loss(self, reason: str):
        """执行止损"""
        try:
            self.logger().warning("=" * 60)
            self.logger().warning(f"🛑 执行止损: {reason}")
            self.logger().warning("=" * 60)

            await self.close_position()

            self.stop_loss_count_today += 1

            # 止损后冷静期：5-10 分钟
            cooldown_minutes = 5
            self.logger().info(f"止损冷静期：{cooldown_minutes} 分钟")
            await asyncio.sleep(cooldown_minutes * 60)

        except Exception as e:
            self.logger().error(f"止损失败: {e}", exc_info=True)

    async def execute_high_frequency_rebalance(self, current_price: Decimal):
        """执行高频再平衡"""
        try:
            self.logger().info("=" * 60)
            self.logger().info(f"⚡ 高频再平衡: 新价格 {current_price:.8f}")
            self.logger().info("=" * 60)

            # 1. 关闭旧仓位
            await self.close_position()

            # 2. 等待
            await asyncio.sleep(3)

            # 3. 在新价格立即开仓（紧跟价格）
            await self.open_position(current_price)

            # 4. 标记执行
            self.rebalance_engine.mark_rebalance_executed()
            self.rebalance_count_today += 1

            self.logger().info(f"✅ 再平衡完成（今日第 {self.rebalance_count_today} 次）")

        except Exception as e:
            self.logger().error(f"再平衡失败: {e}", exc_info=True)

    async def close_position(self):
        """关闭仓位"""
        try:
            if not self.position_id:
                return

            self.logger().info(f"关闭仓位: {self.position_id}")

            order_id = self.connector.remove_liquidity(
                trading_pair=self.config.trading_pair,
                position_address=self.position_id  # ✅ 修复：使用 position_address 参数
            )

            self.logger().info(f"关闭订单: {order_id}")

            self.position_opened = False
            self.position_id = None
            self.position_info = None

        except Exception as e:
            self.logger().error(f"关闭仓位失败: {e}", exc_info=True)

    # ========================================
    # 事件处理
    # ========================================

    def did_fill_order(self, event):
        """订单成交事件"""
        try:
            if not hasattr(event, 'order_id'):
                return

            order_id = event.order_id
            self.logger().info(f"订单成交: {order_id}")

            # 检查是否是开仓订单
            if self.pending_open_order_id and order_id == self.pending_open_order_id:
                self.logger().info(f"✅ 开仓订单成交确认: {order_id}")

                # 设置状态
                self.position_opening = False
                self.position_opened = True

                # 恢复开仓参数
                if hasattr(self, '_pending_open_price'):
                    self.open_price = self._pending_open_price
                if hasattr(self, '_pending_investment'):
                    self.initial_investment = self._pending_investment

                # 重置止损引擎
                self.stop_loss_engine.reset()

                # 清除待处理订单ID
                self.pending_open_order_id = None

                # 异步获取仓位信息
                safe_ensure_future(self.fetch_positions_after_fill())

        except Exception as e:
            self.logger().error(f"处理订单成交事件失败: {e}", exc_info=True)

    def did_fail_order(self, event):
        """订单失败事件"""
        try:
            if not hasattr(event, 'order_id'):
                return

            order_id = event.order_id
            self.logger().warning(f"订单失败: {order_id}")

            # 检查是否是开仓订单失败
            if self.pending_open_order_id and order_id == self.pending_open_order_id:
                self.logger().error(f"❌ 开仓订单失败: {order_id}")
                self.position_opening = False
                self.pending_open_order_id = None

        except Exception as e:
            self.logger().error(f"处理订单失败事件错误: {e}", exc_info=True)

    async def fetch_positions_after_fill(self):
        """订单成交后获取仓位信息"""
        try:
            # 等待链上确认
            await asyncio.sleep(3)

            self.logger().info("开仓成功，获取仓位信息...")
            await self.check_existing_positions()

            if self.position_info:
                self.logger().info(
                    f"仓位信息已获取:\n"
                    f"  仓位ID: {self.position_id}\n"
                    f"  价格区间: [{self.position_info.lower_price:.8f}, {self.position_info.upper_price:.8f}]\n"
                    f"  代币数量: {self.position_info.base_token_amount:.6f} {self.base_token} + "
                    f"{self.position_info.quote_token_amount:.2f} {self.quote_token}"
                )
            else:
                self.logger().warning("未能获取到仓位信息，将在下次监控时重试")

        except Exception as e:
            self.logger().error(f"获取仓位信息失败: {e}", exc_info=True)

    # ========================================
    # 状态展示
    # ========================================

    def format_status(self) -> str:
        """格式化状态"""
        if not self.position_opened or not self.position_info:
            return "无持仓"

        current_price = Decimal(str(self.pool_info.price)) if self.pool_info else Decimal("0")
        price_change = (current_price - self.open_price) / self.open_price * Decimal("100") if self.open_price else Decimal("0")

        return (
            f"\n{'=' * 60}\n"
            f"⚡ Meteora DLMM 高频做市状态\n"
            f"{'=' * 60}\n"
            f"交易对: {self.config.trading_pair}\n"
            f"当前价格: {current_price:.8f} ({price_change:+.2f}%)\n"
            f"区间: [{self.position_info.lower_price:.8f}, {self.position_info.upper_price:.8f}]\n"
            f"今日再平衡: {self.rebalance_count_today} 次\n"
            f"今日止损: {self.stop_loss_count_today} 次\n"
            f"{'=' * 60}\n"
        )


if __name__ == "__main__":
    pass
