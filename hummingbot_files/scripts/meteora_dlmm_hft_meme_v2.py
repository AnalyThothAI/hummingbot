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

# 导入引擎模块（使用新的引擎）
from .engines.stop_loss_engine import StopLossEngine
from .engines.rebalance_engine import RebalanceEngine
from .engines.state_manager import StateManager

# 导入工具模块
from .utils import position_helper, price_helper, swap_helper

# 导入换币管理器
from .swap_manager import SwapManager, should_swap_to_sol


# ========================================
# 配置类
# ========================================

class MeteoraDlmmHftMemeConfig(BaseClientModel):
    """Meteora DLMM 高频做市策略配置（V2 - 重构引擎版）"""

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
        "ACE-SOL",
        json_schema_extra={"prompt": "交易对", "prompt_on_new": True}
    )

    pool_address: str = Field(
        "",
        json_schema_extra={"prompt": "池子地址（可选）", "prompt_on_new": False}
    )

    # ========== 区间配置 ==========
    price_range_pct: Decimal = Field(
        Decimal("10.0"),
        json_schema_extra={"prompt": "基础区间宽度（%）", "prompt_on_new": True}
    )

    enable_asymmetric_range: bool = Field(
        True,
        json_schema_extra={"prompt": "启用非对称区间？", "prompt_on_new": True}
    )

    asymmetric_ratio_strong: Decimal = Field(
        Decimal("3.0"),
        json_schema_extra={"prompt": "强趋势非对称比例（如3.0表示上:下=3:1）", "prompt_on_new": False}
    )

    asymmetric_ratio_medium: Decimal = Field(
        Decimal("2.0"),
        json_schema_extra={"prompt": "中等趋势非对称比例", "prompt_on_new": False}
    )

    # ========== 止损触发配置 ==========
    stop_loss_price: Optional[Decimal] = Field(
        None,
        json_schema_extra={
            "prompt": "止损价格（绝对值，如 0.00004，优先级最高）",
            "prompt_on_new": False
        }
    )

    stop_loss_pct: Decimal = Field(
        Decimal("5.0"),
        json_schema_extra={
            "prompt": "止损百分比（相对开仓价，如 5.0 表示 -5%）",
            "prompt_on_new": True
        }
    )

    stop_loss_on_out_of_range: bool = Field(
        True,
        json_schema_extra={
            "prompt": "跌破LP区间下界时止损？（强烈建议 True）",
            "prompt_on_new": False
        }
    )

    # 防插针设置
    stop_loss_confirmation_seconds: int = Field(
        60,
        json_schema_extra={
            "prompt": "止损确认时长（秒，防止插针误触发）",
            "prompt_on_new": False
        }
    )

    out_of_range_confirmation_seconds: int = Field(
        60,
        json_schema_extra={
            "prompt": "超出区间确认时长（秒）",
            "prompt_on_new": False
        }
    )

    # ========== 再平衡配置 ==========
    rebalance_threshold_pct: Decimal = Field(
        Decimal("85.0"),
        json_schema_extra={"prompt": "再平衡阈值（%）", "prompt_on_new": True}
    )

    rebalance_cooldown_seconds: int = Field(
        180,
        json_schema_extra={"prompt": "再平衡冷却期（秒）", "prompt_on_new": True}
    )

    out_of_range_timeout_seconds: int = Field(
        60,
        json_schema_extra={"prompt": "超出区间确认时长（秒）", "prompt_on_new": False}
    )

    min_profit_for_rebalance: Decimal = Field(
        Decimal("2.0"),
        json_schema_extra={"prompt": "最小再平衡盈利（%）", "prompt_on_new": False}
    )

    # ========== 止损执行配置（强制换SOL模式）==========
    stop_loss_to_sol: bool = Field(
        True,
        json_schema_extra={
            "prompt": "止损后强制换成 SOL？（推荐 True，保护本金）",
            "prompt_on_new": True
        }
    )

    stop_loss_slippage_pct: Decimal = Field(
        Decimal("2.0"),
        json_schema_extra={
            "prompt": "止损换币滑点容忍度（%）",
            "prompt_on_new": False
        }
    )

    # ========== 仓位大小配置（简化版：SOL 统一计价）==========
    position_size_sol: Decimal = Field(
        Decimal("1.0"),
        gt=Decimal("0.1"),
        json_schema_extra={
            "prompt": "投入多少 SOL 做 LP（建议 >= 0.2）",
            "prompt_on_new": True
        }
    )

    # ========== 开仓准备配置 ==========
    auto_prepare_tokens: bool = Field(
        True,
        json_schema_extra={
            "prompt": "开仓前自动准备代币（50:50分配）？",
            "prompt_on_new": False
        }
    )

    prepare_slippage_pct: Decimal = Field(
        Decimal("3.0"),
        json_schema_extra={
            "prompt": "准备代币时的滑点容忍度（%）",
            "prompt_on_new": False
        }
    )

    # ========== 状态持久化 ==========
    enable_state_persistence: bool = Field(
        True,
        json_schema_extra={"prompt_on_new": False}
    )

    state_db_path: str = Field(
        "data/meteora_hft_state.db",
        json_schema_extra={"prompt_on_new": False}
    )

    # ========== 监控配置 ==========
    check_interval_seconds: int = Field(
        20,
        json_schema_extra={"prompt_on_new": False}
    )


# ========================================
# 主策略类
# ========================================
# 注意：StopLossEngine 和 RebalanceEngine 已移至 engines/ 模块

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
        self.pending_order_start_time: Optional[float] = None  # 订单提交时间戳
        self.open_retry_count: int = 0  # 开仓重试计数

        # 仓位信息
        self.position_id: Optional[str] = None
        self.position_info: Optional[CLMMPositionInfo] = None
        self.pool_info: Optional[CLMMPoolInfo] = None

        # 止损引擎（延迟初始化，避免logger未ready）
        self.stop_loss_engine: Optional[StopLossEngine] = None

        # 再平衡引擎（延迟初始化）
        self.rebalance_engine: Optional[RebalanceEngine] = None

        # ========== 新增：风控模块 ==========
        self.state_manager: Optional[StateManager] = None  # 状态持久化管理器
        self.swap_manager: Optional[SwapManager] = None    # 换币管理器
        self.cooldown_until: float = 0  # 冷却期结束时间

        # 统计
        self.daily_start_value: Decimal = Decimal("0")
        self.rebalance_count_today: int = 0
        self.stop_loss_count_today: int = 0

        # 时间追踪
        self.last_check_time: Optional[datetime] = None
        self.position_info_last_update: Optional[float] = None  # 仓位信息缓存时间戳

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
            self.stop_loss_engine = StopLossEngine(self.logger(), self.config)
            self.rebalance_engine = RebalanceEngine(self.logger())

            # ========== 新增：初始化风控模块 ==========
            if self.config.enable_state_persistence:
                self.state_manager = StateManager(
                    db_path=self.config.state_db_path,
                    logger=self.logger()
                )

                # 检查累计亏损状态并恢复开仓价格
                state = self.state_manager.get_state()
                if state["manual_kill"]:
                    self.logger().error(
                        f"🚨 策略已暂停！\n"
                        f"  原因: {state['stop_reason']}\n"
                        f"  累计盈亏: {state['cumulative_pnl']:.6f}\n"
                        f"  请检查后手动重置（state_manager.reset_manual_kill()）"
                    )
                else:
                    self.logger().info(f"✅ 状态已恢复: 累计盈亏 {state['cumulative_pnl']:+.6f}")

                    # 恢复开仓价格（如果存在）
                    if state["current_entry_price"] and state["current_entry_price"] > 0:
                        self.open_price = Decimal(str(state["current_entry_price"]))
                        self.logger().info(f"✅ 恢复开仓价格: {self.open_price:.10f}")

            # 初始化换币管理器
            if self.config.stop_loss_to_sol:
                self.swap_manager = SwapManager(
                    connector=self.swap_connector,  # ✅ 使用 Jupiter swap connector
                    logger=self.logger()
                )

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
            f"超出区间确认: {self.config.out_of_range_timeout_seconds}秒\n"
            f"止损确认: {self.config.stop_loss_confirmation_seconds}秒\n"
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

    # ========================================
    # 辅助方法（私有）
    # ========================================

    def _calculate_price_range(self, center_price: Decimal, range_width_pct: Decimal) -> Tuple[Decimal, Decimal]:
        """
        计算价格区间

        Args:
            center_price: 中心价格
            range_width_pct: 区间宽度百分比（如 5 表示 ±5%）

        Returns:
            (lower_price, upper_price)
        """
        lower_price = center_price * (Decimal("1") - range_width_pct / Decimal("100"))
        upper_price = center_price * (Decimal("1") + range_width_pct / Decimal("100"))
        return lower_price, upper_price

    def _calculate_width_percentages(
        self,
        center_price: Decimal,
        lower_price: Decimal,
        upper_price: Decimal
    ) -> Tuple[float, float]:
        """
        计算上下宽度百分比（用于 Gateway API）

        Args:
            center_price: 中心价格
            lower_price: 下界价格
            upper_price: 上界价格

        Returns:
            (lower_width_pct, upper_width_pct)
        """
        lower_width_pct = float(((center_price - lower_price) / center_price) * 100)
        upper_width_pct = float(((upper_price - center_price) / center_price) * 100)
        return lower_width_pct, upper_width_pct

    async def _refresh_balances(self, connector, wait_seconds: float = 1.0):
        """
        刷新余额并等待更新完成

        Args:
            connector: 连接器实例
            wait_seconds: 等待时间（秒）
        """
        await connector.update_balances(on_interval=False)
        await asyncio.sleep(wait_seconds)

    def _log_operation_error(self, operation: str, error: Exception, **context):
        """
        记录操作错误日志

        Args:
            operation: 操作名称（如 "开仓", "平仓"）
            error: 异常对象
            **context: 额外的上下文信息
        """
        lines = [f"❌ {operation}失败:"]
        lines.append(f"   错误: {error}")
        lines.append(f"   连接器: {self.config.connector}")
        lines.append(f"   交易对: {self.config.trading_pair}")

        for key, value in context.items():
            lines.append(f"   {key}: {value}")

        self.logger().error("\n".join(lines), exc_info=True)

    # ========================================
    # 主循环和状态机
    # ========================================

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
            # 检查订单超时
            safe_ensure_future(self.check_pending_order_timeout())
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

    async def check_pending_order_timeout(self):
        """检查待处理订单是否超时"""
        try:
            if not self.pending_open_order_id or not self.pending_order_start_time:
                return

            # 检查订单是否超时（默认60秒）
            timeout_seconds = 60
            elapsed = time.time() - self.pending_order_start_time

            if elapsed > timeout_seconds:
                self.logger().warning(
                    f"⚠️ 开仓订单超时 ({elapsed:.1f}s > {timeout_seconds}s)\n"
                    f"   订单ID: {self.pending_open_order_id}\n"
                    f"   尝试次数: {self.open_retry_count + 1}"
                )

                # 检查是否超过最大重试次数
                max_retries = 3
                if self.open_retry_count >= max_retries:
                    self.logger().error(
                        f"❌ 开仓失败次数过多 ({self.open_retry_count + 1}次)，暂停开仓\n"
                        f"   请检查:\n"
                        f"   1. 钱包是否有足够的 SOL 用于交易费用\n"
                        f"   2. Gateway 是否正常运行\n"
                        f"   3. RPC 节点是否稳定\n"
                        f"   4. 代币余额是否充足"
                    )
                    # 重置状态但不再重试
                    self.position_opening = False
                    self.pending_open_order_id = None
                    self.pending_order_start_time = None
                    self.open_retry_count = 0
                    return

                # 尝试主动检查订单状态
                try:
                    # 检查是否实际已经开仓成功（可能事件丢失）
                    await self.check_existing_positions()

                    if self.position_opened:
                        self.logger().info("✅ 检测到仓位已开启（订单事件可能丢失）")
                        self.position_opening = False
                        self.pending_open_order_id = None
                        self.pending_order_start_time = None
                        self.open_retry_count = 0
                        return
                except Exception as e:
                    self.logger().debug(f"检查现有仓位失败: {e}")

                # 重置状态并准备重试
                self.position_opening = False
                self.pending_open_order_id = None
                self.pending_order_start_time = None
                self.open_retry_count += 1

                # 等待一段时间后重试
                retry_delay = min(5 * self.open_retry_count, 30)  # 5秒、10秒、15秒...最多30秒
                self.logger().info(f"🔄 将在 {retry_delay} 秒后重试开仓...")
                await asyncio.sleep(retry_delay)

        except Exception as e:
            self.logger().error(f"检查订单超时失败: {e}", exc_info=True)

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

            # 开仓前必须准备双边代币（统一逻辑：每次都检查）
            self.logger().info("检查并准备双边代币...")
            success = await self.prepare_tokens_for_position(current_price)
            if not success:
                self.logger().error("❌ 代币准备失败，跳过本次开仓")
                return

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
        """
        准备代币 - 简化版（SOL 统一计价）

        逻辑：
        1. 计算需要多少 SOL（已在 get_token_amounts 中计算）
        2. 50% SOL 换成 base token
        3. 完成后，持有 50% base + 50% SOL
        """
        try:
            self.logger().info("准备双边代币...")

            # 刷新余额
            await self.swap_connector.update_balances(on_interval=False)

            # 1. 获取配置的投入金额（不再预留 gas，用户配置多少就是多少）
            total_sol = self.config.position_size_sol

            # 2. 50% SOL 换成 base token
            sol_to_swap = total_sol * Decimal("0.5")
            base_amount_needed = sol_to_swap / current_price

            self.logger().info(
                f"📊 准备代币（SOL 统一计价）:\n"
                f"  总投入: {total_sol} SOL\n"
                f"  \n"
                f"  换币计划:\n"
                f"    用 {sol_to_swap:.6f} SOL\n"
                f"    换 {base_amount_needed:.2f} {self.base_token}\n"
                f"    当前价格: {current_price:.10f}\n"
                f"  \n"
                f"  最终持仓:\n"
                f"    {self.base_token}: {base_amount_needed:.2f}\n"
                f"    {self.quote_token}: {sol_to_swap:.6f}"
            )

            # 3. 检查当前是否已经有足够的 base token
            base_balance = self.swap_connector.get_available_balance(self.base_token)

            if base_balance >= base_amount_needed:
                self.logger().info(
                    f"✅ 已有足够的 {self.base_token}:\n"
                    f"   当前: {base_balance:.2f}\n"
                    f"   需要: {base_amount_needed:.2f}\n"
                    f"   无需换币"
                )
                return True

            # 4. 需要换币
            shortage = base_amount_needed - base_balance

            if shortage < Decimal("0.001"):
                self.logger().info("代币数量已接近目标，无需换币")
                return True

            self.logger().info(
                f"需要换币:\n"
                f"  目标: {base_amount_needed:.2f} {self.base_token}\n"
                f"  当前: {base_balance:.2f} {self.base_token}\n"
                f"  缺少: {shortage:.2f} {self.base_token}\n"
                f"  执行: 用 SOL 买入 {shortage * Decimal('1.02'):.2f} {self.base_token} (含 2% buffer)"
            )

            # 执行换币（加 2% buffer 以应对滑点）
            await self.swap_via_jupiter(
                from_token=self.quote_token,
                to_token=self.base_token,
                amount=shortage * Decimal("1.02"),
                side="BUY"
            )

            self.logger().info(f"✅ 换币完成，代币准备就绪")
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
            # 1. 计算价格区间
            range_width_pct = self.config.price_range_pct
            lower_price, upper_price = self._calculate_price_range(center_price, range_width_pct)

            # 2. 获取代币数量
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

            # 计算预估总价值
            estimated_total_value = float(total_base) * float(center_price) + float(total_quote)

            self.logger().info(
                f"📊 开仓计划（高频模式）:\n"
                f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"  中心价格: {center_price:.10f}\n"
                f"  区间宽度: ±{range_width_pct}%\n"
                f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"  目标区间:\n"
                f"    下界: {lower_price:.10f}\n"
                f"    上界: {upper_price:.10f}\n"
                f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"  投入资金:\n"
                f"    {self.base_token}: {total_base:.6f}\n"
                f"    {self.quote_token}: {total_quote:.6f}\n"
                f"  预估总价值: {estimated_total_value:.6f} {self.quote_token}\n"
                f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )

            # 3. 计算 width 百分比
            lower_width_pct, upper_width_pct = self._calculate_width_percentages(center_price, lower_price, upper_price)

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
            self.pending_order_start_time = time.time()  # 记录订单提交时间
            self.logger().info(f"✅ 开仓订单已提交: {order_id}，等待成交确认...")

            # 暂存开仓参数，等待订单成交后使用
            self._pending_open_price = center_price
            self._pending_investment = (total_base * center_price) + total_quote

        except Exception as e:
            self._log_operation_error("开仓", e)
            self.position_opening = False

    async def get_token_amounts(self) -> Tuple[Decimal, Decimal]:
        """
        获取代币数量 - 简化版（SOL 统一计价）

        Returns:
            (base_amount, quote_amount)

        逻辑：
        1. 根据配置计算目标投入数量（严格按配置）
        2. 验证钱包余额是否足够
        3. 返回目标数量（确保投入金额准确）

        注意：
        - 必须在 prepare_tokens_for_position() 之后调用
        - 返回的是目标数量，不是全部钱包余额
        - 确保实际投入 = 配置金额
        """
        try:
            # 刷新余额
            self.logger().info("读取钱包余额...")
            await self._refresh_balances(self.connector)

            # 获取实际余额
            base_balance = self.connector.get_available_balance(self.base_token)
            quote_balance = self.connector.get_available_balance(self.quote_token)

            # 计算目标数量（严格按配置）
            total_sol = self.config.position_size_sol
            target_quote = total_sol * Decimal("0.5")

            # 获取当前价格计算目标 base 数量
            current_price = await self.get_current_price()
            if not current_price:
                self.logger().error("无法获取当前价格")
                return Decimal("0"), Decimal("0")

            target_base = target_quote / current_price

            self.logger().info(
                f"📊 余额检查:\n"
                f"  钱包余额:\n"
                f"    {self.base_token}: {base_balance:.6f}\n"
                f"    {self.quote_token}: {quote_balance:.6f}\n"
                f"  \n"
                f"  目标开仓数量（按配置 {total_sol} SOL）:\n"
                f"    {self.base_token}: {target_base:.6f}\n"
                f"    {self.quote_token}: {target_quote:.6f}"
            )

            # 验证余额是否足够
            if base_balance < target_base:
                self.logger().error(
                    f"❌ {self.base_token} 余额不足\n"
                    f"   需要: {target_base:.6f}\n"
                    f"   实际: {base_balance:.6f}\n"
                    f"   缺少: {target_base - base_balance:.6f}"
                )
                return Decimal("0"), Decimal("0")

            if quote_balance < target_quote:
                self.logger().error(
                    f"❌ {self.quote_token} 余额不足\n"
                    f"   需要: {target_quote:.6f}\n"
                    f"   实际: {quote_balance:.6f}\n"
                    f"   缺少: {target_quote - quote_balance:.6f}"
                )
                return Decimal("0"), Decimal("0")

            # 返回目标数量（严格按配置，不使用全部余额）
            self.logger().info(
                f"✅ 开仓数量（严格按配置）:\n"
                f"  {self.base_token}: {target_base:.6f}\n"
                f"  {self.quote_token}: {target_quote:.6f}\n"
                f"  总价值: {total_sol:.6f} SOL"
            )

            return target_base, target_quote

        except Exception as e:
            self.logger().error(f"获取代币数量失败: {e}", exc_info=True)
            return Decimal("0"), Decimal("0")

    async def check_existing_positions(self):
        """检查现有仓位（带重试机制）"""
        try:
            pool_address = await self.get_pool_address()
            if not pool_address:
                self.logger().warning("无法获取池子地址")
                return

            # 重试机制：最多重试 3 次
            positions = None
            for attempt in range(3):
                try:
                    positions = await self.connector.get_user_positions(pool_address=pool_address)
                    self.logger().debug(
                        f"获取仓位成功（按 pool_address，尝试 {attempt + 1}/3）: "
                        f"{len(positions) if positions else 0} 个"
                    )

                    # 如果返回了数据（即使是空列表），就跳出重试
                    if positions is not None:
                        break

                except Exception as e:
                    if attempt < 2:  # 不是最后一次尝试
                        self.logger().debug(f"获取仓位失败（尝试 {attempt + 1}/3）: {e}")
                        await asyncio.sleep(0.5)  # 等待 0.5 秒后重试
                    else:
                        # 最后一次尝试失败，尝试不传 pool_address
                        self.logger().warning(f"使用 pool_address 获取仓位失败，尝试获取所有仓位: {e}")
                        try:
                            positions = await self.connector.get_user_positions()
                            self.logger().debug(f"获取仓位成功（全部）: {len(positions) if positions else 0} 个")
                        except Exception as e2:
                            self.logger().error(f"获取所有仓位也失败: {e2}")
                            positions = None

            if positions and len(positions) > 0:
                # 过滤出有效仓位（流动性 > 0）
                valid_positions = []
                for pos in positions:
                    base_amt = Decimal(str(pos.base_token_amount))
                    quote_amt = Decimal(str(pos.quote_token_amount))
                    if base_amt > Decimal("0.000001") or quote_amt > Decimal("0.000001"):
                        valid_positions.append(pos)

                if not valid_positions:
                    self.logger().info(f"发现 {len(positions)} 个仓位，但都已关闭（流动性为0）")
                    self.position_opened = False
                    self.position_id = None
                    self.position_info = None
                    return

                # 选择仓位：优先使用已记录的 position_id，否则取第一个
                if self.position_id:
                    position_info = next((p for p in valid_positions if p.address == self.position_id), None)
                    if not position_info:
                        self.logger().warning(
                            f"⚠️ 已记录的仓位 {self.position_id} 未找到，使用第一个有效仓位"
                        )
                        position_info = valid_positions[0]
                else:
                    position_info = valid_positions[0]

                # 检查仓位是否实际有流动性
                base_amount = Decimal(str(position_info.base_token_amount))
                quote_amount = Decimal(str(position_info.quote_token_amount))

                # 仓位有效
                self.position_info = position_info
                self.position_id = position_info.address
                self.position_opened = True

                # 设置开仓价格（如果尚未从状态恢复）
                if not self.open_price and self.pool_info:
                    self.open_price = Decimal(str(self.pool_info.price))
                    self.logger().warning(
                        f"⚠️ 未找到历史开仓价格，使用当前价格: {self.open_price:.10f}"
                    )

                # 只在状态变化时打印详细信息（避免重复日志）
                if not self.position_opened or self.position_id != position_info.address:
                    # 新发现仓位或仓位变化
                    log_msg = (
                        f"✅ 发现有效仓位:\n"
                        f"   地址: {self.position_id}\n"
                        f"   Base: {base_amount:.6f}\n"
                        f"   Quote: {quote_amount:.6f}\n"
                        f"   区间: {position_info.lower_price:.10f} - {position_info.upper_price:.10f}"
                    )
                    if len(valid_positions) > 1:
                        log_msg += f"\n   （共 {len(valid_positions)} 个有效仓位）"
                    if self.open_price:
                        log_msg += f"\n   开仓价格: {self.open_price:.10f}"
                    self.logger().info(log_msg)
                else:
                    # 仓位未变化，只在 debug 级别记录
                    self.logger().debug(
                        f"仓位状态确认: {self.position_id} "
                        f"(Base: {base_amount:.2f}, Quote: {quote_amount:.6f})"
                    )
            else:
                # 返回 0 个仓位：可能是真的没有，也可能是接口暂时失败
                if self.position_opened and self.position_id:
                    # 上次有仓位，这次返回 0 个 → 可能是接口不稳定
                    self.logger().warning(
                        f"⚠️ 接口返回 0 个仓位，但上次有仓位 {self.position_id}\n"
                        f"   可能是接口暂时失败，保持上次状态\n"
                        f"   如果仓位确实关闭，下次检查会更新"
                    )
                    # 保持上次的状态，不修改
                else:
                    # 上次也没仓位，确实是空的
                    self.position_opened = False
                    self.position_id = None
                    self.position_info = None
                    self.logger().info(f"未发现现有仓位（返回 {len(positions) if positions else 0} 个仓位）")
        except Exception as e:
            self.logger().error(f"检查仓位失败（异常）: {e}", exc_info=True)
            # 不修改状态，保持上次的值
            # 这样可以避免因为临时网络问题导致误判

    # ========================================
    # 高频监控和止损逻辑（核心）
    # ========================================

    async def monitor_position_high_frequency(self):
        """高频监控仓位"""
        try:
            # 检查引擎是否已初始化
            if not self.stop_loss_engine or not self.rebalance_engine:
                return

            # 智能更新仓位信息：
            # 1. position_info为None时立即获取
            # 2. 距离上次更新超过60秒时刷新（避免频繁API调用）
            POSITION_INFO_UPDATE_INTERVAL = 60  # 60秒更新一次
            now = time.time()

            should_update_position = (
                not self.position_info or
                (self.position_info_last_update is None) or
                (now - self.position_info_last_update > POSITION_INFO_UPDATE_INTERVAL)
            )

            if should_update_position:
                try:
                    await self.check_existing_positions()
                    self.position_info_last_update = now
                    self.logger().debug(f"仓位信息已更新（间隔: {POSITION_INFO_UPDATE_INTERVAL}秒）")
                except Exception as e:
                    self.logger().warning(f"监控中检查仓位失败: {e}")
                    # 注意：不要return，继续用旧的position_info

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
            should_stop, stop_type, stop_reason = self.stop_loss_engine.check_stop_loss(
                current_price=current_price,
                open_price=self.open_price,
                lower_price=lower_price,
                upper_price=upper_price
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
            should_rebal, rebal_reason = await self.rebalance_engine.should_rebalance(
                current_price=current_price,
                lower_price=lower_price,
                upper_price=upper_price,
                accumulated_fees_value=fees_value,
                position_value=position_value,
                config=self.config
            )

            # 计算超出区间时长（用于日志）
            out_duration = 0.0
            if self.stop_loss_engine.below_range_since:
                out_duration = time.time() - self.stop_loss_engine.below_range_since

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
        """执行止损 - 简化版（强制换SOL模式）

        逻辑：
        1. 关闭LP仓位
        2. 如果启用 stop_loss_to_sol，强制将 base_token 全部换成 SOL
        3. 进入固定冷却期（3分钟）
        """
        try:
            self.logger().warning("=" * 60)
            self.logger().warning(f"🛑 执行止损: {reason}")
            self.logger().warning("=" * 60)

            # 1. 关闭LP仓位
            await self.close_position()
            self.stop_loss_count_today += 1

            # 2. 强制换币到SOL（保护本金，防止继续下跌）
            if self.config.stop_loss_to_sol:
                if not self.swap_manager:
                    self.logger().error("❌ SwapManager 未初始化，无法换币")
                else:
                    self.logger().warning(
                        f"⚠️  止损触发，强制将 {self.base_token} 换成 SOL 以保护本金"
                    )

                    # 等待链上确认完成（确保余额已更新）
                    await asyncio.sleep(3)
                    await self.swap_connector.update_balances(on_interval=False)
                    await asyncio.sleep(1)

                    # 获取当前 base_token 余额
                    base_balance = self.swap_connector.get_available_balance(self.base_token)

                    if base_balance > 0:
                        self.logger().info(f"  {self.base_token} 当前余额: {base_balance:.6f}")

                        # 执行换币
                        success, sol_amount, error_msg = await self.swap_manager.swap_all_to_sol(
                            token=self.base_token,
                            slippage_pct=self.config.stop_loss_slippage_pct,
                            reason="STOP_LOSS",
                            retry_count=2
                        )

                        if success:
                            self.logger().info(
                                f"✅ 成功将 {self.base_token} 换成 SOL\n"
                                f"  换得 SOL: {sol_amount:.6f}"
                            )
                        else:
                            self.logger().error(
                                f"❌ 换币失败: {error_msg}\n"
                                f"  {self.base_token} 仍在钱包中，请手动处理"
                            )
                    else:
                        self.logger().info(f"  {self.base_token} 余额为0，无需换币")
            else:
                self.logger().warning(
                    f"⚠️  换币功能已禁用（stop_loss_to_sol=False），"
                    f"持有 {self.base_token}（风险自负）"
                )

            # 3. 进入固定冷却期（3分钟）
            cooldown_seconds = 180
            self.logger().info(f"💤 止损冷却期: {cooldown_seconds / 60:.1f} 分钟")
            await asyncio.sleep(cooldown_seconds)

        except Exception as e:
            self.logger().error(f"执行止损失败: {e}", exc_info=True)

    async def execute_high_frequency_rebalance(self, current_price: Decimal):
        """执行高频再平衡"""
        try:
            self.logger().info("=" * 60)
            self.logger().info(f"⚡ 高频再平衡: 新价格 {current_price:.8f}")
            self.logger().info("=" * 60)

            # 1. 关闭旧仓位
            await self.close_position()

            # 2. 等待链上确认
            await asyncio.sleep(3)

            # 3. 重新平衡代币（关键！）
            # 移除流动性后，代币比例可能是 80% ACE + 20% SOL
            # 需要重新换成 50:50 才能按配置开仓
            self.logger().info("准备代币（再平衡后需要重新分配）...")
            success = await self.prepare_tokens_for_position(current_price)
            if not success:
                self.logger().error("❌ 再平衡后代币准备失败")
                return

            # 4. 在新价格立即开仓（紧跟价格）
            await self.open_position(current_price)

            # 5. 标记执行
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
            self.position_info_last_update = None  # 重置仓位信息更新时间
            self.open_price = None  # 重置开仓价格
            self.initial_investment = Decimal("0")  # 重置初始投资

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

                # 记录开仓到状态管理器
                if self.state_manager and self.open_price:
                    # 获取仓位ID（异步获取可能还没完成，先用订单ID）
                    temp_position_id = f"pending_{order_id[:8]}"
                    self.state_manager.record_open(
                        position_id=temp_position_id,
                        entry_price=self.open_price
                    )

                # 重置止损引擎
                self.stop_loss_engine.reset()

                # 清除待处理订单ID和时间戳
                self.pending_open_order_id = None
                self.pending_order_start_time = None
                self.open_retry_count = 0  # 重置重试计数

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
                self.pending_order_start_time = None
                # 不重置 open_retry_count，让超时机制来处理重试

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
                # 计算实际总价值
                if self.pool_info and self.pool_info.price:
                    actual_total_value = (
                        Decimal(str(self.position_info.base_token_amount)) *
                        Decimal(str(self.pool_info.price)) +
                        Decimal(str(self.position_info.quote_token_amount))
                    )
                else:
                    actual_total_value = Decimal("0")

                self.logger().info(
                    f"📊 实际结果对照:\n"
                    f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"  仓位ID: {self.position_id}\n"
                    f"  实际区间:\n"
                    f"    下界: {self.position_info.lower_price:.10f}\n"
                    f"    上界: {self.position_info.upper_price:.10f}\n"
                    f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"  实际持仓:\n"
                    f"    {self.base_token}: {self.position_info.base_token_amount:.6f}\n"
                    f"    {self.quote_token}: {self.position_info.quote_token_amount:.6f}\n"
                    f"  实际总价值: {actual_total_value:.6f} {self.quote_token}\n"
                    f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                )
            else:
                self.logger().warning("未能获取到仓位信息，将在下次监控时重试")

        except Exception as e:
            self.logger().error(f"获取仓位信息失败: {e}", exc_info=True)

    # ========================================
    # 状态展示
    # ========================================

    def format_status(self) -> str:
        """格式化状态（优化版 - 清晰展示止损逻辑）"""
        try:
            lines = []
            lines.append("=" * 70)
            lines.append("⚡ Meteora DLMM 高频做市策略 - 实时状态")
            lines.append("=" * 70)

            # === 1. 钱包余额 ===
            lines.append("\n💰 钱包余额:")
            try:
                base_balance = self.connector.get_available_balance(self.base_token)
                quote_balance = self.connector.get_available_balance(self.quote_token)
                lines.append(f"  {self.base_token}: {base_balance:.6f}")
                lines.append(f"  {self.quote_token}: {quote_balance:.6f}")
            except Exception as e:
                lines.append(f"  无法获取余额: {e}")

            # === 2. 仓位状态 ===
            if not self.position_opened or not self.position_info:
                lines.append("\n📊 仓位状态: 无持仓")

                # 显示下次检查时间
                if self.last_check_time:
                    next_check_seconds = self.config.check_interval_seconds - (datetime.now() - self.last_check_time).total_seconds()
                    if next_check_seconds > 0:
                        lines.append(f"⏱️  下次检查: {next_check_seconds:.0f}秒后")

                lines.append("=" * 70)
                return "\n".join(lines)

            lines.append("\n📊 仓位状态: 已开仓")
            lines.append(f"仓位ID: {self.position_id[:10]}...{self.position_id[-10:] if self.position_id else ''}")

            # === 3. 价格和距离信息 ===
            # 获取当前价格，确保有效性
            if self.pool_info and hasattr(self.pool_info, 'price') and self.pool_info.price:
                current_price = Decimal(str(self.pool_info.price))
            elif self.open_price:
                current_price = Decimal(str(self.open_price))
                self.logger().warning("⚠️  池子价格无效，使用开仓价格作为参考")
            else:
                current_price = Decimal("0")

            lower_price = Decimal(str(self.position_info.lower_price))
            upper_price = Decimal(str(self.position_info.upper_price))

            lines.append(f"\n💹 价格信息:")
            lines.append(f"  当前价格: {current_price:.10f}")
            lines.append(f"  价格区间: [{lower_price:.10f}, {upper_price:.10f}]")

            # 计算距离边界
            if lower_price <= current_price <= upper_price:
                distance_to_lower_pct = ((current_price - lower_price) / lower_price) * Decimal("100")
                distance_to_upper_pct = ((upper_price - current_price) / current_price) * Decimal("100")
                lines.append(f"  状态: ✅ 在范围内")
                lines.append(f"  距下界: +{distance_to_lower_pct:.2f}%")
                lines.append(f"  距上界: +{distance_to_upper_pct:.2f}%")
            elif current_price < lower_price:
                out_pct = ((lower_price - current_price) / lower_price) * Decimal("100")
                lines.append(f"  状态: 🔻 超出下界 -{out_pct:.2f}%")
            else:
                out_pct = ((current_price - upper_price) / upper_price) * Decimal("100")
                lines.append(f"  状态: 🔺 超出上界 +{out_pct:.2f}%")

            # === 4. 盈亏信息 ===
            if self.open_price and current_price > 0:
                price_change_pct = ((current_price - self.open_price) / self.open_price) * Decimal("100")
                lines.append(f"\n📈 盈亏分析:")
                lines.append(f"  开仓价格: {self.open_price:.10f}")

                pnl_icon = "📈" if price_change_pct >= 0 else "📉"
                lines.append(f"  价格变化: {pnl_icon} {price_change_pct:+.2f}%")

                # 计算当前仓位价值
                base_amount = Decimal(str(self.position_info.base_token_amount))
                quote_amount = Decimal(str(self.position_info.quote_token_amount))
                current_value = (base_amount * current_price) + quote_amount

                # 计算手续费
                try:
                    base_fees = Decimal(str(self.position_info.base_fee_amount)) if hasattr(self.position_info, 'base_fee_amount') else Decimal("0")
                    quote_fees = Decimal(str(self.position_info.quote_fee_amount)) if hasattr(self.position_info, 'quote_fee_amount') else Decimal("0")
                    fees_value = (base_fees * current_price) + quote_fees
                except Exception:
                    base_fees = Decimal("0")
                    quote_fees = Decimal("0")
                    fees_value = Decimal("0")

                if self.initial_investment > 0:
                    unrealized_pnl = (current_value + fees_value) - self.initial_investment
                    unrealized_pnl_pct = (unrealized_pnl / self.initial_investment) * Decimal("100")

                    pnl_status_icon = "📈" if unrealized_pnl > 0 else "📉"
                    lines.append(f"  {pnl_status_icon} 未实现盈亏: {unrealized_pnl:+.6f} {self.quote_token} ({unrealized_pnl_pct:+.2f}%)")
                    lines.append(f"  初始投资: {self.initial_investment:.6f} {self.quote_token}")
                    lines.append(f"  当前价值: {current_value:.6f} {self.quote_token}")
                    lines.append(f"  累计手续费: {fees_value:.6f} {self.quote_token}")

            # === 5. 止损规则详情（核心优化） ===
            lines.append(f"\n🛡️  止损规则（三级保护）:")

            # Level 1: 幅度止损（最高优先级）
            if self.open_price and current_price > 0:
                price_change_pct = ((current_price - self.open_price) / self.open_price) * Decimal("100")
                stop_loss_threshold = -self.config.stop_loss_pct
                distance_to_hard_stop = price_change_pct - stop_loss_threshold

                if price_change_pct <= stop_loss_threshold:
                    lines.append(f"  🚨 Level 1 - 幅度止损: 已触发！")
                    lines.append(f"    当前下跌: {price_change_pct:.2f}% (阈值: {stop_loss_threshold:.2f}%)")
                else:
                    lines.append(f"  ✅ Level 1 - 幅度止损: {self.config.stop_loss_pct}%")
                    if price_change_pct < 0:
                        lines.append(f"    当前下跌: {price_change_pct:.2f}% (距触发: {distance_to_hard_stop:.2f}%)")
                    else:
                        lines.append(f"    当前上涨: {price_change_pct:+.2f}% (无风险)")

            # Level 2: 超出区间确认
            lines.append(f"  ✅ Level 2 - 超出区间确认: {self.config.out_of_range_timeout_seconds}秒")

            if self.stop_loss_engine and self.stop_loss_engine.below_range_since:
                out_duration = time.time() - self.stop_loss_engine.below_range_since
                remaining = float(self.config.out_of_range_timeout_seconds) - out_duration
                progress = (out_duration / float(self.config.out_of_range_timeout_seconds)) * 100.0

                # 判断方向
                if current_price < lower_price:
                    direction = "下跌"
                    direction_icon = "🔻"
                else:
                    direction = "上涨"
                    direction_icon = "🔺"

                lines.append(f"    {direction_icon} 超出区间 ({direction}): {out_duration:.0f}s / {self.config.out_of_range_timeout_seconds}s ({progress:.0f}%)")

                if remaining > 0:
                    lines.append(f"    剩余时间: {remaining:.0f}s")
                    if remaining <= 10:
                        if direction == "下跌":
                            lines.append(f"    ⚠️  即将触发硬止损！")
                        else:
                            lines.append(f"    ⚠️  即将触发再平衡！")
                else:
                    if direction == "下跌":
                        lines.append(f"    🚨 已触发硬止损！")
                    else:
                        lines.append(f"    🔄 已触发再平衡（非止损）")
            else:
                lines.append(f"    ✅ 价格在区间内")

            # Level 3: 持仓时长
            if self.stop_loss_engine and self.stop_loss_engine.position_opened_at:
                hold_hours = (time.time() - self.stop_loss_engine.position_opened_at) / 3600
                max_hold_hours = float(self.config.max_position_hold_hours)

                if hold_hours >= max_hold_hours:
                    lines.append(f"  ⚠️  Level 3 - 持仓时长: {hold_hours:.1f}h / {max_hold_hours:.1f}h (已触发)")
                else:
                    remaining_hours = max_hold_hours - hold_hours
                    lines.append(f"  ✅ Level 3 - 持仓时长: {hold_hours:.1f}h / {max_hold_hours:.1f}h (剩余: {remaining_hours:.1f}h)")

            # 换币到 SOL 的条件说明
            lines.append(f"\n💱 止损换币:")
            if self.config.stop_loss_to_sol:
                lines.append(f"  ✅ 启用 - 止损触发时强制换成 SOL")
                lines.append(f"  目的: 保护本金，防止继续持有贬值的 {self.base_token}")
                lines.append(f"  滑点容忍: {self.config.stop_loss_slippage_pct}%")
            else:
                lines.append(f"  ❌ 未启用 - 止损后保留 {self.base_token}")

            # === 6. 再平衡状态 ===
            lines.append(f"\n🔄 再平衡状态:")
            lines.append(f"  今日次数: {self.rebalance_count_today}")
            lines.append(f"  触发条件:")
            lines.append(f"    • 距离阈值: {self.config.rebalance_threshold_pct}%")
            lines.append(f"    • 冷却期: {self.config.rebalance_cooldown_seconds}秒")
            lines.append(f"    • 最小盈利: {self.config.min_profit_for_rebalance}%")

            if self.rebalance_engine and self.rebalance_engine.last_rebalance_time:
                remaining_cooldown = self.rebalance_engine._remaining_cooldown(self.config.rebalance_cooldown_seconds)
                if remaining_cooldown > 0:
                    lines.append(f"  ⏳ 冷却中: 剩余 {remaining_cooldown:.0f}s")
                else:
                    lines.append(f"  ✅ 就绪")

            # === 7. 策略配置摘要 ===
            lines.append(f"\n⚙️  策略配置:")
            lines.append(f"  交易对: {self.config.trading_pair}")
            lines.append(f"  区间宽度: ±{self.config.price_range_pct}%")
            lines.append(f"  检查间隔: {self.config.check_interval_seconds}秒")

            # === 8. 今日统计 ===
            lines.append(f"\n📊 今日统计:")
            lines.append(f"  再平衡: {self.rebalance_count_today} 次")
            lines.append(f"  止损: {self.stop_loss_count_today} 次")

            # === 9. 下次检查时间 ===
            if self.last_check_time:
                next_check_seconds = self.config.check_interval_seconds - (datetime.now() - self.last_check_time).total_seconds()
                if next_check_seconds > 0:
                    lines.append(f"\n⏱️  下次检查: {next_check_seconds:.0f}秒后")

            lines.append("=" * 70)
            return "\n".join(lines)
        except Exception as e:
            import traceback
            return f"❌ 状态显示错误: {str(e)}\n{traceback.format_exc()}"


if __name__ == "__main__":
    pass
