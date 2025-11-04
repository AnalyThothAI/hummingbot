"""
止损引擎 - 判断是否应该退出市场

职责：
1. 价格止损检测（绝对价格 + 防插针）
2. 区间止损检测（跌破区间下界 + 60秒规则）
3. 时间止损检测（持仓过久未盈利）

不负责：
- 再平衡判断（交给 RebalanceEngine）
- 换币操作（交给 SwapManager）
- 任何上涨/横盘场景的判断

设计原则：
- 单一职责：只判断"是否应该退出市场"
- 低耦合：不依赖其他引擎
- 防插针：所有判断都需要时间确认
"""

import time
from decimal import Decimal
from typing import Optional, Tuple


class StopLossEngine:
    """止损引擎 - 判断是否应该退出市场"""

    def __init__(self, logger, config):
        """
        初始化止损引擎

        Args:
            logger: 日志记录器
            config: 策略配置对象（需包含止损相关字段）
        """
        self.logger = logger
        self.config = config

        # 止损价格（开仓时计算）
        self.stop_loss_price: Optional[Decimal] = None

        # 防插针计时器
        self.below_stop_loss_since: Optional[float] = None  # 跌破止损价时间
        self.below_range_since: Optional[float] = None      # 跌破区间下界时间

        # 持仓时间
        self.position_opened_at: Optional[float] = None

    # ========================================
    # 初始化和重置
    # ========================================

    def set_position_opened(
        self,
        open_price: Decimal,
        position_lower_price: Decimal
    ):
        """
        开仓时调用，计算止损价格

        Args:
            open_price: 开仓价格
            position_lower_price: LP 区间下界

        止损价格计算优先级：
        1. stop_loss_price（用户指定绝对价格）
        2. stop_loss_below_range_pct（区间下界的百分比）
        3. stop_loss_pct（开仓价格的百分比）
        """
        self.position_opened_at = time.time()
        self.below_stop_loss_since = None
        self.below_range_since = None

        # 计算止损价格（三选一，按优先级）
        if self.config.stop_loss_price:
            # 方式 1：用户指定的绝对价格（最高优先级）
            self.stop_loss_price = Decimal(str(self.config.stop_loss_price))
            self.logger.info(
                f"✅ 止损价格（绝对价格）: {self.stop_loss_price}"
            )

        elif hasattr(self.config, 'stop_loss_below_range_pct') and self.config.stop_loss_below_range_pct:
            # 方式 2：区间下界的百分比
            self.stop_loss_price = position_lower_price * (
                Decimal("1") - Decimal(str(self.config.stop_loss_below_range_pct)) / Decimal("100")
            )
            self.logger.info(
                f"✅ 止损价格（区间下界 -{self.config.stop_loss_below_range_pct}%）: {self.stop_loss_price}\n"
                f"   区间下界: {position_lower_price}"
            )

        else:
            # 方式 3：开仓价格的百分比（默认）
            self.stop_loss_price = open_price * (
                Decimal("1") - Decimal(str(self.config.stop_loss_pct)) / Decimal("100")
            )
            self.logger.info(
                f"✅ 止损价格（开仓价 -{self.config.stop_loss_pct}%）: {self.stop_loss_price}\n"
                f"   开仓价格: {open_price}"
            )

    def reset(self):
        """重置状态（用于测试或手动重置）"""
        self.below_stop_loss_since = None
        self.below_range_since = None

    # ========================================
    # 核心判断逻辑
    # ========================================

    def check_stop_loss(
        self,
        current_price: Decimal,
        open_price: Decimal,
        lower_price: Decimal,
        upper_price: Decimal
    ) -> Tuple[bool, str, str]:
        """
        检查是否触发止损

        Args:
            current_price: 当前价格
            open_price: 开仓价格
            lower_price: LP 区间下界
            upper_price: LP 区间上界

        Returns:
            (是否止损, 止损类型, 原因)

            止损类型:
            - "HARD_STOP": 硬止损，立即退出市场
            - "SOFT_STOP": 软止损，建议退出但不强制
            - "NONE": 无止损触发
        """
        now = time.time()

        # === 防御性检查 ===
        if not open_price or open_price <= 0:
            self.logger.warning("⚠️ 开仓价格无效，跳过止损检查")
            return False, "NONE", ""

        # === 硬止损 1: 价格止损（绝对价格 + 防插针）===
        if self.stop_loss_price and current_price < self.stop_loss_price:
            if self.below_stop_loss_since is None:
                # 首次跌破，开始计时
                self.below_stop_loss_since = now
                self.logger.warning(
                    f"⚠️ 价格跌破止损位 {self.stop_loss_price:.10f}，开始确认...\n"
                    f"   当前价格: {current_price:.10f}\n"
                    f"   需要持续: {self.config.stop_loss_confirmation_seconds}秒"
                )
                return False, "NONE", ""

            # 检查持续时间
            duration = now - self.below_stop_loss_since

            if duration >= self.config.stop_loss_confirmation_seconds:
                # 确认止损
                return (
                    True,
                    "HARD_STOP",
                    f"价格跌破止损位 {self.stop_loss_price:.10f}（确认 {duration:.0f}秒）"
                )
            else:
                # 仍在确认中
                remaining = self.config.stop_loss_confirmation_seconds - duration
                if remaining <= 10:
                    # 距离触发不到 10 秒，警告
                    self.logger.warning(
                        f"⚠️ 即将触发价格止损！剩余 {remaining:.0f}秒"
                    )
                return False, "NONE", ""

        else:
            # 价格回升，重置计时器
            if self.below_stop_loss_since is not None:
                self.logger.info("✅ 价格回升至止损位之上，取消止损确认")
                self.below_stop_loss_since = None

        # === 硬止损 2: 区间止损（跌破下界 + 60秒规则 + 下跌确认）===
        if current_price < lower_price:
            if self.below_range_since is None:
                # 首次跌破区间，开始计时
                self.below_range_since = now
                self.logger.warning(
                    f"⚠️ 价格跌破区间下界 {lower_price:.10f}，开始确认...\n"
                    f"   当前价格: {current_price:.10f}\n"
                    f"   需要持续: {self.config.out_of_range_timeout_seconds}秒"
                )
                return False, "NONE", ""

            # 检查持续时间
            duration = now - self.below_range_since

            if duration >= self.config.out_of_range_timeout_seconds:
                # 检查是否确实下跌（防止横盘突破）
                price_change_pct = (current_price - open_price) / open_price * Decimal("100")

                if price_change_pct < -3:  # 下跌超过 3% 才确认是真下跌
                    return (
                        True,
                        "HARD_STOP",
                        f"跌破区间下界 {duration:.0f}秒（下跌 {abs(price_change_pct):.2f}%）"
                    )
                else:
                    # 下跌幅度不够，可能是横盘或小幅回调，不触发止损
                    self.logger.info(
                        f"💡 价格跌破区间 {duration:.0f}秒，但下跌幅度较小（{price_change_pct:.2f}%），"
                        f"继续观察"
                    )
                    return False, "NONE", ""

            else:
                # 仍在确认中
                remaining = self.config.out_of_range_timeout_seconds - duration
                if remaining <= 10:
                    self.logger.warning(
                        f"⚠️ 即将触发区间止损！剩余 {remaining:.0f}秒"
                    )
                return False, "NONE", ""

        else:
            # 价格回到区间内，重置计时器
            if self.below_range_since is not None:
                self.logger.info("✅ 价格回到区间内，取消止损确认")
                self.below_range_since = None

        # === 硬止损 3: 时间止损（持仓过久 + 亏损）===
        if self.position_opened_at is not None:
            hold_hours = (now - self.position_opened_at) / 3600
            max_hold_hours = float(self.config.max_position_hold_hours)

            if hold_hours >= max_hold_hours:
                price_change_pct = (current_price - open_price) / open_price * Decimal("100")

                if price_change_pct < 0:
                    # 持仓过久且亏损，硬止损
                    return (
                        True,
                        "HARD_STOP",
                        f"持仓 {hold_hours:.1f}h 超过限制（亏损 {abs(price_change_pct):.2f}%）"
                    )
                elif price_change_pct < 2:
                    # 持仓过久但盈利很少，软止损（建议）
                    return (
                        True,
                        "SOFT_STOP",
                        f"持仓 {hold_hours:.1f}h 超过限制（盈利仅 {price_change_pct:.2f}%）"
                    )

        # 无止损触发
        return False, "NONE", ""

    # ========================================
    # 状态查询（用于 status 展示）
    # ========================================

    def get_stop_loss_status(
        self,
        current_price: Decimal,
        open_price: Decimal,
        lower_price: Decimal
    ) -> dict:
        """
        获取止损状态（用于 status 展示）

        Returns:
            {
                "stop_loss_price": Decimal,
                "below_stop_loss_duration": float,
                "below_range_duration": float,
                "hold_hours": float,
                "price_change_pct": Decimal
            }
        """
        now = time.time()

        # 计算持续时间
        below_stop_loss_duration = (
            now - self.below_stop_loss_since
            if self.below_stop_loss_since
            else 0.0
        )

        below_range_duration = (
            now - self.below_range_since
            if self.below_range_since
            else 0.0
        )

        hold_hours = (
            (now - self.position_opened_at) / 3600
            if self.position_opened_at
            else 0.0
        )

        price_change_pct = (
            (current_price - open_price) / open_price * Decimal("100")
            if open_price and open_price > 0
            else Decimal("0")
        )

        return {
            "stop_loss_price": self.stop_loss_price,
            "below_stop_loss_duration": below_stop_loss_duration,
            "below_range_duration": below_range_duration,
            "hold_hours": hold_hours,
            "price_change_pct": price_change_pct
        }
