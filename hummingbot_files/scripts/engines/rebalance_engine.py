"""
再平衡引擎 - 判断是否应该调整做市区间

职责：
1. 检测价格超出上界（上涨趋势）
2. 检测价格接近边界（提前调整）
3. 手续费收益检查（锁定收益）
4. 冷却期管理（防止频繁调整）

不负责：
- 止损判断（交给 StopLossEngine）
- 下跌场景的处理（交给 StopLossEngine）
- 换币操作（交给 SwapManager）

设计原则：
- 单一职责：只判断"是否应该调整做市区间"
- 低耦合：不依赖止损引擎
- 防假突破：需要时间确认（60秒）
"""

import time
from decimal import Decimal
from typing import Optional, Tuple


class RebalanceEngine:
    """再平衡引擎 - 判断是否应该调整做市区间"""

    def __init__(self, logger):
        """
        初始化再平衡引擎

        Args:
            logger: 日志记录器
        """
        self.logger = logger

        # 冷却期管理
        self.last_rebalance_time: Optional[float] = None

        # 防假突破计时器
        self.out_of_range_since: Optional[float] = None

    # ========================================
    # 初始化和重置
    # ========================================

    def reset(self):
        """重置状态（开仓后调用）"""
        self.out_of_range_since = None

    def mark_rebalance_executed(self):
        """标记再平衡已执行（更新冷却时间）"""
        self.last_rebalance_time = time.time()
        self.out_of_range_since = None

    # ========================================
    # 核心判断逻辑
    # ========================================

    async def should_rebalance(
        self,
        current_price: Decimal,
        lower_price: Decimal,
        upper_price: Decimal,
        accumulated_fees_value: Decimal,
        position_value: Decimal,
        config
    ) -> Tuple[bool, str]:
        """
        判断是否需要再平衡

        Args:
            current_price: 当前价格
            lower_price: LP 区间下界
            upper_price: LP 区间上界
            accumulated_fees_value: 累积手续费价值
            position_value: 仓位总价值
            config: 策略配置对象

        Returns:
            (是否再平衡, 原因)

        判断逻辑（按优先级）：
        1. 冷却期检查（防止频繁调整）
        2. 超出上界检查（上涨趋势）
        3. 接近边界检查（提前调整）
        4. 手续费收益检查（锁定收益）
        """

        # === 优先级 0: 冷却期检查 ===
        if not self._is_cooldown_passed(config.rebalance_cooldown_seconds):
            remaining = self._remaining_cooldown(config.rebalance_cooldown_seconds)
            return False, f"冷却期未过（剩余 {remaining:.0f}秒）"

        # === 优先级 1: 超出上界（上涨趋势，积极再平衡）===
        if current_price > upper_price:
            # 开始计时（防假突破）
            if self.out_of_range_since is None:
                self.out_of_range_since = time.time()
                self.logger.info(
                    f"💡 价格超出上界 {upper_price:.10f}，开始确认...\n"
                    f"   当前价格: {current_price:.10f}\n"
                    f"   需要持续: {config.out_of_range_timeout_seconds}秒"
                )
                return False, "超出上界，等待确认"

            # 检查持续时间
            duration = time.time() - self.out_of_range_since

            if duration >= config.out_of_range_timeout_seconds:
                # 确认突破，触发再平衡
                excess_pct = (current_price - upper_price) / upper_price * Decimal("100")
                return True, f"价格超出上界 {excess_pct:.2f}%（确认 {duration:.0f}秒）"
            else:
                # 仍在确认中
                remaining = config.out_of_range_timeout_seconds - duration
                return False, f"超出上界，确认中（剩余 {remaining:.0f}秒）"

        else:
            # 价格回到区间内或跌破区间，重置计时器
            if self.out_of_range_since is not None:
                self.logger.info("✅ 价格回到区间内，取消再平衡确认")
                self.out_of_range_since = None

        # === 优先级 2: 接近边界（提前调整，避免脱离）===
        # 注意：只检查接近上界（上涨方向），不检查接近下界（下跌方向由止损引擎处理）
        if lower_price <= current_price <= upper_price:
            # 价格在区间内，计算距离边界的百分比
            range_width = upper_price - lower_price
            distance_to_upper = upper_price - current_price

            # 计算距离上界的百分比
            upper_pct = (distance_to_upper / range_width) * Decimal("100")

            # 阈值：rebalance_threshold_pct = 85% 意味着剩余 15% 时触发
            threshold = Decimal(str(100 - config.rebalance_threshold_pct))

            if upper_pct < threshold:
                # 接近上界，提前调整
                return True, f"接近上界（剩余 {upper_pct:.1f}%，阈值 {threshold:.1f}%）"

        # === 优先级 3: 手续费收益达标（锁定收益）===
        if accumulated_fees_value > 0 and position_value > 0:
            fees_pct = accumulated_fees_value / position_value * Decimal("100")

            if fees_pct >= Decimal(str(config.min_profit_for_rebalance)):
                return True, f"累积手续费 {fees_pct:.2f}% 达标（阈值 {config.min_profit_for_rebalance}%）"

        # 无需再平衡
        return False, "无需再平衡"

    # ========================================
    # 辅助方法（私有）
    # ========================================

    def _is_cooldown_passed(self, cooldown_seconds: int) -> bool:
        """
        检查冷却期是否已过

        Args:
            cooldown_seconds: 冷却期时长（秒）

        Returns:
            True 如果冷却期已过或从未执行过再平衡
        """
        if self.last_rebalance_time is None:
            return True

        elapsed = time.time() - self.last_rebalance_time
        return elapsed >= cooldown_seconds

    def _remaining_cooldown(self, cooldown_seconds: int) -> float:
        """
        计算剩余冷却时间（秒）

        Args:
            cooldown_seconds: 冷却期时长（秒）

        Returns:
            剩余秒数，如果冷却期已过返回 0
        """
        if self.last_rebalance_time is None:
            return 0

        elapsed = time.time() - self.last_rebalance_time
        return max(0, cooldown_seconds - elapsed)

    # ========================================
    # 状态查询（用于 status 展示）
    # ========================================

    def get_rebalance_status(
        self,
        cooldown_seconds: int
    ) -> dict:
        """
        获取再平衡状态（用于 status 展示）

        Args:
            cooldown_seconds: 配置的冷却期时长

        Returns:
            {
                "is_in_cooldown": bool,
                "remaining_cooldown": float,
                "out_of_range_duration": float
            }
        """
        is_in_cooldown = not self._is_cooldown_passed(cooldown_seconds)
        remaining_cooldown = self._remaining_cooldown(cooldown_seconds)

        out_of_range_duration = (
            time.time() - self.out_of_range_since
            if self.out_of_range_since
            else 0.0
        )

        return {
            "is_in_cooldown": is_in_cooldown,
            "remaining_cooldown": remaining_cooldown,
            "out_of_range_duration": out_of_range_duration
        }
