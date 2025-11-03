"""
高频再平衡决策引擎

负责：
1. 判断是否需要再平衡仓位
2. 冷却期管理
3. 距离边界计算
4. 60秒规则触发
5. 最小盈利检查

返回再平衡决策和原因
"""

import time
from decimal import Decimal
from typing import Optional, Tuple


class HighFrequencyRebalanceEngine:
    """高频再平衡决策引擎"""

    def __init__(self, logger):
        """
        初始化再平衡引擎

        Args:
            logger: 日志记录器
        """
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

        Args:
            current_price: 当前价格
            lower_price: 区间下界
            upper_price: 区间上界
            accumulated_fees_value: 累积手续费价值
            position_value: 仓位总价值
            config: 策略配置对象
            out_duration_seconds: 超出区间的时长（秒）

        Returns:
            (是否再平衡, 原因)
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
        """检查冷却期是否已过"""
        if self.last_rebalance_time is None:
            return True
        return (time.time() - self.last_rebalance_time) >= cooldown_seconds

    def _remaining_cooldown(self, cooldown_seconds: int) -> float:
        """计算剩余冷却时间（秒）"""
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
        """
        计算价格距离边界的相对距离（百分比）

        Returns:
            0.0 (在边界) ~ 0.5 (在中心)
        """
        range_width = upper_price - lower_price
        distance_to_lower = current_price - lower_price
        distance_to_upper = upper_price - current_price
        min_distance = min(distance_to_lower, distance_to_upper)
        return min_distance / range_width

    def mark_rebalance_executed(self):
        """标记再平衡已执行（更新冷却时间）"""
        self.last_rebalance_time = time.time()
