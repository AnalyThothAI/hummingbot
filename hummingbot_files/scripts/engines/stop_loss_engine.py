"""
快速止损引擎

负责：
1. 幅度止损检测（价格下跌超过阈值）
2. 60秒规则检测（价格超出区间持续时间）
3. 持仓时长检测（长期未盈利）

返回止损决策和建议
"""

import time
from decimal import Decimal
from typing import Optional, Tuple


class FastStopLossEngine:
    """快速止损引擎"""

    def __init__(self, logger, config):
        """
        初始化止损引擎

        Args:
            logger: 日志记录器
            config: 策略配置对象（需包含 stop_loss_pct, enable_60s_rule 等字段）
        """
        self.logger = logger
        self.config = config

        # 时间记录
        self.price_out_of_range_since: Optional[float] = None
        self.position_opened_at: Optional[float] = None

    def reset(self):
        """重置状态（开仓时调用）"""
        self.price_out_of_range_since = None
        self.position_opened_at = time.time()

    def check_stop_loss(
        self,
        current_price: Decimal,
        open_price: Decimal,
        lower_price: Decimal,
        upper_price: Decimal
    ) -> Tuple[bool, str, str, float]:
        """
        检查是否触发止损

        Args:
            current_price: 当前价格
            open_price: 开仓价格
            lower_price: 区间下界
            upper_price: 区间上界

        Returns:
            (是否止损, 止损类型, 原因, 超出区间时长)

            止损类型:
            - "HARD_STOP": 立即止损（幅度止损、下跌60秒规则）
            - "SOFT_STOP": 建议止损（持仓过久未盈利）
            - "REBALANCE": 需要再平衡（上涨超出区间）
            - "NONE": 无止损触发
        """

        now = time.time()

        # === 防御性检查：open_price ===
        if not open_price or open_price <= 0:
            self.logger.warning("⚠️  开仓价格无效，部分止损逻辑将跳过")
            # 设置一个默认值，避免后续崩溃
            price_change_pct = Decimal("0")
            has_valid_open_price = False
        else:
            price_change_pct = (current_price - open_price) / open_price * Decimal("100")
            has_valid_open_price = True

        # 计算超出区间时长（在所有逻辑之前）
        is_out_of_range = current_price < lower_price or current_price > upper_price

        if is_out_of_range:
            if self.price_out_of_range_since is None:
                self.price_out_of_range_since = now
            out_duration = now - self.price_out_of_range_since
        else:
            # 价格在区间内
            out_duration = 0.0
            # 重置计时器
            self.price_out_of_range_since = None

        # === Level 1: 幅度止损（最高优先级）===
        if has_valid_open_price and price_change_pct <= -self.config.stop_loss_pct:
            return True, "HARD_STOP", f"下跌 {abs(price_change_pct):.2f}% 超过止损线 {self.config.stop_loss_pct}%", out_duration

        # === Level 2: 60秒规则 + 下跌 ===
        if is_out_of_range:
            if self.config.enable_60s_rule and out_duration >= self.config.out_of_range_timeout_seconds:
                # 超出 60 秒，检查方向
                if current_price < lower_price and has_valid_open_price and price_change_pct < -3:
                    # 下跌方向，立即止损
                    return True, "HARD_STOP", f"下跌超出区间 {out_duration:.0f}秒", out_duration
                else:
                    # 上涨方向，触发再平衡（不是止损）
                    return False, "REBALANCE", f"超出区间 {out_duration:.0f}秒，需要再平衡", out_duration

        # === Level 3: 持仓时长 ===
        if self.position_opened_at is not None:
            hold_hours = (now - self.position_opened_at) / 3600

            if hold_hours >= float(self.config.max_position_hold_hours):
                # 持仓过久且未盈利
                if has_valid_open_price and price_change_pct < 0:
                    return True, "SOFT_STOP", f"持仓 {hold_hours:.1f}h 未盈利", out_duration

        # 无止损触发
        return False, "NONE", "", out_duration
