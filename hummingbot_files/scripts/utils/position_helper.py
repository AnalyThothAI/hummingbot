"""
仓位管理辅助函数

负责：
1. 检查现有仓位
2. 获取仓位信息
3. 计算仓位价值
4. 获取代币数量

简化主策略文件的仓位管理逻辑
"""

import asyncio
from decimal import Decimal
from typing import Optional, Tuple

from hummingbot.connector.gateway.gateway_lp import CLMMPositionInfo


async def check_existing_positions(connector, trading_pair: str, logger) -> Tuple[bool, Optional[str], Optional[CLMMPositionInfo]]:
    """
    检查是否有现有仓位（带流动性检查）

    Args:
        connector: Gateway connector 实例
        trading_pair: 交易对
        logger: 日志记录器

    Returns:
        (是否有仓位, 仓位ID, 仓位信息)
    """
    try:
        logger.debug("检查现有 LP 仓位...")
        positions = await connector.get_clmm_positions(
            trading_pair=trading_pair
        )

        if not positions:
            logger.debug("未检测到现有仓位")
            return False, None, None

        # 获取第一个仓位（策略只管理一个仓位）
        position_info = positions[0]
        position_id = position_info.position_id

        # ✅ 关键修复：检查仓位是否实际有流动性
        base_amount = Decimal(str(position_info.base_token_amount))
        quote_amount = Decimal(str(position_info.quote_token_amount))

        # 仓位存在但流动性为 0，视为已关闭
        if base_amount <= Decimal("0.000001") and quote_amount <= Decimal("0.000001"):
            logger.warning(
                f"⚠️ 发现空仓位（已关闭）:\n"
                f"  ID: {position_id}\n"
                f"  Base: {base_amount}\n"
                f"  Quote: {quote_amount}\n"
                f"  视为无仓位状态"
            )
            return False, None, None

        # 仓位有效
        logger.info(
            f"✅ 检测到有效仓位:\n"
            f"  ID: {position_id}\n"
            f"  区间: {position_info.lower_price:.10f} - {position_info.upper_price:.10f}\n"
            f"  Base: {base_amount:.6f}\n"
            f"  Quote: {quote_amount:.6f}"
        )

        return True, position_id, position_info

    except Exception as e:
        logger.warning(f"检查现有仓位失败: {e}")
        return False, None, None


async def get_token_amounts(
    connector,
    base_token: str,
    quote_token: str,
    config,
    logger
) -> Tuple[Decimal, Decimal]:
    """
    获取用于开仓的代币数量

    Args:
        connector: Gateway connector 实例
        base_token: 基础代币符号
        quote_token: 报价代币符号
        config: 策略配置对象
        logger: 日志记录器

    Returns:
        (base_amount, quote_amount)
    """
    try:
        # 如果配置了固定数量，直接使用
        if config.base_token_amount > 0 or config.quote_token_amount > 0:
            logger.debug(
                f"使用配置的固定数量:\n"
                f"  {base_token}: {config.base_token_amount}\n"
                f"  {quote_token}: {config.quote_token_amount}"
            )
            return config.base_token_amount, config.quote_token_amount

        # 否则使用钱包余额的百分比
        logger.info("强制刷新余额...")
        await connector.update_balances(on_interval=False)
        await asyncio.sleep(1)

        base_balance = connector.get_available_balance(base_token)
        quote_balance = connector.get_available_balance(quote_token)

        logger.info(
            f"当前钱包余额（已刷新）:\n"
            f"  {base_token}: {base_balance}\n"
            f"  {quote_token}: {quote_balance}"
        )

        # 检查余额是否足够
        if base_balance <= 0 and quote_balance <= 0:
            logger.error(
                f"❌ 余额不足，无法开仓\n"
                f"   {base_token}: {base_balance}\n"
                f"   {quote_token}: {quote_balance}"
            )
            return Decimal("0"), Decimal("0")

        # 使用配置的百分比
        use_pct = config.wallet_balance_use_pct / Decimal("100")
        total_base = base_balance * use_pct
        total_quote = quote_balance * use_pct

        logger.info(
            f"使用钱包余额的 {config.wallet_balance_use_pct}%:\n"
            f"  {base_token}: {total_base:.6f}\n"
            f"  {quote_token}: {total_quote:.6f}"
        )

        return total_base, total_quote

    except Exception as e:
        logger.error(f"获取代币数量失败: {e}", exc_info=True)
        return Decimal("0"), Decimal("0")


async def calculate_position_value(
    position_info: Optional[CLMMPositionInfo],
    current_price: Decimal,
    logger
) -> Decimal:
    """
    计算仓位当前价值（以 quote token 计）

    Args:
        position_info: 仓位信息
        current_price: 当前价格
        logger: 日志记录器

    Returns:
        仓位价值（Decimal）
    """
    try:
        if not position_info:
            return Decimal("0")

        base_amount = Decimal(str(position_info.base_token_amount))
        quote_amount = Decimal(str(position_info.quote_token_amount))

        # 计算总价值 = base * price + quote
        total_value = base_amount * current_price + quote_amount

        logger.debug(
            f"仓位价值计算:\n"
            f"  Base: {base_amount:.6f} × {current_price:.10f} = {base_amount * current_price:.6f}\n"
            f"  Quote: {quote_amount:.6f}\n"
            f"  Total: {total_value:.6f}"
        )

        return total_value

    except Exception as e:
        logger.error(f"计算仓位价值失败: {e}", exc_info=True)
        return Decimal("0")


def calculate_price_range(
    center_price: Decimal,
    range_width_pct: Decimal
) -> Tuple[Decimal, Decimal]:
    """
    计算价格区间

    Args:
        center_price: 中心价格
        range_width_pct: 区间宽度百分比（例如 5 表示 ±5%）

    Returns:
        (lower_price, upper_price)
    """
    lower_price = center_price * (Decimal("1") - range_width_pct / Decimal("100"))
    upper_price = center_price * (Decimal("1") + range_width_pct / Decimal("100"))
    return lower_price, upper_price


def calculate_width_percentages(
    center_price: Decimal,
    lower_price: Decimal,
    upper_price: Decimal
) -> Tuple[float, float]:
    """
    计算上下区间宽度百分比（用于 Gateway API）

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
