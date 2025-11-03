"""
换币辅助函数

负责：
1. 通过 Jupiter 执行换币
2. 准备双边代币（自动 swap 功能）
3. 获取 Jupiter 报价

简化主策略文件的换币逻辑
"""

import asyncio
from decimal import Decimal
from typing import Optional

from hummingbot.core.rate_oracle.rate_oracle import RateOracle


async def swap_via_jupiter(
    connector,
    base_token: str,
    quote_token: str,
    from_token: str,
    to_token: str,
    amount: Decimal,
    side: str,
    logger
) -> bool:
    """
    通过 Jupiter 执行换币

    Args:
        connector: Gateway connector 实例（用于 swap）
        base_token: 基础代币符号
        quote_token: 报价代币符号
        from_token: 输入代币符号
        to_token: 输出代币符号
        amount: 数量
        side: "BUY" 或 "SELL"（仅用于日志）
        logger: 日志记录器

    Returns:
        是否成功
    """
    try:
        # 1. 构造 trading_pair（始终是 BASE-QUOTE 格式）
        trading_pair = f"{base_token}-{quote_token}"

        # 2. 判断 is_buy（Gateway API 语义）
        # - is_buy=True: 买入 base_token（卖出 quote_token）
        # - is_buy=False: 卖出 base_token（买入 quote_token）
        if from_token == base_token:
            is_buy = False  # 卖出 base_token
        else:
            is_buy = True   # 买入 base_token

        # 3. 获取 Jupiter 报价
        logger.info("获取 Jupiter 报价...")
        quote_price = await connector.get_quote_price(
            trading_pair=trading_pair,
            is_buy=is_buy,
            amount=amount
        )

        if not quote_price or quote_price <= 0:
            logger.error(f"获取报价失败，返回价格: {quote_price}")
            return False

        # 注入价格到 RateOracle
        try:
            rate_oracle = RateOracle.get_instance()
            rate_oracle.set_price(trading_pair, Decimal(str(quote_price)))
        except Exception as oracle_err:
            logger.debug(f"RateOracle 注入失败: {oracle_err}")

        # 4. 计算预期的 quote_token 数量
        quote_token_amount = amount * Decimal(str(quote_price))

        # 5. 打印兑换信息
        if is_buy:
            logger.info(
                f"执行 Jupiter 兑换（买入 {base_token}）:\n"
                f"   卖出约: {quote_token_amount:.6f} {quote_token}\n"
                f"   买入: {amount:.6f} {base_token}\n"
                f"   价格: {quote_price:.10f} {quote_token}/{base_token}"
            )
        else:
            logger.info(
                f"执行 Jupiter 兑换（卖出 {base_token}）:\n"
                f"   卖出: {amount:.6f} {base_token}\n"
                f"   买入约: {quote_token_amount:.6f} {quote_token}\n"
                f"   价格: {quote_price:.10f} {quote_token}/{base_token}"
            )

        # 6. 执行兑换
        order_id = connector.place_order(
            is_buy=is_buy,
            trading_pair=trading_pair,
            amount=amount,
            price=quote_price
        )

        logger.info(f"Jupiter 兑换订单已提交: {order_id}")

        # 7. 等待订单成交（简化版）
        await asyncio.sleep(5)
        await connector.update_balances(on_interval=False)

        return True

    except Exception as e:
        logger.error(f"换币失败: {e}", exc_info=True)
        return False


async def prepare_tokens_for_position(
    connector,
    base_token: str,
    quote_token: str,
    current_price: Decimal,
    logger
) -> bool:
    """
    准备双边代币（自动调整为 50:50）

    Args:
        connector: Gateway connector 实例（用于 swap）
        base_token: 基础代币符号
        quote_token: 报价代币符号
        current_price: 当前价格
        logger: 日志记录器

    Returns:
        是否成功
    """
    try:
        logger.info("准备双边代币...")

        await connector.update_balances(on_interval=False)

        base_balance = connector.get_available_balance(base_token)
        quote_balance = connector.get_available_balance(quote_token)

        actual_base_value = Decimal(str(base_balance)) * current_price
        actual_quote_value = Decimal(str(quote_balance))
        total_value = actual_base_value + actual_quote_value

        if total_value == 0:
            logger.error("总余额为 0")
            return False

        target_base_amount = total_value * Decimal("0.5") / current_price
        shortage = target_base_amount - Decimal(str(base_balance))

        if abs(shortage) < Decimal("0.001"):
            return True

        if shortage > 0:
            # 需要买入 base_token
            await swap_via_jupiter(
                connector=connector,
                base_token=base_token,
                quote_token=quote_token,
                from_token=quote_token,
                to_token=base_token,
                amount=shortage * Decimal("1.02"),
                side="BUY",
                logger=logger
            )
        else:
            # 需要卖出 base_token
            await swap_via_jupiter(
                connector=connector,
                base_token=base_token,
                quote_token=quote_token,
                from_token=base_token,
                to_token=quote_token,
                amount=abs(shortage) * Decimal("1.02"),
                side="SELL",
                logger=logger
            )

        return True

    except Exception as e:
        logger.error(f"准备代币失败: {e}", exc_info=True)
        return False
