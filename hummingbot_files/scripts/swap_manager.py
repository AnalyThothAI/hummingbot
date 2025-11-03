"""
资产交换管理器
用于通过 Jupiter/DEX 将代币换成 SOL

特性:
- 支持通过 Gateway 调用 Jupiter
- 自动余额刷新
- 滑点保护
- 失败重试
- 详细日志记录
"""

import asyncio
from decimal import Decimal
from typing import Optional, Tuple
from hummingbot.connector.connector_base import ConnectorBase


class SwapManager:
    """资产交换管理器（通过 Gateway + Jupiter）"""

    def __init__(self, connector: ConnectorBase, logger=None):
        """
        初始化交换管理器

        Args:
            connector: Gateway connector 实例
            logger: 日志记录器（可选）
        """
        self.connector = connector
        self.logger = logger

        if self.logger:
            self.logger.info("✅ SwapManager 初始化完成")

    async def swap_all_to_sol(
        self,
        token: str,
        slippage_pct: Decimal = Decimal("2"),
        reason: str = "STOP_LOSS",
        retry_count: int = 2
    ) -> Tuple[bool, Decimal, str]:
        """
        将指定代币全部换成 SOL

        Args:
            token: 要换出的代币符号
            slippage_pct: 滑点容忍度（百分比，默认2%）
            reason: 换币原因（用于日志）
            retry_count: 失败重试次数

        Returns:
            (是否成功, 换得的SOL数量, 错误信息)
        """
        if self.logger:
            self.logger.info(f"🔄 开始换币: {token} → SOL (原因: {reason})")

        # === Step 1: 获取当前余额 ===
        try:
            balance = self.connector.get_available_balance(token)
        except Exception as e:
            error_msg = f"获取 {token} 余额失败: {e}"
            if self.logger:
                self.logger.error(error_msg)
            return False, Decimal("0"), error_msg

        # 如果余额为0，直接返回成功
        if balance <= 0:
            if self.logger:
                self.logger.warning(f"⚠️  {token} 余额为0，无需换币")
            return True, Decimal("0"), "余额为0"

        if self.logger:
            self.logger.info(f"  {token} 余额: {balance:.6f}")

        # === Step 2: 获取换币报价 ===
        trading_pair = f"{token}-SOL"

        for attempt in range(retry_count + 1):
            try:
                # 获取报价（预估能换到多少SOL）
                quote_price = await self.connector.get_quote_price(
                    trading_pair=trading_pair,
                    is_buy=False,  # 卖出 token，买入 SOL
                    amount=balance,
                    slippage_pct=slippage_pct
                )

                if not quote_price or quote_price <= 0:
                    error_msg = f"无效的报价: {quote_price}"
                    if self.logger:
                        self.logger.warning(f"⚠️  {error_msg}")

                    if attempt < retry_count:
                        await asyncio.sleep(2 ** attempt)  # 指数退避
                        continue
                    else:
                        return False, Decimal("0"), error_msg

                # 计算预期得到的 SOL 数量
                expected_sol = balance * quote_price

                if self.logger:
                    self.logger.info(
                        f"  报价: 1 {token} = {quote_price:.8f} SOL\n"
                        f"  预期换得: {expected_sol:.6f} SOL"
                    )

                # === Step 3: 执行换币 ===
                order_id = self.connector.sell(
                    trading_pair=trading_pair,
                    amount=balance,
                    order_type=None,  # Gateway 不需要order_type
                    price=quote_price
                )

                if self.logger:
                    self.logger.info(f"  换币订单已提交: {order_id}")

                # === Step 4: 等待交易确认 ===
                await asyncio.sleep(5)  # Solana 确认通常很快，5秒足够

                # === Step 5: 强制刷新余额 ===
                if self.logger:
                    self.logger.info("  刷新余额中...")

                await self.connector.update_balances(on_interval=False)
                await asyncio.sleep(1)  # 等待余额更新

                # === Step 6: 验证结果 ===
                new_token_balance = self.connector.get_available_balance(token)
                sol_balance = self.connector.get_available_balance("SOL")

                if self.logger:
                    self.logger.info(
                        f"✅ 换币完成！\n"
                        f"  {token} 剩余: {new_token_balance:.6f}\n"
                        f"  SOL 余额: {sol_balance:.6f}"
                    )

                # 成功
                return True, sol_balance, ""

            except asyncio.CancelledError:
                raise

            except Exception as e:
                error_msg = f"换币失败 (尝试 {attempt + 1}/{retry_count + 1}): {e}"

                if self.logger:
                    if attempt < retry_count:
                        self.logger.warning(f"⚠️  {error_msg}，重试中...")
                    else:
                        self.logger.error(f"❌ {error_msg}")

                if attempt < retry_count:
                    await asyncio.sleep(2 ** attempt)  # 指数退避
                    continue
                else:
                    return False, Decimal("0"), str(e)

        # 理论上不会到这里
        return False, Decimal("0"), "未知错误"

    async def swap_to_sol_with_amount(
        self,
        token: str,
        amount: Decimal,
        slippage_pct: Decimal = Decimal("2"),
        reason: str = "MANUAL"
    ) -> Tuple[bool, Decimal, str]:
        """
        将指定数量的代币换成 SOL

        Args:
            token: 要换出的代币符号
            amount: 要换出的数量
            slippage_pct: 滑点容忍度（百分比，默认2%）
            reason: 换币原因

        Returns:
            (是否成功, 换得的SOL数量, 错误信息)
        """
        if self.logger:
            self.logger.info(f"🔄 开始换币: {amount:.6f} {token} → SOL (原因: {reason})")

        # 检查余额是否足够
        try:
            balance = self.connector.get_available_balance(token)
        except Exception as e:
            error_msg = f"获取 {token} 余额失败: {e}"
            if self.logger:
                self.logger.error(error_msg)
            return False, Decimal("0"), error_msg

        if balance < amount:
            error_msg = f"余额不足: {balance:.6f} < {amount:.6f}"
            if self.logger:
                self.logger.error(f"❌ {error_msg}")
            return False, Decimal("0"), error_msg

        # 调用完整换币逻辑（临时修改余额）
        original_balance = balance
        try:
            # 暂时无法直接指定数量，需要通过 Gateway API
            # 这里简化处理：直接使用 sell 方法
            trading_pair = f"{token}-SOL"

            quote_price = await self.connector.get_quote_price(
                trading_pair=trading_pair,
                is_buy=False,
                amount=amount,
                slippage_pct=slippage_pct
            )

            if not quote_price or quote_price <= 0:
                error_msg = f"无效的报价: {quote_price}"
                if self.logger:
                    self.logger.error(f"❌ {error_msg}")
                return False, Decimal("0"), error_msg

            expected_sol = amount * quote_price

            if self.logger:
                self.logger.info(
                    f"  报价: 1 {token} = {quote_price:.8f} SOL\n"
                    f"  预期换得: {expected_sol:.6f} SOL"
                )

            order_id = self.connector.sell(
                trading_pair=trading_pair,
                amount=amount,
                order_type=None,
                price=quote_price
            )

            if self.logger:
                self.logger.info(f"  换币订单已提交: {order_id}")

            await asyncio.sleep(5)
            await self.connector.update_balances(on_interval=False)
            await asyncio.sleep(1)

            sol_balance = self.connector.get_available_balance("SOL")

            if self.logger:
                self.logger.info(f"✅ 换币完成！SOL 余额: {sol_balance:.6f}")

            return True, sol_balance, ""

        except Exception as e:
            error_msg = f"换币失败: {e}"
            if self.logger:
                self.logger.error(f"❌ {error_msg}")
            return False, Decimal("0"), str(e)

    async def get_swap_quote(
        self,
        token: str,
        amount: Decimal,
        slippage_pct: Decimal = Decimal("2")
    ) -> Optional[Decimal]:
        """
        获取换币报价（不执行）

        Args:
            token: 要换出的代币
            amount: 要换出的数量
            slippage_pct: 滑点容忍度

        Returns:
            预期换得的 SOL 数量，失败返回 None
        """
        try:
            trading_pair = f"{token}-SOL"
            quote_price = await self.connector.get_quote_price(
                trading_pair=trading_pair,
                is_buy=False,
                amount=amount,
                slippage_pct=slippage_pct
            )

            if not quote_price or quote_price <= 0:
                return None

            expected_sol = amount * quote_price
            return expected_sol

        except Exception as e:
            if self.logger:
                self.logger.error(f"获取报价失败: {e}")
            return None

    def get_sol_balance(self) -> Decimal:
        """
        获取 SOL 余额

        Returns:
            SOL 余额
        """
        try:
            return self.connector.get_available_balance("SOL")
        except Exception as e:
            if self.logger:
                self.logger.error(f"获取 SOL 余额失败: {e}")
            return Decimal("0")

    def get_token_balance(self, token: str) -> Decimal:
        """
        获取指定代币余额

        Args:
            token: 代币符号

        Returns:
            代币余额
        """
        try:
            return self.connector.get_available_balance(token)
        except Exception as e:
            if self.logger:
                self.logger.error(f"获取 {token} 余额失败: {e}")
            return Decimal("0")


# ========================================
# 工具函数
# ========================================

def should_swap_to_sol(
    current_price: Decimal,
    entry_price: Decimal,
    reason: str,
    threshold_pct: Decimal = Decimal("5")
) -> bool:
    """
    判断是否需要换成 SOL

    Args:
        current_price: 当前价格
        entry_price: 开仓价格
        reason: 平仓原因
        threshold_pct: 价格下跌阈值（默认5%）

    Returns:
        是否需要换 SOL
    """
    # 1. 如果平仓原因包含"下跌"或"下界"关键字，必须换
    downside_keywords = ["下跌", "下界", "止损", "HARD_STOP"]
    if any(keyword in reason for keyword in downside_keywords):
        return True

    # 2. 如果价格跌幅超过阈值，换
    if current_price < entry_price:
        drop_pct = abs((current_price - entry_price) / entry_price) * Decimal("100")
        if drop_pct >= threshold_pct:
            return True

    # 3. 其他情况不换（如上涨再平衡）
    return False


# ========================================
# 测试代码
# ========================================

if __name__ == "__main__":
    print("SwapManager 模块测试")
    print("=" * 50)

    # 测试判断逻辑
    print("\n测试判断逻辑:")

    test_cases = [
        (Decimal("0.8"), Decimal("1.0"), "下跌止损", True),
        (Decimal("0.94"), Decimal("1.0"), "超出下界", True),
        (Decimal("0.9"), Decimal("1.0"), "正常再平衡", True),
        (Decimal("1.2"), Decimal("1.0"), "上涨再平衡", False),
        (Decimal("0.99"), Decimal("1.0"), "手动平仓", False),
    ]

    for current, entry, reason, expected in test_cases:
        result = should_swap_to_sol(current, entry, reason)
        status = "✅" if result == expected else "❌"
        print(f"{status} 价格 {current} (开仓{entry}) - {reason}: {result}")

    print("\n" + "=" * 50)
    print("实际使用时需要连接到 Gateway connector")
