"""
修复后的 swap_via_jupiter 方法

关键理解：
1. Gateway place_order 的 amount 参数是 base_token 的数量（amm_trade_example.py:28）
2. is_buy=True: 买入 amount 数量的 base_token
3. is_buy=False: 卖出 amount 数量的 base_token
4. 我们需要根据 from_token 计算正确的 base_amount
"""

async def swap_via_jupiter(
    self,
    from_token: str,
    to_token: str,
    amount: Decimal,
    max_retries: int = 3
) -> bool:
    """
    通过 Jupiter 兑换代币

    Args:
        from_token: 源代币（要卖出的代币）
        to_token: 目标代币（要买入的代币）
        amount: 源代币数量（要卖出多少）

    Returns:
        True 如果兑换成功
    """
    trading_pair = self.config.trading_pair

    # 确定交易方向
    if from_token == self.base_token:
        # 卖出 base_token
        is_buy = False
        base_amount = amount  # amount 就是 base_token 数量
    else:
        # 卖出 quote_token（买入 base_token）
        is_buy = True
        base_amount = None  # 需要通过报价计算

    retry_delay = 1

    for attempt in range(max_retries):
        try:
            self.logger().info(f"Jupiter 兑换 (尝试 {attempt + 1}/{max_retries}):")
            self.logger().info(f"   卖出: {amount:.6f} {from_token}")
            self.logger().info(f"   买入: {to_token}")

            # 获取报价
            if base_amount is None:
                # 买入 base_token：需要先获取价格，计算能买多少
                temp_price = await self.swap_connector.get_quote_price(
                    trading_pair=trading_pair,
                    is_buy=is_buy,
                    amount=Decimal("1")  # 1个base的价格
                )

                if not temp_price or temp_price <= 0:
                    self.logger().warning(f"获取报价失败")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    return False

                # 计算能买多少 base_token
                base_amount = amount / Decimal(str(temp_price))
                quote_price = temp_price

                self.logger().info(f"   报价: 1 {self.base_token} = {quote_price:.10f} {self.quote_token}")
                self.logger().info(f"   预期买入: {base_amount:.6f} {to_token}")
            else:
                # 卖出 base_token：直接获取报价
                quote_price = await self.swap_connector.get_quote_price(
                    trading_pair=trading_pair,
                    is_buy=is_buy,
                    amount=base_amount
                )

                if not quote_price or quote_price <= 0:
                    self.logger().warning(f"获取报价失败")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    return False

                expected_quote = base_amount * Decimal(str(quote_price))

                self.logger().info(f"   报价: 1 {self.base_token} = {quote_price:.10f} {self.quote_token}")
                self.logger().info(f"   预期得到: {expected_quote:.6f} {to_token}")

            # 注入价格到 RateOracle
            try:
                rate_oracle = RateOracle.get_instance()
                rate_oracle.set_price(trading_pair, Decimal(str(quote_price)))
                self.logger().debug(f"注入价格: {trading_pair} = {quote_price}")
            except Exception as e:
                self.logger().debug(f"RateOracle 注入失败: {e}")

            # 执行兑换 - 关键：amount 必须是 base_token 数量！
            order_id = self.swap_connector.place_order(
                is_buy=is_buy,
                trading_pair=trading_pair,
                amount=base_amount,  # 重要！
                price=quote_price
            )

            self.logger().info(f"Jupiter 订单已提交: {order_id}")

            # 等待订单成交
            self.pending_swap_order_id = order_id
            self.swap_order_filled = False

            max_wait = 30
            elapsed = 0

            while elapsed < max_wait:
                await asyncio.sleep(1)
                elapsed += 1

                if self.swap_order_filled:
                    self.logger().info(f"✅ Jupiter 兑换成功")
                    self.pending_swap_order_id = None
                    return True

                if self.pending_swap_order_id is None:
                    self.logger().error(f"❌ Jupiter 兑换失败")
                    return False

            # 超时
            self.logger().error(f"❌ Jupiter 兑换超时（{max_wait}秒）")
            self.pending_swap_order_id = None
            return False

        except Exception as e:
            self.logger().error(f"Jupiter 兑换异常 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
                continue
            return False

    return False
