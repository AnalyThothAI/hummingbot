# Meteora DLMM Smart LP V2.0.5 - 关键修复

## 修复日期
2025-11-02

## 核心问题

### 问题 1: Swap amount 参数理解错误 ❌
**现象:**
- 想用 0.353 SOL 买 PAYAI
- 实际只买到 0.353 PAYAI（应该买到 2000+ PAYAI）

**根本原因:**
Gateway `place_order` 的 `amount` 参数是 **base_token 的数量**，不是 from_token！

参考 `amm_trade_example.py:28`:
```python
amount: Decimal = Field(Decimal("0.01"), json_schema_extra={
    "prompt": "Order amount (in base token)", "prompt_on_new": True})
```

**错误实现:**
```python
# 错误：直接传 from_token 数量
order_id = self.swap_connector.place_order(
    is_buy=True,
    amount=0.353,  # 这是 SOL 数量（quote_token）
    ...
)
# 结果：Gateway 理解为买 0.353 PAYAI（base_token）
```

**正确实现:**
```python
# 1. 先获取价格
price = await self.swap_connector.get_quote_price(..., amount=Decimal("1"))
# price = 0.000159 (1 PAYAI = 0.000159 SOL)

# 2. 计算能买多少 base_token
base_amount = 0.353 / 0.000159 = 2221 PAYAI

# 3. 传入 base_token 数量
order_id = self.swap_connector.place_order(
    is_buy=True,
    amount=2221,  # ✅ 正确：base_token 数量
    ...
)
```

---

### 问题 2: 重复调用 prepare_tokens ❌

**现象:**
- `initialize_strategy` 调用了 `prepare_tokens`
- `check_and_open_positions` 又调用了 `prepare_tokens`
- 导致重复 swap 和时序混乱

**修复方案:**
```python
async def initialize_strategy(self):
    await self.fetch_pool_info()
    await self.check_existing_positions()

    # 如果没有仓位，直接在这里准备代币并开仓
    if not self.position_opened:
        if self.config.enable_auto_swap:
            success = await self.prepare_tokens_for_multi_layer_position()
            if not success:
                self.logger().error("代币准备失败，策略初始化中止")
                return

        # ✅ 直接开仓，不走 on_tick
        current_price = Decimal(str(self.pool_info.price))
        await self.open_multi_layer_positions(current_price)

    self.logger().info("策略初始化完成")

async def check_and_open_positions(self):
    # ✅ 移除这里的 prepare_tokens 调用
    # 因为初始化时已经处理了
    if self.position_opened or self.position_opening:
        return

    await self.fetch_pool_info()
    current_price = Decimal(str(self.pool_info.price))

    # 直接开仓（假设代币已准备好）
    await self.open_multi_layer_positions(current_price)
```

---

### 问题 3: 开仓时序混乱 ❌

**现象:**
- Swap 在 55秒提交
- Layer 2 在 56秒就开始了（swap还没完成！）
- 导致 Layer 1, 3 失败

**原因:**
并发问题 + 余额计算时机不对

**修复:**
1. 确保 swap 完全成功后再开仓
2. 序列化开仓（不需要并发）
3. 每层开仓后等待确认

---

## 完整修复代码

### 修复 1: swap_via_jupiter 方法

**位置:** `meteora_dlmm_smart_lp_v2.py` 第 563-690 行

**替换为:**
```python
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

    # 确定交易方向和计算 base_token 数量
    if from_token == self.base_token:
        is_buy = False
        base_amount = amount  # 卖出 base，amount 就是 base数量
    else:
        is_buy = True
        base_amount = None  # 买入 base，需要计算

    retry_delay = 1

    for attempt in range(max_retries):
        try:
            self.logger().info(
                f"Jupiter 兑换 (尝试 {attempt + 1}/{max_retries}):\n"
                f"   卖出: {amount:.6f} {from_token}\n"
                f"   买入: {to_token}"
            )

            # 获取报价
            if base_amount is None:
                # 买入 base：先获取1个base的价格
                temp_price = await self.swap_connector.get_quote_price(
                    trading_pair=trading_pair,
                    is_buy=is_buy,
                    amount=Decimal("1")
                )

                if not temp_price or temp_price <= 0:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    return False

                # 计算能买多少 base
                base_amount = amount / Decimal(str(temp_price))
                quote_price = temp_price

                self.logger().info(
                    f"   报价: 1 {self.base_token} = {quote_price:.10f} {self.quote_token}\n"
                    f"   预期买入: {base_amount:.6f} {to_token}"
                )
            else:
                # 卖出 base：直接获取报价
                quote_price = await self.swap_connector.get_quote_price(
                    trading_pair=trading_pair,
                    is_buy=is_buy,
                    amount=base_amount
                )

                if not quote_price or quote_price <= 0:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    return False

                expected_quote = base_amount * Decimal(str(quote_price))
                self.logger().info(
                    f"   报价: 1 {self.base_token} = {quote_price:.10f} {self.quote_token}\n"
                    f"   预期得到: {expected_quote:.6f} {to_token}"
                )

            # 注入价格
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
                amount=base_amount,  # ✅ 重要！
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
            self.logger().error(f"Jupiter 兑换异常: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
                continue
            return False

    return False
```

---

### 修复 2: initialize_strategy 方法

**位置:** `meteora_dlmm_smart_lp_v2.py` 第 343-364 行

**替换为:**
```python
async def initialize_strategy(self):
    """策略初始化"""
    await asyncio.sleep(3)  # 等待连接器初始化

    try:
        # 1. 获取池子信息
        await self.fetch_pool_info()

        # 2. 检查现有仓位
        await self.check_existing_positions()

        # 3. 如果没有仓位，准备代币并开仓
        if not self.position_opened:
            # 准备代币
            if self.config.enable_auto_swap:
                success = await self.prepare_tokens_for_multi_layer_position()
                if not success:
                    self.logger().error("代币准备失败，策略初始化中止")
                    return

            # ✅ 直接开仓
            current_price = Decimal(str(self.pool_info.price))
            await self.open_multi_layer_positions(current_price)

        self.logger().info("策略初始化完成")

    except Exception as e:
        self.logger().error(f"策略初始化失败: {e}")
```

---

### 修复 3: check_and_open_positions 方法

**位置:** `meteora_dlmm_smart_lp_v2.py` 第 888-928 行

**替换为:**
```python
async def check_and_open_positions(self):
    """检查并开仓（仅在初始化失败后重试）"""
    if self.position_opened or self.position_opening:
        return

    try:
        await self.fetch_pool_info()
        if not self.pool_info:
            self.logger().warning("无法获取池子信息")
            return

        current_price = Decimal(str(self.pool_info.price))

        # ✅ 移除重复的 prepare_tokens 调用
        # 假设代币已在 initialize 时准备好
        # 如果仍需准备，说明初始化失败，重新准备
        if self.config.enable_auto_swap:
            # 快速检查余额
            base_balance = self.connector.get_available_balance(self.base_token)
            quote_balance = self.connector.get_available_balance(self.quote_token)

            if base_balance == 0 or quote_balance == 0:
                success = await self.prepare_tokens_for_multi_layer_position()
                if not success:
                    self.logger().error("代币准备失败，等待下次重试")
                    return

        # 开仓
        await self.open_multi_layer_positions(current_price)

    except Exception as e:
        self.logger().error(f"检查开仓失败: {e}")
```

---

## 预期效果

修复后的行为：

1. **Swap 正确:**
   - 用 0.353 SOL 换 PAYAI
   - 成交 ~2000 PAYAI（而不是 0.353）

2. **时序正确:**
   - Swap 完成 → 等待余额更新 → 开仓Layer 1 → Layer 2 → Layer 3

3. **无重复:**
   - 只在 `initialize_strategy` 中准备代币和开仓
   - `on_tick` 只用于监控和再平衡

---

## 测试建议

1. 清空现有仓位
2. 钱包准备：至少 1 SOL（无需PAYAI）
3. 配置：
```yaml
num_layers: 1  # 先测试1层
wallet_allocation_pct: 60.0
enable_auto_swap: true
```

4. 观察日志应该看到：
```
Jupiter 兑换:
   卖出: 0.3XX SOL
   买入: PAYAI
   报价: 1 PAYAI = 0.000159 SOL
   预期买入: 2XXX PAYAI
✅ Jupiter 兑换成功
等待余额更新...
兑换后 PAYAI 余额: 2XXX  ← 应该是2000+，不是0.3
开仓 Layer_1:
   PAYAI: XXX
   SOL: XXX
✅ Layer_1 订单已提交
✅ 策略初始化完成
```

---

## 版本历史

### V2.0.5 (2025-11-02) - 修复 Swap 数量错误
- ✅ **关键修复**: `place_order` amount 参数必须是 base_token 数量
- ✅ 买入时先计算能买多少 base_token
- ✅ 卖出时直接使用 base_token 数量
- ✅ 修复重复调用 `prepare_tokens`
- ✅ 简化初始化流程：init 中直接开仓
- ✅ 改进日志：清晰显示报价和预期数量
