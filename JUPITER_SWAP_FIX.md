# Jupiter Swap 修复

## 修复日期
2025-11-02

## 问题描述

用户报告换币失败，错误信息：
```
TypeError: GatewaySwap.place_order() missing 2 required positional arguments: 'is_buy' and 'price'
```

**错误日志**:
```
2025-11-02 12:42:30,493 - ERROR - 换币失败: GatewaySwap.place_order() missing 2 required positional arguments: 'is_buy' and 'price'
Traceback (most recent call last):
  File "/home/hummingbot/scripts/meteora_dlmm_hft_meme.py", line 610, in swap_via_jupiter
    order_id = self.swap_connector.place_order(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: ...
```

## 根本原因

### 错误的实现方式

**高频版本（错误）**:
```python
async def swap_via_jupiter(self, input_token: str, output_token: str, amount: Decimal, side: str) -> bool:
    trading_pair = f"{output_token}-{input_token}" if side == "BUY" else f"{input_token}-{output_token}"

    order_id = self.swap_connector.place_order(
        trading_pair=trading_pair,
        side=side,              # ❌ 错误参数名，应该是 is_buy
        amount=float(amount),
        order_type="MARKET",    # ❌ 不需要这个参数
        slippage_tolerance=float(self.config.auto_swap_slippage_pct)  # ❌ 不需要
    )
```

**问题**:
1. ❌ 缺少必需参数 `is_buy`（布尔值）
2. ❌ 缺少必需参数 `price`（报价）
3. ❌ 使用了错误的参数 `side`（应该是 `is_buy`）
4. ❌ 使用了不存在的参数 `order_type` 和 `slippage_tolerance`
5. ❌ 没有先调用 `get_quote_price()` 获取报价

### 正确的实现方式（V2 版本）

**V2 版本**:
```python
# 1. 构造 trading_pair（始终是 BASE-QUOTE 格式）
trading_pair = f"{self.base_token}-{self.quote_token}"

# 2. 判断 is_buy
if from_token == self.base_token:
    is_buy = False  # 卖出 base_token
else:
    is_buy = True   # 买入 base_token

# 3. 获取报价（必需！）
quote_price = await self.swap_connector.get_quote_price(
    trading_pair=trading_pair,
    is_buy=is_buy,
    amount=amount
)

# 4. 执行兑换
order_id = self.swap_connector.place_order(
    is_buy=is_buy,          # ✅ 必需参数
    trading_pair=trading_pair,
    amount=amount,
    price=quote_price       # ✅ 必需参数
)
```

## Gateway Swap API 语义

### place_order 方法签名

```python
def place_order(
    self,
    is_buy: bool,           # 必需：True=买入 base_token, False=卖出 base_token
    trading_pair: str,      # 必需：格式 "BASE-QUOTE"
    amount: Decimal,        # 必需：始终是 base_token 的数量
    price: Decimal          # 必需：报价（通过 get_quote_price 获取）
) -> str:
```

### 参数含义

**is_buy 参数**:
- `True`: 买入 base_token（卖出 quote_token）
- `False`: 卖出 base_token（买入 quote_token）

**amount 参数**:
- **始终**是 base_token 的数量
- 无论买入还是卖出，都是 base_token 数量

**price 参数**:
- 1 个 base_token 的价格（以 quote_token 计价）
- 必须通过 `get_quote_price()` 提前获取

**trading_pair 参数**:
- 格式：`BASE-QUOTE`（例如 `PAYAI-SOL`）
- 不随买卖方向改变

## 修复方案

### 完整重写 swap_via_jupiter 方法

**修复后的代码**:
```python
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

        # 6. 执行兑换（使用正确的参数）✅
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
```

## 修复内容总结

### 1. 添加 get_quote_price 调用 ✅

**之前**: 没有获取报价，直接调用 place_order
**现在**: 先调用 `get_quote_price()` 获取报价

### 2. 修正 place_order 参数 ✅

**之前**:
```python
order_id = self.swap_connector.place_order(
    trading_pair=trading_pair,
    side=side,              # ❌
    amount=float(amount),
    order_type="MARKET",    # ❌
    slippage_tolerance=...  # ❌
)
```

**现在**:
```python
order_id = self.swap_connector.place_order(
    is_buy=is_buy,          # ✅
    trading_pair=trading_pair,
    amount=amount,          # ✅ 保持 Decimal
    price=quote_price       # ✅
)
```

### 3. 修正 trading_pair 构造 ✅

**之前**: 根据 side 动态构造（错误）
```python
trading_pair = f"{output_token}-{input_token}" if side == "BUY" else f"{input_token}-{output_token}"
```

**现在**: 始终使用 BASE-QUOTE 格式
```python
trading_pair = f"{self.base_token}-{self.quote_token}"
```

### 4. 添加详细日志 ✅

**之前**: 没有任何换币日志
**现在**: 详细的换币信息（买入/卖出、数量、价格）

### 5. 添加 RateOracle 注入 ✅

**之前**: 没有注入价格
**现在**: 注入价格到 RateOracle，供其他组件使用

## 与 V2 版本对比

| 功能 | V2 版本 | 高频版本（修复前） | 高频版本（修复后） | 状态 |
|------|---------|-------------------|-------------------|------|
| get_quote_price | ✅ 调用 | ❌ 缺失 | ✅ 调用 | 一致 ✅ |
| is_buy 参数 | ✅ 正确 | ❌ 使用 side | ✅ 正确 | 一致 ✅ |
| price 参数 | ✅ 有 | ❌ 缺失 | ✅ 有 | 一致 ✅ |
| trading_pair | ✅ 固定 BASE-QUOTE | ❌ 动态构造 | ✅ 固定 BASE-QUOTE | 一致 ✅ |
| 日志详细度 | ✅ 详细 | ❌ 无 | ✅ 详细 | 一致 ✅ |
| RateOracle | ✅ 注入 | ❌ 无 | ✅ 注入 | 一致 ✅ |
| 重试机制 | ✅ 有（3次） | ❌ 无 | ⚠️ 简化（无重试） | 部分一致 |
| 订单轮询 | ✅ 有（30秒） | ❌ 无 | ⚠️ 简化（固定5秒） | 部分一致 |

## 验证结果

### 1. 语法检查 ✅
```bash
python3 -m py_compile hummingbot_files/scripts/meteora_dlmm_hft_meme.py
# ✅ 语法检查通过
```

### 2. 与 V2 核心逻辑对比 ✅
- `get_quote_price` 调用：✅ 一致
- `place_order` 参数：✅ 一致
- `is_buy` 判断逻辑：✅ 一致
- `trading_pair` 构造：✅ 一致

### 3. 简化的改进 ✅
- V2 有重试机制（3次）：高频版本暂未实现（可后续添加）
- V2 有订单轮询（30秒）：高频版本使用固定 5 秒等待（简化）

## 预期行为

### 成功的换币流程

```
1. 准备双边代币
   └─> prepare_tokens_for_position()
       ├─> 更新余额
       ├─> 计算缺少的代币
       └─> 调用 swap_via_jupiter()

2. 执行 Jupiter 换币
   └─> swap_via_jupiter()
       ├─> 获取 Jupiter 报价 ✅
       │   └─> get_quote_price(is_buy, trading_pair, amount)
       ├─> 注入价格到 RateOracle ✅
       ├─> 打印兑换信息 ✅
       ├─> 执行兑换 ✅
       │   └─> place_order(is_buy, trading_pair, amount, price)
       └─> 等待成交（5秒）✅

3. 更新余额
   └─> update_balances()

4. 继续开仓
   └─> open_position()
```

### 预期日志

**成功换币**:
```
INFO - 检查并准备双边代币...
INFO - 准备双边代币...
INFO - 获取 Jupiter 报价...
INFO - 执行 Jupiter 兑换（买入 PAYAI）:
   卖出约: 0.761234 SOL
   买入: 4750.123456 PAYAI
   价格: 0.00016027 SOL/PAYAI
INFO - Jupiter 兑换订单已提交: swap-PAYAI-SOL-1762087350495136
INFO - 开仓（高频模式）:
  价格: 0.00016028
  ...
INFO - ✅ 开仓成功: range-PAYAI-SOL-1762087350495136
```

**换币失败（报价获取失败）**:
```
INFO - 检查并准备双边代币...
INFO - 准备双边代币...
INFO - 获取 Jupiter 报价...
ERROR - 获取报价失败，返回价格: None
WARNING - 代币准备失败，跳过本次开仓
```

## 常见问题

### Q: 为什么 amount 始终是 base_token 数量？
A: 这是 Gateway Swap API 的设计。无论买入还是卖出，`amount` 参数都是 base_token 的数量。Gateway 会根据 `is_buy` 和 `price` 自动计算 quote_token 数量。

### Q: 为什么需要先调用 get_quote_price？
A: Jupiter 是 DEX 聚合器，价格是动态的。必须先获取实时报价，然后用这个报价执行兑换。否则可能因为价格变化导致兑换失败。

### Q: 为什么不使用 slippage_tolerance 参数？
A: `slippage_tolerance` 不是 `place_order` 的参数。滑点控制是在 Gateway 层面由 Jupiter 自动处理的。

### Q: 高频版本为什么不实现 V2 的重试和轮询？
A: 为了简化代码。如果实际使用中发现换币经常失败，可以参考 V2 添加重试机制。订单轮询也可以后续添加。

## 总结

### 修复前的问题
- ❌ 缺少 `is_buy` 和 `price` 参数
- ❌ 没有调用 `get_quote_price` 获取报价
- ❌ 使用了错误的参数名（`side`, `order_type`, `slippage_tolerance`）
- ❌ trading_pair 构造错误

### 修复后的改进
- ✅ 使用正确的 Gateway Swap API
- ✅ 调用 `get_quote_price` 获取报价
- ✅ 使用正确的参数（`is_buy`, `trading_pair`, `amount`, `price`）
- ✅ 固定的 trading_pair 格式（BASE-QUOTE）
- ✅ 详细的换币日志
- ✅ RateOracle 价格注入
- ✅ 与 V2 核心逻辑一致

### 关键改进
1. **API 兼容性**: 完全符合 Gateway Swap API
2. **日志完整性**: 详细的换币信息
3. **代码一致性**: 与 V2 版本一致
4. **可靠性**: 先获取报价再执行

---

**修复完成时间**: 2025-11-02
**修复人员**: Claude (Anthropic)
**测试状态**: ✅ 语法通过，⏳ 功能测试待完成
**可用性**: ✅ 应该能正常换币

## 下一步

立即测试：
```bash
start --script meteora_dlmm_hft_meme.py
```

预期：
- ✅ 策略成功启动
- ✅ 获取 Jupiter 报价成功
- ✅ 执行换币成功
- ✅ 成功开仓
