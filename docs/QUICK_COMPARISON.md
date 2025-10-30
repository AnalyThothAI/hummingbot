# 🔍 V1 vs V2 新闻狙击策略 - 快速对比

## 核心代码对比

### 下单逻辑

#### V1 版本
```python
# 手动管理订单和价格计算
async def _execute_trade(self, side, base, quote, amount, slippage):
    # 1. 获取价格
    current_price = await self.connector.get_quote_price(...)

    # 2. 计算滑点价格
    if is_buy:
        final_price = current_price / ((Decimal("1") + slippage) * self.config.gas_buffer)
    else:
        final_price = current_price * ((Decimal("1") + slippage) * self.config.gas_buffer)

    # 3. 下单
    order_id = self.connector.place_order(
        is_buy=is_buy,
        trading_pair=trading_pair,
        amount=base_amount,
        price=final_price
    )

    # 4. 手动追踪订单
    self.pending_orders[order_id] = {
        "side": side,
        "amount": base_amount,
        "price": final_price,
        ...
    }

    # 5. 需要手动实现止盈止损 ❌
    # 6. 需要手动处理订单失败 ❌
    # 7. 需要手动重试 ❌
```

#### V2 版本
```python
# 声明式配置，Executor 自动处理一切
async def _process_signal(self, side, base, quote, amount, slippage):
    # 1. 获取价格
    mid_price = self.market_data_provider.get_price_by_type(...)

    # 2. 计算入场价格（考虑滑点）
    entry_price = mid_price * (1 + slippage) if is_buy else mid_price * (1 - slippage)

    # 3. 创建 Executor 配置
    executor_config = PositionExecutorConfig(
        side=trade_type,
        amount=base_amount,
        entry_price=entry_price,
        triple_barrier_config=TripleBarrierConfig(
            stop_loss=Decimal("0.10"),    # ✅ 自动止损
            take_profit=Decimal("0.05"),  # ✅ 自动止盈
            time_limit=300                # ✅ 自动超时平仓
        )
    )

    # 4. 提交执行动作
    action = CreateExecutorAction(executor_config=executor_config)
    self.executor_orchestrator.execute_actions([action])

    # ✅ Executor 自动管理订单生命周期
    # ✅ 自动止盈止损
    # ✅ 自动重试
    # ✅ 自动状态追踪
```

---

### 订单成交处理

#### V1 版本
```python
def did_fill_order(self, event: OrderFilledEvent):
    """手动处理订单成交"""
    order_id = event.order_id

    if order_id in self.pending_orders:
        order_info = self.pending_orders[order_id]

        # 手动记录统计
        self.stats["success"] += 1

        # 手动清理订单
        del self.pending_orders[order_id]

        # ❌ 需要手动实现止盈止损监控
        # ❌ 需要启动一个任务持续监控价格
```

#### V2 版本
```python
# 无需手动处理！Executor 自动管理
# 订单成交后，Executor 会：
# 1. ✅ 自动监控价格
# 2. ✅ 触及止损时自动平仓
# 3. ✅ 触及止盈时自动平仓
# 4. ✅ 超时后自动平仓
# 5. ✅ 记录完整的交易统计
```

---

### 订单失败处理

#### V1 版本
```python
def did_fail_order(self, event: MarketOrderFailureEvent):
    """手动处理订单失败"""
    order_id = event.order_id

    if order_id in self.pending_orders:
        # 手动实现重试逻辑
        if self.retry_count < MAX_RETRIES:
            self.retry_count += 1
            await asyncio.sleep(RETRY_DELAY)
            await self._execute_trade_with_retry(...)  # 递归重试
        else:
            self.logger().error("Max retries reached")
            self.stats["failed"] += 1

        del self.pending_orders[order_id]
```

#### V2 版本
```python
# 无需手动处理！Executor 内置重试机制
# ✅ 自动重试失败的订单
# ✅ 智能退避策略
# ✅ 自动错误恢复
# ✅ 详细的失败日志
```

---

### 止盈止损

#### V1 版本
```python
# 需要手动实现价格监控
async def _monitor_position(self, order_id):
    """手动监控止盈止损"""
    order_info = self.pending_orders[order_id]
    entry_price = order_info["price"]

    while order_id in self.pending_orders:
        current_price = await self.connector.get_quote_price(...)

        # 手动计算盈亏
        if order_info["side"] == "BUY":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        # 手动检查止盈止损
        if pnl_pct <= -0.10:  # 止损
            await self._close_position(order_id, "STOP_LOSS")
            break
        elif pnl_pct >= 0.05:  # 止盈
            await self._close_position(order_id, "TAKE_PROFIT")
            break

        await asyncio.sleep(1)  # 每秒检查一次
```

#### V2 版本
```python
# 配置即可，无需手动实现
triple_barrier_config=TripleBarrierConfig(
    stop_loss=Decimal("0.10"),    # 10% 自动止损
    take_profit=Decimal("0.05"),  # 5% 自动止盈
    time_limit=300,               # 5分钟自动平仓
    trailing_stop=TrailingStop(   # 移动止损
        activation_price=Decimal("0.03"),
        trailing_delta=Decimal("0.01")
    )
)

# ✅ Executor 自动监控
# ✅ 自动触发平仓
# ✅ 支持多种订单类型
# ✅ 智能滑点处理
```

---

### 状态管理

#### V1 版本
```python
# 手动维护所有状态
class Strategy:
    def __init__(self):
        self.pending_orders = {}      # 手动追踪
        self.filled_orders = []       # 手动记录
        self.stats = {                # 手动统计
            "signals": 0,
            "success": 0,
            "failed": 0,
            "retries": 0
        }

    def format_status(self):
        # 手动格式化状态
        return f"""
        Pending Orders: {len(self.pending_orders)}
        Success: {self.stats['success']}
        Failed: {self.stats['failed']}
        """
```

#### V2 版本
```python
# Executor 自动管理状态
class Strategy:
    def format_status(self):
        # 获取执行器状态（自动管理）
        active_executors = self.filter_executors(
            executors=self.get_all_executors(),
            filter_func=lambda e: e.is_active
        )

        for executor in active_executors:
            # ✅ 丰富的内置信息
            print(f"ID: {executor.id}")
            print(f"Status: {executor.status}")
            print(f"PnL: {executor.net_pnl_quote}")
            print(f"Fill Ratio: {executor.filled_amount_quote}")
            print(f"Entry Time: {executor.start_timestamp}")
            print(f"Close Reason: {executor.close_type}")
```

---

## 功能对比表

| 功能 | V1 实现 | V2 实现 | 说明 |
|------|---------|---------|------|
| **下单** | 手动调用 `place_order()` | 创建 `PositionExecutorConfig` | V2 声明式配置 |
| **订单追踪** | 手动维护 `pending_orders` 字典 | Executor 自动追踪 | V2 无需手动管理 |
| **止损** | 需自己实现价格监控 + 平仓逻辑 | `stop_loss` 参数 | V2 自动执行 |
| **止盈** | 需自己实现价格监控 + 平仓逻辑 | `take_profit` 参数 | V2 自动执行 |
| **时间限制** | 需自己实现定时器 | `time_limit` 参数 | V2 自动执行 |
| **移动止损** | 需自己实现复杂逻辑 | `trailing_stop` 参数 | V2 自动执行 |
| **订单重试** | 需手动实现重试逻辑 | Executor 内置重试 | V2 自动重试 |
| **错误处理** | 手动捕获和处理 | Executor 自动恢复 | V2 更健壮 |
| **状态统计** | 手动记录和计算 | Executor 自动统计 | V2 信息更丰富 |
| **信号去重** | 需手动实现（V1+Redis） | 需手动实现（V2+Redis） | 两者都需要 |
| **代码量** | ~600 行 | ~400 行 | V2 更简洁 |

---

## 性能对比

| 指标 | V1 | V2 | 说明 |
|------|----|----|------|
| **订单执行速度** | ⭐⭐⭐ | ⭐⭐⭐ | 相同 |
| **止盈止损响应** | ⭐⭐ | ⭐⭐⭐ | V2 更快，专门优化 |
| **内存占用** | 低 | 中 | V2 多一层抽象 |
| **CPU 占用** | 低 | 中 | V2 Executor 持续监控 |
| **错误恢复** | ⭐ | ⭐⭐⭐ | V2 内置重试 |
| **可维护性** | ⭐ | ⭐⭐⭐ | V2 代码更清晰 |

---

## 适用场景

### V1 适合：
- ✅ 简单的买卖逻辑
- ✅ 无需止盈止损
- ✅ 学习和实验
- ✅ 快速原型验证
- ✅ 对性能要求极致的场景

### V2 适合：
- ✅ 需要自动止盈止损
- ✅ 复杂的交易逻辑
- ✅ 生产环境使用
- ✅ 长期运行的策略
- ✅ 需要详细统计和监控

---

## 迁移建议

### 从 V1 迁移到 V2

#### 1. 订单管理
```python
# V1
self.pending_orders[order_id] = {...}

# V2
# 不需要手动管理，Executor 自动追踪
```

#### 2. 事件处理
```python
# V1
def did_fill_order(self, event):
    # 处理成交

def did_fail_order(self, event):
    # 处理失败

# V2
# 不需要这些方法，Executor 自动处理
```

#### 3. 止盈止损
```python
# V1
async def _monitor_position(self, order_id):
    while True:
        # 检查价格
        # 判断止盈止损
        # 执行平仓

# V2
triple_barrier_config=TripleBarrierConfig(...)
# Executor 自动处理
```

#### 4. 策略逻辑
```python
# V1
def on_tick(self):
    if should_trade():
        self._execute_trade(...)

# V2
def create_actions_proposal(self):
    if should_trade():
        return [CreateExecutorAction(...)]
    return []
```

---

## 实际例子对比

### 场景：收到 MQTT 信号，执行买入并设置 10% 止损、5% 止盈

#### V1 实现（简化版）
```python
async def _execute_trade_with_retry(self, side, base, quote, amount, slippage):
    # 1. 获取价格
    price = await self.connector.get_quote_price(...)

    # 2. 计算滑点价格
    final_price = price * (1 + slippage)

    # 3. 下单
    order_id = self.connector.place_order(...)

    # 4. 记录订单
    self.pending_orders[order_id] = {
        "entry_price": final_price,
        "amount": amount
    }

    # 5. 启动监控任务
    asyncio.create_task(self._monitor_position(order_id))

async def _monitor_position(self, order_id):
    # 持续监控价格
    while order_id in self.pending_orders:
        current_price = await self.connector.get_quote_price(...)
        entry_price = self.pending_orders[order_id]["entry_price"]

        # 计算盈亏
        pnl_pct = (current_price - entry_price) / entry_price

        # 检查止盈止损
        if pnl_pct <= -0.10:
            await self._close_position(order_id)
            break
        elif pnl_pct >= 0.05:
            await self._close_position(order_id)
            break

        await asyncio.sleep(1)

# 总计：需要实现 3-4 个方法，约 100 行代码
```

#### V2 实现
```python
async def _process_signal(self, side, base, quote, amount, slippage):
    # 1. 获取价格
    mid_price = self.market_data_provider.get_price_by_type(...)

    # 2. 计算入场价格
    entry_price = mid_price * (1 + slippage)

    # 3. 创建 Executor
    executor_config = PositionExecutorConfig(
        side=TradeType.BUY,
        amount=amount,
        entry_price=entry_price,
        triple_barrier_config=TripleBarrierConfig(
            stop_loss=Decimal("0.10"),    # 10% 止损
            take_profit=Decimal("0.05")   # 5% 止盈
        )
    )

    # 4. 提交执行
    action = CreateExecutorAction(executor_config=executor_config)
    self.executor_orchestrator.execute_actions([action])

# 总计：1 个方法，约 20 行代码
# ✅ Executor 自动处理监控、止盈止损、平仓
```

---

## 最终建议

### 新闻狙击策略

**推荐使用 V2** ✅

**理由：**
1. ✅ **自动止盈止损** - 新闻交易波动大，必须有风控
2. ✅ **时间限制** - 新闻热度有限，5分钟后自动平仓
3. ✅ **自动重试** - 网络问题时自动恢复
4. ✅ **代码更少** - 减少出错概率
5. ✅ **易于维护** - 未来扩展更方便

**V1 仅适用于：**
- ❌ 你不需要止盈止损
- ❌ 你只想快速测试想法
- ❌ 你对性能有极致要求

---

## 快速决策

```
需要止盈止损？
├─ 是 → 使用 V2 ✅
└─ 否 →
    需要自动重试？
    ├─ 是 → 使用 V2 ✅
    └─ 否 →
        需要详细统计？
        ├─ 是 → 使用 V2 ✅
        └─ 否 → 使用 V1
```

---

**结论：对于新闻狙击策略，V2 是更好的选择！** 🚀
