# 🏗️ Hummingbot 策略架构指南

## 📖 目录
- [V1 vs V2 架构对比](#v1-vs-v2-架构对比)
- [Executor 详解](#executor-详解)
- [Controller 详解](#controller-详解)
- [选择建议](#选择建议)
- [最佳实践](#最佳实践)

---

## V1 vs V2 架构对比

### V1 架构 (ScriptStrategyBase)

**核心理念：** 事件驱动 + 手动管理

```python
class MyStrategy(ScriptStrategyBase):
    def __init__(self, connectors, config):
        super().__init__(connectors)
        self.pending_orders = {}  # 手动追踪订单

    def on_tick(self):
        # 每个 tick 检查条件
        if should_trade():
            order_id = self.connector.place_order(...)
            self.pending_orders[order_id] = {...}

    def did_fill_order(self, event):
        # 订单成交，手动处理
        order_id = event.order_id
        if order_id in self.pending_orders:
            # 手动实现止盈止损逻辑
            self._check_stop_loss_take_profit(order_id)

    def did_fail_order(self, event):
        # 订单失败，手动重试
        if should_retry():
            self._retry_order(...)
```

**优点：**
- ✅ 简单直接，易于理解
- ✅ 完全控制每个细节
- ✅ 适合快速原型开发
- ✅ 无额外抽象层，性能开销小

**缺点：**
- ❌ 需要手动管理订单生命周期
- ❌ 止盈止损需要自己实现
- ❌ 重试机制需要手动编写
- ❌ 代码量大，容易出错
- ❌ 难以复用逻辑

**适用场景：**
- 简单的买卖策略
- 快速测试想法
- 学习和实验
- 无需复杂风控的场景

---

### V2 架构 (StrategyV2Base + Executor)

**核心理念：** 声明式 + 自动化管理

```python
class MyStrategy(StrategyV2Base):
    def __init__(self, connectors, config):
        super().__init__(connectors, config)
        # Executor 自动追踪订单

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        # 声明式：告诉系统"我想做什么"
        if should_trade():
            return [CreateExecutorAction(
                executor_config=PositionExecutorConfig(
                    side=TradeType.BUY,
                    amount=amount,
                    entry_price=price,
                    triple_barrier_config=TripleBarrierConfig(
                        stop_loss=Decimal("0.10"),
                        take_profit=Decimal("0.05"),
                        time_limit=300
                    )
                )
            )]
        return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        # 条件触发时停止执行器
        return []

    # 订单管理、止盈止损、重试全部由 Executor 自动处理！
```

**优点：**
- ✅ Executor 自动管理订单生命周期
- ✅ 内置止盈止损（Triple Barrier）
- ✅ 自动重试和错误恢复
- ✅ 统一的订单状态追踪
- ✅ 代码更简洁，更少出错
- ✅ 易于组合多个策略（Controller 模式）

**缺点：**
- ❌ 学习曲线稍高
- ❌ 抽象层增加了复杂度
- ❌ 需要理解 Executor 工作原理

**适用场景：**
- 需要止盈止损的策略
- 复杂的多步骤交易
- 生产环境使用
- 需要风险管理的场景
- 多策略组合

---

## Executor 详解

### 什么是 Executor？

**Executor** 是 V2 架构的核心组件，负责执行一个完整的交易任务。

**类比理解：**
- **V1**: 你是出租车司机，手动换挡、刹车、加速
- **V2**: 你是乘客，告诉自动驾驶系统目的地，它自动处理一切

### Executor 类型

Hummingbot 提供多种内置 Executor：

| Executor | 用途 | 特点 |
|----------|------|------|
| **PositionExecutor** | 方向性交易 | 止盈止损、时间限制、移动止损 |
| **ArbitrageExecutor** | 套利交易 | 同时在两个市场买卖 |
| **DCAExecutor** | 定投策略 | 分批建仓 |
| **GridExecutor** | 网格交易 | 在价格区间内高抛低吸 |
| **TWAPExecutor** | TWAP 执行 | 时间加权平均价格执行 |
| **OrderExecutor** | 简单订单 | 单个订单执行 |

### PositionExecutor 工作流程

```
1. 创建 Executor
   ↓
2. 下入场订单（Market/Limit）
   ↓
3. 等待订单成交
   ↓
4. 监控价格和时间
   ├─ 价格 ≤ 入场价 * (1 - stop_loss) → 止损
   ├─ 价格 ≥ 入场价 * (1 + take_profit) → 止盈
   ├─ 时间 ≥ time_limit → 超时平仓
   └─ 移动止损触发 → 平仓
   ↓
5. 下平仓订单
   ↓
6. 等待平仓成交
   ↓
7. 标记 Executor 完成
   ↓
8. 记录盈亏和统计
```

### Triple Barrier 配置详解

```python
TripleBarrierConfig(
    # 止损：价格跌破入场价 10% 时平仓
    stop_loss=Decimal("0.10"),

    # 止盈：价格涨过入场价 5% 时平仓
    take_profit=Decimal("0.05"),

    # 时间限制：300秒（5分钟）后自动平仓
    time_limit=300,

    # 移动止损（可选）
    trailing_stop=TrailingStop(
        activation_price=Decimal("0.03"),  # 盈利3%后启动
        trailing_delta=Decimal("0.01")     # 从最高点回落1%时平仓
    ),

    # 订单类型
    open_order_type=OrderType.MARKET,         # 入场用市价单
    take_profit_order_type=OrderType.LIMIT,   # 止盈用限价单
    stop_loss_order_type=OrderType.MARKET,    # 止损用市价单
    time_limit_order_type=OrderType.MARKET    # 超时用市价单
)
```

### Executor 状态追踪

```python
# 获取所有执行器
all_executors = self.get_all_executors()

# 过滤活跃的执行器
active_executors = self.filter_executors(
    executors=all_executors,
    filter_func=lambda e: e.is_active
)

# 按交易对过滤
pair_executors = self.filter_executors(
    executors=all_executors,
    filter_func=lambda e: e.trading_pair == "TOKEN-WBNB"
)

# 检查执行器状态
for executor in active_executors:
    print(f"Trading Pair: {executor.trading_pair}")
    print(f"Side: {executor.side}")
    print(f"Status: {executor.status}")
    print(f"PnL: {executor.net_pnl_quote}")
```

---

## Controller 详解

### 什么是 Controller？

**Controller** 是可重用的策略模块，专注于特定的交易逻辑。

**类比理解：**
- **Script**: 完整的应用程序（包含所有功能）
- **Controller**: 可重用的库/组件（专注单一职责）
- **StrategyV2Base**: 应用容器（可以组合多个 Controller）

### Controller vs Script

| 维度 | Script | Controller |
|------|--------|-----------|
| **定义** | 完整的策略 | 可重用的策略模块 |
| **文件位置** | `scripts/` | `controllers/` |
| **配置** | 直接编码或单个配置文件 | YAML 配置文件 |
| **可重用性** | 低 | 高 |
| **组合能力** | 无 | 可以组合多个 Controller |
| **适用场景** | 独立策略 | 策略组件、多策略系统 |

### Controller 架构示例

```python
# controllers/generic/mqtt_news_sniping_controller.py
class MQTTNewsSnipingController(ControllerBase):
    def __init__(self, config, market_data_provider, actions_queue):
        super().__init__(config, market_data_provider, actions_queue)
        self._setup_mqtt()

    def determine_executor_actions(self) -> List[ExecutorAction]:
        # 根据 MQTT 信号决定创建哪些 Executor
        if has_signal():
            return [CreateExecutorAction(...)]
        return []

# 配置文件: mqtt_news_bsc.yml
controller_type: generic
controller_name: mqtt_news_sniping_controller
connector_name: pancakeswap
trading_pair: WBNB-USDT
mqtt_broker: localhost
mqtt_topic: trading/bsc/snipe

# 策略脚本
class MultiSourceNewsSniping(StrategyV2Base):
    # 组合多个 Controller
    controllers_config = [
        "mqtt_news_bsc.yml",      # BSC 新闻
        "mqtt_news_eth.yml",      # ETH 新闻
        "twitter_monitor.yml",    # Twitter 监控
    ]
```

### 何时使用 Controller？

**使用 Controller 的场景：**
- ✅ 需要在多个策略中重用逻辑
- ✅ 需要组合多种策略（如套利 + 做市）
- ✅ 需要统一的风险管理和资金分配
- ✅ 团队协作，不同人开发不同模块
- ✅ 生产环境，需要灵活配置

**使用 Script 的场景：**
- ✅ 单一策略，不需要复用
- ✅ 快速原型和测试
- ✅ 学习和实验
- ✅ 策略逻辑紧密耦合，不适合拆分

---

## 选择建议

### 决策树

```
开始
 ↓
需要止盈止损？
├─ 否 → V1 Script ✅
└─ 是 ↓
    需要组合多个策略？
    ├─ 否 → V2 Script + Executor ✅
    └─ 是 ↓
        策略逻辑可重用？
        ├─ 否 → V2 Script + Executor
        └─ 是 → V2 + Controller ✅
```

### 具体建议

#### 新闻狙击策略（你的场景）

**推荐：V2 Script + Executor**

理由：
- ✅ 需要自动止盈止损（Triple Barrier）
- ✅ 单一信号源（MQTT）
- ✅ 不需要组合其他策略
- ❌ 暂时不需要 Controller 的复杂性

**未来扩展（V2 + Controller）：**
- 当你需要多个信号源（Twitter, Telegram, Discord）
- 当你需要同时在多条链上运行（BSC, ETH, Polygon）
- 当你需要与其他策略组合（套利 + 新闻狙击）

#### 其他场景建议

| 场景 | 推荐架构 | 理由 |
|------|---------|------|
| 简单 TWAP | V1 Script | 无需止盈止损，逻辑简单 |
| 方向性交易策略 | V2 + Executor | 需要 Triple Barrier |
| 跨交易所套利 | V2 + Executor (ArbitrageExecutor) | 需要同时管理两个订单 |
| 多策略组合 | V2 + Controller | 需要统一风控和资金管理 |
| 做市 + 套利 | V2 + Controller | 组合两种策略 |
| 网格交易 | V2 + Executor (GridExecutor) | 内置网格逻辑 |

---

## 最佳实践

### 1. 策略开发流程

```
阶段 1: 原型开发
- 使用 V1 Script 快速验证想法
- 手动实现核心逻辑
- 小额测试
↓
阶段 2: 功能完善
- 迁移到 V2 + Executor
- 添加止盈止损
- 完善错误处理
↓
阶段 3: 生产部署
- 考虑是否需要 Controller
- 添加监控和告警
- 风险管理优化
↓
阶段 4: 扩展和组合
- 重构为 Controller（如果需要复用）
- 组合多个策略
- 统一风控系统
```

### 2. 代码组织

```
scripts/
├── v1_simple_strategies/       # V1 简单策略
│   ├── simple_buy_sell.py
│   └── basic_arbitrage.py
├── v2_strategies/              # V2 策略
│   ├── v2_news_sniping_strategy.py
│   ├── v2_directional_rsi.py
│   └── v2_with_controllers.py
└── utility/                    # 工具脚本
    └── test_news_signal_sender.py

controllers/
├── generic/                    # 通用 Controller
│   ├── arbitrage_controller.py
│   └── mqtt_news_controller.py
└── market_making/              # 做市 Controller
    └── pmm_controller.py

conf/
├── scripts/                    # Script 配置
│   └── v2_news_sniping.yml
└── controllers/                # Controller 配置
    ├── mqtt_news_bsc.yml
    └── mqtt_news_eth.yml
```

### 3. 测试策略

```python
# 1. 单元测试（V1 和 V2 通用）
def test_normalize_token():
    strategy = NewsSnipingV2(...)
    assert strategy._normalize_token_symbol("BNB") == "WBNB"

# 2. 集成测试（V2 更容易）
async def test_position_executor():
    strategy = NewsSnipingV2(...)
    actions = await strategy._process_signal("BUY", "TOKEN", "WBNB", Decimal("0.001"))
    assert len(strategy.get_all_executors()) == 1

# 3. 回测（V2 有更好支持）
# Hummingbot 的回测框架对 V2 支持更好
```

### 4. 监控和调试

```python
# V1: 手动记录状态
def format_status(self):
    return f"Orders: {len(self.pending_orders)}"

# V2: 丰富的内置状态
def format_status(self):
    active = self.filter_executors(
        executors=self.get_all_executors(),
        filter_func=lambda e: e.is_active
    )

    for executor in active:
        print(f"Executor: {executor.id}")
        print(f"PnL: {executor.net_pnl_quote}")
        print(f"Status: {executor.status}")
        print(f"Fill Ratio: {executor.filled_amount_quote / executor.config.amount}")
```

### 5. 错误处理

```python
# V1: 手动处理每个错误
def did_fail_order(self, event):
    if self.retry_count < MAX_RETRIES:
        self._retry_order(event.order_id)
    else:
        self.logger().error("Max retries reached")

# V2: Executor 自动重试
# 只需配置重试策略，Executor 自动处理
```

### 6. 性能优化

```python
# V1: 需要优化 on_tick 逻辑
def on_tick(self):
    # 避免频繁检查
    if self.current_timestamp - self.last_check < CHECK_INTERVAL:
        return
    # ...

# V2: create_actions_proposal 自动控制频率
# 框架会在合适的时机调用，无需手动优化
```

---

## 总结

### 快速决策表

| 需求 | V1 | V2 + Executor | V2 + Controller |
|------|----|--------------|----|
| 学习 Hummingbot | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| 快速原型 | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| 止盈止损 | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 复杂交易逻辑 | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 多策略组合 | ⭐ | ⭐⭐ | ⭐⭐⭐ |
| 代码可维护性 | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 生产环境使用 | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

### 新闻狙击策略总结

**你的场景：**
- MQTT 信号驱动
- 需要止盈止损
- 信号去重
- 单一策略

**最佳选择：V2 + Executor ✅**

已实现的功能：
- ✅ PositionExecutor 自动管理订单
- ✅ Triple Barrier 止盈止损
- ✅ Redis 信号去重
- ✅ 完整的状态监控
- ✅ 灵活的配置系统

**未来可升级到 Controller（当需要时）：**
- 多个信号源
- 多条链支持
- 与其他策略组合
- 团队协作开发

---

**希望这份指南能帮助你理解 Hummingbot 的架构选择！** 🚀
