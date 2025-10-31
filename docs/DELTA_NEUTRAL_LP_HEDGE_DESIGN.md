# Delta 中性 LP 对冲策略 - 设计文档

## 📋 目录

- [策略概述](#策略概述)
- [核心原理](#核心原理)
- [策略流程](#策略流程)
- [风险对冲计算](#风险对冲计算)
- [再平衡逻辑](#再平衡逻辑)
- [风险控制](#风险控制)
- [技术架构](#技术架构)
- [配置参数](#配置参数)
- [实现细节](#实现细节)

---

## 策略概述

### 什么是 Delta 中性 LP 对冲？

在 DEX 上提供流动性（LP）可以赚取交易手续费，但面临**无常损失**风险。通过在 CEX 上开对冲仓位，可以：

✅ **消除价格风险**：LP 持仓价值变化 = CEX 对冲收益
✅ **保留手续费收入**：继续赚取 LP 手续费
✅ **Delta 中性**：总仓位对价格变化不敏感

### 策略目标

- **主要收益**：DEX LP 手续费
- **风险控制**：通过 CEX 合约对冲，消除价格波动风险
- **资金效率**：使用杠杆合约，提高资金利用率

### 适用场景

- 高波动币种（如 CAKE、UNI、DYDX）
- 高交易量池子（手续费收入高）
- 有 CEX 合约市场的币种
- 长期持有策略（赚取持续手续费）

---

## 核心原理

### 1. LP 持仓动态平衡

以你的例子：
```
总资金：5000 USDT
LP 持仓：
  - 4000 USDT 价值的 CAKE
  - 1000 USDT

假设 CAKE 价格 = 2 USDT
  - CAKE 数量 = 4000 / 2 = 2000 CAKE
```

### 2. 风险敞口计算

**风险敞口**：如果 CAKE 价格上涨/下跌，你的资产价值会变化

```
风险敞口 = CAKE 持仓价值 = 4000 USDT
```

如果 CAKE 涨 10%：
- LP 中的 CAKE 会减少（被交易者买走）
- LP 中的 USDT 会增加
- 但总价值变化 ≈ 资产价值变化 - 无常损失

### 3. CEX 对冲

为了对冲这个风险：

```
CEX 空单 = LP 中 CAKE 价值 × 对冲比例
         = 4000 × 95%
         = 3800 USDT 名义价值

使用 10x 杠杆，实际占用保证金 = 3800 / 10 = 380 USDT
```

**为什么是 95% 而不是 100%？**

1. **缓冲空间**：避免过度对冲
2. **手续费成本**：减少再平衡频率
3. **无常损失特性**：LP 的 Delta 不是线性的

### 4. Delta 中性效果

| 场景 | LP 变化 | CEX 对冲收益 | 总收益 |
|------|---------|-------------|--------|
| CAKE +10% | -200 USDT (无常损失) | +380 USDT (空单盈利) | +180 USDT |
| CAKE -10% | -200 USDT (无常损失) | -380 USDT (空单亏损) | -580 USDT |

> 注：实际情况更复杂，需要动态再平衡

---

## 策略流程

### 整体流程图

```
┌─────────────────────────────────────────────────────────┐
│                      启动策略                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  1. 获取 LP 持仓信息   │
         │     - 池子地址         │
         │     - Token 数量       │
         │     - 当前价值         │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  2. 计算风险敞口       │
         │     Base Delta        │
         │     Quote Delta       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  3. 检查 CEX 持仓      │
         │     当前合约仓位       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  4. 计算对冲缺口       │
         │     目标仓位           │
         │     实际仓位           │
         │     调整量             │
         └───────────┬───────────┘
                     │
                     ▼
              ┌─────┴─────┐
              │ 缺口超过    │
              │ 阈值？      │
              └─────┬─────┘
                    │
          ┌─────────┼─────────┐
          │ 是               │ 否
          ▼                  ▼
┌──────────────────┐   ┌──────────────┐
│ 5. 调整 CEX 仓位  │   │ 6. 等待下一轮 │
│    开仓/加仓      │   │    检查周期   │
│    减仓/平仓      │   └──────┬───────┘
└─────────┬────────┘          │
          │                   │
          └─────────┬─────────┘
                    │
                    ▼
          ┌──────────────────┐
          │ 7. 记录统计数据   │
          │    LP 手续费      │
          │    对冲成本       │
          │    净收益         │
          └─────────┬────────┘
                    │
                    └─────► 每 50 秒循环
```

---

## 风险对冲计算

### 1. LP 持仓分析

#### Uniswap V3 / CLMM 类型

```python
# LP 仓位信息
position = {
    "lower_price": 1.8,      # 价格下限
    "upper_price": 2.2,      # 价格上限
    "base_amount": 2000,     # CAKE 数量
    "quote_amount": 1000,    # USDT 数量
    "current_price": 2.0     # 当前价格
}

# 计算持仓价值
base_value = base_amount * current_price  # 2000 * 2 = 4000 USDT
quote_value = quote_amount                # 1000 USDT
total_value = base_value + quote_value    # 5000 USDT
```

#### AMM 类型（Uniswap V2）

```python
# AMM LP
# x * y = k (恒定乘积)
# 当前价格 P = y / x = quote_amount / base_amount

base_amount = sqrt(k / P)
quote_amount = sqrt(k * P)

# 价值
base_value = base_amount * P
quote_value = quote_amount
total_value = base_value + quote_value
```

### 2. Delta 计算

**Delta** = 持仓对价格的敏感度

#### 简化计算（推荐用于实际实现）

```python
# 方法 1: 直接使用 Base Token 价值
delta = base_amount * current_price * hedge_ratio

# 例子
delta = 2000 * 2.0 * 0.95 = 3800 USDT
```

#### 精确计算（CLMM）

对于 Uniswap V3 concentrated liquidity：

```python
def calculate_clmm_delta(position, current_price):
    """
    计算 CLMM LP 的 Delta

    在价格区间内，Delta 会随价格变化：
    - 价格上涨 → CAKE 减少 → Delta 降低
    - 价格下跌 → CAKE 增加 → Delta 升高
    """
    lower = position.lower_price
    upper = position.upper_price
    P = current_price

    # 在区间内
    if lower <= P <= upper:
        # 计算当前流动性 L
        L = position.liquidity

        # Base token 数量（Delta 的主要来源）
        base_amount = L * (1/sqrt(P) - 1/sqrt(upper))

        # Delta（USDT 价值）
        delta = base_amount * P

    # 价格在区间下方（全是 Base）
    elif P < lower:
        delta = position.base_amount * P

    # 价格在区间上方（全是 Quote，无 Base）
    else:
        delta = 0

    return delta
```

### 3. 对冲比例选择

不同场景下的推荐对冲比例：

| 场景 | 对冲比例 | 理由 |
|------|---------|------|
| **低波动**（稳定币对） | 90-95% | 价格稳定，无需频繁调整 |
| **中等波动**（主流币） | 95-100% | 平衡保护和成本 |
| **高波动**（山寨币） | 100-105% | 需要更强保护 |
| **CLMM 窄区间** | 95-100% | 价格移动快速改变 Delta |
| **AMM 全区间** | 100% | Delta 变化较慢 |

---

## 再平衡逻辑

### 1. 何时触发再平衡？

#### 方法 A：固定时间间隔（推荐）

```python
# 每 50 秒检查一次
check_interval = 50  # 秒

# 优点：
# - 简单可靠
# - 成本可预测
# - 适合大多数场景

# 缺点：
# - 可能在不需要时也调整
```

#### 方法 B：Delta 偏差触发

```python
# 当对冲偏差超过阈值时触发
deviation_threshold = 0.05  # 5%

current_delta = get_lp_delta()
current_hedge = get_cex_position()
target_hedge = current_delta * hedge_ratio

deviation = abs(current_hedge - target_hedge) / target_hedge

if deviation > deviation_threshold:
    rebalance()
```

#### 方法 C：混合模式（最优）

```python
# 定时检查 + 偏差触发
check_interval = 50  # 秒
min_deviation = 0.03  # 3% 最小调整阈值

# 每 50 秒检查
# 只有偏差 > 3% 时才调整
```

### 2. 再平衡计算

```python
def calculate_rebalance():
    """
    计算需要调整的仓位
    """
    # 1. 获取当前状态
    lp_info = get_lp_position()
    base_amount = lp_info.base_amount
    current_price = lp_info.current_price

    # 2. 计算目标对冲
    target_delta = base_amount * current_price
    target_hedge = target_delta * hedge_ratio  # 95%

    # 3. 获取当前 CEX 仓位
    cex_position = get_cex_position()
    current_hedge = abs(cex_position.notional_value)

    # 4. 计算调整量
    adjustment = target_hedge - current_hedge

    # 5. 检查是否需要调整
    deviation_pct = abs(adjustment) / target_hedge

    if deviation_pct < min_rebalance_threshold:
        return None  # 不需要调整

    return {
        "action": "INCREASE" if adjustment > 0 else "DECREASE",
        "amount": abs(adjustment),
        "target_hedge": target_hedge,
        "current_hedge": current_hedge,
        "deviation": deviation_pct
    }
```

### 3. 调整操作

```python
def execute_rebalance(rebalance_plan):
    """
    执行再平衡
    """
    action = rebalance_plan["action"]
    amount = rebalance_plan["amount"]

    if action == "INCREASE":
        # 增加空单（LP 中 CAKE 增加了）
        # 场景：价格下跌，CAKE 流入 LP
        place_short_order(amount)

    elif action == "DECREASE":
        # 减少空单（LP 中 CAKE 减少了）
        # 场景：价格上涨，CAKE 从 LP 流出
        reduce_short_position(amount)
```

### 4. 再平衡成本估算

```python
# 示例：50 秒调整一次
rebalance_interval = 50  # 秒
daily_rebalances = 86400 / 50  # 1728 次/天

# 假设每次调整 100 USDT
adjustment_size = 100  # USDT
cex_fee_rate = 0.0005  # 0.05% taker 费率

# 每次成本
cost_per_rebalance = adjustment_size * cex_fee_rate
                   = 100 * 0.0005
                   = 0.05 USDT

# 每日成本
daily_cost = cost_per_rebalance * daily_rebalances
           = 0.05 * 1728
           = 86.4 USDT

# ⚠️ 这个成本太高了！
```

**优化方案**：

```python
# 方案 1: 降低调整频率
rebalance_interval = 300  # 5 分钟
daily_rebalances = 86400 / 300 = 288 次
daily_cost = 0.05 * 288 = 14.4 USDT ✅

# 方案 2: 提高调整阈值
min_rebalance_threshold = 0.05  # 5%
# 只有偏差 > 5% 才调整
# 预计每天调整 50-100 次
daily_cost = 0.05 * 75 = 3.75 USDT ✅

# 方案 3: 使用 Maker 订单
use_limit_orders = True
cex_fee_rate = 0.0002  # 0.02% maker 费率
daily_cost = 0.02 * 288 = 5.76 USDT ✅
```

---

## 风险控制

### 1. LP 层面风险

#### 风险 A: 无常损失

虽然有对冲，但对冲不是 100% 完美：

```python
# 监控无常损失
def calculate_impermanent_loss():
    initial_value = 5000  # 初始投入

    # 获取当前 LP 价值
    current_lp_value = get_lp_value()

    # 如果持有不动的价值
    hodl_value = initial_base * current_price + initial_quote

    # 无常损失
    il = current_lp_value - hodl_value
    il_pct = il / hodl_value

    return il, il_pct
```

**对策**：
- 定期检查 IL
- 如果 IL > 阈值（如 -2%），考虑关闭策略

#### 风险 B: 价格快速移出区间（CLMM）

```python
# CLMM 特有风险
position_range = [1.8, 2.2]
current_price = 2.5  # 超出上限

# 此时：
# - LP 中全是 USDT，无 CAKE
# - 但 CEX 还有空单
# - 不再 Delta 中性！

# 对策
def check_price_range():
    if current_price > upper_price * 1.05:
        logger.warning("价格超出区间 5% 以上！")
        # 选项 1: 平掉所有 CEX 仓位
        # 选项 2: 移除 LP，重新开仓
        # 选项 3: 继续监控，等待价格回归
```

### 2. CEX 层面风险

#### 风险 A: 爆仓风险

```python
# 杠杆仓位的爆仓风险
leverage = 10
position_size = 3800  # USDT
margin = position_size / leverage = 380  # USDT

entry_price = 2.0  # CAKE 价格
# 爆仓价格（10x 杠杆）
liquidation_price = entry_price * (1 + 1/leverage)
                  = 2.0 * 1.1
                  = 2.2 USDT

# 对策
def manage_leverage():
    # 1. 使用低杠杆（3-5x）
    leverage = 5
    liquidation_price = 2.0 * 1.2 = 2.4  # 更安全

    # 2. 维持充足保证金
    target_margin_ratio = 0.3  # 30%
    required_margin = position_size * target_margin_ratio

    # 3. 动态调整
    if margin_ratio < 0.2:  # 20%
        add_margin()  # 追加保证金
```

#### 风险 B: 资金费率

```python
# 永续合约资金费率
funding_rate = 0.0001  # 0.01% 每 8 小时
daily_funding = funding_rate * 3  # 3 次/天

# 持有 3800 USDT 空单
daily_cost = 3800 * daily_funding
           = 3800 * 0.0003
           = 1.14 USDT/天

# 年化成本
yearly_cost = 1.14 * 365 = 416 USDT

# 对策
# 1. 监控资金费率
# 2. 如果资金费率持续为正（空方支付）
#    考虑切换到交割合约
```

### 3. 系统层面风险

#### 风险 A: 网络延迟

```python
# Gateway 调用可能延迟
max_acceptable_delay = 5  # 秒

async def get_lp_with_timeout():
    try:
        lp_info = await asyncio.wait_for(
            get_lp_position(),
            timeout=max_acceptable_delay
        )
        return lp_info
    except asyncio.TimeoutError:
        logger.error("获取 LP 信息超时！")
        # 使用缓存的数据
        return cached_lp_info
```

#### 风险 B: 对冲失败

```python
# 场景：LP 持仓变化，但 CEX 对冲失败
def execute_rebalance_with_retry():
    max_retries = 3

    for attempt in range(max_retries):
        try:
            result = place_hedge_order()
            if result.success:
                return True
        except Exception as e:
            logger.error(f"对冲失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            await asyncio.sleep(1)

    # 所有重试都失败
    logger.critical("⚠️ 对冲失败！存在风险敞口！")
    send_alert("对冲失败警报")

    # 紧急措施：关闭 LP？
    if emergency_close_lp:
        close_lp_position()
```

---

## 技术架构

### 模块设计

```
delta_neutral_lp_hedge/
├── core/
│   ├── lp_monitor.py          # LP 持仓监控
│   ├── delta_calculator.py    # Delta 计算
│   ├── hedge_executor.py      # CEX 对冲执行
│   └── rebalancer.py          # 再平衡逻辑
├── models/
│   ├── lp_position.py         # LP 仓位模型
│   ├── cex_position.py        # CEX 仓位模型
│   └── hedge_plan.py          # 对冲计划模型
├── utils/
│   ├── price_feed.py          # 价格获取
│   ├── risk_metrics.py        # 风险指标
│   └── statistics.py          # 统计分析
└── strategy.py                # 主策略
```

### 1. LP 监控模块

```python
class LpMonitor:
    """
    监控 DEX LP 持仓
    """

    async def get_position_info(self) -> LpPosition:
        """
        获取 LP 持仓信息

        Returns:
            LpPosition:
                - pool_address: 池子地址
                - base_amount: Base token 数量
                - quote_amount: Quote token 数量
                - lower_price: 价格下限（CLMM）
                - upper_price: 价格上限（CLMM）
                - liquidity: 流动性（CLMM）
                - unclaimed_fees: 未领取手续费
        """
        pass

    async def get_position_value(self) -> Decimal:
        """
        计算 LP 持仓总价值（USDT）
        """
        pass

    async def get_fees_earned(self) -> Dict[str, Decimal]:
        """
        获取已赚取的手续费
        """
        pass
```

### 2. Delta 计算模块

```python
class DeltaCalculator:
    """
    计算 LP 持仓的 Delta（风险敞口）
    """

    def calculate_base_delta(
        self,
        position: LpPosition,
        current_price: Decimal
    ) -> Decimal:
        """
        计算 Base token 的 Delta（USDT 价值）

        简化方法：
        delta = base_amount * current_price

        精确方法（CLMM）：
        考虑价格区间和流动性分布
        """
        # 简化实现
        return position.base_amount * current_price

    def calculate_target_hedge(
        self,
        delta: Decimal,
        hedge_ratio: Decimal = Decimal("0.95")
    ) -> Decimal:
        """
        计算目标对冲仓位

        Args:
            delta: LP Delta
            hedge_ratio: 对冲比例（默认 95%）

        Returns:
            目标 CEX 空单大小（USDT 名义价值）
        """
        return delta * hedge_ratio
```

### 3. 对冲执行模块

```python
class HedgeExecutor:
    """
    在 CEX 上执行对冲操作
    """

    async def get_current_position(self) -> CexPosition:
        """
        获取当前 CEX 仓位

        Returns:
            CexPosition:
                - side: LONG / SHORT
                - size: 仓位大小（币数量）
                - notional_value: 名义价值（USDT）
                - entry_price: 开仓均价
                - unrealized_pnl: 未实现盈亏
                - margin: 保证金
                - leverage: 杠杆
        """
        pass

    async def adjust_position(
        self,
        target_notional: Decimal,
        current_price: Decimal
    ) -> bool:
        """
        调整仓位到目标大小

        Args:
            target_notional: 目标名义价值（USDT）
            current_price: 当前价格

        Returns:
            是否成功
        """
        pass

    async def place_hedge_order(
        self,
        side: str,  # "BUY" or "SELL"
        amount: Decimal,
        use_limit: bool = False
    ) -> str:
        """
        下对冲订单

        Args:
            side: 方向
            amount: 数量（USDT 价值）
            use_limit: 是否使用限价单（降低费率）

        Returns:
            订单 ID
        """
        pass
```

### 4. 再平衡模块

```python
class Rebalancer:
    """
    管理再平衡逻辑
    """

    def __init__(
        self,
        lp_monitor: LpMonitor,
        delta_calculator: DeltaCalculator,
        hedge_executor: HedgeExecutor,
        config: RebalanceConfig
    ):
        self.lp_monitor = lp_monitor
        self.delta_calc = delta_calculator
        self.hedge_exec = hedge_executor
        self.config = config

    async def check_and_rebalance(self) -> Optional[RebalanceResult]:
        """
        检查并执行再平衡

        Returns:
            RebalanceResult 如果执行了再平衡
            None 如果不需要再平衡
        """
        # 1. 获取 LP 持仓
        lp_position = await self.lp_monitor.get_position_info()
        current_price = await self.get_current_price()

        # 2. 计算目标对冲
        delta = self.delta_calc.calculate_base_delta(
            lp_position,
            current_price
        )
        target_hedge = self.delta_calc.calculate_target_hedge(
            delta,
            self.config.hedge_ratio
        )

        # 3. 获取当前 CEX 仓位
        cex_position = await self.hedge_exec.get_current_position()
        current_hedge = abs(cex_position.notional_value)

        # 4. 计算偏差
        deviation = abs(target_hedge - current_hedge)
        deviation_pct = deviation / target_hedge if target_hedge > 0 else 0

        # 5. 检查是否需要调整
        if deviation_pct < self.config.min_rebalance_threshold:
            self.logger().info(
                f"无需调整 - 偏差 {deviation_pct:.2%} "
                f"< 阈值 {self.config.min_rebalance_threshold:.2%}"
            )
            return None

        # 6. 执行调整
        success = await self.hedge_exec.adjust_position(
            target_hedge,
            current_price
        )

        return RebalanceResult(
            success=success,
            target_hedge=target_hedge,
            current_hedge=current_hedge,
            adjustment=target_hedge - current_hedge,
            deviation_pct=deviation_pct
        )
```

---

## 配置参数

### 配置文件示例

```yaml
# ========================================
# Delta 中性 LP 对冲策略配置
# ========================================

# ========== 交易所配置 ==========

# DEX 配置
dex_exchange: pancakeswap/clmm
dex_network: bsc
trading_pair: CAKE-USDT

# CEX 配置
cex_exchange: binance_perpetual
cex_trading_pair: CAKEUSDT

# ========== LP 配置 ==========

# LP 池子地址（可选，自动检测）
pool_address: ""

# 是否自动领取手续费
auto_claim_fees: true

# 手续费领取阈值（USDT）
# 手续费价值 > 此值时才领取
fee_claim_threshold: 10

# ========== 对冲配置 ==========

# 对冲比例（0-1）
# 0.95 = 对冲 95% 的 Base token 价值
hedge_ratio: 0.95

# 杠杆倍数
leverage: 5

# 对冲方向（一般是 SHORT）
hedge_side: SHORT

# ========== 再平衡配置 ==========

# 检查间隔（秒）
check_interval: 300  # 5 分钟

# 最小再平衡阈值（百分比）
# 只有偏差 > 此值时才调整
min_rebalance_threshold: 0.03  # 3%

# 最大再平衡阈值（百分比）
# 偏差 > 此值时立即调整，不等待下个周期
max_rebalance_threshold: 0.10  # 10%

# 使用限价单（降低费率，但可能不成交）
use_limit_orders: false

# 限价单超时（秒）
limit_order_timeout: 30

# ========== 风险控制 ==========

# 最大 Delta 敞口（USDT）
# 如果对冲后仍有敞口 > 此值，暂停策略
max_delta_exposure: 500

# 最小保证金率
# 低于此值时追加保证金
min_margin_ratio: 0.25  # 25%

# 最大无常损失容忍度（百分比）
# IL > 此值时发出警告
max_il_tolerance: 0.05  # 5%

# 价格区间监控（CLMM）
# 价格超出区间此百分比时警告
price_range_warning_pct: 0.05  # 5%

# ========== 费用估算 ==========

# CEX Taker 费率
cex_taker_fee: 0.0005  # 0.05%

# CEX Maker 费率
cex_maker_fee: 0.0002  # 0.02%

# Gas 成本估算（每次调整）
gas_cost_per_rebalance: 0.5  # USDT

# ========== 统计记录 ==========

# 是否启用详细日志
verbose_logging: true

# 日志文件路径
log_file: logs/delta_neutral_lp_hedge.log

# 是否记录到数据库
save_to_database: true

# 数据库配置
database_url: sqlite:///delta_neutral_stats.db

# ========== 告警配置 ==========

# 是否启用告警
enable_alerts: true

# 告警方式（telegram/email/webhook）
alert_method: telegram

# Telegram Bot Token
telegram_bot_token: ""
telegram_chat_id: ""

# 告警触发条件
alert_triggers:
  - delta_exposure_exceeded      # Delta 敞口超限
  - margin_ratio_low             # 保证金率过低
  - rebalance_failed             # 再平衡失败
  - il_exceeded                  # 无常损失超限
  - price_out_of_range           # 价格超出区间
```

---

## 实现细节

### 主策略类结构

```python
class DeltaNeutralLpHedgeStrategy(ScriptStrategyBase):
    """
    Delta 中性 LP 对冲策略

    核心逻辑：
    1. 监控 DEX LP 持仓
    2. 计算 Delta 风险敞口
    3. 在 CEX 开对冲仓位
    4. 定期再平衡
    5. 赚取 LP 手续费 - 对冲成本 = 净收益
    """

    def __init__(self, connectors, config):
        super().__init__(connectors)
        self.config = config

        # 连接器
        self.dex_connector = connectors[config.dex_exchange]
        self.cex_connector = connectors[config.cex_exchange]

        # 模块
        self.lp_monitor = LpMonitor(self.dex_connector, config)
        self.delta_calc = DeltaCalculator(config)
        self.hedge_exec = HedgeExecutor(self.cex_connector, config)
        self.rebalancer = Rebalancer(
            self.lp_monitor,
            self.delta_calc,
            self.hedge_exec,
            config
        )

        # 状态
        self.initialized = False
        self.lp_position = None
        self.cex_position = None

        # 统计
        self.stats = {
            "total_fees_earned": Decimal("0"),
            "total_hedge_cost": Decimal("0"),
            "net_profit": Decimal("0"),
            "rebalance_count": 0,
            "rebalance_failed_count": 0
        }

    async def on_tick(self):
        """
        主循环 - 每个 tick 调用一次
        """
        try:
            # 初始化
            if not self.initialized:
                await self.initialize()
                return

            # 检查时间间隔
            current_time = time.time()
            if current_time - self.last_check_time < self.config.check_interval:
                return

            self.last_check_time = current_time

            # 执行再平衡检查
            result = await self.rebalancer.check_and_rebalance()

            # 记录结果
            if result:
                await self.record_rebalance(result)

            # 更新统计
            await self.update_statistics()

        except Exception as e:
            self.logger().error(f"策略执行错误: {e}", exc_info=True)

    async def initialize(self):
        """
        初始化策略
        """
        self.logger().info("🚀 初始化 Delta 中性 LP 对冲策略...")

        # 1. 检查 LP 持仓
        self.lp_position = await self.lp_monitor.get_position_info()
        if not self.lp_position:
            self.logger().error("❌ 未找到 LP 持仓！")
            return

        self.logger().info(
            f"✅ LP 持仓:\n"
            f"   Base: {self.lp_position.base_amount} CAKE\n"
            f"   Quote: {self.lp_position.quote_amount} USDT\n"
            f"   价值: {self.lp_position.total_value} USDT"
        )

        # 2. 检查 CEX 仓位
        self.cex_position = await self.hedge_exec.get_current_position()

        if self.cex_position:
            self.logger().info(
                f"✅ CEX 持仓:\n"
                f"   方向: {self.cex_position.side}\n"
                f"   大小: {self.cex_position.notional_value} USDT\n"
                f"   杠杆: {self.cex_position.leverage}x"
            )
        else:
            self.logger().info("ℹ️  无 CEX 持仓，将创建初始对冲")
            # 创建初始对冲
            await self.create_initial_hedge()

        # 3. 设置初始检查时间
        self.last_check_time = time.time()

        self.initialized = True
        self.logger().info("✅ 策略初始化完成！")

    async def create_initial_hedge(self):
        """
        创建初始对冲仓位
        """
        # 计算目标对冲
        current_price = await self.get_current_price()
        delta = self.delta_calc.calculate_base_delta(
            self.lp_position,
            current_price
        )
        target_hedge = self.delta_calc.calculate_target_hedge(delta)

        self.logger().info(
            f"创建初始对冲:\n"
            f"   Delta: {delta} USDT\n"
            f"   目标对冲: {target_hedge} USDT"
        )

        # 执行对冲
        success = await self.hedge_exec.adjust_position(
            target_hedge,
            current_price
        )

        if success:
            self.logger().info("✅ 初始对冲创建成功！")
        else:
            self.logger().error("❌ 初始对冲创建失败！")

    def format_status(self) -> str:
        """
        格式化状态显示
        """
        lines = []

        lines.append("=" * 70)
        lines.append("Delta 中性 LP 对冲策略".center(70))
        lines.append("=" * 70)

        # LP 信息
        if self.lp_position:
            lines.append("\n📊 LP 持仓")
            lines.append("-" * 70)
            lines.append(f"Base: {self.lp_position.base_amount} CAKE")
            lines.append(f"Quote: {self.lp_position.quote_amount} USDT")
            lines.append(f"总价值: {self.lp_position.total_value} USDT")
            lines.append(f"未领取手续费: {self.lp_position.unclaimed_fees_usdt} USDT")

        # CEX 对冲信息
        if self.cex_position:
            lines.append("\n🔒 CEX 对冲")
            lines.append("-" * 70)
            lines.append(f"方向: {self.cex_position.side}")
            lines.append(f"名义价值: {self.cex_position.notional_value} USDT")
            lines.append(f"杠杆: {self.cex_position.leverage}x")
            lines.append(f"保证金: {self.cex_position.margin} USDT")
            lines.append(f"未实现盈亏: {self.cex_position.unrealized_pnl} USDT")

        # 统计信息
        lines.append("\n💰 收益统计")
        lines.append("-" * 70)
        lines.append(f"累计手续费: {self.stats['total_fees_earned']} USDT")
        lines.append(f"对冲成本: {self.stats['total_hedge_cost']} USDT")
        lines.append(f"净收益: {self.stats['net_profit']} USDT")
        lines.append(f"再平衡次数: {self.stats['rebalance_count']}")

        lines.append("=" * 70)

        return "\n".join(lines)
```

### 关键计算示例

```python
# 示例 1: 初始状态
"""
LP 持仓:
  - 2000 CAKE (价格 2.0 USDT)
  - 1000 USDT
  - 总价值: 5000 USDT

Delta: 2000 * 2.0 = 4000 USDT
目标对冲: 4000 * 0.95 = 3800 USDT

CEX 空单: 3800 USDT (10x 杠杆)
保证金: 380 USDT
"""

# 示例 2: 价格上涨到 2.2
"""
LP 持仓变化（AMM）:
  - CAKE 被买走，减少到约 1900 CAKE
  - USDT 增加到约 1100 USDT
  - 总价值: 5078 USDT (减少了无常损失)

Delta: 1900 * 2.2 = 4180 USDT
目标对冲: 4180 * 0.95 = 3971 USDT
当前对冲: 3800 USDT

偏差: (3971 - 3800) / 3971 = 4.3%
> 3% 阈值，触发再平衡

调整: 增加 171 USDT 空单

CEX 盈亏:
  - 空单从 2.0 涨到 2.2，亏损
  - 亏损 = 3800 * (2.2 - 2.0) / 2.0 = -380 USDT

LP 价值变化:
  - 如果持有不动: 2000 * 2.2 + 1000 = 5400 USDT
  - 实际 LP 价值: 5078 USDT
  - 无常损失: 5078 - 5400 = -322 USDT

总盈亏: -322 (IL) - 380 (CEX 亏损) = -702 USDT

⚠️ 看起来亏了，但这是因为：
1. 对冲比例不是 100%
2. 还有 LP 手续费收入（未计入）
"""

# 示例 3: 加上手续费
"""
假设期间赚取手续费: 50 USDT

总盈亏: -702 + 50 = -652 USDT

这仍然是负的，说明：
1. 单次价格波动的影响
2. 需要长期运行，手续费累积
3. 或者提高对冲比例到 100%
"""
```

---

## 总结

### 策略优势

✅ **稳定收益**：赚取 LP 手续费，不依赖价格波动
✅ **风险对冲**：CEX 对冲消除价格风险
✅ **资金效率**：使用杠杆，提高资金利用率
✅ **自动化**：无需手动管理，自动再平衡

### 策略劣势

❌ **对冲成本**：频繁调整产生手续费
❌ **复杂性**：需要同时管理 DEX 和 CEX
❌ **资金费率**：永续合约可能有资金费用
❌ **技术要求**：需要稳定的系统和 API 连接

### 适用场景

- ✅ 高交易量池子（手续费高）
- ✅ 中等波动币种（Delta 变化不太剧烈）
- ✅ 有 CEX 合约市场
- ✅ 长期运营（手续费累积）

### 预期收益

```
年化收益 = LP 手续费 APR - 对冲成本 - 资金费率

示例（CAKE-USDT）:
  LP 手续费 APR: 15%
  对冲成本: -3%
  资金费率: -2%
  净 APR: 10%

5000 USDT 投入
预期年收益: 500 USDT
```

### 下一步

1. **实现代码**：基于此设计实现完整策略
2. **回测**：使用历史数据验证策略
3. **小额测试**：先用小额资金测试
4. **优化参数**：调整对冲比例、再平衡阈值
5. **监控运营**：持续跟踪收益和风险

---

## 参考资料

- [Uniswap V3 流动性数学](https://uniswap.org/whitepaper-v3.pdf)
- [无常损失计算器](https://dailydefi.org/tools/impermanent-loss-calculator/)
- [Delta 对冲策略](https://www.investopedia.com/terms/d/deltahedging.asp)
- [币安永续合约文档](https://binance-docs.github.io/apidocs/futures/en/)

---

**设计版本**: v1.0
**创建日期**: 2025-10-30
**作者**: Claude Code
