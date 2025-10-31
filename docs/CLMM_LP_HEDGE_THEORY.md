# CLMM LP 对冲策略 - 理论与实践

## 📚 理论基础

### 为什么简单的 Delta 对冲不够？

**问题 1：CLMM LP 不是简单的持仓**

传统思维：
```
LP 中有 2000 CAKE → 开 2000 CAKE 空单 → Delta 中性 ❌
```

**错误原因**：
1. LP 持仓是**动态的**，价格变化时 CAKE 数量会变
2. LP 持仓有**Gamma**（Delta 的变化率），不是线性的
3. 价格出界后，LP 变成纯单币，Delta 突变

**问题 2：对冲比例应该是多少？**

- 95%？为什么不是 100%？
- 100%？那为什么不是 105%？
- **答案**：取决于你的目标和 LP 的数学特性

---

## 🎯 核心理论：LP 是什么？

### Uniswap V3 CLMM 数学模型

#### 1. 流动性公式

在价格区间 [Pa, Pb] 内，流动性 L 与资产关系：

```
当价格 P 在区间内：

x = L * (1/√P - 1/√Pb)    # Base token 数量（如 CAKE）
y = L * (√P - √Pa)         # Quote token 数量（如 USDT）

其中：
- L = 流动性常数（不变）
- Pa = 价格下限
- Pb = 价格上限
- P = 当前价格
```

#### 2. 价值函数

LP 持仓的总价值 V(P)：

```
V(P) = x * P + y

当 Pa ≤ P ≤ Pb：
V(P) = L * (√P - √Pa) + L * (1/√P - 1/√Pb) * P
     = L * (2√P - √Pa - P/√Pb)
```

**关键发现**：V(P) 是 P 的**非线性函数**！

#### 3. Delta（一阶导数）

Delta = dV/dP，表示价值对价格的敏感度：

```
Δ = dV/dP = L * (1/√P - 1/√Pb)
          = x（当前 Base token 数量）

这不是巧合！LP 的 Delta = 持有的 Base token 数量
```

**实际意义**：
- 如果 LP 中有 2000 CAKE，Delta = 2000 CAKE
- 价格涨 1 USDT，LP 价值增加约 2000 USDT
- **但**：价格涨后，CAKE 数量减少，Delta 也减少

#### 4. Gamma（二阶导数）

Gamma = d²V/dP² = dΔ/dP，表示 Delta 的变化率：

```
Γ = d²V/dP² = -L / (2 * P^(3/2))

Gamma 永远是负的！
```

**实际意义**：
- 价格上涨 → CAKE 减少 → Delta 减少（负 Gamma）
- 价格下跌 → CAKE 增加 → Delta 增加（负 Gamma）
- **这就是无常损失的根源！**

---

## 🔬 科学的对冲策略

### 策略 A：完全 Delta 对冲（传统做法）

#### 原理

始终保持：**LP Delta + CEX Delta = 0**

```python
# 实时计算
lp_delta = lp_position.base_amount  # LP 中的 CAKE 数量
cex_position = -lp_delta             # CEX 开等量空单

# 例子
当前价格 P = 2.0 USDT
LP 中: 2000 CAKE
CEX 空单: 2000 CAKE (价值 4000 USDT)
总 Delta = 0 ✅
```

#### 价格变化时

**场景 1：价格涨到 2.1 USDT**

```
LP 变化（根据 CLMM 公式）:
- CAKE 被买走，减少到约 1950 CAKE
- USDT 增加

新 Delta:
LP Delta = 1950 CAKE
CEX Delta = -2000 CAKE
总 Delta = -50 CAKE ❌ 不再中性！

需要调整：
买入 50 CAKE，平掉部分空单
新 CEX 空单 = 1950 CAKE ✅
```

#### 优点
✅ 理论上完美对冲价格风险
✅ 逻辑清晰，易于理解

#### 缺点
❌ **频繁调整**：价格每次变化都需要调整
❌ **高手续费**：每次调整都要付 Taker 费
❌ **无常损失没有完全对冲**：因为有 Gamma

#### 成本分析

```python
# 假设参数
price_volatility = 0.02  # 2% 日波动
rebalance_threshold = 0.01  # 1% 偏差触发
avg_rebalances_per_day = 50  # 约每半小时一次

# 每次调整成本
position_size = 4000  # USDT
adjustment_ratio = 0.01  # 平均调整 1%
fee_rate = 0.0005  # 0.05% taker

cost_per_rebalance = position_size * adjustment_ratio * fee_rate
                   = 4000 * 0.01 * 0.0005
                   = 0.02 USDT

# 每日成本
daily_cost = 0.02 * 50 = 1 USDT

# 年化成本
yearly_cost = 1 * 365 = 365 USDT
cost_ratio = 365 / 5000 = 7.3% 年化成本！
```

**结论**：如果 LP 手续费 APR < 7.3%，亏本！

---

### 策略 B：Delta-Gamma 对冲（进阶）

#### 原理

同时对冲 Delta 和 Gamma：
- Delta 对冲：消除一阶风险
- Gamma 对冲：减少 Delta 变化速度

#### 问题

在 CEX 上，我们只有线性工具（现货、永续合约）：
- 现货/合约：Delta ≠ 0, Gamma = 0
- 期权：Delta ≠ 0, Gamma ≠ 0（但 DeFi 期权流动性差）

**结论**：纯 CEX 工具无法完美 Gamma 对冲

---

### 策略 C：带容忍度的 Delta 对冲（实用）

#### 核心思想

**不追求完美对冲，而是平衡成本与风险**

```python
# 参数
hedge_ratio = 0.95          # 只对冲 95%
rebalance_threshold = 0.05  # 5% 偏差才调整
rebalance_interval = 3600   # 最快 1 小时调整一次

# 目标
target_cex_position = lp_delta * hedge_ratio

# 调整条件（AND 关系）
deviation = abs(current_cex - target_cex) / target_cex
time_since_last = now - last_rebalance_time

should_rebalance = (
    deviation > rebalance_threshold AND
    time_since_last > rebalance_interval
)
```

#### 为什么是 95%？

**理论依据**：

1. **手续费成本**：完全对冲成本 > 收益
2. **Gamma 风险**：5% 缓冲吸收 Gamma 造成的偏差
3. **单边风险偏好**：如果看好 CAKE，可以保留多头敞口

**数学推导**（简化）：

```
最优对冲比例 h* 满足：

边际成本 = 边际收益

MC = 2 * transaction_fee * rebalance_frequency
MR = -risk_aversion * variance_reduction

h* ≈ 1 - (transaction_fee * rebalance_freq) / (risk_aversion * variance)

典型值：
h* ≈ 0.90 - 0.98
```

---

## ⚠️ 边界情况处理（关键！）

### 情况 1：价格接近边界

**问题**：价格接近 Pa 或 Pb 时，LP 构成快速变化

```python
# 例子：价格区间 [1.8, 2.2]
# 当前价格 2.15，接近上限 2.2

# 此时：
# - CAKE 数量很少（大部分变成 USDT）
# - 微小价格变化导致 CAKE 数量大变
# - Delta 快速下降

# 如果价格继续涨到 2.25（超出上限）
# - LP 变成 100% USDT
# - CAKE = 0
# - Delta = 0！

# 但 CEX 还有空单 → 巨大风险！
```

#### 科学处理方案

**方案 1：提前预警**

```python
def check_range_proximity(price, pa, pb):
    """
    检查价格是否接近边界
    """
    upper_buffer = 0.05  # 5% 缓冲
    lower_buffer = 0.05

    upper_warning = pb * (1 - upper_buffer)  # 2.2 * 0.95 = 2.09
    lower_warning = pa * (1 + lower_buffer)  # 1.8 * 1.05 = 1.89

    if price >= upper_warning:
        return "NEAR_UPPER", (price - upper_warning) / (pb - upper_warning)
    elif price <= lower_warning:
        return "NEAR_LOWER", (lower_warning - price) / (lower_warning - pa)
    else:
        return "SAFE", 0

# 使用
status, proximity = check_range_proximity(2.15, 1.8, 2.2)
if status == "NEAR_UPPER":
    if proximity > 0.5:  # 超过一半缓冲区
        logger.warning(f"⚠️ 价格接近上限！Proximity: {proximity:.1%}")
        # 措施：
        # 1. 更频繁地检查（每 30 秒）
        # 2. 降低调整阈值（3% → 1%）
        # 3. 准备关闭策略
```

**方案 2：动态调整对冲比例**

```python
def calculate_adaptive_hedge_ratio(price, pa, pb, base_ratio=0.95):
    """
    根据价格位置动态调整对冲比例

    原理：
    - 在区间中心：使用基础比例（95%）
    - 接近边界：提高比例（接近 100%）
    - 原因：边界附近 Delta 变化快，需要更紧密跟踪
    """
    # 计算价格在区间中的位置（0-1）
    range_width = pb - pa
    position = (price - pa) / range_width  # 0 = 下限, 1 = 上限

    # 距离中心的偏离度（0-0.5）
    center_deviation = abs(position - 0.5)

    # 调整比例：中心 95%，边界 100%
    adjustment = center_deviation * 0.1  # 最多增加 5%
    adaptive_ratio = min(base_ratio + adjustment, 1.0)

    return adaptive_ratio

# 例子
price = 2.15, pa = 1.8, pb = 2.2
position = (2.15 - 1.8) / 0.4 = 0.875  # 接近上限
center_deviation = abs(0.875 - 0.5) = 0.375
adjustment = 0.375 * 0.1 = 0.0375
adaptive_ratio = 0.95 + 0.0375 = 0.9875 ≈ 99%

# 更接近 100% 对冲！
```

**方案 3：价格出界时的处理**

```python
def handle_out_of_range(price, pa, pb, lp_position, cex_position):
    """
    价格超出区间时的处理
    """
    if price > pb:
        # LP 变成 100% USDT，Delta = 0
        logger.warning(f"🔴 价格超出上限！{price} > {pb}")

        # 选项 A：平掉所有 CEX 空单
        if config.close_hedge_when_out_of_range:
            logger.info("平掉所有 CEX 对冲仓位")
            close_all_cex_positions()

        # 选项 B：保持 CEX 仓位，等待价格回归
        elif config.wait_for_price_return:
            logger.info("保持 CEX 仓位，等待价格回归区间")
            # 风险：价格持续上涨，CEX 空单亏损
            # 收益：如果价格回落，恢复对冲

        # 选项 C：移除 LP，重新开仓在新区间
        elif config.reposition_lp:
            logger.info("移除当前 LP，在新价格区间重新开仓")
            remove_lp_position()
            # 计算新区间：以当前价格为中心
            new_pa = price * 0.95
            new_pb = price * 1.05
            add_lp_position(new_pa, new_pb)
            # 同时调整 CEX 对冲

    elif price < pa:
        # LP 变成 100% CAKE，Delta = 全部 CAKE
        logger.warning(f"🔴 价格跌破下限！{price} < {pa}")

        # 类似处理...
```

### 情况 2：价格剧烈波动

**问题**：价格在短时间内大幅波动（如 ±10%）

```python
# Flash crash 场景
t0: price = 2.0, lp_delta = 2000 CAKE, cex = -1900 CAKE
t1: price = 1.8 (-10%), lp_delta = 2200 CAKE (增加 200)
t2: price = 2.0 (回归), lp_delta = 2000 CAKE (减少 200)

# 如果在 t1 调整：
# - 增加空单 200 CAKE @ 1.8
# 在 t2：
# - 减少空单 200 CAKE @ 2.0
# 亏损 = 200 * (2.0 - 1.8) = 40 USDT

# 如果不调整：
# 在 t1 时刻有 300 CAKE 的多头敞口
# 风险敞口 = 300 * 1.8 = 540 USDT
```

#### 科学处理

**方案 1：波动率自适应**

```python
def calculate_volatility_adjusted_threshold(
    recent_volatility: float,
    base_threshold: float = 0.05
) -> float:
    """
    根据波动率调整再平衡阈值

    高波动：提高阈值（减少调整）
    低波动：降低阈值（更精确对冲）
    """
    if recent_volatility > 0.05:  # 5% 高波动
        # 提高到 10%，避免在噪音中频繁交易
        return base_threshold * 2
    elif recent_volatility < 0.01:  # 1% 低波动
        # 降低到 3%，更精确跟踪
        return base_threshold * 0.6
    else:
        return base_threshold
```

**方案 2：时间加权调整**

```python
def time_weighted_rebalance_decision(
    deviation: float,
    time_since_last: int,
    deviation_threshold: float,
    min_time_interval: int
) -> bool:
    """
    综合偏差和时间的决策

    逻辑：
    - 如果偏差很大（>10%），立即调整
    - 如果偏差中等（5-10%），等待一段时间观察
    - 如果偏差小（<5%），等待更长时间
    """
    # 紧急情况：偏差太大
    if deviation > deviation_threshold * 2:
        return True

    # 正常情况：时间 + 偏差都满足
    if deviation > deviation_threshold and time_since_last > min_time_interval:
        return True

    # 长时间未调整：即使偏差小也调整（防止漂移累积）
    if time_since_last > min_time_interval * 3:
        return True

    return False
```

### 情况 3：LP 费率变化

**问题**：不同价格区间的费率不同

```python
# Uniswap V3 有多个费率等级
fee_tiers = {
    "0.01%": "稳定币对",
    "0.05%": "相关资产",
    "0.30%": "主流币",
    "1.00%": "高风险币"
}

# CAKE-USDT 可能在 0.30% 池子
# 24h 交易量 = 1,000,000 USDT
# 你的 LP 占比 = 5000 / 500000 = 1%
# 你的手续费 = 1,000,000 * 0.003 * 0.01 = 30 USDT/天

# 但如果价格移出你的区间：
# - 交易不再经过你的流动性
# - 手续费收入 = 0
# - 但对冲成本仍在！
```

#### 科学评估

```python
def estimate_fee_apr_in_range(
    pool_24h_volume: Decimal,
    pool_tvl: Decimal,
    your_liquidity: Decimal,
    fee_tier: Decimal,
    in_range_probability: float
) -> float:
    """
    估算实际 APR（考虑价格在区间内的概率）

    in_range_probability: 历史数据统计价格在区间内的时间比例
    """
    # 基础 APR（假设永远在区间内）
    base_apr = (pool_24h_volume * 365 * fee_tier) / pool_tvl

    # 你的预期 APR
    your_apr = base_apr * in_range_probability

    return your_apr

# 例子
pool_24h_volume = 1_000_000
pool_tvl = 500_000
your_liquidity = 5_000
fee_tier = 0.003
in_range_prob = 0.7  # 70% 时间在区间内

base_apr = (1_000_000 * 365 * 0.003) / 500_000 = 2.19 = 219%
your_apr = 219% * 0.7 = 153%  # 仍然很高

# 但要减去对冲成本！
hedging_cost_apr = 0.10  # 10% (之前计算)
net_apr = 153% - 10% = 143%  # 仍然可观 ✅

# 但如果价格经常出界：
in_range_prob = 0.3  # 只有 30% 在区间内
your_apr = 219% * 0.3 = 66%
net_apr = 66% - 10% = 56%  # 还行

# 极端情况：
in_range_prob = 0.1  # 90% 时间在区间外
your_apr = 219% * 0.1 = 22%
net_apr = 22% - 10% = 12%  # 勉强
```

---

## 📊 科学的策略设计

### 综合策略框架

```python
class ScientificClmmHedgeStrategy:
    """
    科学的 CLMM LP 对冲策略

    核心原则：
    1. 基于理论的 Delta 计算
    2. 动态调整对冲比例
    3. 智能处理边界情况
    4. 成本-收益优化
    """

    def __init__(self, config):
        # 基础参数
        self.base_hedge_ratio = 0.95
        self.base_rebalance_threshold = 0.05
        self.min_rebalance_interval = 3600  # 1 小时

        # 边界参数
        self.range_proximity_buffer = 0.05  # 5%
        self.out_of_range_action = "WAIT"  # CLOSE / WAIT / REPOSITION

        # 风险参数
        self.max_delta_exposure = 1000  # USDT
        self.emergency_deviation_threshold = 0.15  # 15%

    async def calculate_lp_delta(self, position, price):
        """
        精确计算 LP Delta

        使用 Uniswap V3 数学公式
        """
        pa = position.lower_price
        pb = position.upper_price

        if price < pa:
            # 价格低于下限：100% Base token
            delta = position.total_base_if_below_range * price
        elif price > pb:
            # 价格高于上限：100% Quote token (Delta = 0)
            delta = 0
        else:
            # 在区间内
            L = position.liquidity
            sqrt_p = math.sqrt(price)
            sqrt_pb = math.sqrt(pb)

            base_amount = L * (1/sqrt_p - 1/sqrt_pb)
            delta = base_amount * price

        return delta

    async def get_adaptive_hedge_ratio(self, price, pa, pb, volatility):
        """
        动态对冲比例

        考虑：
        1. 价格在区间中的位置
        2. 近期波动率
        3. 历史调整成本
        """
        # 基于位置调整
        range_position = (price - pa) / (pb - pa)
        center_deviation = abs(range_position - 0.5)

        position_adjustment = center_deviation * 0.1

        # 基于波动率调整
        if volatility > 0.05:
            volatility_adjustment = -0.05  # 降低对冲，减少成本
        else:
            volatility_adjustment = 0.05  # 提高对冲，更精确

        adaptive_ratio = self.base_hedge_ratio + position_adjustment + volatility_adjustment
        adaptive_ratio = max(0.85, min(1.0, adaptive_ratio))  # 限制在 85-100%

        return adaptive_ratio

    async def get_adaptive_threshold(self, volatility, time_since_last):
        """
        动态调整阈值
        """
        base = self.base_rebalance_threshold

        # 高波动 → 高阈值
        if volatility > 0.05:
            threshold = base * 1.5
        elif volatility > 0.02:
            threshold = base
        else:
            threshold = base * 0.7

        # 长时间未调整 → 降低阈值（防止漂移）
        if time_since_last > self.min_rebalance_interval * 3:
            threshold *= 0.5

        return threshold

    async def check_and_rebalance(self):
        """
        主逻辑
        """
        # 1. 获取 LP 状态
        lp_position = await self.get_lp_position()
        current_price = await self.get_current_price()

        # 2. 边界检查
        range_status = self.check_range_status(
            current_price,
            lp_position.lower_price,
            lp_position.upper_price
        )

        if range_status == "OUT_OF_RANGE":
            return await self.handle_out_of_range()

        if range_status == "NEAR_BOUNDARY":
            # 提高检查频率和对冲精度
            self.min_rebalance_interval = 300  # 5 分钟
            self.base_rebalance_threshold = 0.02  # 2%

        # 3. 计算 Delta
        lp_delta = await self.calculate_lp_delta(lp_position, current_price)

        # 4. 动态参数
        volatility = await self.calculate_recent_volatility()
        hedge_ratio = await self.get_adaptive_hedge_ratio(
            current_price,
            lp_position.lower_price,
            lp_position.upper_price,
            volatility
        )

        # 5. 目标仓位
        target_hedge = lp_delta * hedge_ratio

        # 6. 当前仓位
        current_hedge = await self.get_cex_position_notional()

        # 7. 偏差分析
        deviation = abs(target_hedge - current_hedge) / target_hedge
        time_since_last = time.time() - self.last_rebalance_time

        # 8. 动态阈值
        threshold = await self.get_adaptive_threshold(volatility, time_since_last)

        # 9. 决策
        should_rebalance = (
            deviation > threshold or  # 偏差超过阈值
            deviation > self.emergency_deviation_threshold or  # 紧急情况
            time_since_last > self.min_rebalance_interval * 5  # 太久未调整
        )

        if should_rebalance:
            return await self.execute_rebalance(target_hedge, current_hedge)

        return None
```

---

## 💡 最优实践建议

### 1. 初始设置

```yaml
# 保守型（推荐新手）
hedge_ratio: 0.90
rebalance_threshold: 0.08
rebalance_interval: 7200  # 2 小时
leverage: 3

# 平衡型（推荐）
hedge_ratio: 0.95
rebalance_threshold: 0.05
rebalance_interval: 3600  # 1 小时
leverage: 5

# 激进型（专业）
hedge_ratio: 0.98
rebalance_threshold: 0.03
rebalance_interval: 1800  # 30 分钟
leverage: 5
adaptive_hedging: true
```

### 2. 区间设置

**基于历史波动率**

```python
def calculate_optimal_range(
    current_price: float,
    daily_volatility: float,
    target_days_in_range: int = 7
) -> Tuple[float, float]:
    """
    计算最优价格区间

    目标：价格有 X% 概率在区间内停留 Y 天

    假设价格遵循几何布朗运动（GBM）
    """
    # 使用标准差的倍数
    # 1σ ≈ 68% 概率
    # 2σ ≈ 95% 概率
    # 3σ ≈ 99.7% 概率

    # 对于 7 天，使用 2σ
    sigma_multiplier = 2
    adjusted_vol = daily_volatility * math.sqrt(target_days_in_range)

    range_width_pct = sigma_multiplier * adjusted_vol

    lower = current_price * (1 - range_width_pct)
    upper = current_price * (1 + range_width_pct)

    return lower, upper

# 例子
current_price = 2.0
daily_vol = 0.05  # 5% 日波动
target_days = 7

lower, upper = calculate_optimal_range(current_price, daily_vol, target_days)
# lower = 2.0 * (1 - 2 * 0.05 * √7) = 2.0 * (1 - 0.265) = 1.47
# upper = 2.0 * (1 + 0.265) = 2.53
# 区间 [1.47, 2.53]，约 ±26.5%

# 实际使用可以更窄（提高费率但增加出界风险）
# 推荐：±10% 到 ±20%
```

### 3. 监控指标

**关键指标**

```python
class StrategyMetrics:
    """策略监控指标"""

    # 收益指标
    total_lp_fees_earned: Decimal  # LP 手续费收入
    total_hedging_cost: Decimal    # 对冲成本（手续费 + 滑点）
    total_funding_cost: Decimal    # 资金费率成本
    net_pnl: Decimal               # 净盈亏

    # 风险指标
    current_delta_exposure: Decimal  # 当前 Delta 敞口
    max_delta_exposure: Decimal      # 历史最大敞口
    time_in_range_pct: float         # 价格在区间内的时间比例
    avg_deviation: float             # 平均对冲偏差

    # 运营指标
    rebalance_count: int            # 调整次数
    avg_rebalance_interval: float   # 平均调整间隔
    avg_rebalance_cost: Decimal     # 平均调整成本

    # 效率指标
    lp_fee_apr: float               # LP 费率 APR
    hedging_cost_apr: float         # 对冲成本 APR
    net_apr: float                  # 净 APR
    sharpe_ratio: float             # 夏普比率（如果可算）
```

### 4. 风险控制

**硬性限制**

```python
# 每日检查
daily_checks = {
    "max_delta_exposure": 1000,       # USDT
    "min_margin_ratio": 0.30,         # 30%
    "max_daily_hedging_cost": 50,     # USDT
    "min_net_apr": 0.05,              # 5%
}

# 触发暂停的条件
emergency_stop_conditions = [
    "delta_exposure > max_delta_exposure",
    "margin_ratio < min_margin_ratio",
    "consecutive_rebalance_failures > 3",
    "net_apr < min_net_apr for 7 days",
    "price_out_of_range for > 24 hours",
]
```

---

## 📈 回测与验证

### 必须回测的场景

1. **正常波动**：价格在区间内正常波动
2. **单边行情**：价格持续上涨或下跌
3. **震荡行情**：价格反复穿越中心
4. **Flash crash**：价格急剧下跌后反弹
5. **出界场景**：价格移出区间并停留
6. **区间穿越**：价格快速穿越整个区间

### 验证指标

- ✅ 净 APR > 15%（CAKE-USDT）
- ✅ 最大 Delta 敞口 < 1000 USDT
- ✅ 对冲成本 < LP 手续费的 30%
- ✅ 出界时间 < 20%
- ✅ 极端情况下不爆仓

---

## 🎯 总结：科学对冲的关键

### 1. **理解数学**
- LP 不是简单持仓，是动态的 Delta + 负 Gamma
- 必须精确计算 Delta，不能估算

### 2. **成本为王**
- 过度对冲 = 破产
- 最优对冲比例是成本-收益的平衡点

### 3. **边界至关重要**
- 价格接近/超出边界时，策略逻辑完全不同
- 必须有明确的处理方案

### 4. **动态调整**
- 静态参数无法适应市场变化
- 基于波动率、位置、时间的动态参数

### 5. **风险第一**
- 盈利是目标，但不爆仓是前提
- 严格的风控和熔断机制

### 6. **持续监控**
- 自动化不等于无人化
- 关键指标必须每日检查

---

**这才是科学的、可持续的 CLMM LP 对冲策略！**
