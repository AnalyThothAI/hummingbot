# 🔄 CEX-DEX LP 套利策略设计方案

## 日期
2025-10-30

---

## 📋 目录

1. [策略概述](#策略概述)
2. [核心原理](#核心原理)
3. [架构设计](#架构设计)
4. [详细流程](#详细流程)
5. [关键组件](#关键组件)
6. [配置参数](#配置参数)
7. [实现细节](#实现细节)
8. [风险控制](#风险控制)
9. [代码结构](#代码结构)
10. [测试方案](#测试方案)

---

## 策略概述

### 核心思想

**在 DEX 上被动做 LP Maker，在 CEX 上主动做 Taker，通过价格差和 LP 费用赚取套利收益**

### 工作模式

```
CEX (Taker - 主动交易)  ←→  DEX (LP Maker - 被动成交)
       ↓                           ↓
  对冲锁定利润              赚取交易费用 + 价差
```

### 适用场景

- ✅ CEX 流动性好，价格稳定
- ✅ DEX 有交易量，LP 能被成交
- ✅ 两边价差 > 手续费 + gas + 目标利润
- ✅ 不需要高频操作（LP 开关有成本）

---

## 核心原理

### 套利逻辑（仅卖方示例）

#### 阶段 1: 开仓条件

```python
# 在 CEX 上获取最佳买入价（我们能以此价格买入 Token A）
taker_best_ask = get_cex_best_ask()  # 例如: 100 USDT

# 计算目标盈利卖价（我们需要在 DEX LP 卖出的价格）
profitability_sell_price = taker_best_ask * (1 + target_profitability)
# 例如: 100 * 1.02 = 102 USDT (目标 2% 利润)

# 开 LP 仓位条件：
# LP 价格区间下限 >= 目标盈利卖价
if lp_lower_price_bound >= profitability_sell_price:
    # ✅ 可以开仓
    # 在 DEX 上开 LP，卖出 Token A
    open_lp_position(
        token_a_amount=order_size,
        price_range=(lp_lower_price_bound, lp_upper_price_bound)
    )
```

**说明**:
- 我们在 DEX LP 中放入 Token A
- 如果 LP 被成交，我们会收到 Token B (USDT)
- 成交均价约为 `(lower_bound + upper_bound) / 2`
- 只要均价 >= 102，我们就能达到 2% 目标利润

---

#### 阶段 2: 持仓监控

```python
# 持续监控 CEX 价格
taker_best_ask = get_cex_best_ask()  # 实时更新

# 计算止损线（最低可接受卖价）
cutoff_sell_price = taker_best_ask * (1 + min_profitability)
# 例如: 100 * 1.005 = 100.5 USDT (最低 0.5% 利润)

# LP 预期成交均价
average_price_of_limit_order = (lp_lower_price_bound + lp_upper_price_bound) / 2

# 检查是否触发止损
if average_price_of_limit_order < cutoff_sell_price:
    # ❌ LP 如果现在成交，利润不足，需要关闭
    close_lp_position()
    # 原因：CEX 价格上涨了，或者 LP 区间不合适了
```

**说明**:
- 如果 CEX 价格大幅上涨，我们的 LP 卖价变得不够高
- 例如 CEX 涨到 105，则需要 LP 均价 >= 105.525 才能保证 0.5% 利润
- 如果当前 LP 区间是 102-104，均价 103，低于 105.525，触发止损

---

#### 阶段 3: LP 成交后

```python
# Event: LP 订单被成交（有人在 DEX 买走了我们的 Token A）
def on_lp_filled(event):
    # 1. 获取成交详情
    received_token_b = event.amount  # 收到的 USDT
    sold_token_a = event.base_amount  # 卖出的 Token A

    # 2. 立即关闭 LP 仓位（提取剩余流动性）
    close_lp_position()

    # 3. 在 CEX 对冲（买回 Token A）
    cex_buy_token_a(
        amount=sold_token_a,
        price=taker_best_ask  # 以当前最佳价买入
    )

    # 4. 计算实际利润
    # 收入: received_token_b (从 DEX LP 卖出得到)
    # 成本: sold_token_a * taker_best_ask (在 CEX 买回的成本)
    # 费用: lp_fees + cex_fees + gas
    profit = received_token_b - (sold_token_a * taker_best_ask) - fees
```

---

### 完整流程图

```
┌─────────────────────────────────────────────────────────┐
│                     初始状态                              │
│            持有 Token A，无 LP 仓位                        │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│                  On Tick 事件                             │
│  1. 获取 CEX 最佳买价 (taker_best_ask)                     │
│  2. 检查 LP 仓位状态                                       │
└────────────────────┬────────────────────────────────────┘
                     ↓
            ┌────────┴────────┐
            │  有 LP 仓位？   │
            └────────┬────────┘
                     │
         ┌───────────┴───────────┐
         │ Yes                   │ No
         ↓                       ↓
┌──────────────────┐    ┌──────────────────┐
│   持仓监控模式    │    │   开仓判断模式    │
└──────────────────┘    └──────────────────┘
         │                       │
         │                       ↓
         │              计算 profitability_sell_price
         │              = taker_best_ask * (1 + target_profit)
         │                       │
         │                       ↓
         │              检查: lp_lower_bound >= profitability_sell_price?
         │                       │
         │              ┌────────┴────────┐
         │              │ Yes             │ No
         │              ↓                 ↓
         │         开 LP 仓位          继续等待
         │              │
         │              └─────────┐
         ↓                       ↓
    计算 cutoff_sell_price      更新仓位信息
    = taker_best_ask * (1 + min_profit)
         │
         ↓
    计算 avg_lp_price
    = (lower_bound + upper_bound) / 2
         │
         ↓
    检查: avg_lp_price >= cutoff_sell_price?
         │
    ┌────┴────┐
    │ Yes     │ No
    ↓         ↓
 继续持有   关闭 LP
              │
              └────────→ 回到初始状态

┌─────────────────────────────────────────────────────────┐
│              LP 订单成交事件                              │
│  1. LP 被部分或全部成交                                    │
│  2. 关闭 LP 仓位                                          │
│  3. 在 CEX 对冲（买回 Token A）                            │
│  4. 计算并记录利润                                         │
└────────────────────┬────────────────────────────────────┘
                     ↓
              回到初始状态
```

---

## 架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│           CexDexLpArbitrageStrategy                      │
│         (V2 StrategyV2Base 框架)                         │
└────────────┬────────────────────────────────────────────┘
             │
             ├─→ MarketDataProvider
             │   ├─ CEX 价格数据 (Binance, OKX, etc.)
             │   └─ DEX 池子信息 (Uniswap, PancakeSwap, etc.)
             │
             ├─→ LpPositionManager
             │   ├─ 开仓逻辑
             │   ├─ 关仓逻辑
             │   ├─ 监控逻辑
             │   └─ 仓位信息跟踪
             │
             ├─→ CexHedgeExecutor
             │   ├─ CEX 对冲下单
             │   ├─ 订单状态跟踪
             │   └─ 成交确认
             │
             ├─→ ProfitabilityCalculator
             │   ├─ 开仓盈利计算
             │   ├─ 持仓监控计算
             │   ├─ 费用估算
             │   └─ 实际 PnL 统计
             │
             └─→ RiskController
                 ├─ 余额检查
                 ├─ 价格滑点保护
                 ├─ 止损逻辑
                 └─ 异常处理
```

---

### 类设计

#### 1. 主策略类

```python
class CexDexLpArbitrageStrategy(StrategyV2Base):
    """
    CEX-DEX LP 套利策略

    特点:
    - 在 DEX 上做 LP (被动成交)
    - 在 CEX 上对冲 (主动交易)
    - 赚取价差 + LP 手续费
    """

    def __init__(self, config: CexDexLpArbitrageConfig):
        # 初始化连接器
        self.cex_connector = connectors[config.cex_exchange]
        self.dex_connector = connectors[config.dex_exchange]

        # 初始化子模块
        self.lp_manager = LpPositionManager(...)
        self.hedge_executor = CexHedgeExecutor(...)
        self.profit_calculator = ProfitabilityCalculator(...)
        self.risk_controller = RiskController(...)

        # 状态跟踪
        self.lp_position_opened = False
        self.lp_position_info = None
        self.pending_hedge_order_id = None
```

---

#### 2. LP 仓位管理器

```python
class LpPositionManager:
    """
    管理 DEX LP 仓位

    职责:
    - 计算最优 LP 价格区间
    - 开仓/关仓操作
    - 监控 LP 状态
    - 处理 LP 成交事件
    """

    async def calculate_lp_range(
        self,
        center_price: Decimal,
        target_profitability: Decimal
    ) -> Tuple[Decimal, Decimal]:
        """计算 LP 价格区间"""

    async def open_lp_position(
        self,
        token_amount: Decimal,
        price_range: Tuple[Decimal, Decimal]
    ) -> str:
        """开 LP 仓位"""

    async def close_lp_position(self) -> str:
        """关 LP 仓位"""

    async def monitor_lp_position(self) -> bool:
        """监控 LP 仓位，返回是否需要关仓"""
```

---

#### 3. CEX 对冲执行器

```python
class CexHedgeExecutor:
    """
    在 CEX 上执行对冲交易

    职责:
    - LP 成交后立即对冲
    - 跟踪对冲订单状态
    - 处理失败重试
    """

    async def hedge_lp_fill(
        self,
        lp_fill_event: LpFillEvent
    ) -> str:
        """
        LP 成交后执行对冲

        例如:
        - LP 卖出了 1 BTC
        - 在 CEX 买入 1 BTC
        """

    async def cancel_hedge_if_needed(self, order_id: str):
        """取消对冲订单（如果 LP 未成交）"""
```

---

#### 4. 盈利计算器

```python
class ProfitabilityCalculator:
    """
    计算套利盈利性

    职责:
    - 开仓前计算预期利润
    - 持仓中计算实时盈亏
    - 成交后计算实际利润
    """

    def calculate_target_lp_price(
        self,
        cex_price: Decimal,
        target_profitability: Decimal,
        fees: Decimal
    ) -> Decimal:
        """
        计算目标 LP 价格

        公式:
        target_lp_price = cex_price * (1 + target_profitability + total_fees)
        """

    def calculate_min_lp_price(
        self,
        cex_price: Decimal,
        min_profitability: Decimal,
        fees: Decimal
    ) -> Decimal:
        """计算最低可接受 LP 价格（止损线）"""

    def calculate_actual_profit(
        self,
        lp_filled_amount: Decimal,
        lp_filled_price: Decimal,
        hedge_filled_price: Decimal,
        fees: Dict[str, Decimal]
    ) -> Decimal:
        """计算实际利润"""
```

---

## 详细流程

### 流程 1: 主循环 (on_tick)

```python
async def on_tick(self):
    """策略主循环"""

    # 1. 获取最新市场数据
    cex_best_ask = await self.get_cex_best_ask()
    cex_best_bid = await self.get_cex_best_bid()
    dex_pool_info = await self.get_dex_pool_info()

    # 2. 检查余额
    if not await self.risk_controller.check_sufficient_balance():
        self.logger().warning("余额不足，暂停策略")
        return

    # 3. 根据 LP 仓位状态执行不同逻辑
    if self.lp_position_opened:
        # 持仓监控模式
        await self._monitor_existing_position(cex_best_ask, cex_best_bid)
    else:
        # 寻找开仓机会
        await self._check_opening_opportunity(cex_best_ask, cex_best_bid, dex_pool_info)
```

---

### 流程 2: 开仓判断

```python
async def _check_opening_opportunity(
    self,
    cex_best_ask: Decimal,
    cex_best_bid: Decimal,
    dex_pool_info: PoolInfo
):
    """检查是否有开仓机会"""

    # 计算目标利润价格
    sell_side_target_price = self.profit_calculator.calculate_target_lp_price(
        cex_price=cex_best_ask,  # CEX 买入价
        target_profitability=self.config.target_profitability,
        fees=self._estimate_total_fees()
    )

    buy_side_target_price = self.profit_calculator.calculate_target_lp_price(
        cex_price=cex_best_bid,  # CEX 卖出价
        target_profitability=self.config.target_profitability,
        fees=self._estimate_total_fees()
    )

    # 检查卖方机会
    # 我们在 DEX LP 卖出 Token A，在 CEX 买入 Token A
    if self._check_sell_side_opportunity(sell_side_target_price, dex_pool_info):
        await self._open_sell_side_position(sell_side_target_price)
        return

    # 检查买方机会
    # 我们在 DEX LP 买入 Token A，在 CEX 卖出 Token A
    if self._check_buy_side_opportunity(buy_side_target_price, dex_pool_info):
        await self._open_buy_side_position(buy_side_target_price)
        return
```

---

### 流程 3: 开 LP 仓位

```python
async def _open_sell_side_position(self, target_price: Decimal):
    """
    开卖方 LP 仓位

    步骤:
    1. 计算 LP 价格区间
    2. 在 DEX 上开 LP（放入 Token A）
    3. 记录仓位信息
    """

    # 1. 计算 LP 区间
    # 让 LP 区间下限 = 目标价格
    # 上限 = 下限 * (1 + spread_pct)
    lower_bound = target_price
    upper_bound = target_price * (1 + self.config.lp_spread_pct)

    self.logger().info(
        f"开卖方 LP 仓位:\n"
        f"   Token A 数量: {self.config.lp_token_amount}\n"
        f"   价格区间: {lower_bound} - {upper_bound}\n"
        f"   预期均价: {(lower_bound + upper_bound) / 2}\n"
        f"   CEX 对冲价: {await self.get_cex_best_ask()}"
    )

    # 2. 开 LP
    lp_order_id = await self.lp_manager.open_lp_position(
        token_amount=self.config.lp_token_amount,
        price_range=(lower_bound, upper_bound),
        is_buy=False  # 卖方 LP
    )

    # 3. 记录信息
    self.lp_position_opened = True
    self.lp_position_info = {
        "order_id": lp_order_id,
        "side": "SELL",
        "token_amount": self.config.lp_token_amount,
        "price_range": (lower_bound, upper_bound),
        "open_time": time.time(),
        "open_cex_price": await self.get_cex_best_ask(),
    }
```

---

### 流程 4: 持仓监控

```python
async def _monitor_existing_position(
    self,
    cex_best_ask: Decimal,
    cex_best_bid: Decimal
):
    """监控现有 LP 仓位"""

    if not self.lp_position_info:
        return

    side = self.lp_position_info["side"]
    lower_bound, upper_bound = self.lp_position_info["price_range"]
    avg_lp_price = (lower_bound + upper_bound) / 2

    # 计算止损价格
    if side == "SELL":
        # 卖方: LP 卖出 Token A，CEX 买入 Token A
        current_cex_price = cex_best_ask
        cutoff_price = self.profit_calculator.calculate_min_lp_price(
            cex_price=current_cex_price,
            min_profitability=self.config.min_profitability,
            fees=self._estimate_total_fees()
        )

        # 检查止损
        if avg_lp_price < cutoff_price:
            self.logger().warning(
                f"触发止损:\n"
                f"   LP 均价: {avg_lp_price}\n"
                f"   止损线: {cutoff_price}\n"
                f"   CEX 价格: {current_cex_price}\n"
                f"   原因: CEX 价格上涨，LP 卖价不够高"
            )
            await self._close_lp_position("STOP_LOSS")

    else:  # "BUY"
        # 买方: LP 买入 Token A，CEX 卖出 Token A
        current_cex_price = cex_best_bid
        cutoff_price = self.profit_calculator.calculate_min_lp_price(
            cex_price=current_cex_price,
            min_profitability=self.config.min_profitability,
            fees=self._estimate_total_fees()
        )

        if avg_lp_price > cutoff_price:
            self.logger().warning(
                f"触发止损:\n"
                f"   LP 均价: {avg_lp_price}\n"
                f"   止损线: {cutoff_price}\n"
                f"   CEX 价格: {current_cex_price}\n"
                f"   原因: CEX 价格下跌，LP 买价太高"
            )
            await self._close_lp_position("STOP_LOSS")

    # 检查超时
    elapsed = time.time() - self.lp_position_info["open_time"]
    if elapsed > self.config.lp_timeout_seconds:
        self.logger().info(f"LP 仓位超时 ({elapsed}秒)，关闭")
        await self._close_lp_position("TIMEOUT")
```

---

### 流程 5: LP 成交处理

```python
async def on_lp_filled(self, event: LpFillEvent):
    """
    LP 订单成交事件

    关键步骤:
    1. 立即关闭 LP（提取剩余流动性）
    2. 在 CEX 对冲（锁定利润）
    3. 计算实际利润
    """

    if not self.lp_position_info:
        return

    side = self.lp_position_info["side"]
    filled_amount = event.amount
    filled_price = event.price

    self.logger().info(
        f"LP 订单成交:\n"
        f"   方向: {side}\n"
        f"   数量: {filled_amount}\n"
        f"   价格: {filled_price}"
    )

    # 1. 关闭 LP（如果还有剩余流动性）
    await self._close_lp_position("FILLED")

    # 2. 在 CEX 对冲
    if side == "SELL":
        # LP 卖出了 Token A，需要在 CEX 买回
        hedge_order_id = await self.hedge_executor.hedge_lp_fill(
            is_buy=True,
            amount=filled_amount,
            reason=f"对冲 LP 卖单，LP 价格: {filled_price}"
        )
    else:  # "BUY"
        # LP 买入了 Token A，需要在 CEX 卖出
        hedge_order_id = await self.hedge_executor.hedge_lp_fill(
            is_buy=False,
            amount=filled_amount,
            reason=f"对冲 LP 买单，LP 价格: {filled_price}"
        )

    self.pending_hedge_order_id = hedge_order_id

    # 3. 等待对冲成交后计算利润（在 on_hedge_filled 中）
```

---

### 流程 6: 对冲成交处理

```python
async def on_hedge_filled(self, event: OrderFilledEvent):
    """
    CEX 对冲订单成交

    计算并记录实际利润
    """

    if event.order_id != self.pending_hedge_order_id:
        return

    hedge_price = event.price
    hedge_amount = event.amount

    # 获取 LP 成交信息（从历史记录中）
    lp_fill_info = self.lp_position_info.get("last_fill")
    if not lp_fill_info:
        self.logger().error("找不到 LP 成交信息")
        return

    lp_price = lp_fill_info["price"]
    lp_amount = lp_fill_info["amount"]
    side = self.lp_position_info["side"]

    # 计算利润
    if side == "SELL":
        # LP 卖出，CEX 买入
        revenue = lp_amount * lp_price  # LP 收到的 USDT
        cost = hedge_amount * hedge_price  # CEX 买入成本
        lp_fee_earned = lp_fill_info.get("fee_earned", 0)

        gross_profit = revenue - cost + lp_fee_earned

        # 扣除费用
        cex_fee = event.trade_fee.amount
        gas_cost = self._estimate_gas_cost()

        net_profit = gross_profit - cex_fee - gas_cost
        profit_pct = (net_profit / cost) * 100 if cost > 0 else 0

    else:  # "BUY"
        # LP 买入，CEX 卖出
        cost = lp_amount * lp_price  # LP 买入成本
        revenue = hedge_amount * hedge_price  # CEX 卖出收入
        lp_fee_earned = lp_fill_info.get("fee_earned", 0)

        gross_profit = revenue - cost + lp_fee_earned

        cex_fee = event.trade_fee.amount
        gas_cost = self._estimate_gas_cost()

        net_profit = gross_profit - cex_fee - gas_cost
        profit_pct = (net_profit / cost) * 100 if cost > 0 else 0

    # 记录统计
    self.stats["total_profit"] += net_profit
    self.stats["completed_cycles"] += 1

    self.logger().info(
        f"套利完成:\n"
        f"   LP {side}: {lp_amount} @ {lp_price}\n"
        f"   CEX {'BUY' if side == 'SELL' else 'SELL'}: {hedge_amount} @ {hedge_price}\n"
        f"   毛利: {gross_profit:.4f}\n"
        f"   CEX 手续费: {cex_fee:.4f}\n"
        f"   Gas 成本: {gas_cost:.4f}\n"
        f"   净利润: {net_profit:.4f} ({profit_pct:.2f}%)\n"
        f"   累计利润: {self.stats['total_profit']:.4f}"
    )

    # 清理状态
    self.lp_position_opened = False
    self.lp_position_info = None
    self.pending_hedge_order_id = None
```

---

## 配置参数

### 配置类

```python
class CexDexLpArbitrageConfig(BaseClientModel):
    """CEX-DEX LP 套利配置"""

    # ========== 交易所配置 ==========
    cex_exchange: str = Field(
        "binance",
        description="CEX 交易所（用于对冲）"
    )

    dex_exchange: str = Field(
        "uniswap/clmm",
        description="DEX 交易所（用于 LP）"
    )

    trading_pair: str = Field(
        "WETH-USDC",
        description="交易对"
    )

    # ========== LP 配置 ==========
    lp_token_amount: Decimal = Field(
        Decimal("0.1"),
        description="LP 单边 Token 数量"
    )

    lp_spread_pct: Decimal = Field(
        Decimal("0.01"),
        description="LP 价格区间宽度（百分比）"
    )

    lp_timeout_seconds: int = Field(
        300,
        description="LP 最长持有时间（秒），超时强制关闭"
    )

    # ========== 盈利目标 ==========
    target_profitability: Decimal = Field(
        Decimal("0.02"),
        description="目标利润率（2%）"
    )

    min_profitability: Decimal = Field(
        Decimal("0.005"),
        description="最低利润率（0.5%），低于此触发止损"
    )

    # ========== 费用估算 ==========
    cex_taker_fee_pct: Decimal = Field(
        Decimal("0.001"),
        description="CEX Taker 手续费率（0.1%）"
    )

    dex_lp_fee_pct: Decimal = Field(
        Decimal("0.003"),
        description="DEX LP 手续费率（0.3%），这是我们能赚的"
    )

    gas_cost_usdt: Decimal = Field(
        Decimal("5"),
        description="预估 Gas 成本（USDT）"
    )

    # ========== 风控配置 ==========
    max_position_size: Decimal = Field(
        Decimal("1"),
        description="最大单笔交易量"
    )

    min_balance_reserve_pct: Decimal = Field(
        Decimal("0.1"),
        description="最低余额保留比例（10%）"
    )

    # ========== 策略配置 ==========
    enable_buy_side: bool = Field(
        True,
        description="启用买方套利（DEX LP 买入，CEX 卖出）"
    )

    enable_sell_side: bool = Field(
        True,
        description="启用卖方套利（DEX LP 卖出，CEX 买入）"
    )

    check_interval_seconds: int = Field(
        5,
        description="主循环检查间隔（秒）"
    )
```

---

## 关键组件实现

### 费用计算

```python
def _estimate_total_fees(self) -> Decimal:
    """
    估算总费用

    包括:
    - CEX Taker 手续费
    - DEX Gas 费用
    - 价格滑点预留
    """

    # CEX 手续费（按交易额计算）
    cex_fee_pct = self.config.cex_taker_fee_pct

    # Gas 成本（转换为百分比）
    # 假设交易 10000 USDT，gas 5 USDT，则 gas_pct = 0.05%
    trade_value = self.config.lp_token_amount * self.current_price
    gas_pct = self.config.gas_cost_usdt / trade_value if trade_value > 0 else 0

    # 滑点预留（1%）
    slippage_pct = Decimal("0.01")

    # 总费用
    total_fees = cex_fee_pct + gas_pct + slippage_pct

    return total_fees
```

---

### LP 价格区间计算

```python
def _calculate_lp_range_for_sell_side(
    self,
    cex_buy_price: Decimal,
    target_profitability: Decimal
) -> Tuple[Decimal, Decimal]:
    """
    计算卖方 LP 价格区间

    目标: LP 卖出价格 > CEX 买入价格 * (1 + 目标利润 + 费用)

    返回: (lower_bound, upper_bound)
    """

    # 计算目标价格
    total_fees = self._estimate_total_fees()
    target_price = cex_buy_price * (1 + target_profitability + total_fees)

    # LP 区间
    # 让下限 = 目标价格（保守）
    lower_bound = target_price

    # 上限 = 下限 * (1 + spread)
    upper_bound = lower_bound * (1 + self.config.lp_spread_pct)

    return (lower_bound, upper_bound)
```

---

### 开仓机会判断

```python
def _check_sell_side_opportunity(
    self,
    target_price: Decimal,
    dex_pool_info: PoolInfo
) -> bool:
    """
    检查是否有卖方套利机会

    条件:
    1. 当前 DEX 池子价格 >= target_price (市场愿意付高价)
    2. 池子流动性充足
    3. 我们有足够的 Token A
    """

    current_dex_price = Decimal(str(dex_pool_info.price))

    # 检查价格
    if current_dex_price < target_price:
        self.logger().debug(
            f"DEX 价格 ({current_dex_price}) < 目标价格 ({target_price})，"
            f"暂无卖方机会"
        )
        return False

    # 检查流动性
    pool_liquidity = dex_pool_info.total_liquidity
    min_liquidity = self.config.lp_token_amount * current_dex_price * 10

    if pool_liquidity < min_liquidity:
        self.logger().debug(f"池子流动性不足: {pool_liquidity} < {min_liquidity}")
        return False

    # 检查余额
    token_a = self.trading_pair.split("-")[0]
    available_balance = self.cex_connector.get_available_balance(token_a)

    if available_balance < self.config.lp_token_amount:
        self.logger().warning(f"Token A 余额不足: {available_balance} < {self.config.lp_token_amount}")
        return False

    return True
```

---

## 风险控制

### 1. 余额检查

```python
async def check_sufficient_balance(self) -> bool:
    """
    检查余额是否充足

    要求:
    - Token A: 足够开 LP + CEX 对冲
    - Token B (USDT): 足够支付 gas + CEX 手续费
    """

    token_a, token_b = self.trading_pair.split("-")

    # CEX 余额
    cex_token_a = self.cex_connector.get_available_balance(token_a)
    cex_token_b = self.cex_connector.get_available_balance(token_b)

    # DEX 余额
    dex_token_a = self.dex_connector.get_available_balance(token_a)
    dex_token_b = self.dex_connector.get_available_balance(token_b)

    # 检查 Token A（需要足够开 LP）
    required_token_a = self.config.lp_token_amount
    if dex_token_a < required_token_a:
        self.logger().warning(
            f"DEX {token_a} 余额不足: {dex_token_a} < {required_token_a}"
        )
        return False

    # 检查 Token B（需要足够支付 gas 和对冲）
    estimated_gas = self.config.gas_cost_usdt
    estimated_hedge_cost = required_token_a * self.current_price * 1.1  # 预留 10%
    required_token_b = estimated_gas + estimated_hedge_cost

    if cex_token_b < required_token_b:
        self.logger().warning(
            f"CEX {token_b} 余额不足: {cex_token_b} < {required_token_b}"
        )
        return False

    return True
```

---

### 2. 滑点保护

```python
def _apply_slippage_protection(
    self,
    price: Decimal,
    is_buy: bool,
    slippage_pct: Decimal = Decimal("0.01")
) -> Decimal:
    """
    应用滑点保护

    买入: 最高愿意支付的价格 = price * (1 + slippage)
    卖出: 最低愿意接受的价格 = price * (1 - slippage)
    """

    if is_buy:
        return price * (1 + slippage_pct)
    else:
        return price * (1 - slippage_pct)
```

---

### 3. 异常处理

```python
async def _handle_lp_open_failure(self, error: Exception):
    """处理 LP 开仓失败"""

    self.logger().error(f"LP 开仓失败: {error}")

    # 重置状态
    self.lp_position_opened = False
    self.lp_position_info = None

    # 记录失败统计
    self.stats["lp_open_failures"] += 1

    # 如果连续失败，暂停策略
    if self.stats["lp_open_failures"] >= 3:
        self.logger().error("LP 开仓连续失败 3 次，暂停策略")
        self.stop()

async def _handle_hedge_failure(self, error: Exception):
    """处理对冲失败"""

    self.logger().error(f"CEX 对冲失败: {error}")

    # 这是严重问题，因为 LP 已经成交但对冲失败
    # 我们处于裸露风险中

    # 重试对冲
    retry_count = 0
    max_retries = 5

    while retry_count < max_retries:
        try:
            self.logger().info(f"重试对冲 ({retry_count + 1}/{max_retries})")
            await asyncio.sleep(2)
            await self._execute_hedge()
            self.logger().info("对冲重试成功")
            return
        except Exception as e:
            retry_count += 1
            self.logger().error(f"对冲重试失败: {e}")

    # 所有重试都失败
    self.logger().error("对冲重试全部失败，请手动处理风险！")
    self.notify_user("对冲失败，存在裸露风险，请立即手动处理！")
    self.stop()
```

---

## 代码结构

### 文件组织

```
hummingbot_files/
├── scripts/
│   └── cex_dex_lp_arbitrage.py           # 主策略文件
│
├── conf/scripts/
│   └── cex_dex_lp_arbitrage.yml          # 配置文件
│
└── lib/
    └── cex_dex_lp_arbitrage/
        ├── __init__.py
        ├── lp_position_manager.py        # LP 仓位管理
        ├── cex_hedge_executor.py         # CEX 对冲执行
        ├── profitability_calculator.py   # 盈利计算
        └── risk_controller.py            # 风险控制
```

---

## 测试方案

### 单元测试

```python
# test_profitability_calculator.py

def test_target_price_calculation():
    """测试目标价格计算"""

    calc = ProfitabilityCalculator(config)

    # 场景: CEX 买入价 100，目标利润 2%，费用 1%
    target_price = calc.calculate_target_lp_price(
        cex_price=Decimal("100"),
        target_profitability=Decimal("0.02"),
        fees=Decimal("0.01")
    )

    # 预期: 100 * (1 + 0.02 + 0.01) = 103
    assert target_price == Decimal("103")

def test_min_price_calculation():
    """测试最低价格计算"""

    calc = ProfitabilityCalculator(config)

    # 场景: CEX 买入价 100，最低利润 0.5%，费用 1%
    min_price = calc.calculate_min_lp_price(
        cex_price=Decimal("100"),
        min_profitability=Decimal("0.005"),
        fees=Decimal("0.01")
    )

    # 预期: 100 * (1 + 0.005 + 0.01) = 101.5
    assert min_price == Decimal("101.5")
```

---

### 集成测试

```python
# test_arbitrage_flow.py

async def test_sell_side_arbitrage_flow():
    """测试完整卖方套利流程"""

    strategy = CexDexLpArbitrageStrategy(config)

    # 模拟市场数据
    mock_cex_best_ask = Decimal("100")  # CEX 买入价
    mock_dex_pool_price = Decimal("105")  # DEX 价格更高，有套利机会

    # 1. 检查开仓机会
    opportunity = await strategy._check_sell_side_opportunity(...)
    assert opportunity is True

    # 2. 开 LP 仓位
    await strategy._open_sell_side_position(...)
    assert strategy.lp_position_opened is True

    # 3. 模拟 LP 成交
    lp_fill_event = LpFillEvent(
        order_id=strategy.lp_position_info["order_id"],
        amount=Decimal("1"),
        price=Decimal("104")  # LP 成交价
    )
    await strategy.on_lp_filled(lp_fill_event)

    # 4. 检查对冲订单已创建
    assert strategy.pending_hedge_order_id is not None

    # 5. 模拟对冲成交
    hedge_fill_event = OrderFilledEvent(
        order_id=strategy.pending_hedge_order_id,
        price=Decimal("100.5"),  # CEX 买入价
        amount=Decimal("1")
    )
    await strategy.on_hedge_filled(hedge_fill_event)

    # 6. 检查利润计算
    # LP 卖出: 1 @ 104 = 104 USDT
    # CEX 买入: 1 @ 100.5 = 100.5 USDT
    # 毛利: 3.5 USDT (扣除费用后应该 > 2 USDT)
    assert strategy.stats["total_profit"] > Decimal("2")
```

---

## 总结

### 策略优势

✅ **被动做市**: DEX LP 被动等待成交，无需主动追价
✅ **对冲锁定**: CEX 立即对冲，锁定利润，降低风险
✅ **赚取双重收益**: 价差 + LP 手续费
✅ **风险可控**: 有止损机制，超时保护

---

### 策略劣势

❌ **LP 开关成本**: 每次开关 LP 需要 gas，频繁操作不划算
❌ **LP 可能不成交**: 如果价格不到 LP 区间，一直不成交
❌ **对冲时机风险**: LP 成交到 CEX 对冲之间有时间差，价格可能变化
❌ **流动性要求**: DEX 和 CEX 都需要足够流动性

---

### 适用场景

- ✅ **中长期套利**: 单次持仓 5-60 分钟
- ✅ **稳定币对**: USDC-USDT 这类价差小但稳定的
- ✅ **主流币种**: ETH, BTC 等流动性好的
- ❌ **不适合高频**: LP 开关成本高
- ❌ **不适合波动大的币**: 价格变化太快，止损频繁

---

### 下一步

1. **实现核心代码**: 按照架构实现主策略类
2. **单元测试**: 测试盈利计算、LP 区间计算等
3. **回测**: 使用历史数据验证策略有效性
4. **纸盘测试**: 模拟环境测试完整流程
5. **小额实盘**: 从小资金开始实盘测试
6. **优化迭代**: 根据实盘结果优化参数

---

**设计日期**: 2025-10-30
**设计状态**: ✅ 完成
**参考示例**: `lp_manage_position.py`, `arbitrage_controller.py`
