# 🐛 V2 Hybrid 价格调整一致性修复

## 问题描述

**日期**: 2025-10-30

**问题**: 监控价格与入场价格不一致，导致 PnL 计算错误

### 症状

```
入场价: 32672.043006235965193 TOKEN/WBNB
监控价: 45117.622519 TOKEN/WBNB
PnL: 38.09%
```

**异常表现**:
- 买入后立即显示 38% 利润
- 但实际价格没有上涨
- 价格差异约为 38% (45117 / 32672 = 1.38)

---

## 根本原因

### 价格计算不一致

#### 入场价计算 (lines 383-398)

```python
# 1. 获取原始报价
entry_price = await self.connector.get_quote_price(
    trading_pair=trading_pair,
    is_buy=is_buy,
    amount=base_amount
)
# 结果: 44200 TOKEN/WBNB (原始市场价)

# 2. 应用调整（滑点 + gas buffer）
if is_buy:
    entry_price = entry_price / ((Decimal("1") + slippage) * self.config.gas_buffer)
    # = 44200 / (1.02 * 1.15)
    # = 44200 / 1.173
    # = 37679 TOKEN/WBNB ✅ 调整后的价格
```

#### 监控价格计算 (原始代码 lines 561-565)

```python
# ❌ 错误：只获取原始价格，没有应用调整
current_price = await self.connector.get_quote_price(
    trading_pair=trading_pair,
    is_buy=is_buy,
    amount=position["fill_amount"]
)
# 结果: 44300 TOKEN/WBNB (原始市场价，未调整)
```

### 问题所在

**入场价和监控价使用了不同的计算方法**:

| 阶段 | 原始价格 | 调整系数 | 最终价格 | 说明 |
|------|---------|---------|---------|------|
| 入场 | 44200 | ÷ 1.173 | **37679** | 应用了调整 ✅ |
| 监控 | 44300 | 无 | **44300** | 未应用调整 ❌ |

**PnL 计算 (错误)**:
```python
pnl = (44300 - 37679) / 37679 = 17.6%  # 错误的 17.6% 利润！
```

实际上价格只涨了:
```python
真实涨幅 = (44300 - 44200) / 44200 = 0.23%  # 只涨了 0.23%
```

---

## 解决方案

### 修复: 监控价格应用相同的调整

```python
# ✅ 正确：获取原始价格后，应用与入场价相同的调整

# 1. 获取原始报价
current_price_raw = await self.connector.get_quote_price(
    trading_pair=trading_pair,
    is_buy=is_buy,
    amount=position["fill_amount"]
)

if not current_price_raw or current_price_raw <= 0:
    self.logger().warning(f"⚠️  无法获取 {trading_pair} 当前价格，等待下次检查")
    await asyncio.sleep(1)
    continue

# 2. 应用与入场价相同的调整（滑点 + gas buffer）
slippage = Decimal(str(self.config.slippage))
if is_buy:
    current_price = current_price_raw / ((Decimal("1") + slippage) * self.config.gas_buffer)
else:
    current_price = current_price_raw * ((Decimal("1") + slippage) * self.config.gas_buffer)

# 现在 current_price 和 entry_price 使用相同的计算方法 ✅
```

---

## 详细分析

### 为什么需要价格调整？

在 AMM DEX 上交易时，我们需要考虑：

1. **滑点 (Slippage)**: 2%
   - 价格会因为交易量而变化
   - 例如：买入会推高价格

2. **Gas Buffer**: 1.15 (15%)
   - Gas 费用的预留
   - 防止交易失败

**总调整系数**: 1.02 × 1.15 = **1.173** (17.3%)

### 调整的方向

#### 买入 (BUY)

```python
# 我们需要更多的 WBNB 来买相同数量的 TOKEN
adjusted_price = market_price / 1.173

例如:
- 市场价: 44200 TOKEN/WBNB
- 调整后: 37679 TOKEN/WBNB
- 含义: 我们用 1 WBNB 可以买到 37679 个 TOKEN (更少)
```

#### 卖出 (SELL)

```python
# 我们会收到更少的 WBNB
adjusted_price = market_price * 1.173

例如:
- 市场价: 44200 TOKEN/WBNB
- 调整后: 51850 TOKEN/WBNB
- 含义: 我们卖出 51850 个 TOKEN 才能得到 1 WBNB (需要更多)
```

---

## 测试验证

### 场景 1: 价格上涨 0.5%

```python
# 入场
market_price_entry = 44200 TOKEN/WBNB
entry_price = 44200 / 1.173 = 37679 TOKEN/WBNB

# 3秒后，市场价上涨 0.5%
market_price_now = 44200 * 1.005 = 44421 TOKEN/WBNB
current_price = 44421 / 1.173 = 37867 TOKEN/WBNB

# PnL 计算 (正确)
pnl = (37867 - 37679) / 37679 = +0.50% ✅

# 日志输出
📊 持仓状态: buy-TOKEN-WBNB-xxx | PnL: +0.50% | Price: 37867.0
```

---

### 场景 2: 价格下跌 5%

```python
# 入场
entry_price = 37679 TOKEN/WBNB

# 5秒后，市场价下跌 5%
market_price_now = 44200 * 0.95 = 41990 TOKEN/WBNB
current_price = 41990 / 1.173 = 35795 TOKEN/WBNB

# PnL 计算 (正确)
pnl = (35795 - 37679) / 37679 = -5.00% ✅

# 日志输出
📊 持仓状态: buy-TOKEN-WBNB-xxx | PnL: -5.00% | Price: 35795.0
```

---

### 场景 3: 修复前 vs 修复后

#### 修复前 (错误)

```python
入场价: 37679 TOKEN/WBNB (调整后)
监控价: 44300 TOKEN/WBNB (原始，未调整)
PnL: (44300 - 37679) / 37679 = +17.6% ❌ 错误！

# 实际市场只涨了 0.23%，但显示涨了 17.6%
```

#### 修复后 (正确)

```python
入场价: 37679 TOKEN/WBNB (调整后)
监控价: 44300 / 1.173 = 37764 TOKEN/WBNB (调整后)
PnL: (37764 - 37679) / 37679 = +0.23% ✅ 正确！

# 正确反映市场涨幅
```

---

## 为什么这个 Bug 很隐蔽？

### 原因 1: 数值看起来合理

```python
入场价: 37679
监控价: 44300
PnL: 17.6%
```

- 如果不知道调整系数，这些数字看起来没问题
- 17.6% 的利润在波动大的市场中似乎可能

### 原因 2: 方向是对的

```python
# 价格确实在上涨，只是幅度被放大了
真实涨幅: +0.23%
显示涨幅: +17.6%  # 放大了 76 倍！
```

### 原因 3: 代码逻辑看起来一致

```python
# 两处都是:
current_price = await self.connector.get_quote_price(
    trading_pair=trading_pair,
    is_buy=is_buy,  # ✅ 方向一致
    amount=xxx
)
```

- 表面上看，代码结构一致
- 但缺少了调整步骤

---

## 关键教训

### 1. 价格计算的一致性原则

**黄金法则**:
> 入场价和监控价必须使用**完全相同**的计算方法

这包括:
- ✅ 相同的调整系数
- ✅ 相同的方向 (is_buy)
- ✅ 相同的单位转换

### 2. AMM DEX 的特殊性

CEX (中心化交易所):
- 订单簿模式
- 价格相对稳定
- 滑点可预测

AMM DEX (自动做市商):
- 流动性池模式
- 价格依赖于交易量
- 需要预留滑点和 gas

### 3. 测试的重要性

这个 Bug 在以下情况下会被发现:
- ✅ 价格没变化时，PnL 应该接近 0%
- ✅ 价格小幅变化时，PnL 应该对应
- ❌ 但如果只测试大幅波动，可能会忽略

---

## 相关修改

### 文件

`/hummingbot_files/scripts/v2_news_sniping_hybrid.py`

### 修改位置 (lines 559-578)

```python
# 修改前
current_price = await self.connector.get_quote_price(...)

# 修改后
current_price_raw = await self.connector.get_quote_price(...)
# ... 检查 ...
slippage = Decimal(str(self.config.slippage))
if is_buy:
    current_price = current_price_raw / ((Decimal("1") + slippage) * self.config.gas_buffer)
else:
    current_price = current_price_raw * ((Decimal("1") + slippage) * self.config.gas_buffer)
```

---

## 同步修复: Binance Oracle 警告

在 `__init__` 方法中添加 (lines 152-157):

```python
# ========== 抑制 Binance Oracle 警告 ==========
import logging
performance_logger = logging.getLogger("hummingbot.client.performance")
performance_logger.setLevel(logging.ERROR)
```

**原因**: 对于链上代币，Binance 没有汇率数据，会产生频繁的 WARNING

---

## 测试建议

### 1. 零变化测试

```bash
# 1. 发送信号
python scripts/utility/test_news_signal_sender.py --side BUY --base TOKEN --amount 0.001

# 2. 立即检查 PnL（价格未变）
# 预期: PnL 应该在 -0.5% 到 +0.5% 之间
# 实际: 如果显示 17% 或更高，说明仍有问题
```

### 2. 小幅波动测试

```bash
# 观察监控日志，如果价格波动 1%
# PnL 应该显示约 1%，而不是 18% 或 16%
```

### 3. 完整流程验证

```bash
# 1. 启动策略
start --script v2_news_sniping_hybrid

# 2. 发送买入信号
python scripts/utility/test_news_signal_sender.py --side BUY --base TOKEN --amount 0.001

# 3. 观察日志
# ✅ 入场价: 37679 TOKEN/WBNB (调整后)
# ✅ 监控价: 37700 TOKEN/WBNB (也是调整后)
# ✅ PnL: +0.06% (合理)
# ✅ 无 Binance Oracle 警告
```

---

## 总结

### 问题原因

**价格调整不一致**: 入场价应用了调整系数 (÷ 1.173)，但监控价没有应用

### 修复方法

**监控价格应用相同调整**:
```python
current_price = current_price_raw / adjustment_factor  # for BUY
```

### 效果

- ✅ PnL 计算准确
- ✅ 止盈止损触发正确
- ✅ 价格单位一致
- ✅ 日志清晰

---

**修复日期**: 2025-10-30
**影响范围**: PnL 计算、止盈止损触发
**严重程度**: 🔴 高 (影响核心交易逻辑)
**修复状态**: ✅ 已完成
