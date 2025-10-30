# 📘 CEX-DEX LP 套利策略使用指南

## 目录

1. [快速开始](#快速开始)
2. [策略原理](#策略原理)
3. [配置说明](#配置说明)
4. [使用步骤](#使用步骤)
5. [监控和调试](#监控和调试)
6. [常见问题](#常见问题)
7. [优化建议](#优化建议)

---

## 快速开始

### 5 分钟上手

```bash
# 1. 启动 Hummingbot
cd /path/to/hummingbot
./start

# 2. 在 Hummingbot 中启动策略
start --script cex_dex_lp_arbitrage

# 3. 查看状态
status

# 4. 停止策略
stop
```

---

## 策略原理

### 核心思想

**在 DEX 上被动做 LP Maker，在 CEX 上主动做 Taker，赚取价差 + LP 手续费**

### 工作流程

```
┌─────────────────────────────────────────────┐
│  1. 监控 CEX 价格和 DEX 池子                 │
│     每 10 秒检查一次                          │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│  2. 发现套利机会                             │
│     条件: DEX 价格 > CEX 价格 + 费用 + 2%    │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│  3. 在 DEX 开 LP 仓位                        │
│     放入 0.1 WETH，价格区间: 2040-2060       │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│  4. 等待 LP 被成交（被动）                   │
│     有人在 DEX 买走我们的 WETH               │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│  5. LP 成交后立即在 CEX 对冲                 │
│     在 Binance 买回 0.1 WETH                │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│  6. 计算利润                                 │
│     利润 = (DEX 卖价 - CEX 买价) + LP 费     │
└─────────────────────────────────────────────┘
```

### 卖方套利示例

```
初始状态:
- 持有: 0.1 WETH
- CEX (Binance) 价格: 2000 USDC
- DEX (Uniswap) 价格: 2050 USDC

第 1 步: 计算目标价格
target_price = 2000 * (1 + 0.02 + 0.015) = 2070 USDC
说明: 需要至少 2070 USDC 才能达到 2% 利润（扣除 1.5% 费用）

第 2 步: 检查机会
DEX 价格 2050 < target_price 2070
结果: 暂无机会，继续等待

--- 5 分钟后，DEX 价格上涨 ---

DEX 价格: 2080 USDC
DEX 价格 2080 > target_price 2070 ✅
结果: 有套利机会！

第 3 步: 开 LP 仓位
在 Uniswap 开 LP，价格区间: 2070-2090
放入: 0.1 WETH
预期卖价: 2080 USDC

第 4 步: 等待成交
10 分钟后，有人以 2085 USDC 买走了我们的 WETH

第 5 步: CEX 对冲
立即在 Binance 买入 0.1 WETH @ 2010 USDC

第 6 步: 计算利润
收入:
  - DEX 卖出: 0.1 WETH @ 2085 = 208.5 USDC
  - LP 手续费: 0.1 * 0.003 * 2085 = 0.63 USDC
  - 总收入: 209.13 USDC

成本:
  - CEX 买入: 0.1 WETH @ 2010 = 201 USDC
  - CEX 手续费: 201 * 0.001 = 0.20 USDC
  - Gas 费: 5 USDC
  - 总成本: 206.20 USDC

净利润: 209.13 - 206.20 = 2.93 USDC (1.42%)
```

---

## 配置说明

### 配置文件位置

```
hummingbot_files/conf/scripts/cex_dex_lp_arbitrage.yml
```

### 关键参数

#### 1. 交易所配置

```yaml
# CEX 交易所（对冲）
cex_exchange: binance

# DEX 交易所（LP）
dex_exchange: uniswap/clmm

# 交易对
trading_pair: WETH-USDC
```

**注意事项**:
- CEX 和 DEX 必须支持相同交易对
- DEX 通常使用 WETH 而不是 ETH
- 检查 DEX 格式: `name/type`（如 `uniswap/clmm`, `pancakeswap/amm`）

---

#### 2. LP 参数

```yaml
# LP Token 数量
lp_token_amount: 0.1

# LP 价格区间宽度
lp_spread_pct: 0.01  # 1%

# LP 超时时间
lp_timeout_seconds: 300  # 5 分钟
```

**说明**:
- `lp_token_amount`: 卖方放 Base Token，买方放 Quote Token
- `lp_spread_pct`: 区间越窄越快成交，但价格优势越小
- `lp_timeout_seconds`: 超时强制关闭，防止资金占用

**建议**:
- 新手: `lp_spread_pct: 0.01` (1%)
- 激进: `lp_spread_pct: 0.005` (0.5%)
- 保守: `lp_spread_pct: 0.02` (2%)

---

#### 3. 盈利目标

```yaml
# 目标利润率
target_profitability: 0.02  # 2%

# 最低利润率（止损）
min_profitability: 0.005  # 0.5%
```

**说明**:
- `target_profitability`: 开仓门槛，预期利润必须 > 此值
- `min_profitability`: 持仓中止损线，低于此值关仓

**示例**:
```
开仓: 预期 2% 利润，开 LP
持仓中: CEX 价格上涨，预期利润降到 0.4% < 0.5% → 触发止损
```

---

#### 4. 费用估算

```yaml
cex_taker_fee_pct: 0.001  # 0.1%
dex_lp_fee_pct: 0.003     # 0.3%
gas_cost_quote: 5         # 5 USDC
```

**不同链的 Gas 成本**:
| 链 | Gas 成本 (USDC) | 备注 |
|----|-----------------|------|
| Ethereum Mainnet | 5-20 | Gas 高，适合大额 |
| Polygon | 0.1-1 | Gas 低，适合小额 |
| BSC | 0.5-2 | 中等 |
| Arbitrum | 0.5-2 | Layer 2，较低 |
| Optimism | 0.5-2 | Layer 2，较低 |

---

#### 5. 策略开关

```yaml
enable_sell_side: true   # 卖方套利
enable_buy_side: false   # 买方套利
check_interval_seconds: 10
```

**说明**:
- `enable_sell_side`: DEX LP 卖出，CEX 买入
- `enable_buy_side`: DEX LP 买入，CEX 卖出
- 可以同时启用两个方向

---

## 使用步骤

### 第 1 步: 准备账户和余额

#### CEX 账户 (Binance)

```bash
# 1. 连接 Binance
connect binance

# 2. 检查余额
balance

# 需要:
# - Base Token (WETH): 至少 lp_token_amount
# - Quote Token (USDC): 至少 lp_token_amount * price + 手续费
```

#### DEX 账户 (Uniswap)

```bash
# 1. 连接 Gateway
gateway connect uniswap

# 2. 检查余额
balance

# 需要:
# - Base Token (WETH): 至少 lp_token_amount
# - Gas Token (ETH): 足够支付 gas
```

**最低余额要求**:
```
假设: WETH = 2000 USDC, lp_token_amount = 0.1

CEX (Binance):
- WETH: 0.1 (用于对冲)
- USDC: 250 (0.1 * 2000 * 1.1 + 预留)

DEX (Uniswap):
- WETH: 0.1 (用于 LP)
- ETH: 0.01 (gas 费)
```

---

### 第 2 步: 配置策略

编辑配置文件:

```bash
# 1. 打开配置文件
nano hummingbot_files/conf/scripts/cex_dex_lp_arbitrage.yml

# 2. 修改参数（至少修改这些）:
cex_exchange: binance
dex_exchange: uniswap/clmm
trading_pair: WETH-USDC
lp_token_amount: 0.1
target_profitability: 0.02
gas_cost_quote: 5  # 根据实际 gas 调整

# 3. 保存并退出
Ctrl + X, Y, Enter
```

---

### 第 3 步: 启动策略

```bash
# 在 Hummingbot 中
start --script cex_dex_lp_arbitrage
```

你会看到:

```
CEX-DEX LP 套利策略启动:
   CEX: binance
   DEX: uniswap/clmm
   交易对: WETH-USDC
   LP 数量: 0.1 WETH
   目标利润: 2.00%
   最低利润: 0.50%
   卖方套利: 启用
   买方套利: 禁用
```

---

### 第 4 步: 监控策略

#### 查看状态

```bash
status
```

输出示例（无仓位）:

```
CEX-DEX LP 套利策略
CEX: binance | DEX: uniswap/clmm
交易对: WETH-USDC

⏳ 等待开仓机会...

统计:
   完成周期: 0
   累计利润: 0.0000 USDC
   LP 开仓失败: 0
   对冲失败: 0
```

输出示例（有仓位）:

```
📊 LP 仓位状态:
   方向: SELL
   价格区间: 2070.0000 - 2090.0000
   均价: 2080.0000
   数量: 0.1
   持仓时间: 120秒
   开仓 CEX 价: 2000.0000

统计:
   完成周期: 1
   累计利润: 2.9300 USDC
   LP 开仓失败: 0
   对冲失败: 0
```

---

#### 查看日志

```bash
# 实时日志
tail -f logs/logs_strategy.log

# 搜索特定事件
grep "套利机会" logs/logs_strategy.log
grep "对冲完成" logs/logs_strategy.log
```

---

### 第 5 步: 停止策略

```bash
stop
```

---

## 监控和调试

### 关键日志事件

#### 1. 发现套利机会

```
发现卖方套利机会:
   DEX 价格: 2080.0000
   目标价格: 2070.0000
   CEX 买价: 2000.0000
```

**说明**: DEX 价格 > 目标价格，可以开仓

---

#### 2. 开 LP 仓位

```
开 CLMM LP 仓位:
   方向: SELL
   中心价: 2080.0000
   价格区间: 2070.0000 - 2090.0000
   Base: 0.1, Quote: 0.0

LP 开仓成交: buy-WETH-USDC-1234567890
```

**说明**: LP 已开，等待成交

---

#### 3. 持仓监控

```
# 正常监控
LP 仓位正常，当前 CEX 价格: 2005.0000

# 触发止损
触发止损:
   LP 均价: 2080.0000
   止损线: 2090.0000
   CEX 价格: 2015.0000

LP 仓位已关闭，原因: STOP_LOSS
```

**说明**: CEX 价格上涨，LP 卖价不够高，止损

---

#### 4. LP 成交 + 对冲

```
LP 订单成交: xxx
   数量: 0.1
   价格: 2085.0000

CEX 对冲:
   方向: BUY
   数量: 0.1
   限价: 2010.0000

对冲完成:
   价格: 2008.5000
   数量: 0.1
   手续费: 0.2000
```

**说明**: LP 成交后立即对冲，完成一个套利周期

---

### 监控指标

| 指标 | 说明 | 正常范围 |
|------|------|----------|
| 完成周期 | 完成的套利次数 | 每天 5-20 次 |
| 累计利润 | 总利润（扣除所有费用） | 正值 |
| LP 开仓失败 | 开 LP 失败次数 | < 10% |
| 对冲失败 | CEX 对冲失败次数 | 0 |

**警告信号**:
- ❌ 对冲失败 > 0 → 严重！可能有裸露风险
- ⚠️  LP 开仓失败 > 20% → 检查 gas 费、余额
- ⚠️  完成周期 = 0（运行 > 1 小时）→ 检查价差、目标利润率

---

## 常见问题

### Q1: 策略一直没有开仓？

**可能原因**:

1. **价差不够大**
   ```
   检查: DEX 价格 vs CEX 价格
   解决: 降低 target_profitability（如从 0.02 降到 0.015）
   ```

2. **Gas 成本太高**
   ```
   检查: gas_cost_quote 是否过高
   解决: 使用 Layer 2（Polygon, Arbitrum）或降低 gas 估算
   ```

3. **余额不足**
   ```
   检查: balance
   解决: 充值足够的 Token
   ```

---

### Q2: LP 频繁触发止损？

**可能原因**:

1. **CEX 价格波动大**
   ```
   问题: CEX 价格快速上涨，LP 卖价变得不够高
   解决: 放宽 min_profitability（如从 0.005 升到 0.01）
   ```

2. **LP 区间设置不当**
   ```
   问题: lp_spread_pct 太小，区间太窄
   解决: 增大 lp_spread_pct（如从 0.005 升到 0.01）
   ```

---

### Q3: LP 很久不成交？

**可能原因**:

1. **LP 价格太高**
   ```
   问题: 我们的卖价高于市场价，没人买
   解决: 降低 target_profitability 或增大 lp_spread_pct
   ```

2. **DEX 交易量小**
   ```
   问题: 池子没有交易
   解决: 换一个交易量大的池子或交易对
   ```

3. **LP 超时**
   ```
   问题: 超过 lp_timeout_seconds，自动关闭
   解决: 增大超时时间或降低开仓门槛
   ```

---

### Q4: 对冲失败怎么办？

**这是最严重的问题！**

**现象**:
```
CEX 对冲失败: Connection timeout
对冲重试 (1/5)
```

**原因**:
- CEX API 连接问题
- 余额不足
- 价格超出限价

**处理**:
1. 策略会自动重试 5 次
2. 如果全部失败，策略停止
3. **立即手动对冲！**（在 CEX 上手动下单）

---

### Q5: 利润率低于预期？

**检查项**:

1. **实际费用 > 估算费用**
   ```
   检查日志中的实际 gas 费用
   如果实际 > 估算，增大 gas_cost_quote
   ```

2. **价格滑点**
   ```
   CEX 对冲时可能有滑点
   使用限价单而不是市价单
   ```

3. **LP 手续费收入被高估**
   ```
   检查 dex_lp_fee_pct 是否正确
   不同池子费率不同（0.05%, 0.3%, 1%）
   ```

---

## 优化建议

### 1. 选择合适的链

| 链 | 优点 | 缺点 | 适合场景 |
|----|------|------|----------|
| Ethereum Mainnet | 流动性最好 | Gas 高（5-20 USDC） | 大额交易 (> 1000 USDC) |
| Polygon | Gas 低（< 1 USDC） | 流动性一般 | 中小额交易 |
| Arbitrum | Gas 低，流动性好 | 生态较新 | 推荐！|
| BSC | Gas 低，流动性好 | 中心化风险 | 推荐！|

**建议**: 新手从 Polygon 或 BSC 开始，Gas 低，试错成本小

---

### 2. 选择合适的交易对

**✅ 推荐**:
- 主流币: WETH-USDC, WBTC-USDC
- 稳定币: USDC-USDT, DAI-USDC
- 大市值: MATIC-USDC, BNB-USDC

**❌ 不推荐**:
- 小市值新币（波动大，风险高）
- 低流动性池子（LP 难成交）
- 跨链桥代币（价格可能脱钩）

---

### 3. 参数调优

#### 保守配置（新手）

```yaml
lp_token_amount: 0.05          # 小额
target_profitability: 0.02     # 2% 利润
min_profitability: 0.005       # 0.5% 止损
lp_spread_pct: 0.01            # 1% 区间
lp_timeout_seconds: 300        # 5 分钟
```

**特点**: 安全，慢，利润稳定

---

#### 激进配置（经验丰富）

```yaml
lp_token_amount: 0.5           # 大额
target_profitability: 0.015    # 1.5% 利润（降低门槛）
min_profitability: 0.003       # 0.3% 止损（更宽容）
lp_spread_pct: 0.005           # 0.5% 区间（更快成交）
lp_timeout_seconds: 180        # 3 分钟
```

**特点**: 激进，快，但风险更高

---

#### 稳定币配置

```yaml
target_profitability: 0.005    # 0.5%（价差小）
min_profitability: 0.001       # 0.1%
lp_spread_pct: 0.005           # 0.5%
lp_timeout_seconds: 600        # 10 分钟（容忍更长时间）
```

**特点**: 适合 USDC-USDT 等稳定币对

---

### 4. 最佳实践

#### ✅ DO

1. **从小额开始**
   - 先用 0.01-0.1 测试
   - 确认策略运行正常后再加大

2. **监控前 24 小时**
   - 密切关注日志
   - 检查是否有异常

3. **定期检查余额**
   - 确保 CEX 和 DEX 都有足够余额
   - 预留 10-20% 作为 buffer

4. **记录实际数据**
   - 记录每次套利的实际利润
   - 对比预期 vs 实际，优化参数

5. **使用限价保护**
   - CEX 对冲使用限价单
   - 防止价格剧烈波动时吃亏

---

#### ❌ DON'T

1. **不要盲目追求高利润**
   - target_profitability 太高 → 永远不开仓
   - 2-3% 是合理目标

2. **不要忽视 Gas 成本**
   - Gas 占比超过 50% 利润 → 不划算
   - 小额交易在 Mainnet 上不合适

3. **不要在波动大的时候运行**
   - 价格剧烈波动 → 频繁止损
   - 等市场稳定后再启动

4. **不要同时运行多个策略**
   - 可能导致余额不足
   - 订单冲突

5. **不要忽视对冲失败**
   - 对冲失败 = 裸露风险
   - 必须立即手动处理

---

## 总结

### 策略优势

✅ **被动做市**: LP 被动等待，无需高频操作
✅ **立即对冲**: CEX 对冲锁定利润
✅ **双重收益**: 价差 + LP 手续费
✅ **风险可控**: 止损、超时保护

---

### 适用场景

- ✅ 中大额交易（> 1000 USDC）
- ✅ 主流币种（WETH, WBTC）
- ✅ 稳定市场（价差稳定）
- ✅ 低 Gas 链（Polygon, BSC, Arbitrum）

---

### 不适用场景

- ❌ 小额交易（< 100 USDC）
- ❌ 高波动币种（新币、山寨币）
- ❌ 高 Gas 链 + 小额（Mainnet < 500 USDC）
- ❌ 低流动性池子

---

### 预期收益

**理想情况**（Polygon, WETH-USDC）:
- 单次: 1-3% 净利润
- 每天: 5-10 次
- 月利润: 20-50%（高风险高收益）

**一般情况**:
- 单次: 0.5-1.5% 净利润
- 每天: 2-5 次
- 月利润: 10-20%

---

### 下一步

1. ✅ 阅读完本指南
2. ⏳ 准备账户和余额
3. ⏳ 配置策略参数
4. ⏳ 小额测试（0.01-0.1）
5. ⏳ 监控 24 小时
6. ⏳ 优化参数
7. ⏳ 逐步加大资金

---

**编写日期**: 2025-10-30
**策略版本**: v1.0
**文档状态**: ✅ 完成

如有问题，请参考:
- 设计文档: `CEX_DEX_LP_ARBITRAGE_DESIGN.md`
- 策略代码: `cex_dex_lp_arbitrage.py`
- 配置文件: `cex_dex_lp_arbitrage.yml`
