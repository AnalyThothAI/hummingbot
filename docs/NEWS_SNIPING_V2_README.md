# 📰 新闻狙击策略 V2 - 使用指南

## 🎯 核心改进

V2 版本相比 V1 的主要改进：

| 功能 | V1 (ScriptStrategyBase) | V2 (StrategyV2Base + Executor) |
|------|-------------------------|--------------------------------|
| **订单管理** | 手动管理订单状态 | PositionExecutor 自动管理 |
| **止盈止损** | 无自动止盈止损 | ✅ Triple Barrier（止损/止盈/时间限制/移动止损） |
| **重试机制** | 手动实现重试逻辑 | Executor 内置重试和错误恢复 |
| **信号去重** | 无 | ✅ Redis 去重（防止重复执行） |
| **订单追踪** | 手动追踪 pending_orders | ✅ 统一的 Executor 状态管理 |
| **风险控制** | 基础滑点控制 | ✅ 多层风险控制（止损/止盈/时间限制） |
| **适用场景** | 简单快速交易 | 完整的新闻交易策略 |

## 🏗️ 架构设计

```
MQTT信号源
    ↓
信号去重 (Redis)
    ↓
策略逻辑 (NewsSnipingV2)
    ↓
创建 PositionExecutor ←→ Triple Barrier 配置
    ↓                      ├─ 止损 (Stop Loss)
订单执行 (Connector)       ├─ 止盈 (Take Profit)
    ↓                      ├─ 时间限制 (Time Limit)
自动监控和平仓             └─ 移动止损 (Trailing Stop)
```

## 📋 前置要求

### 1. 依赖库

```bash
# 安装 MQTT 库
pip install paho-mqtt

# 安装 Redis 库（用于信号去重）
pip install redis
```

### 2. 基础设施

确保 `docker-compose.yml` 中已配置并运行：

- ✅ **Redis** (端口 6379) - 信号去重
- ✅ **EMQX** (端口 1883) - MQTT broker
- ✅ **Gateway** - DEX 连接

```bash
# 启动服务
docker-compose up -d redis emqx gateway

# 检查状态
docker-compose ps
```

## 🚀 快速开始

### 步骤 1：创建策略配置

在 Hummingbot CLI 中创建策略：

```bash
create --script-config v2_news_sniping_strategy
```

按照提示配置以下参数：

#### 基础配置
- **connector**: `pancakeswap` (或其他 DEX)
- **trading_pair**: `WBNB-USDT` (默认交易对)

#### MQTT 配置
- **mqtt_broker**: `localhost`
- **mqtt_port**: `1883`
- **mqtt_topic**: `trading/bsc/snipe`
- **mqtt_username**: 如果 EMQX 需要认证
- **mqtt_password**: 如果 EMQX 需要认证

#### 交易配置
- **default_trade_amount**: `0.001` (BUY=BNB数量, SELL=代币数量)
- **default_quote_token**: `WBNB`
- **slippage**: `0.02` (2%)

#### 止盈止损配置 ⭐
- **stop_loss_pct**: `0.10` (10% 止损)
- **take_profit_pct**: `0.05` (5% 止盈)
- **time_limit_seconds**: `300` (5分钟超时自动平仓)
- **enable_trailing_stop**: `False` (是否启用移动止损)

#### Redis 去重配置
- **enable_signal_deduplication**: `True`
- **redis_host**: `localhost`
- **redis_port**: `6379`
- **signal_dedup_window_seconds**: `60` (60秒内相同信号去重)

### 步骤 2：启动策略

```bash
start --script v2_news_sniping_strategy
```

### 步骤 3：发送测试信号

在另一个终端中：

```bash
# 进入项目目录
cd /Users/qinghuan/Documents/code/hummingbot

# 发送买入信号
python scripts/utility/test_news_signal_sender.py \
  --side BUY \
  --base TOKEN \
  --quote WBNB \
  --amount 0.001

# 发送卖出信号
python scripts/utility/test_news_signal_sender.py \
  --side SELL \
  --base TOKEN \
  --quote WBNB \
  --amount 100
```

## 📡 MQTT 信号格式

### 买入信号示例

```json
{
  "side": "BUY",
  "base_token": "TOKEN",
  "quote_token": "WBNB",
  "amount": "0.001",
  "slippage": "0.02"
}
```

**说明：**
- `amount` = 花费的 WBNB 数量
- 策略会自动计算能买入多少 TOKEN

### 卖出信号示例

```json
{
  "side": "SELL",
  "base_token": "TOKEN",
  "quote_token": "WBNB",
  "amount": "100",
  "slippage": "0.02"
}
```

**说明：**
- `amount` = 卖出的 TOKEN 数量
- 策略会自动计算能换回多少 WBNB

### 可选参数

- `slippage`: 覆盖配置中的滑点设置

## 🎮 PositionExecutor 工作流程

当策略接收到信号后，会自动创建 PositionExecutor：

```
1. 接收信号
   ↓
2. Redis 去重检查
   ↓
3. 获取当前价格
   ↓
4. 计算交易数量和入场价格
   ↓
5. 创建 PositionExecutor
   ├─ 配置: entry_price, amount, triple_barrier
   └─ 下市价单入场
   ↓
6. Executor 自动监控
   ├─ 价格触及止盈 → 自动平仓 ✅
   ├─ 价格触及止损 → 自动平仓 ❌
   ├─ 超过时间限制 → 自动平仓 ⏰
   └─ 移动止损触发 → 自动平仓 📈
   ↓
7. 订单完成，记录结果
```

## 🛡️ 风险管理特性

### 1. Triple Barrier（三重屏障）

**止损 (Stop Loss)**
- 价格下跌超过设定百分比时自动卖出
- 例如：10% 止损，入场价 $1.00，价格跌到 $0.90 时自动平仓

**止盈 (Take Profit)**
- 价格上涨达到目标时自动卖出锁定利润
- 例如：5% 止盈，入场价 $1.00，价格涨到 $1.05 时自动平仓

**时间限制 (Time Limit)**
- 超过指定时间后自动平仓，避免长时间持仓
- 例如：300秒（5分钟）后自动平仓

**移动止损 (Trailing Stop)** - 可选
- 价格上涨时动态调整止损位置
- 激活价格：上涨 3%
- 回落幅度：1% 时触发平仓

### 2. 信号去重

使用 Redis 实现时间窗口内的信号去重：

```python
# 相同的信号指纹
signal_key = {
    "side": "BUY",
    "base_token": "TOKEN",
    "quote_token": "WBNB"
}

# 60秒内重复信号会被忽略
# 防止因网络问题或系统错误导致的重复执行
```

### 3. 滑点保护

```python
# 买入时接受更高价格
entry_price = mid_price * (1 + slippage)

# 卖出时接受更低价格
entry_price = mid_price * (1 - slippage)
```

## 📊 监控和调试

### 查看策略状态

在 Hummingbot CLI 中输入 `status` 查看：

```
🎯 新闻狙击策略 V2 - 运行状态
======================================================================

📋 配置:
  Connector: pancakeswap
  Trading Pair: WBNB-USDT
  Default Amount: 0.001
  Slippage: 2.0%
  Stop Loss: 10.0%
  Take Profit: 5.0%
  Time Limit: 300s
  Trailing Stop: Disabled
  Signal Dedup: Enabled

🔌 连接状态:
  Connector: 🟢 就绪
  MQTT: 🟢 连接
  Redis: 🟢 连接

📊 统计:
  信号接收: 10
  信号去重: 3
  Executor 创建: 7

🔄 活跃订单:
  - TOKEN-WBNB BUY | Amount: 100.000000 | Entry: 0.000100
```

### 查看日志

```bash
# 查看 Hummingbot 日志
docker logs -f hummingbot

# 查看最近 100 行
docker logs --tail 100 hummingbot
```

### 常见日志输出

```
📩 信号: {'side': 'BUY', 'base_token': 'TOKEN', 'amount': '0.001'}
🎯 处理信号: BUY TOKEN-WBNB, Amount: 0.001, Slippage: 2.0%
💰 买入: 用 0.001 WBNB 买入约 10.000000 TOKEN
✅ Executor 已创建
   交易对: TOKEN-WBNB
   方向: BUY
   数量: 10.000000 TOKEN
   入场价: 0.000102
   止损: 10.0%
   止盈: 5.0%
```

## 🔧 故障排查

### MQTT 连接失败

**问题：** `❌ MQTT 连接失败，错误码: 5`

**解决：**
1. 检查 EMQX 是否运行：`docker ps | grep emqx`
2. 检查认证信息是否正确
3. 测试连接：
   ```bash
   # 使用 mosquitto_pub 测试
   mosquitto_pub -h localhost -p 1883 -t "test" -m "hello" -u admin -P password
   ```

### Redis 连接失败

**问题：** `⚠️  Redis 连接失败，信号去重已禁用`

**解决：**
1. 检查 Redis 是否运行：`docker ps | grep redis`
2. 测试连接：
   ```bash
   docker exec -it redis redis-cli ping
   # 应返回: PONG
   ```
3. 如果不需要去重，可以在配置中禁用：`enable_signal_deduplication: False`

### 信号未触发交易

**问题：** 收到信号但未创建 Executor

**检查清单：**
1. ✅ Connector 是否就绪？
2. ✅ 信号格式是否正确？
3. ✅ 是否被去重过滤？
4. ✅ 交易对是否存在？
5. ✅ 查看日志中的错误信息

### 订单执行失败

**问题：** Executor 创建了但订单失败

**可能原因：**
1. 余额不足
2. 交易对不存在或流动性不足
3. Gateway 连接问题
4. Gas 费用不足

## 🎛️ 高级配置

### 针对不同场景的配置建议

#### 场景 1：高波动新币狙击
```yaml
slippage: 0.05              # 5% 大滑点
stop_loss_pct: 0.20         # 20% 止损
take_profit_pct: 0.50       # 50% 高止盈
time_limit_seconds: 180     # 3分钟快速平仓
enable_trailing_stop: True  # 启用移动止损
```

#### 场景 2：稳定币套利
```yaml
slippage: 0.005             # 0.5% 小滑点
stop_loss_pct: 0.01         # 1% 止损
take_profit_pct: 0.005      # 0.5% 止盈
time_limit_seconds: 600     # 10分钟
enable_trailing_stop: False
```

#### 场景 3：新闻事件交易
```yaml
slippage: 0.03              # 3% 中等滑点
stop_loss_pct: 0.15         # 15% 止损
take_profit_pct: 0.10       # 10% 止盈
time_limit_seconds: 300     # 5分钟
enable_trailing_stop: True
trailing_stop_activation_pct: 0.05  # 5% 后启动
trailing_stop_delta_pct: 0.02       # 回落 2% 平仓
```

## 🧪 测试流程

### 1. 测试信号去重

```bash
# 发送第一个信号
python scripts/utility/test_news_signal_sender.py --side BUY --base TOKEN --amount 0.001

# 立即发送相同信号（应该被去重）
python scripts/utility/test_news_signal_sender.py --side BUY --base TOKEN --amount 0.001

# 等待 60 秒后再发送（应该被执行）
sleep 60
python scripts/utility/test_news_signal_sender.py --side BUY --base TOKEN --amount 0.001
```

### 2. 测试止盈止损

1. 创建买入订单
2. 在 Hummingbot status 中观察 Executor 状态
3. 等待价格变化触发止盈或止损
4. 查看日志确认自动平仓

### 3. 测试时间限制

1. 设置短时间限制（如 60 秒）
2. 创建订单
3. 观察 60 秒后是否自动平仓

## 📚 扩展开发

### 集成其他信号源

除了 MQTT，你可以集成：

1. **Telegram Bot**
   ```python
   # 在策略中添加 Telegram 监听器
   # 解析消息并调用 _process_signal()
   ```

2. **WebSocket API**
   ```python
   # 监听 WebSocket 消息
   # 转换为统一格式调用 _process_signal()
   ```

3. **Twitter/X 监控**
   ```python
   # 使用 Twitter API 监控关键词
   # 解析推文生成信号
   ```

### 自定义 Triple Barrier 逻辑

你可以根据不同代币动态调整止盈止损：

```python
def get_custom_triple_barrier(self, token: str, volatility: Decimal) -> TripleBarrierConfig:
    # 根据代币和波动率定制配置
    if volatility > Decimal("0.5"):  # 高波动
        return TripleBarrierConfig(
            stop_loss=Decimal("0.20"),
            take_profit=Decimal("0.50"),
            ...
        )
    else:  # 低波动
        return TripleBarrierConfig(
            stop_loss=Decimal("0.05"),
            take_profit=Decimal("0.10"),
            ...
        )
```

## 🆚 V1 vs V2 迁移指南

如果你之前使用 V1 版本，主要变化：

| V1 | V2 |
|----|-----|
| `did_fill_order(event)` | ✅ Executor 自动处理 |
| `did_fail_order(event)` | ✅ Executor 自动重试 |
| `_execute_trade_with_retry()` | ✅ Executor 内置重试 |
| 手动止盈止损 | ✅ Triple Barrier 自动管理 |
| `pending_orders` 字典 | ✅ `get_all_executors()` |

**迁移步骤：**
1. 保留 V1 配置作为备份
2. 创建 V2 配置
3. 小额测试 V2 版本
4. 验证止盈止损逻辑
5. 完全切换到 V2

## 📞 支持

- **问题反馈**: GitHub Issues
- **文档**: Hummingbot V2 Strategy 文档
- **社区**: Hummingbot Discord

## ⚠️ 风险提示

- DEX 交易存在滑点和 MEV 风险
- 新币交易波动极大，可能快速归零
- 止损不能保证完全避免损失
- 在测试网或小额资金测试后再使用
- 不构成投资建议，盈亏自负

---

**祝交易顺利！** 🚀
