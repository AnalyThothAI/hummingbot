# 配置文件优化说明

## ✅ 已优化的配置文件

**文件路径**: `/Users/qinghuan/Documents/code/hummingbot/hummingbot_files/conf/scripts/conf_meteora_dlmm_hft_meme_1.yml`

---

## 修改对比

### 移除的参数
```yaml
# ❌ 已移除（与新参数重复）
max_daily_loss_pct: '15.0'  # 被 total_loss_limit_pct 取代
```

**原因**: `max_daily_loss_pct` 是旧版本的参数，功能与新的 `total_loss_limit_pct` 重复。新版本使用累计亏损（而非每日），更合理。

### 新增的参数
```yaml
# ✅ 新增：风控参数
total_loss_limit_pct: '15.0'              # 累计亏损 15% 暂停策略
enable_swap_on_downside: true             # 下跌止损时自动换成 SOL
swap_slippage_pct: '2.0'                  # 止损换币滑点 2%
downside_cooldown_seconds: 300            # 下跌止损后冷却 5分钟
upside_cooldown_seconds: 60               # 上涨再平衡后冷却 1分钟
enable_state_persistence: true            # 启用状态持久化
state_db_path: data/meteora_hft_state.db  # 数据库路径
```

### 保留但优化的参数
```yaml
# ✅ 保留并优化注释
rebalance_cooldown_seconds: 180           # 从 1800 改为 180（3分钟更合理）
price_range_pct: '3.0'                    # 保持 3%（极窄区间）
max_position_hold_hours: '6.0'            # 保持 6小时
```

---

## 完整参数列表（按功能分组）

### 1. 基础配置
| 参数 | 值 | 说明 |
|------|-----|------|
| `connector` | meteora/clmm | LP 连接器 |
| `swap_connector` | jupiter/router | Swap 连接器 |
| `trading_pair` | PAYAI-SOL | 交易对 |
| `pool_address` | 7hMhU5... | 池子地址 |

### 2. 高频参数
| 参数 | 值 | 说明 |
|------|-----|------|
| `price_range_pct` | 3.0 | 价格区间宽度 ±3% |
| `rebalance_threshold_pct` | 75.0 | 再平衡阈值 75% |
| `rebalance_cooldown_seconds` | 180 | 再平衡冷却期 3分钟 |
| `min_profit_for_rebalance` | 2.0 | 最小再平衡盈利 2% |

### 3. 止损配置
| 参数 | 值 | 说明 |
|------|-----|------|
| `enable_60s_rule` | true | 启用 60秒规则 |
| `out_of_range_timeout_seconds` | 60 | 超出区间 60秒触发 |
| `stop_loss_pct` | 5.0 | 幅度止损 5% |
| `enable_volume_monitoring` | true | 启用交易量监控 |
| `volume_drop_threshold_pct` | 80.0 | 交易量骤降阈值 80% |

### 4. 资金配置
| 参数 | 值 | 说明 |
|------|-----|------|
| `base_token_amount` | 0.0 | 使用钱包余额 |
| `quote_token_amount` | 0.0 | 使用钱包余额 |
| `wallet_allocation_pct` | 80.0 | 使用钱包余额的 80% |

### 5. 换币配置
| 参数 | 值 | 说明 |
|------|-----|------|
| `enable_auto_swap` | true | 开仓前自动换币 |
| `auto_swap_slippage_pct` | 3.0 | 开仓换币滑点 3% |

### 6. 🆕 风控参数（新增）
| 参数 | 值 | 说明 |
|------|-----|------|
| `total_loss_limit_pct` | 15.0 | 累计亏损 15% 暂停策略 ⭐ |
| `enable_swap_on_downside` | true | 下跌止损时换 SOL ⭐ |
| `swap_slippage_pct` | 2.0 | 止损换币滑点 2% |
| `downside_cooldown_seconds` | 300 | 下跌止损后冷却 5分钟 ⭐ |
| `upside_cooldown_seconds` | 60 | 上涨再平衡后冷却 1分钟 ⭐ |
| `enable_state_persistence` | true | 启用状态持久化 ⭐ |
| `state_db_path` | data/meteora_hft_state.db | 数据库路径 |

### 7. 监控配置
| 参数 | 值 | 说明 |
|------|-----|------|
| `check_interval_seconds` | 10 | 检查间隔 10秒 |
| `max_position_hold_hours` | 6.0 | 最长持仓 6小时 |

---

## 参数详解

### 核心风控参数

#### `total_loss_limit_pct: 15.0`
**作用**: 累计亏损保护
- 追踪所有平仓的累计盈亏
- 当 `累计亏损 / 初始资金 >= 15%` 时
- 自动暂停策略（设置 `manual_kill = True`）
- 需要手动检查后才能重启

**示例**:
```
初始资金: 100 SOL
第1次平仓: -5 SOL  → 累计: -5 SOL  (5%)
第2次平仓: -7 SOL  → 累计: -12 SOL (12%)
第3次平仓: -3 SOL  → 累计: -15 SOL (15%) → 🚨 暂停策略
```

#### `enable_swap_on_downside: true`
**作用**: 下跌止损自动换 SOL
- 价格跌破下界触发止损时
- 资产已全部变成 base token (meme币)
- 自动通过 Jupiter 换成 SOL
- 避免继续持有贬值的 meme 币

**判断逻辑**:
```python
需要换SOL的情况:
1. 平仓原因包含"下跌"、"下界"、"止损"关键字
2. 价格跌幅 >= 5%

不需要换SOL的情况:
1. 上涨再平衡
2. 价格跌幅 < 5%
```

#### `downside_cooldown_seconds: 300` / `upside_cooldown_seconds: 60`
**作用**: 防止频繁开仓

**下跌止损冷却（5分钟）**:
- 价格跌破下界止损后
- 冷却期内不开仓
- 避免立即重入下跌趋势

**上涨再平衡冷却（1分钟）**:
- 价格突破上界再平衡后
- 短冷却期，快速重入
- 跟随上涨趋势

#### `enable_state_persistence: true`
**作用**: 状态持久化
- 所有开仓/平仓/换币事件写入 SQLite
- 重启后自动恢复累计盈亏
- 防止重复亏损

**数据库位置**: `data/meteora_hft_state.db`

---

## 参数调优建议

### 保守配置（低风险）
```yaml
price_range_pct: '5.0'              # 更宽区间，减少再平衡
stop_loss_pct: '3.0'                # 更严格止损
total_loss_limit_pct: '10.0'        # 更低亏损限制
downside_cooldown_seconds: 600      # 更长冷却期（10分钟）
```

### 激进配置（高风险）
```yaml
price_range_pct: '2.0'              # 极窄区间，高频交易
stop_loss_pct: '8.0'                # 更宽松止损
total_loss_limit_pct: '20.0'        # 更高亏损容忍
downside_cooldown_seconds: 120      # 更短冷却期（2分钟）
```

### 当前配置（平衡型）⭐ 推荐
```yaml
price_range_pct: '3.0'              # 平衡区间
stop_loss_pct: '5.0'                # 标准止损
total_loss_limit_pct: '15.0'        # 合理限制
downside_cooldown_seconds: 300      # 标准冷却（5分钟）
```

---

## 配置验证清单

在启动策略前，请确认：

- [ ] `pool_address` 是否正确
- [ ] `trading_pair` 是否匹配池子
- [ ] `price_range_pct` 是否适合当前波动率
- [ ] `total_loss_limit_pct` 是否符合风险承受能力
- [ ] `state_db_path` 目录是否存在
- [ ] 钱包中是否有足够的 SOL（用于Gas费）
- [ ] 是否在 devnet 测试过

---

## 常见问题

### Q1: 为什么移除 `max_daily_loss_pct`？
**A**: 新版本使用 `total_loss_limit_pct` 追踪累计亏损（而非每日），更合理。累计亏损不会因为日期变化而重置。

### Q2: `rebalance_cooldown_seconds` 为什么从 1800 改为 180？
**A**: 1800秒（30分钟）太长，不适合高频策略。180秒（3分钟）更合理，能快速响应价格变化。

### Q3: 数据库文件会占用多少空间？
**A**: 很小，通常 < 1MB。每次平仓只记录一行数据（约100字节）。

### Q4: 如何清空历史数据重新开始？
**A**: 删除或重命名数据库文件：
```bash
mv data/meteora_hft_state.db data/meteora_hft_state_backup.db
```

### Q5: 如果数据库丢失会怎样？
**A**: 累计盈亏会重置为0，但不影响策略运行。建议定期备份数据库文件。

---

## 配置文件位置

**当前配置**: `/Users/qinghuan/Documents/code/hummingbot/hummingbot_files/conf/scripts/conf_meteora_dlmm_hft_meme_1.yml`

**备份建议**:
```bash
# 创建备份
cp conf_meteora_dlmm_hft_meme_1.yml conf_meteora_dlmm_hft_meme_1_backup_$(date +%Y%m%d).yml

# 查看历史备份
ls -lt conf_meteora_dlmm_hft_meme_1*.yml
```

---

**配置文件已优化完成！✅**
