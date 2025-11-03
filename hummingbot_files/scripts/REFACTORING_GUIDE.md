# Meteora DLMM HFT 策略重构指南

## 📋 重构目标

将 1599 行的单文件策略重构为符合 Hummingbot 最佳实践的模块化结构。

### 当前问题
- ❌ **过长**：1599 行远超官方复杂脚本的 500-645 行范围
- ❌ **职责混乱**：包含 3 个独立的引擎类 + 1 个策略类
- ❌ **难以维护**：止损逻辑、再平衡逻辑、换币逻辑都在一个文件中
- ❌ **不符合最佳实践**：官方脚本是单文件但高内聚，当前是单文件但低内聚

### 重构目标
- ✅ **降低主文件复杂度**：从 1599 行降至 ~400-500 行
- ✅ **职责清晰**：每个模块单一职责
- ✅ **符合官方实践**：模块化但保持主策略文件的自包含性
- ✅ **易于测试和维护**：引擎和工具函数可以独立测试

---

## 📁 新的目录结构

```
hummingbot_files/scripts/
├── meteora_dlmm_hft_meme.py          # 主策略文件 (目标: ~400行)
│   ├── Config: MeteoraDlmmHftMemeConfig
│   └── Strategy: MeteoraDlmmHftMeme
│       ├── Lifecycle: on_tick(), on_start(), on_stop()
│       ├── Position Management: open_position(), close_position()
│       ├── Monitoring: monitor_position_high_frequency()
│       ├── Event Handlers: did_fill_order(), did_fail_order()
│       └── Status: format_status()
│
├── engines/                           # 独立引擎模块
│   ├── __init__.py                   # 导出所有引擎
│   ├── stop_loss_engine.py           # 止损引擎 (~120行)
│   ├── rebalance_engine.py           # 再平衡引擎 (~140行)
│   └── state_manager.py              # 状态持久化 (~150行)
│
├── utils/                             # 工具模块
│   ├── __init__.py
│   ├── position_helper.py            # 仓位辅助函数 (~200行)
│   ├── price_helper.py               # 价格获取和计算 (~140行)
│   └── swap_helper.py                # 换币辅助函数 (~170行)
│
└── swap_manager.py                    # 换币管理器（已存在，保持不变）
```

---

## 🔧 模块职责划分

### 1. `meteora_dlmm_hft_meme.py` (主策略文件)

**职责**：
- 策略配置（`MeteoraDlmmHftMemeConfig`）
- 策略生命周期管理（`on_tick`, `on_start`, `on_stop`）
- 状态机逻辑（开仓 → 持仓 → 平仓 → 再平衡）
- 事件处理（`did_fill_order`, `did_fail_order`）
- 状态显示（`format_status`）

**保留的方法**：
```python
class MeteoraDlmmHftMeme(ScriptStrategyBase):
    @classmethod
    def init_markets(cls, config)

    def __init__(self, connectors, config)
    async def initialize_strategy(self)

    # 生命周期
    def on_start(self)
    def on_tick(self)
    async def on_stop(self)

    # 核心流程
    async def check_pending_order_timeout(self)
    async def check_and_open_position(self)
    async def open_position(self, center_price)
    async def monitor_position_high_frequency(self)
    async def execute_stop_loss(self, reason)
    async def execute_high_frequency_rebalance(self, current_price)
    async def close_position(self)

    # 事件处理
    def did_fill_order(self, event)
    def did_fail_order(self, event)
    async def fetch_positions_after_fill(self)

    # 状态显示
    def format_status(self) -> str
```

**预计行数**：~400-500 行

---

### 2. `engines/stop_loss_engine.py`

**职责**：
- 幅度止损检测（价格下跌超过阈值）
- 60秒规则检测（价格超出区间持续时间）
- 持仓时长检测（长期未盈利）
- 返回止损决策和建议

**导出**：
```python
class FastStopLossEngine:
    def __init__(self, logger, config)
    def reset(self)
    def check_stop_loss(
        current_price, open_price, lower_price, upper_price
    ) -> Tuple[bool, str, str, float]
```

**预计行数**：~120 行

---

### 3. `engines/rebalance_engine.py`

**职责**：
- 判断是否需要再平衡仓位
- 冷却期管理
- 距离边界计算
- 60秒规则触发
- 最小盈利检查

**导出**：
```python
class HighFrequencyRebalanceEngine:
    def __init__(self, logger)
    async def should_rebalance(
        current_price, lower_price, upper_price,
        accumulated_fees_value, position_value,
        config, out_duration_seconds
    ) -> Tuple[bool, str]
    def mark_rebalance_executed(self)

    # 私有方法
    def _is_cooldown_passed(self, cooldown_seconds) -> bool
    def _remaining_cooldown(self, cooldown_seconds) -> float
    def _calculate_distance_from_edge(
        current_price, lower_price, upper_price
    ) -> Decimal
```

**预计行数**：~140 行

---

### 4. `engines/state_manager.py`

**职责**：
- 状态持久化到 SQLite
- 加载历史状态
- 记录交易历史
- 统计数据管理

**导出**：
```python
class StateManager:
    def __init__(self, db_path, logger)
    async def load_state(self) -> dict
    async def save_state(self, state: dict)
    async def record_trade(self, trade_data: dict)
    async def get_daily_stats(self) -> dict
    async def close(self)
```

**预计行数**：~150 行（已存在）

---

### 5. `utils/position_helper.py`

**职责**：
- 检查现有仓位
- 获取仓位信息
- 计算仓位价值
- 获取代币数量
- 价格区间计算

**导出函数**：
```python
async def check_existing_positions(
    connector, trading_pair, logger
) -> Tuple[bool, Optional[str], Optional[CLMMPositionInfo]]

async def get_token_amounts(
    connector, base_token, quote_token, config, logger
) -> Tuple[Decimal, Decimal]

async def calculate_position_value(
    position_info, current_price, logger
) -> Decimal

def calculate_price_range(
    center_price, range_width_pct
) -> Tuple[Decimal, Decimal]

def calculate_width_percentages(
    center_price, lower_price, upper_price
) -> Tuple[float, float]
```

**预计行数**：~200 行

---

### 6. `utils/price_helper.py`

**职责**：
- 获取当前价格（多重降级策略）
- 获取池子信息
- 获取池子地址
- 注入价格到 RateOracle

**导出函数**：
```python
async def get_current_price(
    connector, trading_pair, logger
) -> Optional[Decimal]

async def fetch_pool_info(
    connector, trading_pair, logger
) -> Optional[CLMMPoolInfo]

async def get_pool_address(
    connector, trading_pair, pool_address_from_config, logger
) -> Optional[str]
```

**预计行数**：~140 行

---

### 7. `utils/swap_helper.py`

**职责**：
- 通过 Jupiter 执行换币
- 准备双边代币（自动 swap 功能）
- 获取 Jupiter 报价

**导出函数**：
```python
async def swap_via_jupiter(
    connector, base_token, quote_token,
    from_token, to_token, amount, side, logger
) -> bool

async def prepare_tokens_for_position(
    connector, base_token, quote_token, current_price, logger
) -> bool
```

**预计行数**：~170 行

---

## 🔄 重构步骤

### 阶段 1：创建模块结构（已完成 ✅）

```bash
# 1. 创建目录
mkdir -p hummingbot_files/scripts/engines
mkdir -p hummingbot_files/scripts/utils

# 2. 创建 __init__.py 文件
# - engines/__init__.py
# - utils/__init__.py

# 3. 提取引擎类
# - engines/stop_loss_engine.py
# - engines/rebalance_engine.py
# - engines/state_manager.py (复制自现有文件)

# 4. 创建工具模块
# - utils/position_helper.py
# - utils/price_helper.py
# - utils/swap_helper.py
```

### 阶段 2：重构主策略文件（待执行）

1. **备份原文件**
   ```bash
   cp meteora_dlmm_hft_meme.py meteora_dlmm_hft_meme.py.backup
   ```

2. **修改导入语句**
   ```python
   # 添加到主文件顶部
   from engines import FastStopLossEngine, HighFrequencyRebalanceEngine, StateManager
   from utils import position_helper, price_helper, swap_helper
   from swap_manager import SwapManager
   ```

3. **删除已提取的类**
   - 删除 `FastStopLossEngine` 类定义（line 184-264）
   - 删除 `HighFrequencyRebalanceEngine` 类定义（line 271-366）

4. **替换函数调用**

   **示例 1：获取当前价格**
   ```python
   # 原来（内联实现）
   async def get_current_price(self) -> Optional[Decimal]:
       try:
           pool_info = await self.connector.get_pool_info(...)
           # ... 40+ 行代码
       except:
           # ... 错误处理

   # 重构后（调用工具函数）
   async def get_current_price(self) -> Optional[Decimal]:
       return await price_helper.get_current_price(
           self.connector,
           self.config.trading_pair,
           self.logger()
       )
   ```

   **示例 2：检查现有仓位**
   ```python
   # 原来（内联实现）
   async def check_existing_positions(self):
       try:
           positions = await self.connector.get_clmm_positions(...)
           # ... 30+ 行代码
       except:
           # ... 错误处理

   # 重构后（调用工具函数）
   async def check_existing_positions(self):
       has_position, position_id, position_info = await position_helper.check_existing_positions(
           self.connector,
           self.config.trading_pair,
           self.logger()
       )

       if has_position:
           self.position_opened = True
           self.position_id = position_id
           self.position_info = position_info
   ```

   **示例 3：准备代币**
   ```python
   # 原来（内联实现）
   async def prepare_tokens_for_position(self, current_price):
       # ... 40+ 行代币准备逻辑

   # 重构后（调用工具函数）
   async def prepare_tokens_for_position(self, current_price: Decimal) -> bool:
       return await swap_helper.prepare_tokens_for_position(
           self.swap_connector,
           self.base_token,
           self.quote_token,
           current_price,
           self.logger()
       )
   ```

5. **验证功能**
   - 运行策略测试
   - 检查日志输出
   - 验证开仓、平仓、再平衡逻辑

### 阶段 3：测试和优化（待执行）

1. **单元测试**（可选）
   - 测试各个引擎的独立功能
   - 测试工具函数的正确性

2. **集成测试**
   - 在 devnet 环境运行完整策略
   - 验证所有流程正常工作

3. **文档更新**
   - 更新 README
   - 添加模块使用说明

---

## 📊 重构前后对比

| 指标 | 重构前 | 重构后 | 改进 |
|------|--------|--------|------|
| **主文件行数** | 1599 行 | ~400-500 行 | ⬇️ 70% |
| **单一职责** | ❌ 混合 | ✅ 清晰 | 质的提升 |
| **可测试性** | ⚠️ 困难 | ✅ 容易 | 模块化 |
| **可维护性** | ⚠️ 低 | ✅ 高 | 职责分离 |
| **符合最佳实践** | ❌ 否 | ✅ 是 | 参考官方 |

---

## 🎯 预期效果

### 1. 降低复杂度
- 主策略文件从 1599 行降至 ~400 行
- 每个模块不超过 200 行
- 符合官方 500-645 行的复杂策略标准

### 2. 职责清晰
- **引擎模块**：决策逻辑（止损、再平衡、状态）
- **工具模块**：辅助函数（价格、仓位、换币）
- **主策略**：流程编排和事件处理

### 3. 易于维护
- 修改止损逻辑只需改 `stop_loss_engine.py`
- 修改价格获取只需改 `price_helper.py`
- 主策略文件专注于业务流程

### 4. 易于测试
- 每个引擎和工具函数可以独立测试
- 减少回归测试的复杂度

### 5. 符合 Hummingbot 最佳实践
- 单文件策略类（主文件）
- 辅助模块分离（engines/ 和 utils/）
- 清晰的导入和依赖关系

---

## ⚠️ 注意事项

### 1. 保持向后兼容
- 配置文件格式不变
- 策略行为逻辑不变
- 只改变代码组织方式

### 2. 日志和调试
- 所有工具函数接收 logger 参数
- 保持相同的日志级别和格式
- 便于问题追踪

### 3. 错误处理
- 每个工具函数包含完整的错误处理
- 返回明确的错误状态
- 不影响主策略的稳定性

### 4. 性能影响
- 函数调用开销可忽略不计
- 异步函数保持异步
- 不影响策略的实时性

---

## 📝 下一步行动

1. ✅ **已完成**：创建所有模块文件
2. ⏳ **待执行**：重构主策略文件
   - 备份原文件
   - 修改导入语句
   - 删除已提取的类
   - 替换函数调用为模块调用
   - 测试验证

3. ⏳ **可选**：编写单元测试
4. ⏳ **可选**：更新文档

---

## 🔗 相关文件

- 主策略：`meteora_dlmm_hft_meme.py`
- 引擎模块：`engines/*.py`
- 工具模块：`utils/*.py`
- 换币管理器：`swap_manager.py`
- 配置示例：`conf/scripts/conf_meteora_dlmm_hft_meme_*.yml`

---

## 📚 参考资料

- Hummingbot 官方脚本示例：`scripts/lp_manage_position.py`
- 最佳实践分析：Agent 分析报告
- 策略设计文档：`MEME_HIGH_FREQUENCY_STRATEGY.md`
