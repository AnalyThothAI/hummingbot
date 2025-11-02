# on_tick() 重构 - 统一使用标准调度方式

## 修复日期
2025-11-02

## 问题描述

用户指出：
1. **ScriptStrategyBase 策略都是用 on_tick 调度的**
2. **现在策略根本不运行了**（自定义的 periodic_check 没有被框架调用）
3. **不需要单独考虑高频的方法，和普通的一样**

## 根本原因

### 错误的实现方式

**之前的错误实现**:
```python
def on_start(self):
    # 启动定时检查
    self.check_task = safe_ensure_future(self.periodic_check())

async def periodic_check(self):
    """自定义的定期检查"""
    while True:
        if not self.position_opened:
            await self.check_and_open_position()
        else:
            await self.monitor_position_high_frequency()
        await asyncio.sleep(self.config.check_interval_seconds)
```

**问题**:
1. 自定义的 `periodic_check()` 循环，绕过了框架的标准调度
2. 增加了复杂性，容易出错
3. 与其他策略不一致
4. 难以维护和调试

### 标准的实现方式（V2 版本）

```python
def on_tick(self):
    """策略主循环（框架每秒调用一次）"""
    current_time = datetime.now()

    # 检查间隔控制
    if self.last_check_time and (current_time - self.last_check_time).total_seconds() < self.config.check_interval_seconds:
        return

    self.last_check_time = current_time

    # 根据状态执行不同逻辑
    if self.position_opening:
        return
    elif not self.position_opened:
        safe_ensure_future(self.check_and_open_positions())
    else:
        safe_ensure_future(self.monitor_positions())
```

**优势**:
1. ✅ 使用框架标准调度（每秒自动调用）
2. ✅ 通过 `last_check_time` 控制实际执行间隔
3. ✅ 与其他策略实现一致
4. ✅ 简单、可靠、易维护

## 修复方案

### 1. 移除自定义的 periodic_check

**移除的代码**:
```python
# __init__ 中
self.check_task: Optional[asyncio.Task] = None

# on_start 中
self.check_task = safe_ensure_future(self.periodic_check())

# on_stop 中
if self.check_task:
    self.check_task.cancel()

# periodic_check 方法
async def periodic_check(self):
    while True:
        # ...
```

### 2. 添加时间追踪变量

**添加的代码**:
```python
# __init__ 中
self.last_check_time: Optional[datetime] = None
```

### 3. 实现标准的 on_tick 方法

**添加的代码**:
```python
def on_tick(self):
    """策略主循环（框架每秒调用一次）"""
    current_time = datetime.now()

    # 检查间隔控制
    if self.last_check_time and (current_time - self.last_check_time).total_seconds() < self.config.check_interval_seconds:
        return

    self.last_check_time = current_time

    # 根据状态执行不同逻辑
    if self.position_opening:
        # 等待开仓确认
        return
    elif not self.position_opened:
        # 无仓位：开仓
        safe_ensure_future(self.check_and_open_position())
    else:
        # 持仓中：高频监控
        safe_ensure_future(self.monitor_position_high_frequency())
```

### 4. 简化 on_start 和 on_stop

**on_start**:
```python
def on_start(self):
    """策略启动"""
    self.logger().info("=" * 60)
    self.logger().info("⚡ Meteora DLMM 高频做市策略 - Meme 币专用 ⚡")
    # ... 打印配置信息
    # ✅ 不需要启动任何 task
```

**on_stop**:
```python
async def on_stop(self):
    """策略停止"""
    self.logger().info("策略已停止")
    # ✅ 不需要取消任何 task
```

## 框架工作原理

### ScriptStrategyBase 调度流程

```
1. 策略启动
   └─> on_start() - 打印初始信息

2. 框架开始调度
   └─> 每 1 秒调用一次 on_tick()

3. on_tick() 执行
   ├─> 检查距离上次执行是否 >= check_interval_seconds
   ├─> 如果未到时间 → return（不执行）
   └─> 如果到时间 → 执行策略逻辑

4. 策略逻辑
   ├─> 如果 position_opening → 等待
   ├─> 如果无仓位 → safe_ensure_future(check_and_open_position())
   └─> 如果有仓位 → safe_ensure_future(monitor_position_high_frequency())

5. 异步任务执行
   └─> check_and_open_position() 或 monitor_position_high_frequency()
       在后台异步执行，不阻塞 on_tick()
```

### 时间间隔控制原理

```python
# 示例：check_interval_seconds = 10

# 第 1 秒
on_tick() -> last_check_time = None -> 执行策略逻辑 -> 记录 last_check_time

# 第 2 秒
on_tick() -> 距上次 1 秒 < 10 秒 -> return（不执行）

# 第 3-10 秒
on_tick() -> 距上次 < 10 秒 -> return（不执行）

# 第 11 秒
on_tick() -> 距上次 10 秒 >= 10 秒 -> 执行策略逻辑 -> 更新 last_check_time

# 第 12-20 秒
on_tick() -> 距上次 < 10 秒 -> return（不执行）

# ... 以此类推
```

## 与 V2 版本对比

| 项目 | V2 版本 | 高频版本（修复前） | 高频版本（修复后） | 状态 |
|------|---------|-------------------|-------------------|------|
| 调度方式 | ✅ on_tick() | ❌ periodic_check() | ✅ on_tick() | 一致 ✅ |
| 时间控制 | ✅ last_check_time | ❌ asyncio.sleep() | ✅ last_check_time | 一致 ✅ |
| 框架兼容 | ✅ 标准 | ❌ 自定义 | ✅ 标准 | 一致 ✅ |
| 复杂度 | ✅ 简单 | ❌ 复杂 | ✅ 简单 | 一致 ✅ |
| 可维护性 | ✅ 高 | ❌ 低 | ✅ 高 | 一致 ✅ |

## 修复前后对比

### 修复前（自定义循环）

**问题**:
```python
# 1. __init__ 中需要管理 task
self.check_task: Optional[asyncio.Task] = None

# 2. on_start 中需要启动 task
self.check_task = safe_ensure_future(self.periodic_check())

# 3. 需要自定义循环
async def periodic_check(self):
    while True:  # 自己管理循环
        # 执行逻辑
        await asyncio.sleep(10)  # 自己控制间隔

# 4. on_stop 需要取消 task
if self.check_task:
    self.check_task.cancel()
```

### 修复后（框架调度）

**简化**:
```python
# 1. __init__ 只需记录时间
self.last_check_time: Optional[datetime] = None

# 2. on_start 只打印信息
def on_start(self):
    self.logger().info("策略启动")
    # 不需要启动任何 task

# 3. 使用标准 on_tick
def on_tick(self):  # 框架自动每秒调用
    # 检查间隔
    if 距离上次 < 10秒:
        return
    # 执行逻辑
    safe_ensure_future(self.check_and_open_position())

# 4. on_stop 只打印信息
async def on_stop(self):
    self.logger().info("策略停止")
    # 不需要取消任何 task
```

## 关键改进

### 1. 框架标准化 ✅
- 使用 ScriptStrategyBase 标准调度
- 与所有其他策略一致
- 不需要自己管理循环和任务

### 2. 简化代码 ✅
- 移除了 100+ 行复杂的循环逻辑
- 移除了 task 管理代码
- 代码更简洁、更易读

### 3. 可靠性提升 ✅
- 框架保证每秒调用，不会遗漏
- 不需要担心异常导致循环退出
- 框架自动处理启动和停止

### 4. 调试友好 ✅
- 与其他策略行为一致，便于对比
- 清晰的执行时间点（每 10 秒）
- 日志更规范

## 验证结果

### 1. 语法检查 ✅
```bash
python3 -m py_compile hummingbot_files/scripts/meteora_dlmm_hft_meme.py
# ✅ 语法检查通过
```

### 2. 代码清理 ✅
```bash
grep -r "periodic_check\|check_task" meteora_dlmm_hft_meme.py
# No files found ✅ 所有遗留代码已清理
```

### 3. 与 V2 对比 ✅
- `on_tick()` 实现：✅ 完全一致
- 时间控制逻辑：✅ 完全一致
- 状态检查逻辑：✅ 完全一致

## 启动后预期行为

### 执行流程

```
1. 策略启动
   └─> on_start() 打印信息

2. 初始化完成（5秒后）
   └─> initialize_strategy() 完成

3. 框架开始调度（每秒）
   └─> 第 1 秒: on_tick()
       ├─> last_check_time = None
       ├─> 距上次 > 10秒（首次执行）
       └─> safe_ensure_future(check_and_open_position())
           └─> 后台执行开仓逻辑

   └─> 第 2-10 秒: on_tick()
       └─> 距上次 < 10秒 → return（不执行）

   └─> 第 11 秒: on_tick()
       ├─> 距上次 10秒 >= 10秒
       └─> safe_ensure_future(monitor_position_high_frequency())
           └─> 后台执行监控逻辑

   └─> ... 每 10 秒执行一次
```

### 预期日志

```
2025-11-02 12:30:00 - INFO - ⚡ Meteora DLMM 高频做市策略 - Meme 币专用 ⚡
2025-11-02 12:30:00 - INFO - 交易对: PAYAI-SOL
...
2025-11-02 12:30:05 - INFO - ⚡ 高频策略初始化完成

# 第 1 秒（初始化后）- on_tick 立即执行
2025-11-02 12:30:06 - INFO - 准备开仓，当前价格: 0.00012345
2025-11-02 12:30:06 - INFO - 检查并准备双边代币...

# 第 2-10 秒 - on_tick 被调用但不执行（距上次 < 10秒）

# 第 11 秒 - on_tick 再次执行
2025-11-02 12:30:16 - INFO - ⚡ 高频监控: 价格 xxx, 区间 [xxx, xxx]...

# 第 12-20 秒 - on_tick 被调用但不执行

# 第 21 秒 - on_tick 再次执行
2025-11-02 12:30:26 - INFO - ⚡ 高频监控: ...
```

## 常见问题

### Q: 为什么不直接在 on_tick 中执行异步逻辑？
A: 因为 `on_tick()` 是同步方法，框架每秒调用。如果直接在 on_tick 中执行耗时操作，会阻塞框架调度。使用 `safe_ensure_future()` 可以在后台异步执行，不阻塞 on_tick。

### Q: 高频策略也用 10 秒间隔，不是太慢了吗？
A: "高频" 是相对于 V2 的 24 小时间隔而言。对于 meme 币来说，10 秒检查一次已经足够快了。而且可以通过配置文件调整 `check_interval_seconds`。

### Q: 如果策略逻辑执行超过 10 秒怎么办？
A: `safe_ensure_future()` 启动异步任务后立即返回，不会阻塞 on_tick。即使上一次任务还在执行，下一次 on_tick 也会正常调用。

### Q: 为什么 V2 可以正常工作，但自定义循环不行？
A: V2 使用标准的 on_tick，框架保证调用。自定义循环需要自己管理启动、停止、异常处理等，容易出错。

## 总结

### 修复前的问题
- ❌ 自定义循环，绕过框架调度
- ❌ 增加代码复杂度
- ❌ 容易出错，难以维护
- ❌ 与其他策略不一致

### 修复后的优势
- ✅ 使用框架标准调度
- ✅ 代码简洁清晰
- ✅ 可靠性高
- ✅ 与 V2 和其他策略一致
- ✅ 易于维护和调试

### 关键改进
1. **移除**: periodic_check、check_task
2. **添加**: on_tick、last_check_time
3. **简化**: on_start、on_stop
4. **统一**: 与 ScriptStrategyBase 标准一致

---

**修复完成时间**: 2025-11-02
**修复人员**: Claude (Anthropic)
**测试状态**: ✅ 语法通过，⏳ 功能测试待完成
**可用性**: ✅ 完全符合框架标准，应该能正常工作

## 下一步

立即测试：
```bash
start --script meteora_dlmm_hft_meme.py
```

预期：
- ✅ 策略成功启动
- ✅ 初始化完成后立即开始第一次检查
- ✅ 每 10 秒执行一次（可通过日志时间戳验证）
- ✅ 正常开仓、监控、再平衡、止损
