# 风控模块集成指南

## 已完成的集成步骤

### 1. 模块文件
- ✅ `state_manager.py` - 状态持久化管理器
- ✅ `swap_manager.py` - 换币管理器

### 2. 配置参数（已添加）
```python
# meteora_dlmm_hft_meme.py (line 131-166)

total_loss_limit_pct: Decimal = Decimal("15.0")  # 累计亏损限制
enable_swap_on_downside: bool = True  # 下跌换SOL
swap_slippage_pct: Decimal = Decimal("2.0")  # 换币滑点
downside_cooldown_seconds: int = 300  # 下跌冷却5分钟
upside_cooldown_seconds: int = 60  # 上涨冷却1分钟
enable_state_persistence: bool = True  # 启用持久化
state_db_path: str = "data/meteora_hft_state.db"  # 数据库路径
```

### 3. 模块初始化（已添加）
```python
# meteora_dlmm_hft_meme.py (__init__ 方法, line 448-451)

self.state_manager: Optional[StateManager] = None
self.swap_manager: Optional[SwapManager] = None
self.cooldown_until: float = 0

# initialize_strategy 方法 (line 478-502)
# - 初始化 StateManager
# - 检查累计亏损状态
# - 初始化 SwapManager
```

---

## 待集成的关键逻辑

### 4. 修改 `monitor_position_high_frequency()` 方法

**位置**: 约 line 976

**需要添加的检查**:

```python
async def monitor_position_high_frequency(self):
    """高频监控（60秒规则）"""

    # ===== 新增 Step 0: 检查策略是否被暂停 =====
    if self.state_manager and self.state_manager.get_state()["manual_kill"]:
        self.logger().error("❌ 策略已暂停：达到累计亏损限制")
        return

    # ===== 新增 Step 0.5: 检查冷却期 =====
    if time.time() < self.cooldown_until:
        remaining = int(self.cooldown_until - time.time())
        self.logger().debug(f"冷却期中，剩余 {remaining}秒")
        return

    # ===== 新增 Step 1: 检查累计亏损限制（最高优先级）=====
    if self.state_manager:
        is_limited, reason = self.state_manager.check_loss_limit(
            self.config.total_loss_limit_pct
        )
        if is_limited:
            self.logger().error(
                f"🚨 达到累计亏损限制 {self.config.total_loss_limit_pct}%！\n"
                f"  原因: {reason}"
            )
            await self._execute_emergency_stop()
            return

    # === 原有逻辑：更新仓位信息 ===
    await self._update_position_info_if_needed()

    # ... 继续原有的止损检查和再平衡逻辑 ...
```

### 5. 新增 `_execute_emergency_stop()` 方法

**位置**: 在 `close_position()` 方法后添加

```python
async def _execute_emergency_stop(self):
    """紧急止损（累计亏损限制）"""
    self.logger().error("🚨🚨🚨 触发紧急止损 🚨🚨🚨")

    # 1. 平仓
    if self.position_opened:
        await self.close_position()
        await asyncio.sleep(3)  # 等待平仓完成

    # 2. 全部换成 SOL
    if self.swap_manager:
        base_balance = self.connector.get_available_balance(self.base_token)
        if base_balance > Decimal("0.01"):  # 避免太小的余额
            self.logger().warning(f"⚠️  准备将 {base_balance:.6f} {self.base_token} 换成 SOL")

            success, sol_amount, error = await self.swap_manager.swap_all_to_sol(
                token=self.base_token,
                slippage_pct=self.config.swap_slippage_pct,
                reason="EMERGENCY_TOTAL_LOSS_LIMIT"
            )

            if success:
                self.logger().info(f"✅ 已换成 {sol_amount:.6f} SOL")

                # 记录换币事件
                if self.state_manager:
                    self.state_manager.record_swap(
                        from_token=self.base_token,
                        from_amount=base_balance,
                        to_amount=sol_amount,
                        reason="EMERGENCY_TOTAL_LOSS_LIMIT"
                    )
            else:
                self.logger().error(f"❌ 换币失败: {error}")

    # 3. 显示摘要
    if self.state_manager:
        self.logger().error(f"\n{self.state_manager.get_summary()}")

    self.logger().error(
        "策略已暂停，请手动检查并决定是否继续\n"
        "重置方法: state_manager.reset_manual_kill()"
    )
```

### 6. 修改 `close_position()` 方法

**位置**: 约 line 1118

**需要添加**:

```python
async def close_position(self):
    """关闭仓位"""
    try:
        if not self.position_id:
            return

        self.logger().info(f"关闭仓位: {self.position_id}")

        # ===== 新增：平仓前记录信息（用于计算盈亏）=====
        entry_price = self.open_price
        current_price = Decimal(str(self.pool_info.price)) if self.pool_info else entry_price

        # 计算实际盈亏
        exit_value = self._calculate_position_value()
        realized_pnl = exit_value - self.initial_investment

        order_id = self.connector.remove_liquidity(
            trading_pair=self.config.trading_pair,
            position_address=self.position_id
        )

        self.logger().info(f"关闭订单: {order_id}")

        # ===== 原有的状态重置 =====
        self.position_opened = False
        self.position_id = None
        self.position_info = None
        self.tokens_prepared = False
        self.position_info_last_update = None
        self.open_price = None
        self.initial_investment = Decimal("0")

        # ===== 新增：记录到状态管理器 =====
        if self.state_manager and entry_price:
            self.state_manager.record_close(
                position_id=order_id,
                entry_price=entry_price,
                exit_price=current_price,
                realized_pnl=realized_pnl,
                reason="MANUAL_CLOSE",  # 会在调用时传入具体原因
                swapped_to_sol=False  # 会在后续判断
            )

    except Exception as e:
        self.logger().error(f"关闭仓位失败: {e}", exc_info=True)

# 新增辅助方法
def _calculate_position_value(self) -> Decimal:
    """计算当前仓位价值"""
    if not self.position_info or not self.pool_info:
        return Decimal("0")

    base_amount = Decimal(str(self.position_info.base_token_amount))
    quote_amount = Decimal(str(self.position_info.quote_token_amount))
    current_price = Decimal(str(self.pool_info.price))

    return (base_amount * current_price) + quote_amount
```

### 7. 新增 `_execute_stop_loss_with_swap()` 方法

**位置**: 在 `close_position()` 后添加

```python
async def _execute_stop_loss_with_swap(self, reason: str):
    """执行止损并判断是否换SOL"""

    # 1. 记录平仓前的信息
    entry_price = self.open_price
    current_price = Decimal(str(self.pool_info.price)) if self.pool_info else entry_price
    exit_value = self._calculate_position_value()
    realized_pnl = exit_value - self.initial_investment

    # 2. 平仓
    await self.close_position()
    await asyncio.sleep(3)  # 等待平仓完成

    # 3. 判断是否需要换SOL
    need_swap = False
    if self.config.enable_swap_on_downside and entry_price and current_price:
        need_swap = should_swap_to_sol(
            current_price=current_price,
            entry_price=entry_price,
            reason=reason,
            threshold_pct=Decimal("5")
        )

    # 4. 执行换币
    if need_swap and self.swap_manager:
        self.logger().warning(f"⚠️  止损触发，准备换成 SOL (原因: {reason})")

        base_balance = self.connector.get_available_balance(self.base_token)
        if base_balance > Decimal("0.01"):
            success, sol_amount, error = await self.swap_manager.swap_all_to_sol(
                token=self.base_token,
                slippage_pct=self.config.swap_slippage_pct,
                reason=reason
            )

            if success:
                self.logger().info(f"✅ 已换成 {sol_amount:.6f} SOL")

                # 记录换币
                if self.state_manager:
                    self.state_manager.record_swap(
                        from_token=self.base_token,
                        from_amount=base_balance,
                        to_amount=sol_amount,
                        reason=reason
                    )
            else:
                self.logger().error(f"❌ 换币失败: {error}")

    # 5. 记录平仓盈亏
    if self.state_manager and entry_price:
        cumulative_pnl = self.state_manager.record_close(
            position_id=self.position_id or "unknown",
            entry_price=entry_price,
            exit_price=current_price,
            realized_pnl=realized_pnl,
            reason=reason,
            swapped_to_sol=need_swap
        )

        self.logger().info(
            f"📊 平仓记录:\n"
            f"  本次盈亏: {realized_pnl:+.6f} SOL\n"
            f"  累计盈亏: {cumulative_pnl:+.6f} SOL\n"
            f"  换 SOL: {'是' if need_swap else '否'}"
        )

    # 6. 设置冷却期
    if need_swap:
        self.cooldown_until = time.time() + self.config.downside_cooldown_seconds
        self.logger().info(f"冷却期 {self.config.downside_cooldown_seconds}秒")
    else:
        self.cooldown_until = time.time() + self.config.upside_cooldown_seconds
        self.logger().info(f"冷却期 {self.config.upside_cooldown_seconds}秒")
```

### 8. 修改开仓逻辑

**位置**: `open_position()` 方法开头

```python
async def open_position(self, center_price: Decimal):
    """开仓"""

    # ===== 新增：检查是否在冷却期 =====
    if time.time() < self.cooldown_until:
        remaining = int(self.cooldown_until - time.time())
        self.logger().warning(f"⚠️  冷却期中，剩余 {remaining}秒，跳过开仓")
        return

    # ===== 新增：检查累计亏损限制 =====
    if self.state_manager:
        is_limited, reason = self.state_manager.check_loss_limit(
            self.config.total_loss_limit_pct
        )
        if is_limited:
            self.logger().error(f"❌ 达到累计亏损限制，拒绝开仓: {reason}")
            return

    # ===== 新增：记录初始资金（仅首次）=====
    if self.state_manager:
        total_value = (total_base * center_price) + total_quote
        self.state_manager.set_initial_capital(total_value)

    # ... 继续原有开仓逻辑 ...

    # ===== 新增：开仓成功后记录 =====
    if self.state_manager and order_id:
        self.state_manager.record_open(
            position_id=order_id,
            entry_price=center_price
        )
```

### 9. 修改 status 显示

**位置**: `format_status()` 方法末尾

```python
def format_status(self) -> str:
    """格式化状态（增强版）"""
    try:
        # ... 原有status显示 ...

        # ===== 新增：显示风控状态 =====
        if self.state_manager:
            lines.append(f"\n{self.state_manager.get_summary()}")

        # ===== 新增：显示冷却期 =====
        if self.cooldown_until > time.time():
            remaining = int(self.cooldown_until - time.time())
            lines.append(f"\n❄️  冷却期: 剩余 {remaining}秒")

        lines.append("=" * 70)
        return "\n".join(lines)

    except Exception as e:
        import traceback
        return f"❌ 状态显示错误: {str(e)}\n{traceback.format_exc()}"
```

---

## 调用关系图

```
on_tick()
  ↓
check_and_open_position()
  ↓
monitor_position_high_frequency()
  ├─ [新增] 检查 manual_kill 状态
  ├─ [新增] 检查冷却期
  ├─ [新增] 检查累计亏损限制
  │    └─ _execute_emergency_stop()
  │         ├─ close_position()
  │         ├─ swap_manager.swap_all_to_sol()
  │         └─ state_manager.record_swap()
  ├─ check_stop_loss()
  │    └─ _execute_stop_loss_with_swap()  [新增]
  │         ├─ close_position()
  │         ├─ should_swap_to_sol()  [新增判断逻辑]
  │         ├─ swap_manager.swap_all_to_sol()  [新增]
  │         ├─ state_manager.record_close()  [新增]
  │         └─ 设置冷却期  [新增]
  └─ should_rebalance()
       └─ execute_high_frequency_rebalance()
            └─ _execute_stop_loss_with_swap()  [新增]
```

---

## 测试建议

### 1. 单元测试

```bash
# 测试 StateManager
cd /Users/qinghuan/Documents/code/hummingbot/hummingbot_files/scripts
python3 state_manager.py

# 测试 SwapManager
python3 swap_manager.py
```

### 2. 集成测试

在 `meteora_dlmm_hft_meme.py` 末尾添加:

```python
if __name__ == "__main__":
    # 测试状态管理器
    from state_manager import StateManager
    manager = StateManager("test.db")
    manager.set_initial_capital(Decimal("100"))
    print(manager.get_summary())
```

### 3. 实盘测试步骤

1. 先在 devnet 测试
2. 设置小额资金（< $10）
3. 观察以下事件:
   - 开仓是否记录到数据库
   - 下跌止损是否换SOL
   - 累计亏损是否正确追踪
   - 达到15%限制是否暂停
   - 重启后是否恢复状态

---

## 配置示例

```yaml
# config/meteora_dlmm_hft_meme_1.yml

connector: "meteora/clmm"
swap_connector: "jupiter/router"
trading_pair: "BONK-USDC"
price_range_pct: 8.0

# 止损参数
stop_loss_pct: 5.0
enable_60s_rule: true
out_of_range_timeout_seconds: 60

# 新增风控参数
total_loss_limit_pct: 15.0
enable_swap_on_downside: true
swap_slippage_pct: 2.0
downside_cooldown_seconds: 300
upside_cooldown_seconds: 60

# 持久化
enable_state_persistence: true
state_db_path: "data/meteora_hft_state.db"
```

---

## 重要提示

1. **数据库路径**: 确保 `data` 目录存在，或使用绝对路径
2. **滑点设置**: Meme币流动性差，建议2-3%
3. **冷却期**: 可根据市场波动调整
4. **累计亏损**: 15%是保守值，可根据风险承受能力调整
5. **重启恢复**: 确保数据库文件不丢失

---

## 待完成清单

- [ ] 将上述代码片段集成到 `meteora_dlmm_hft_meme.py`
- [ ] 测试 StateManager 功能
- [ ] 测试 SwapManager 功能
- [ ] Devnet 集成测试
- [ ] 小额实盘测试
- [ ] 监控和调优

---

## 快速集成脚本（可选）

如果想快速完成集成，可以执行以下步骤：

1. 复制 `state_manager.py` 和 `swap_manager.py` 到 scripts 目录
2. 修改 `meteora_dlmm_hft_meme.py`：
   - 添加导入（已完成）
   - 添加配置参数（已完成）
   - 添加模块初始化（已完成）
   - 修改监控逻辑（参考上文）
   - 添加新方法（参考上文）
3. 测试运行

完成后，策略将具备完整的风控能力！
