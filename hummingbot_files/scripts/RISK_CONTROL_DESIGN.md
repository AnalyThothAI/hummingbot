# Meteora DLMM HFT 风控改进方案

## 当前风控逻辑分析

### 现状问题
**入场示例**：价格 1.0，区间 [0.9, 1.2]

**当前止损规则**：
1. **幅度止损**：跌破 -5%（价格 0.95）立即止损
2. **60秒规则**：跌破下界 0.9 持续 60 秒 + 下跌 >3% → 止损
3. **交易量骤降**：跌 >80% 建议止损
4. **持仓过久**：>24h 且未盈利建议止损

**问题**：
- ❌ 跌破下界 0.9 → 0.8 时，资产已全部变成 base token（meme币）
- ❌ 止损平仓后，仍然持有 base token，继续承受下跌风险
- ❌ 没有"换成 SOL 止损"的逻辑
- ❌ 重启后无法恢复累计亏损状态，可能重复入场

---

## 改进方案设计

### 核心原则
1. **下穿止损必须换 SOL**：避免持有贬值的 meme 币
2. **累计亏损保护**：记录总亏损，达到阈值暂停策略
3. **状态持久化**：重启后能恢复状态，避免重复亏损
4. **简洁实用**：使用 SQLite（Hummingbot 内置）+ Redis（可选）

---

## 详细设计

### 1. 多级止损机制

#### Level 0: 硬止损（新增）- 最高优先级
```
触发条件：累计亏损 >= 15% 初始资金
动作：
  1. 立即平仓
  2. 全部资产换成 SOL
  3. 暂停策略（manual_kill = True）
  4. 记录到数据库：stop_reason = "TOTAL_LOSS_LIMIT"
```

#### Level 1: 幅度止损（强化）
```
触发条件：单次仓位亏损 >= 5%
动作：
  1. 立即平仓
  2. 如果价格 < 开仓价格：换成 SOL（下跌止损）
  3. 如果价格 > 开仓价格：保持当前资产（上涨再平衡）
  4. 记录亏损到累计账本
```

#### Level 2: 下穿止损（新增）- 关键改进
```
触发条件：价格跌破下界 && 持续 60 秒
动作：
  1. 立即平仓（此时已全部变成 base token）
  2. **强制换成 SOL**（避免继续承受下跌）
  3. 冷却期：300 秒（5分钟）
  4. 记录亏损到累计账本
```

#### Level 3: 上穿再平衡
```
触发条件：价格突破上界 && 持续 60 秒
动作：
  1. 平仓（此时已全部变成 quote token/SOL）
  2. 保持 SOL，不换币
  3. 冷却期：60 秒（短冷却，快速重入）
  4. 记录盈利到累计账本
```

---

### 2. 换币逻辑（新增模块）

```python
class SwapManager:
    """资产交换管理器（通过 Jupiter）"""

    async def swap_all_to_sol(
        self,
        token: str,
        reason: str = "STOP_LOSS"
    ) -> Tuple[bool, Decimal]:
        """
        将指定代币全部换成 SOL

        返回: (是否成功, 换得的 SOL 数量)
        """
        # 1. 获取余额
        balance = self.connector.get_available_balance(token)

        if balance <= 0:
            return True, Decimal("0")

        # 2. 通过 Jupiter 换币
        swap_result = await self.connector.swap(
            from_token=token,
            to_token="SOL",
            amount=balance,
            slippage=0.02  # 2% 滑点容忍
        )

        # 3. 等待确认
        await asyncio.sleep(3)

        # 4. 强制刷新余额
        await self.connector.update_balances(on_interval=False)

        sol_balance = self.connector.get_available_balance("SOL")

        # 5. 记录到数据库
        self._log_swap(token, balance, sol_balance, reason)

        return swap_result.success, sol_balance
```

---

### 3. 状态持久化（新增模块）

#### 方案选择：SQLite（推荐，简洁）

```python
class StateManager:
    """策略状态持久化管理器"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 累计盈亏表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_pnl (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp BIGINT NOT NULL,
                event_type TEXT NOT NULL,  -- OPEN/CLOSE/SWAP/STOP
                position_id TEXT,
                entry_price REAL,
                exit_price REAL,
                realized_pnl REAL NOT NULL,
                cumulative_pnl REAL NOT NULL,
                quote_token TEXT,
                reason TEXT,
                swap_to_sol BOOLEAN DEFAULT 0
            )
        """)

        # 策略状态表（单例记录）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                cumulative_pnl REAL DEFAULT 0,
                initial_capital REAL,
                total_open_count INTEGER DEFAULT 0,
                total_close_count INTEGER DEFAULT 0,
                total_swap_count INTEGER DEFAULT 0,
                last_updated BIGINT,
                manual_kill BOOLEAN DEFAULT 0,
                stop_reason TEXT
            )
        """)

        # 插入默认状态
        cursor.execute("""
            INSERT OR IGNORE INTO strategy_state (id, cumulative_pnl, initial_capital)
            VALUES (1, 0, 0)
        """)

        conn.commit()
        conn.close()

    def get_state(self) -> Dict:
        """获取当前策略状态"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM strategy_state WHERE id = 1")
        row = cursor.fetchone()
        conn.close()

        if not row:
            return self._default_state()

        return {
            "cumulative_pnl": row[1],
            "initial_capital": row[2],
            "total_open_count": row[3],
            "total_close_count": row[4],
            "total_swap_count": row[5],
            "last_updated": row[6],
            "manual_kill": bool(row[7]),
            "stop_reason": row[8]
        }

    def update_state(self, **kwargs):
        """更新策略状态"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
        values = list(kwargs.values()) + [int(time.time())]

        cursor.execute(f"""
            UPDATE strategy_state
            SET {set_clause}, last_updated = ?
            WHERE id = 1
        """, values)

        conn.commit()
        conn.close()

    def record_close(
        self,
        position_id: str,
        entry_price: Decimal,
        exit_price: Decimal,
        realized_pnl: Decimal,
        reason: str,
        swapped_to_sol: bool = False
    ):
        """记录平仓事件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取当前累计盈亏
        state = self.get_state()
        new_cumulative_pnl = state["cumulative_pnl"] + float(realized_pnl)

        # 插入记录
        cursor.execute("""
            INSERT INTO strategy_pnl (
                timestamp, event_type, position_id, entry_price, exit_price,
                realized_pnl, cumulative_pnl, quote_token, reason, swap_to_sol
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(time.time()),
            "CLOSE",
            position_id,
            float(entry_price),
            float(exit_price),
            float(realized_pnl),
            new_cumulative_pnl,
            "SOL",
            reason,
            swapped_to_sol
        ))

        # 更新累计盈亏
        cursor.execute("""
            UPDATE strategy_state
            SET cumulative_pnl = ?, total_close_count = total_close_count + 1
            WHERE id = 1
        """, (new_cumulative_pnl,))

        conn.commit()
        conn.close()

        return new_cumulative_pnl

    def check_loss_limit(self, loss_limit_pct: Decimal = Decimal("15")) -> bool:
        """检查是否达到累计亏损限制"""
        state = self.get_state()

        if state["initial_capital"] <= 0:
            return False

        loss_pct = abs(state["cumulative_pnl"] / state["initial_capital"]) * 100

        if state["cumulative_pnl"] < 0 and loss_pct >= float(loss_limit_pct):
            # 达到亏损限制，停止策略
            self.update_state(
                manual_kill=True,
                stop_reason=f"TOTAL_LOSS_LIMIT_{loss_pct:.1f}%"
            )
            return True

        return False
```

---

### 4. 改进后的主流程

```python
async def monitor_position_high_frequency(self):
    """高频监控（60秒规则）"""

    # 0. 检查是否被暂停
    if self.state_manager.get_state()["manual_kill"]:
        self.logger().error("❌ 策略已暂停：达到累计亏损限制")
        return

    # 1. 更新仓位信息（60秒缓存）
    await self._update_position_info_if_needed()

    # 2. 检查累计亏损限制（最高优先级）
    if self.state_manager.check_loss_limit(self.config.total_loss_limit_pct):
        self.logger().error(
            f"🚨 达到累计亏损限制 {self.config.total_loss_limit_pct}%，立即止损"
        )
        await self._execute_emergency_stop()
        return

    # 3. 检查止损
    should_stop, stop_type, reason, out_duration = self.stop_loss_engine.check_stop_loss(
        current_price=Decimal(str(self.pool_info.price)),
        open_price=self.open_price,
        lower_price=Decimal(str(self.position_info.lower_price)),
        upper_price=Decimal(str(self.position_info.upper_price))
    )

    if should_stop:
        if stop_type == "HARD_STOP":
            # 立即止损
            await self._execute_stop_loss(reason)
        else:
            self.logger().warning(f"⚠️  建议止损：{reason}")

    # 4. 检查再平衡
    elif stop_type == "REBALANCE":
        await self._execute_rebalance(reason)

async def _execute_stop_loss(self, reason: str):
    """执行止损"""

    # 1. 平仓
    await self.close_position()

    # 2. 判断是否需要换 SOL
    current_price = Decimal(str(self.pool_info.price)) if self.pool_info else self.open_price

    need_swap = False
    if "下跌" in reason or "超出下界" in reason:
        # 下跌止损，必须换 SOL
        need_swap = True
    elif current_price < self.open_price * Decimal("0.95"):
        # 价格跌幅 > 5%，换 SOL
        need_swap = True

    # 3. 执行换币
    if need_swap:
        self.logger().warning(f"⚠️  止损触发，准备换成 SOL")
        success, sol_amount = await self.swap_manager.swap_all_to_sol(
            self.base_token,
            reason=reason
        )
        if success:
            self.logger().info(f"✅ 已换成 {sol_amount:.6f} SOL")

    # 4. 计算实际盈亏并记录
    exit_value = self._calculate_position_value()
    realized_pnl = exit_value - self.initial_investment

    cumulative_pnl = self.state_manager.record_close(
        position_id=self.position_id,
        entry_price=self.open_price,
        exit_price=current_price,
        realized_pnl=realized_pnl,
        reason=reason,
        swapped_to_sol=need_swap
    )

    self.logger().info(
        f"📊 平仓记录：\n"
        f"  本次盈亏: {realized_pnl:+.6f} SOL\n"
        f"  累计盈亏: {cumulative_pnl:+.6f} SOL\n"
        f"  换 SOL: {'是' if need_swap else '否'}"
    )

    # 5. 进入冷却期
    if need_swap:
        self.cooldown_until = time.time() + 300  # 5分钟
    else:
        self.cooldown_until = time.time() + 60   # 1分钟

async def _execute_emergency_stop(self):
    """紧急止损（累计亏损限制）"""

    self.logger().error("🚨🚨🚨 触发紧急止损 🚨🚨🚨")

    # 1. 平仓
    if self.position_opened:
        await self.close_position()

    # 2. 全部换成 SOL
    base_balance = self.connector.get_available_balance(self.base_token)
    if base_balance > 0:
        await self.swap_manager.swap_all_to_sol(
            self.base_token,
            reason="EMERGENCY_TOTAL_LOSS_LIMIT"
        )

    # 3. 状态已在 check_loss_limit() 中更新

    self.logger().error(
        f"策略已暂停，请手动检查并决定是否继续\n"
        f"累计亏损: {self.state_manager.get_state()['cumulative_pnl']:.6f} SOL"
    )
```

---

### 5. 配置参数（新增）

```python
@dataclass
class MeteoraDlmmHftMemeConfig(BaseClientModel):
    # ... 现有参数 ...

    # ========== 新增：风控参数 ==========
    total_loss_limit_pct: Decimal = Field(
        default=Decimal("15"),
        description="累计亏损限制（%），超过则暂停策略"
    )

    enable_swap_on_downside: bool = Field(
        default=True,
        description="下跌止损时是否自动换成 SOL"
    )

    swap_slippage_pct: Decimal = Field(
        default=Decimal("2"),
        description="换币时的滑点容忍度（%）"
    )

    downside_cooldown_seconds: int = Field(
        default=300,
        description="下跌止损后的冷却期（秒）"
    )

    upside_cooldown_seconds: int = Field(
        default=60,
        description="上涨再平衡后的冷却期（秒）"
    )

    # ========== 新增：持久化参数 ==========
    enable_state_persistence: bool = Field(
        default=True,
        description="是否启用状态持久化"
    )

    state_db_path: str = Field(
        default="data/meteora_hft_state.db",
        description="状态数据库路径"
    )
```

---

## 总结对比

### 改进前
```
入场：价格 1.0，区间 [0.9, 1.2]
  ↓
价格跌至 0.8（跌破下界 0.9）
  ↓
60秒后触发止损
  ↓
平仓（此时已全部变成 meme 币）
  ↓
继续持有 meme 币（❌ 继续下跌风险）
  ↓
重启策略，不知道之前亏损（❌ 可能重复亏损）
```

### 改进后
```
入场：价格 1.0，区间 [0.9, 1.2]
  ↓
价格跌至 0.8（跌破下界 0.9）
  ↓
60秒后触发下穿止损
  ↓
平仓（此时已全部变成 meme 币）
  ↓
✅ 立即换成 SOL（避免继续下跌）
  ↓
✅ 记录亏损到数据库（累计 -X SOL）
  ↓
冷却 5 分钟后重新评估
  ↓
✅ 检查累计亏损是否 >= 15%
  - 是 → 暂停策略（manual_kill）
  - 否 → 继续运行
  ↓
重启策略后
  ↓
✅ 从数据库恢复累计亏损状态
  ↓
✅ 如果累计亏损 >= 15% → 拒绝开仓
```

---

## 实施步骤

### Phase 1: 核心风控（必须）
1. ✅ 添加 `SwapManager` 模块
2. ✅ 修改 `check_stop_loss()` 区分上下穿
3. ✅ 修改 `close_position()` 后增加换币逻辑
4. ✅ 测试换币功能

### Phase 2: 状态持久化（推荐）
1. ✅ 添加 `StateManager` 模块（SQLite）
2. ✅ 初始化时恢复状态
3. ✅ 平仓时记录累计盈亏
4. ✅ 累计亏损检查

### Phase 3: 优化（可选）
1. ⚠️  添加 Redis 分布式状态（多实例需要）
2. ⚠️  添加盈亏报表导出
3. ⚠️  添加 Telegram 告警

---

## 风险提示

1. **换币滑点**：在低流动性时，换币可能产生较大滑点（建议设置 2-5%）
2. **Gas 费用**：每次换币会产生额外 Gas 费（Solana 约 0.00025 SOL）
3. **重启恢复**：必须确保数据库路径正确，否则无法恢复状态
4. **手动干预**：达到累计亏损限制后，需要手动重置 `manual_kill` 才能继续

---

## 推荐配置

```yaml
# config/meteora_dlmm_hft_meme_1.yml

# 基础参数
trading_pair: "MEME-SOL"
price_range_pct: 10.0  # ±10% 区间

# 止损参数
stop_loss_pct: 5.0  # 单次亏损 5% 止损
enable_60s_rule: true
out_of_range_timeout_seconds: 60

# 新增风控参数
total_loss_limit_pct: 15.0  # 累计亏损 15% 暂停
enable_swap_on_downside: true  # 下穿换 SOL
swap_slippage_pct: 2.0  # 2% 滑点
downside_cooldown_seconds: 300  # 下穿冷却 5 分钟
upside_cooldown_seconds: 60  # 上穿冷却 1 分钟

# 持久化参数
enable_state_persistence: true
state_db_path: "data/meteora_hft_state.db"
```
