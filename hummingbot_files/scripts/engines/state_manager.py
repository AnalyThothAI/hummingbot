"""
状态持久化管理器
用于记录和恢复策略的累计盈亏、开仓历史等状态

特性:
- SQLite 数据库持久化
- 累计盈亏追踪
- 亏损限制检查
- 重启后自动恢复状态
"""

import sqlite3
import time
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class StateManager:
    """策略状态持久化管理器"""

    def __init__(self, db_path: str, logger=None):
        """
        初始化状态管理器

        Args:
            db_path: 数据库文件路径
            logger: 日志记录器（可选）
        """
        self.db_path = db_path
        self.logger = logger

        # 确保数据库目录存在
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # 初始化数据库表
        self._init_db()

        if self.logger:
            self.logger.info(f"✅ StateManager 初始化完成: {db_path}")

    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # === 表1: 累计盈亏记录表 ===
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_pnl (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp BIGINT NOT NULL,
                event_type TEXT NOT NULL,
                position_id TEXT,
                entry_price REAL,
                exit_price REAL,
                realized_pnl REAL NOT NULL,
                cumulative_pnl REAL NOT NULL,
                quote_token TEXT,
                reason TEXT,
                swap_to_sol BOOLEAN DEFAULT 0,
                base_amount REAL,
                quote_amount REAL
            )
        """)

        # 创建索引加速查询
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pnl_timestamp
            ON strategy_pnl(timestamp DESC)
        """)

        # === 表2: 策略状态表（单例记录）===
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                cumulative_pnl REAL DEFAULT 0,
                initial_capital REAL DEFAULT 0,
                total_open_count INTEGER DEFAULT 0,
                total_close_count INTEGER DEFAULT 0,
                total_swap_count INTEGER DEFAULT 0,
                last_updated BIGINT,
                manual_kill BOOLEAN DEFAULT 0,
                stop_reason TEXT,
                current_position_id TEXT,
                current_entry_price REAL
            )
        """)

        # 插入默认状态（仅在表为空时）
        cursor.execute("""
            INSERT OR IGNORE INTO strategy_state (
                id, cumulative_pnl, initial_capital, last_updated
            ) VALUES (1, 0, 0, ?)
        """, (int(time.time()),))

        conn.commit()
        conn.close()

    def get_state(self) -> Dict:
        """
        获取当前策略状态

        Returns:
            状态字典，包含累计盈亏、开仓次数等信息
        """
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
            "stop_reason": row[8] or "",
            "current_position_id": row[9],
            "current_entry_price": row[10]
        }

    def _default_state(self) -> Dict:
        """返回默认状态"""
        return {
            "cumulative_pnl": 0.0,
            "initial_capital": 0.0,
            "total_open_count": 0,
            "total_close_count": 0,
            "total_swap_count": 0,
            "last_updated": int(time.time()),
            "manual_kill": False,
            "stop_reason": "",
            "current_position_id": None,
            "current_entry_price": None
        }

    def update_state(self, **kwargs):
        """
        更新策略状态

        Args:
            **kwargs: 要更新的字段及其值
        """
        if not kwargs:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 构建 UPDATE 语句
        set_clauses = []
        values = []
        for key, value in kwargs.items():
            set_clauses.append(f"{key} = ?")
            values.append(value)

        set_clause = ", ".join(set_clauses)
        values.append(int(time.time()))

        cursor.execute(f"""
            UPDATE strategy_state
            SET {set_clause}, last_updated = ?
            WHERE id = 1
        """, values)

        conn.commit()
        conn.close()

        if self.logger:
            self.logger.debug(f"状态已更新: {kwargs}")

    def set_initial_capital(self, capital: Decimal):
        """
        设置初始资金（仅在首次开仓时设置）

        Args:
            capital: 初始资金数量
        """
        state = self.get_state()
        if state["initial_capital"] <= 0:
            self.update_state(initial_capital=float(capital))
            if self.logger:
                self.logger.info(f"✅ 初始资金已设置: {capital:.6f}")

    def record_open(self, position_id: str, entry_price: Decimal):
        """
        记录开仓事件

        Args:
            position_id: 仓位ID
            entry_price: 开仓价格
        """
        self.update_state(
            total_open_count=self.get_state()["total_open_count"] + 1,
            current_position_id=position_id,
            current_entry_price=float(entry_price)
        )

        if self.logger:
            self.logger.info(f"📝 记录开仓: {position_id[:10]}... @ {entry_price:.8f}")

    def record_close(
        self,
        position_id: str,
        entry_price: Decimal,
        exit_price: Decimal,
        realized_pnl: Decimal,
        reason: str,
        swapped_to_sol: bool = False,
        base_amount: Optional[Decimal] = None,
        quote_amount: Optional[Decimal] = None
    ) -> Decimal:
        """
        记录平仓事件并更新累计盈亏

        Args:
            position_id: 仓位ID
            entry_price: 开仓价格
            exit_price: 平仓价格
            realized_pnl: 实际盈亏（以quote token计价）
            reason: 平仓原因
            swapped_to_sol: 是否换成了SOL
            base_amount: base token数量
            quote_amount: quote token数量

        Returns:
            更新后的累计盈亏
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取当前累计盈亏
        state = self.get_state()
        new_cumulative_pnl = state["cumulative_pnl"] + float(realized_pnl)

        # 插入盈亏记录
        cursor.execute("""
            INSERT INTO strategy_pnl (
                timestamp, event_type, position_id, entry_price, exit_price,
                realized_pnl, cumulative_pnl, quote_token, reason, swap_to_sol,
                base_amount, quote_amount
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(time.time()),
            "CLOSE",
            position_id,
            float(entry_price),
            float(exit_price),
            float(realized_pnl),
            new_cumulative_pnl,
            "SOL",  # 假设以 SOL 计价
            reason,
            swapped_to_sol,
            float(base_amount) if base_amount else None,
            float(quote_amount) if quote_amount else None
        ))

        # 更新累计盈亏和计数器
        cursor.execute("""
            UPDATE strategy_state
            SET cumulative_pnl = ?,
                total_close_count = total_close_count + 1,
                current_position_id = NULL,
                current_entry_price = NULL,
                last_updated = ?
            WHERE id = 1
        """, (new_cumulative_pnl, int(time.time())))

        conn.commit()
        conn.close()

        if self.logger:
            pnl_icon = "📈" if realized_pnl > 0 else "📉"
            self.logger.info(
                f"{pnl_icon} 记录平仓: {position_id[:10]}...\n"
                f"  本次盈亏: {realized_pnl:+.6f}\n"
                f"  累计盈亏: {new_cumulative_pnl:+.6f}\n"
                f"  原因: {reason}\n"
                f"  换SOL: {'是' if swapped_to_sol else '否'}"
            )

        return Decimal(str(new_cumulative_pnl))

    def record_swap(self, from_token: str, from_amount: Decimal, to_amount: Decimal, reason: str):
        """
        记录换币事件

        Args:
            from_token: 源代币
            from_amount: 源代币数量
            to_amount: 换得的SOL数量
            reason: 换币原因
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO strategy_pnl (
                timestamp, event_type, realized_pnl, cumulative_pnl,
                quote_token, reason, swap_to_sol, base_amount, quote_amount
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(time.time()),
            "SWAP",
            0,  # 换币本身不计入盈亏
            self.get_state()["cumulative_pnl"],
            "SOL",
            reason,
            True,
            float(from_amount),
            float(to_amount)
        ))

        # 更新换币计数
        cursor.execute("""
            UPDATE strategy_state
            SET total_swap_count = total_swap_count + 1, last_updated = ?
            WHERE id = 1
        """, (int(time.time()),))

        conn.commit()
        conn.close()

        if self.logger:
            self.logger.info(
                f"🔄 记录换币: {from_amount:.6f} {from_token} → {to_amount:.6f} SOL\n"
                f"  原因: {reason}"
            )

    def check_loss_limit(self, loss_limit_pct: Decimal = Decimal("15")) -> Tuple[bool, str]:
        """
        检查是否达到累计亏损限制

        Args:
            loss_limit_pct: 亏损限制百分比（默认15%）

        Returns:
            (是否达到限制, 详细信息)
        """
        state = self.get_state()

        # 如果已经被暂停
        if state["manual_kill"]:
            return True, state["stop_reason"]

        # 如果没有初始资金，无法判断
        if state["initial_capital"] <= 0:
            return False, ""

        cumulative_pnl = state["cumulative_pnl"]
        initial_capital = state["initial_capital"]

        # 计算亏损百分比
        loss_pct = abs(cumulative_pnl / initial_capital) * 100

        # 检查是否达到限制（仅在亏损时）
        if cumulative_pnl < 0 and loss_pct >= float(loss_limit_pct):
            # 设置暂停标志
            stop_reason = f"TOTAL_LOSS_LIMIT_{loss_pct:.2f}%"
            self.update_state(
                manual_kill=True,
                stop_reason=stop_reason
            )

            if self.logger:
                self.logger.error(
                    f"🚨 达到累计亏损限制！\n"
                    f"  累计亏损: {cumulative_pnl:.6f} ({loss_pct:.2f}%)\n"
                    f"  初始资金: {initial_capital:.6f}\n"
                    f"  限制阈值: {loss_limit_pct}%"
                )

            return True, stop_reason

        return False, ""

    def reset_manual_kill(self):
        """重置暂停标志（需要手动调用）"""
        self.update_state(manual_kill=False, stop_reason="")
        if self.logger:
            self.logger.warning("⚠️  手动重置暂停标志，策略将继续运行")

    def get_recent_pnl(self, limit: int = 10) -> List[Dict]:
        """
        获取最近的盈亏记录

        Args:
            limit: 返回记录数量

        Returns:
            盈亏记录列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT timestamp, event_type, position_id, entry_price, exit_price,
                   realized_pnl, cumulative_pnl, reason, swap_to_sol
            FROM strategy_pnl
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        records = []
        for row in rows:
            records.append({
                "timestamp": row[0],
                "event_type": row[1],
                "position_id": row[2],
                "entry_price": row[3],
                "exit_price": row[4],
                "realized_pnl": row[5],
                "cumulative_pnl": row[6],
                "reason": row[7],
                "swap_to_sol": bool(row[8])
            })

        return records

    def get_summary(self) -> str:
        """
        获取状态摘要（用于显示）

        Returns:
            格式化的状态摘要字符串
        """
        state = self.get_state()

        lines = [
            "=" * 50,
            "📊 策略状态摘要",
            "=" * 50,
            f"累计盈亏: {state['cumulative_pnl']:+.6f} SOL",
            f"初始资金: {state['initial_capital']:.6f} SOL",
        ]

        if state['initial_capital'] > 0:
            pnl_pct = (state['cumulative_pnl'] / state['initial_capital']) * 100
            lines.append(f"收益率: {pnl_pct:+.2f}%")

        lines.extend([
            f"开仓次数: {state['total_open_count']}",
            f"平仓次数: {state['total_close_count']}",
            f"换币次数: {state['total_swap_count']}",
            f"策略状态: {'🔴 已暂停' if state['manual_kill'] else '🟢 运行中'}",
        ])

        if state['manual_kill']:
            lines.append(f"暂停原因: {state['stop_reason']}")

        if state['current_position_id']:
            lines.append(f"\n当前仓位: {state['current_position_id'][:10]}...")
            lines.append(f"开仓价格: {state['current_entry_price']:.8f}")

        lines.append("=" * 50)

        return "\n".join(lines)


# ========================================
# 测试代码
# ========================================

if __name__ == "__main__":
    import logging

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # 测试
    manager = StateManager("test_state.db", logger)

    # 设置初始资金
    manager.set_initial_capital(Decimal("100"))

    # 记录开仓
    manager.record_open("pos_123", Decimal("1.0"))

    # 记录平仓（亏损）
    manager.record_close(
        position_id="pos_123",
        entry_price=Decimal("1.0"),
        exit_price=Decimal("0.9"),
        realized_pnl=Decimal("-5"),
        reason="下跌止损",
        swapped_to_sol=True
    )

    # 检查亏损限制
    is_limited, reason = manager.check_loss_limit(Decimal("15"))
    print(f"\n亏损限制: {is_limited}, 原因: {reason}")

    # 显示摘要
    print(f"\n{manager.get_summary()}")

    # 获取最近记录
    print("\n最近盈亏记录:")
    for record in manager.get_recent_pnl(5):
        print(f"  {record['event_type']}: {record['realized_pnl']:+.2f}")
