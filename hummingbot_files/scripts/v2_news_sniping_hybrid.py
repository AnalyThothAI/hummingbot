# -*- coding: utf-8 -*-
"""
BSC 新闻狙击策略 - V2 Hybrid 版本

V2 Hybrid 架构：
- ✅ 使用 StrategyV2Base 框架
- ✅ 使用 V2 的 market_data_provider
- ❌ 不使用 PositionExecutor（手动管理订单）
- ✅ 手动实现止盈止损逻辑
- ✅ Gateway DEX 兼容

适用场景：
- 学习 V2 框架
- 需要完全控制订单逻辑
- Gateway DEX 交易
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from decimal import Decimal
from typing import Dict, List, Optional

from pydantic import Field, field_validator

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    MarketOrderFailureEvent,
    OrderCancelledEvent,
    OrderFilledEvent,
    SellOrderCompletedEvent,
)
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction

# MQTT 支持（可选）
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

# Redis 支持（可选）
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class NewsSnipingV2HybridConfig(StrategyV2ConfigBase):
    """V2 Hybrid 配置类"""

    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))

    # ========== V2 必需配置 ==========
    markets: Dict[str, set] = {}
    candles_config: List[CandlesConfig] = []
    controllers_config: List[str] = []

    # ========== 连接器配置 ==========
    connector: str = Field(
        default="pancakeswap",
        json_schema_extra={
            "prompt": "Enter the connector name (e.g., pancakeswap, uniswap)",
            "prompt_on_new": True
        }
    )

    trading_pair: str = Field(
        default="WBNB-USDT",
        json_schema_extra={
            "prompt": "Enter the default trading pair (e.g., WBNB-USDT)",
            "prompt_on_new": True
        }
    )

    # ========== MQTT 配置 ==========
    mqtt_broker: str = Field(default="localhost")
    mqtt_port: int = Field(default=1883)
    mqtt_topic: str = Field(default="trading/bsc/snipe")
    mqtt_username: str = Field(default="")
    mqtt_password: str = Field(default="")

    # ========== 交易配置 ==========
    default_trade_amount: Decimal = Field(default=Decimal("0.001"))
    default_quote_token: str = Field(default="WBNB")
    slippage: Decimal = Field(default=Decimal("0.02"))
    gas_buffer: Decimal = Field(default=Decimal("1.15"))

    # ========== 手动止盈止损配置 ==========
    stop_loss_pct: Decimal = Field(default=Decimal("0.10"))
    take_profit_pct: Decimal = Field(default=Decimal("0.05"))
    time_limit_seconds: int = Field(default=300)

    enable_manual_tp_sl: bool = Field(
        default=True,
        json_schema_extra={
            "prompt": "Enable manual take-profit/stop-loss monitoring?",
            "prompt_on_new": True
        }
    )

    # ========== Redis 去重配置 ==========
    enable_signal_deduplication: bool = Field(default=True)
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    signal_dedup_window_seconds: int = Field(default=60)

    # ========== 参数验证 ==========
    @field_validator("slippage", "stop_loss_pct", "take_profit_pct")
    @classmethod
    def validate_percentage(cls, v):
        if v < Decimal("0") or v > Decimal("1"):
            raise ValueError("Percentage must be between 0 and 1")
        return v

    @field_validator("gas_buffer")
    @classmethod
    def validate_gas_buffer(cls, v):
        if v < Decimal("1.0") or v > Decimal("2.0"):
            raise ValueError("Gas buffer must be between 1.0 and 2.0")
        return v


class NewsSnipingV2Hybrid(StrategyV2Base):
    """
    BSC 新闻狙击策略 - V2 Hybrid 版本

    使用 V2 框架但手动管理订单，不使用 PositionExecutor
    """

    @classmethod
    def init_markets(cls, config: NewsSnipingV2HybridConfig):
        """初始化市场连接器"""
        cls.markets = {config.connector: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: NewsSnipingV2HybridConfig):
        """初始化策略"""
        super().__init__(connectors, config)
        self.config = config
        self.connector = self.connectors[self.config.connector]

        # 事件循环引用
        self._event_loop = asyncio.get_event_loop()

        # ========== 手动订单管理 ==========
        self.pending_orders: Dict[str, dict] = {}  # 未成交订单
        self.active_positions: Dict[str, dict] = {}  # 已成交持仓
        self.position_monitors: Dict[str, asyncio.Task] = {}  # 监控任务

        # 统计
        self.stats = {
            "signals_received": 0,
            "signals_deduplicated": 0,
            "orders_created": 0,
            "orders_filled": 0,
            "orders_failed": 0,
            "tp_triggered": 0,
            "sl_triggered": 0,
            "timeout_triggered": 0,
        }

        # Redis
        self.redis_client = None
        if config.enable_signal_deduplication and REDIS_AVAILABLE:
            self._setup_redis()

        # MQTT
        self.mqtt_client = None
        if MQTT_AVAILABLE:
            self._setup_mqtt()
        else:
            self.logger().warning("⚠️  MQTT 库未安装")

        total_adj = ((Decimal("1") + self.config.slippage) * self.config.gas_buffer - Decimal("1")) * 100

        self.logger().info(
            f"🚀 新闻狙击策略 V2 Hybrid 已启动\n"
            f"   架构: V2 Base + 手动订单管理\n"
            f"   Connector: {self.config.connector}\n"
            f"   Trading Pair: {self.config.trading_pair}\n"
            f"   Slippage: {self.config.slippage * 100}%\n"
            f"   Gas Buffer: {(self.config.gas_buffer - 1) * 100}%\n"
            f"   Total Adjustment: {total_adj:.2f}%\n"
            f"   Manual TP/SL: {'Enabled' if self.config.enable_manual_tp_sl else 'Disabled'}\n"
            f"   Stop Loss: {self.config.stop_loss_pct * 100}%\n"
            f"   Take Profit: {self.config.take_profit_pct * 100}%\n"
            f"   Time Limit: {self.config.time_limit_seconds}s"
        )

    # ========== Redis ==========
    def _setup_redis(self):
        """初始化 Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            self.logger().info("✅ Redis 已连接")
        except Exception as e:
            self.logger().warning(f"⚠️  Redis 连接失败: {e}")
            self.redis_client = None

    def _is_signal_duplicate(self, signal_data: dict) -> bool:
        """检查信号是否重复"""
        if not self.redis_client:
            return False

        try:
            signal_key_data = {
                "side": signal_data.get("side", "").upper(),
                "base_token": signal_data.get("base_token", "").upper(),
                "quote_token": signal_data.get("quote_token", self.config.default_quote_token).upper(),
            }
            signal_fingerprint = hashlib.md5(
                json.dumps(signal_key_data, sort_keys=True).encode()
            ).hexdigest()

            redis_key = f"signal:news_snipe_hybrid:{signal_fingerprint}"

            if self.redis_client.exists(redis_key):
                self.logger().info(f"🔄 重复信号已忽略: {signal_key_data}")
                return True

            self.redis_client.setex(
                redis_key,
                self.config.signal_dedup_window_seconds,
                int(time.time())
            )
            return False

        except Exception as e:
            self.logger().warning(f"⚠️  Redis 去重检查失败: {e}")
            return False

    # ========== MQTT ==========
    def _setup_mqtt(self):
        """初始化 MQTT"""
        try:
            self.mqtt_client = mqtt.Client(
                client_id=f"news_snipe_hybrid_{int(time.time())}",
                clean_session=True
            )

            if self.config.mqtt_username and self.config.mqtt_password:
                self.mqtt_client.username_pw_set(
                    self.config.mqtt_username,
                    self.config.mqtt_password
                )

            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_mqtt_message

            self.mqtt_client.connect_async(
                self.config.mqtt_broker,
                self.config.mqtt_port
            )
            self.mqtt_client.loop_start()

            self.logger().info("📡 MQTT 连接中...")
        except Exception as e:
            self.logger().error(f"❌ MQTT 连接失败: {e}")

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT 连接回调"""
        if rc == 0:
            client.subscribe(self.config.mqtt_topic)
            self.logger().info(f"✅ MQTT 已订阅: {self.config.mqtt_topic}")
        else:
            self.logger().error(f"❌ MQTT 连接失败，错误码: {rc}")

    def _on_mqtt_message(self, client, userdata, msg):
        """处理 MQTT 消息"""
        try:
            data = json.loads(msg.payload.decode())
            self.stats["signals_received"] += 1
            self.logger().info(f"📩 信号: {data}")

            # 去重
            if self.config.enable_signal_deduplication:
                if self._is_signal_duplicate(data):
                    self.stats["signals_deduplicated"] += 1
                    return

            # 解析信号
            side = data.get("side", "BUY").upper()
            base = self._normalize_token_symbol(data.get("base_token", ""))
            quote = self._normalize_token_symbol(
                data.get("quote_token", self.config.default_quote_token)
            )
            amount = Decimal(str(data.get("amount", self.config.default_trade_amount)))
            slippage = data.get("slippage")
            if slippage is not None:
                slippage = Decimal(str(slippage))
            else:
                slippage = self.config.slippage

            # 验证
            if not base or side not in ["BUY", "SELL"]:
                self.logger().error(f"❌ 无效信号: {data}")
                return

            # 调度到主事件循环
            asyncio.run_coroutine_threadsafe(
                self._process_signal(side, base, quote, amount, slippage),
                self._event_loop
            )

        except Exception as e:
            self.logger().error(f"❌ 处理 MQTT 消息失败: {e}")

    def _normalize_token_symbol(self, token: str) -> str:
        """统一代币符号"""
        if not token:
            return token

        if token.startswith("0x") or token.startswith("0X"):
            return token

        token_upper = token.upper()

        if token_upper == "BNB":
            self.logger().info("ℹ️  自动将 BNB 转换为 WBNB")
            return "WBNB"

        return token_upper

    # ========== 手动订单管理（核心改进）==========
    async def _process_signal(
        self,
        side: str,
        base: str,
        quote: str,
        amount: Decimal,
        slippage: Decimal
    ):
        """
        处理交易信号 - 手动下单（不使用 PositionExecutor）
        """
        try:
            trading_pair = f"{base}-{quote}"
            is_buy = (side == "BUY")
            trade_type = TradeType.BUY if is_buy else TradeType.SELL

            self.logger().info(
                f"🎯 处理信号: {side} {trading_pair}, Amount: {amount}, Slippage: {slippage * 100}%"
            )

            # 计算交易数量
            if is_buy:
                # 获取参考价格
                temp_price = await self.connector.get_quote_price(
                    trading_pair=trading_pair,
                    is_buy=True,
                    amount=Decimal("1")
                )

                if not temp_price or temp_price <= 0:
                    self.logger().error(f"❌ 无法获取 {trading_pair} 价格")
                    return

                base_amount = amount * temp_price
                self.logger().info(f"💰 买入: 用 {amount} {quote} 买入约 {base_amount:.6f} {base}")
            else:
                base_amount = amount
                self.logger().info(f"💰 卖出: {base_amount} {base}")

            # 量化
            base_amount = self.connector.quantize_order_amount(trading_pair, base_amount)

            # 获取精确报价
            entry_price = await self.connector.get_quote_price(
                trading_pair=trading_pair,
                is_buy=is_buy,
                amount=base_amount
            )

            if not entry_price or entry_price <= 0:
                self.logger().error(f"❌ 无法获取 {trading_pair} 精确报价")
                return

            # 调整价格（滑点 + gas buffer）
            if is_buy:
                entry_price = entry_price / ((Decimal("1") + slippage) * self.config.gas_buffer)
            else:
                entry_price = entry_price * ((Decimal("1") + slippage) * self.config.gas_buffer)

            # 量化价格
            entry_price = self.connector.quantize_order_price(trading_pair, entry_price)

            # ========== 手动下单（替代 PositionExecutor）==========
            # 创建 market_trading_pair_tuple
            from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
            market_trading_pair_tuple = MarketTradingPairTuple(
                self.connector,
                trading_pair,
                *trading_pair.split("-")
            )

            if is_buy:
                order_id = self.buy_with_specific_market(
                    market_trading_pair_tuple=market_trading_pair_tuple,
                    amount=base_amount,
                    order_type=OrderType.LIMIT,
                    price=entry_price
                )
            else:
                order_id = self.sell_with_specific_market(
                    market_trading_pair_tuple=market_trading_pair_tuple,
                    amount=base_amount,
                    order_type=OrderType.LIMIT,
                    price=entry_price
                )

            # 追踪订单
            self.pending_orders[order_id] = {
                "side": side,
                "trading_pair": trading_pair,
                "base": base,
                "quote": quote,
                "amount": base_amount,
                "entry_price": entry_price,
                "timestamp": time.time(),
                "slippage": slippage,
            }

            self.stats["orders_created"] += 1

            total_adj = ((Decimal("1") + slippage) * self.config.gas_buffer - Decimal("1")) * 100

            self.logger().info(
                f"✅ 订单已创建 (手动管理)\n"
                f"   Order ID: {order_id}\n"
                f"   交易对: {trading_pair}\n"
                f"   方向: {side}\n"
                f"   数量: {base_amount:.6f} {base}\n"
                f"   入场价: {entry_price:.6f}\n"
                f"   滑点: {slippage * 100}%\n"
                f"   Gas Buffer: {(self.config.gas_buffer - 1) * 100}%\n"
                f"   总调整: {total_adj:.2f}%"
            )

        except Exception as e:
            self.logger().error(f"❌ 处理信号失败: {e}", exc_info=True)
            self.stats["orders_failed"] += 1

    # ========== 订单事件回调（手动管理）==========
    def did_fill_order(self, event: OrderFilledEvent):
        """订单成交回调 - 启动止盈止损监控"""
        order_id = event.order_id

        if order_id in self.pending_orders:
            order_info = self.pending_orders[order_id]

            self.logger().info(
                f"🎉 订单成交: {order_id}\n"
                f"   交易对: {order_info['trading_pair']}\n"
                f"   方向: {order_info['side']}\n"
                f"   数量: {event.amount}\n"
                f"   价格: {event.price}"
            )

            # 移到 active_positions
            self.active_positions[order_id] = {
                **order_info,
                "fill_price": event.price,
                "fill_amount": event.amount,
                "fill_timestamp": time.time(),
            }
            del self.pending_orders[order_id]

            self.stats["orders_filled"] += 1

            # ========== 启动手动止盈止损监控 ==========
            if self.config.enable_manual_tp_sl:
                monitor_task = safe_ensure_future(self._monitor_position(order_id))
                self.position_monitors[order_id] = monitor_task
                self.logger().info(f"📊 已启动持仓监控: {order_id}")

    def did_complete_buy_order(self, event: BuyOrderCompletedEvent):
        """买单完成"""
        self.logger().info(f"✅ 买单完成: {event.order_id}")

    def did_complete_sell_order(self, event: SellOrderCompletedEvent):
        """卖单完成"""
        self.logger().info(f"✅ 卖单完成: {event.order_id}")

    def did_fail_order(self, event: MarketOrderFailureEvent):
        """订单失败"""
        order_id = event.order_id
        self.logger().error(f"❌ 订单失败: {order_id}")

        if order_id in self.pending_orders:
            del self.pending_orders[order_id]

        self.stats["orders_failed"] += 1

    def did_cancel_order(self, event: OrderCancelledEvent):
        """订单取消"""
        order_id = event.order_id
        self.logger().warning(f"⚠️  订单已取消: {order_id}")

        if order_id in self.pending_orders:
            del self.pending_orders[order_id]

    # ========== 手动止盈止损逻辑 ==========
    async def _monitor_position(self, order_id: str):
        """
        监控持仓的止盈止损

        这是 V2 Hybrid 的核心：手动实现 PositionExecutor 的功能
        """
        try:
            if order_id not in self.active_positions:
                return

            position = self.active_positions[order_id]
            entry_price = position["fill_price"]
            trading_pair = position["trading_pair"]
            side = position["side"]
            is_buy = (side == "BUY")
            start_time = position["fill_timestamp"]

            self.logger().info(
                f"👁️  开始监控持仓: {order_id}\n"
                f"   交易对: {trading_pair}\n"
                f"   方向: {side}\n"
                f"   入场价: {entry_price}\n"
                f"   止损: {self.config.stop_loss_pct * 100}%\n"
                f"   止盈: {self.config.take_profit_pct * 100}%\n"
                f"   超时: {self.config.time_limit_seconds}s"
            )

            while order_id in self.active_positions:
                try:
                    # 检查超时
                    if self.config.time_limit_seconds > 0:
                        elapsed = time.time() - start_time
                        if elapsed > self.config.time_limit_seconds:
                            self.logger().info(f"⏰ 持仓超时，准备平仓: {order_id}")
                            await self._close_position(order_id, "TIMEOUT")
                            break

                    # 获取当前价格（用于 PnL 计算）
                    # 重要：需要获取与入场时相同方向的价格进行比较
                    current_price = await self.connector.get_quote_price(
                        trading_pair=trading_pair,
                        is_buy=is_buy,  # 与入场方向一致！
                        amount=position["fill_amount"]
                    )

                    if not current_price or current_price <= 0:
                        self.logger().warning(f"⚠️  无法获取 {trading_pair} 当前价格，等待下次检查")
                        await asyncio.sleep(1)
                        continue

                    # 计算 PnL
                    # 注意：price 是 base/quote 汇率（例如 32178 TOKEN/WBNB）
                    if is_buy:
                        # 买入后：current_price > entry_price 表示盈利
                        # 例如：入场 32178 TOKEN/WBNB，现在 35000 TOKEN/WBNB，盈利
                        pnl_pct = (current_price - entry_price) / entry_price
                    else:
                        # 卖出后：current_price < entry_price 表示盈利
                        # 例如：入场卖出 32178，现在 30000，盈利
                        pnl_pct = (entry_price - current_price) / entry_price

                    # 检查止损
                    if pnl_pct <= -self.config.stop_loss_pct:
                        self.logger().info(
                            f"🛑 触发止损: {order_id}\n"
                            f"   入场价: {entry_price}\n"
                            f"   当前价: {current_price}\n"
                            f"   PnL: {pnl_pct * 100:.2f}%"
                        )
                        await self._close_position(order_id, "STOP_LOSS")
                        break

                    # 检查止盈
                    elif pnl_pct >= self.config.take_profit_pct:
                        self.logger().info(
                            f"🎯 触发止盈: {order_id}\n"
                            f"   入场价: {entry_price}\n"
                            f"   当前价: {current_price}\n"
                            f"   PnL: {pnl_pct * 100:.2f}%"
                        )
                        await self._close_position(order_id, "TAKE_PROFIT")
                        break

                    # 定期输出状态
                    if int(time.time()) % 10 == 0:  # 每10秒
                        self.logger().info(
                            f"📊 持仓状态: {order_id} | "
                            f"PnL: {pnl_pct * 100:.2f}% | "
                            f"Price: {current_price:.6f}"
                        )

                    await asyncio.sleep(1)  # 每秒检查一次

                except asyncio.CancelledError:
                    self.logger().info(f"⚠️  监控任务已取消: {order_id}")
                    break
                except Exception as e:
                    self.logger().error(f"❌ 监控循环出错: {e}")
                    await asyncio.sleep(1)

        except Exception as e:
            self.logger().error(f"❌ 监控持仓失败: {e}", exc_info=True)
        finally:
            # 清理
            if order_id in self.position_monitors:
                del self.position_monitors[order_id]

    async def _close_position(self, order_id: str, reason: str):
        """
        平仓
        """
        if order_id not in self.active_positions:
            return

        try:
            position = self.active_positions[order_id]
            trading_pair = position["trading_pair"]
            side = position["side"]
            amount = position["fill_amount"]
            is_buy = (side == "BUY")

            self.logger().info(
                f"🔄 平仓: {order_id}\n"
                f"   原因: {reason}\n"
                f"   交易对: {trading_pair}\n"
                f"   方向: {'SELL' if is_buy else 'BUY'}\n"
                f"   数量: {amount}"
            )

            # 获取平仓价格
            close_price = await self.connector.get_quote_price(
                trading_pair=trading_pair,
                is_buy=not is_buy,
                amount=amount
            )

            if not close_price or close_price <= 0:
                self.logger().error(f"❌ 无法获取平仓价格")
                return

            # 应用滑点
            if not is_buy:
                # 原买入，现卖出
                close_price = close_price / (Decimal("1") + self.config.slippage)
            else:
                # 原卖出，现买入
                close_price = close_price * (Decimal("1") + self.config.slippage)

            close_price = self.connector.quantize_order_price(trading_pair, close_price)

            # 创建 market_trading_pair_tuple
            from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
            market_trading_pair_tuple = MarketTradingPairTuple(
                self.connector,
                trading_pair,
                *trading_pair.split("-")
            )

            # 下平仓单
            if is_buy:
                # 原买入，现卖出
                close_order_id = self.sell_with_specific_market(
                    market_trading_pair_tuple=market_trading_pair_tuple,
                    amount=amount,
                    order_type=OrderType.LIMIT,
                    price=close_price
                )
            else:
                # 原卖出，现买入
                close_order_id = self.buy_with_specific_market(
                    market_trading_pair_tuple=market_trading_pair_tuple,
                    amount=amount,
                    order_type=OrderType.LIMIT,
                    price=close_price
                )

            self.logger().info(f"✅ 平仓单已创建: {close_order_id}")

            # 更新统计
            if reason == "STOP_LOSS":
                self.stats["sl_triggered"] += 1
            elif reason == "TAKE_PROFIT":
                self.stats["tp_triggered"] += 1
            elif reason == "TIMEOUT":
                self.stats["timeout_triggered"] += 1

            # 移除持仓
            del self.active_positions[order_id]

            # 取消监控任务
            if order_id in self.position_monitors:
                self.position_monitors[order_id].cancel()

        except Exception as e:
            self.logger().error(f"❌ 平仓失败: {e}", exc_info=True)

    # ========== V2 必需方法（返回空）==========
    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        """
        V2 必需方法

        我们不使用 Executor，所以返回空列表
        订单由 MQTT 信号触发，手动管理
        """
        return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        """
        V2 必需方法

        订单由手动监控管理，不需要自动停止
        """
        return []

    # ========== 状态显示 ==========
    def format_status(self) -> str:
        """格式化状态显示"""
        if not self.ready_to_trade:
            return "⚠️ 未就绪"

        lines = []
        lines.append("=" * 70)
        lines.append("🎯 新闻狙击策略 V2 Hybrid - 运行状态")
        lines.append("=" * 70)

        # 架构说明
        lines.append("\n🏗️  架构:")
        lines.append("  ✅ V2 StrategyV2Base")
        lines.append("  ❌ 不使用 PositionExecutor")
        lines.append("  ✅ 手动订单管理")
        lines.append("  ✅ 手动止盈止损监控")

        # 配置
        lines.append("\n📋 配置:")
        lines.append(f"  Connector: {self.config.connector}")
        lines.append(f"  Trading Pair: {self.config.trading_pair}")
        lines.append(f"  Slippage: {self.config.slippage * 100}%")
        lines.append(f"  Gas Buffer: {(self.config.gas_buffer - 1) * 100}%")
        total_adj = ((Decimal("1") + self.config.slippage) * self.config.gas_buffer - Decimal("1")) * 100
        lines.append(f"  Total Adjustment: {total_adj:.2f}%")
        lines.append(f"  Stop Loss: {self.config.stop_loss_pct * 100}%")
        lines.append(f"  Take Profit: {self.config.take_profit_pct * 100}%")
        lines.append(f"  Time Limit: {self.config.time_limit_seconds}s")

        # 连接状态
        lines.append("\n🔌 连接状态:")
        connector_status = "🟢 就绪" if self.connector.ready else "🔴 未就绪"
        lines.append(f"  Connector: {connector_status}")

        if self.mqtt_client:
            mqtt_status = "🟢 连接" if self.mqtt_client.is_connected() else "🔴 断开"
        else:
            mqtt_status = "⚪ 未启用"
        lines.append(f"  MQTT: {mqtt_status}")

        if self.redis_client:
            try:
                self.redis_client.ping()
                redis_status = "🟢 连接"
            except:
                redis_status = "🔴 断开"
        else:
            redis_status = "⚪ 未启用"
        lines.append(f"  Redis: {redis_status}")

        # 统计
        lines.append("\n📊 统计:")
        lines.append(f"  信号接收: {self.stats['signals_received']}")
        lines.append(f"  信号去重: {self.stats['signals_deduplicated']}")
        lines.append(f"  订单创建: {self.stats['orders_created']}")
        lines.append(f"  订单成交: {self.stats['orders_filled']}")
        lines.append(f"  订单失败: {self.stats['orders_failed']}")
        lines.append(f"  止盈触发: {self.stats['tp_triggered']}")
        lines.append(f"  止损触发: {self.stats['sl_triggered']}")
        lines.append(f"  超时触发: {self.stats['timeout_triggered']}")

        # 未成交订单
        if self.pending_orders:
            lines.append(f"\n⏳ 未成交订单: {len(self.pending_orders)}")
            for oid, info in list(self.pending_orders.items())[:3]:
                lines.append(
                    f"  - {info['trading_pair']} {info['side']} | "
                    f"Amount: {info['amount']:.6f} | "
                    f"Price: {info['entry_price']:.6f}"
                )

        # 活跃持仓
        if self.active_positions:
            lines.append(f"\n🔄 活跃持仓: {len(self.active_positions)}")
            for oid, info in list(self.active_positions.items())[:3]:
                elapsed = int(time.time() - info['fill_timestamp'])
                lines.append(
                    f"  - {info['trading_pair']} {info['side']} | "
                    f"Amount: {info['fill_amount']:.6f} | "
                    f"Entry: {info['fill_price']:.6f} | "
                    f"Time: {elapsed}s"
                )

        # 余额
        lines.append("\n💰 余额:")
        try:
            balance_df = self.get_balance_df()
            for line in balance_df.to_string(index=False).split("\n"):
                lines.append(f"  {line}")
        except Exception as e:
            lines.append(f"  无法获取余额: {e}")

        lines.append("=" * 70)
        return "\n".join(lines)

    # ========== 生命周期 ==========
    def start(self, clock: Clock, timestamp: float) -> None:
        """启动策略"""
        super().start(clock, timestamp)
        self.logger().info("✅ V2 Hybrid 策略已启动")

    async def on_stop(self):
        """停止策略"""
        # 取消所有监控任务
        for order_id, task in self.position_monitors.items():
            if not task.done():
                task.cancel()
                self.logger().info(f"⚠️  取消监控任务: {order_id}")

        # 停止 MQTT
        if self.mqtt_client:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
                self.logger().info("📡 MQTT 已断开")
            except Exception as e:
                self.logger().warning(f"⚠️  MQTT 断开时出错: {e}")

        # 关闭 Redis
        if self.redis_client:
            try:
                self.redis_client.close()
                self.logger().info("💾 Redis 已关闭")
            except Exception as e:
                self.logger().warning(f"⚠️  Redis 关闭时出错: {e}")

        self.logger().info(
            f"🛑 V2 Hybrid 策略已停止\n"
            f"   总信号: {self.stats['signals_received']}\n"
            f"   订单创建: {self.stats['orders_created']}\n"
            f"   订单成交: {self.stats['orders_filled']}\n"
            f"   止盈: {self.stats['tp_triggered']}\n"
            f"   止损: {self.stats['sl_triggered']}\n"
            f"   超时: {self.stats['timeout_triggered']}"
        )
