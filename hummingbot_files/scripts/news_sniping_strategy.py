# -*- coding: utf-8 -*-
"""
BSC 新闻狙击策略

功能：
- 通过 MQTT 接收交易信号
- 支持滑点控制
- 支持自动重试
- BUY: amount 表示花费的 BNB/WBNB 数量
- SELL: amount 表示卖出的代币数量
- 自动识别 BNB/WBNB
"""

import asyncio
import json
import logging
import os
import time
from decimal import Decimal
from typing import Dict

from pydantic import Field, field_validator

from hummingbot.client.config.config_data_types import BaseClientModel
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.event.events import OrderFilledEvent, MarketOrderFailureEvent
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase

# 尝试导入 MQTT，如果失败则禁用 MQTT 功能
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False


class BscNewsSnipeConfig(BaseClientModel):
    """配置类"""

    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))

    # 连接器配置
    connector: str = Field(
        default="pancakeswap/router",
        json_schema_extra={
            "prompt": "Enter the connector name (e.g., pancakeswap/router)",
            "prompt_on_new": True
        }
    )

    trading_pair: str = Field(
        default="WBNB-USDT",
        json_schema_extra={
            "prompt": "Enter the trading pair (e.g., WBNB-USDT)",
            "prompt_on_new": True
        }
    )

    # MQTT 配置
    mqtt_broker: str = Field(
        default="localhost",
        json_schema_extra={
            "prompt": "Enter MQTT broker address",
            "prompt_on_new": True
        }
    )

    mqtt_port: int = Field(
        default=1883,
        json_schema_extra={
            "prompt": "Enter MQTT broker port",
            "prompt_on_new": True
        }
    )

    mqtt_topic: str = Field(
        default="trading/bsc/snipe",
        json_schema_extra={
            "prompt": "Enter MQTT topic to subscribe",
            "prompt_on_new": True
        }
    )

    mqtt_username: str = Field(
        default="",
        json_schema_extra={
            "prompt": "Enter MQTT username (leave empty if none)",
            "prompt_on_new": True
        }
    )

    mqtt_password: str = Field(
        default="",
        json_schema_extra={
            "prompt": "Enter MQTT password (leave empty if none)",
            "prompt_on_new": True
        }
    )

    # 交易配置
    default_trade_amount: Decimal = Field(
        default=Decimal("0.001"),
        json_schema_extra={
            "prompt": "Enter default trade amount (BUY=BNB amount, SELL=token amount)",
            "prompt_on_new": True
        }
    )

    default_quote_token: str = Field(
        default="WBNB",
        json_schema_extra={
            "prompt": "Enter default quote token (e.g., WBNB, USDT)",
            "prompt_on_new": True
        }
    )

    gas_buffer: Decimal = Field(
        default=Decimal("1.15"),
        json_schema_extra={
            "prompt": "Enter gas buffer multiplier (e.g., 1.15 for 15% buffer)",
            "prompt_on_new": True
        }
    )

    slippage: Decimal = Field(
        default=Decimal("0.15"),
        json_schema_extra={
            "prompt": "Enter slippage tolerance (e.g., 0.15 for 15%)",
            "prompt_on_new": True
        }
    )

    max_retries: int = Field(
        default=3,
        json_schema_extra={
            "prompt": "Enter max retry attempts for failed orders",
            "prompt_on_new": True
        }
    )

    retry_delay: Decimal = Field(
        default=Decimal("2.0"),
        json_schema_extra={
            "prompt": "Enter delay between retries in seconds",
            "prompt_on_new": True
        }
    )

    # 参数验证
    @field_validator("default_trade_amount", "gas_buffer", "slippage", "retry_delay")
    @classmethod
    def validate_decimal(cls, v):
        if isinstance(v, str):
            return Decimal(v)
        return v

    @field_validator("slippage")
    @classmethod
    def validate_slippage(cls, v):
        if v < Decimal("0") or v > Decimal("1"):
            raise ValueError("Slippage must be between 0 and 1 (0% to 100%)")
        return v

    @field_validator("max_retries")
    @classmethod
    def validate_retries(cls, v):
        if v < 0 or v > 10:
            raise ValueError("Max retries must be between 0 and 10")
        return v

    @field_validator("mqtt_port")
    @classmethod
    def validate_port(cls, v):
        if v < 1 or v > 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v


class BscNewsSnipeWithConfig(ScriptStrategyBase):
    """
    BSC 新闻狙击策略

    交易逻辑：
    - BUY: amount 表示花费的 BNB/WBNB 数量
    - SELL: amount 表示卖出的代币数量
    """

    @classmethod
    def init_markets(cls, config: BscNewsSnipeConfig):
        """初始化市场连接器"""
        cls.markets = {config.connector: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: BscNewsSnipeConfig):
        """初始化策略"""
        super().__init__(connectors)
        self.config = config
        self.connector = self.connectors[self.config.connector]

        # 保存事件循环引用（MQTT 回调需要）
        self._event_loop = asyncio.get_event_loop()

        # 订单追踪
        self.pending_orders = {}

        # 统计
        self.stats = {
            "signals": 0,
            "success": 0,
            "failed": 0,
            "retries": 0,
        }

        # MQTT 客户端
        self.mqtt_client = None
        if MQTT_AVAILABLE:
            self._setup_mqtt()
        else:
            self.log_with_clock(logging.WARNING, "⚠️  MQTT 库未安装，无法接收信号")

        self.log_with_clock(
            logging.INFO,
            f"🚀 BSC 新闻狙击已启动\n"
            f"   Connector: {self.config.connector}\n"
            f"   Trading Pair: {self.config.trading_pair}\n"
            f"   MQTT: {self.config.mqtt_broker}:{self.config.mqtt_port}\n"
            f"   Default Amount: {self.config.default_trade_amount}\n"
            f"   Gas Buffer: {self.config.gas_buffer}\n"
            f"   Slippage: {self.config.slippage * 100}%\n"
            f"   Max Retries: {self.config.max_retries}"
        )

    # ========== MQTT ==========
    def _setup_mqtt(self):
        """初始化 MQTT 连接"""
        try:
            self.mqtt_client = mqtt.Client(
                client_id=f"bsc_snipe_{int(time.time())}",
                clean_session=True
            )

            if self.config.mqtt_username and self.config.mqtt_password:
                self.mqtt_client.username_pw_set(
                    self.config.mqtt_username,
                    self.config.mqtt_password
                )

            self.mqtt_client.on_connect = self._on_connect
            self.mqtt_client.on_message = self._on_message

            self.mqtt_client.connect_async(
                self.config.mqtt_broker,
                self.config.mqtt_port
            )
            self.mqtt_client.loop_start()

            self.log_with_clock(logging.INFO, "📡 MQTT 已连接")
        except Exception as e:
            self.log_with_clock(logging.ERROR, f"❌ MQTT 连接失败: {e}")

    def _on_connect(self, client, userdata, flags, rc):
        """MQTT 连接回调"""
        if rc == 0:
            client.subscribe(self.config.mqtt_topic)
            self.log_with_clock(logging.INFO, f"✅ 订阅: {self.config.mqtt_topic}")
        else:
            self.log_with_clock(logging.ERROR, f"❌ MQTT 连接失败，错误码: {rc}")

    def _on_message(self, client, userdata, msg):
        """处理 MQTT 消息"""
        try:
            data = json.loads(msg.payload.decode())
            self.log_with_clock(logging.INFO, f"📩 信号: {data}")

            # 解析信号
            side = data.get("side", "BUY").upper()
            base = data.get("base_token", "")
            quote = data.get("quote_token", self.config.default_quote_token)
            amount = Decimal(str(data.get("amount", self.config.default_trade_amount)))

            # 统一处理 BNB/WBNB
            quote = self._normalize_token_symbol(quote)
            base = self._normalize_token_symbol(base)

            # 获取滑点（可选）
            slippage = data.get("slippage")
            if slippage is not None:
                slippage = Decimal(str(slippage))
            else:
                slippage = self.config.slippage

            # 验证信号
            if not base or side not in ["BUY", "SELL"]:
                self.log_with_clock(logging.ERROR, "❌ 无效信号")
                return

            # 从 MQTT 线程调度到主事件循环
            # 注意：不能使用 safe_ensure_future()，因为 MQTT 回调在单独的线程中
            asyncio.run_coroutine_threadsafe(
                self._execute_trade_with_retry(side, base, quote, amount, slippage),
                self._event_loop
            )

        except Exception as e:
            self.log_with_clock(logging.ERROR, f"❌ 处理消息失败: {e}")

    def _normalize_token_symbol(self, token: str) -> str:
        """
        统一代币符号，自动处理 BNB/WBNB

        Args:
            token: 代币符号或地址

        Returns:
            标准化后的符号
        """
        if not token:
            return token

        # 地址直接返回
        if token.startswith("0x") or token.startswith("0X"):
            return token

        # 符号转大写
        token_upper = token.upper()

        # BNB 自动转 WBNB
        if token_upper == "BNB":
            self.log_with_clock(logging.INFO, "ℹ️  自动将 BNB 转换为 WBNB")
            return "WBNB"

        return token_upper

    # ========== 交易执行 ==========
    async def _execute_trade_with_retry(
        self,
        side: str,
        base: str,
        quote: str,
        amount: Decimal,
        slippage: Decimal,
        retry_count: int = 0
    ):
        """
        执行交易（支持重试）

        Args:
            side: 交易方向 (BUY/SELL)
            base: 基础代币
            quote: 计价代币
            amount: BUY=quote数量, SELL=base数量
            slippage: 滑点
            retry_count: 当前重试次数
        """
        if retry_count == 0:
            self.stats["signals"] += 1

        try:
            result = await self._execute_trade(side, base, quote, amount, slippage)

            if result:
                return

            # 重试逻辑
            if retry_count < self.config.max_retries:
                self.stats["retries"] += 1
                self.log_with_clock(
                    logging.WARNING,
                    f"⚠️  交易失败，{float(self.config.retry_delay)}秒后重试 "
                    f"({retry_count + 1}/{self.config.max_retries})"
                )
                await asyncio.sleep(float(self.config.retry_delay))
                await self._execute_trade_with_retry(
                    side, base, quote, amount, slippage, retry_count + 1
                )
            else:
                self.log_with_clock(
                    logging.ERROR,
                    f"❌ 已达到最大重试次数 ({self.config.max_retries})"
                )
                self.stats["failed"] += 1

        except Exception as e:
            self.log_with_clock(logging.ERROR, f"❌ 重试逻辑失败: {e}")
            self.stats["failed"] += 1

    async def _execute_trade(
        self,
        side: str,
        base: str,
        quote: str,
        amount: Decimal,
        slippage: Decimal
    ) -> bool:
        """
        执行单次交易

        计价规则：
        - BUY: amount 表示 quote token (WBNB) 数量
        - SELL: amount 表示 base token 数量

        Args:
            side: 交易方向
            base: 基础代币
            quote: 计价代币
            amount: 交易数量
            slippage: 滑点

        Returns:
            成功返回 True，失败返回 False
        """
        try:
            # 检查连接器状态
            if not self.connector.ready:
                self.log_with_clock(logging.ERROR, "❌ Connector 未就绪")
                return False

            # 构建交易对
            trading_pair = f"{base}-{quote}"
            is_buy = (side == "BUY")

            # 根据交易方向计算 base token 数量
            if is_buy:
                # BUY: amount 是 quote 数量，需要转换
                quote_amount = amount

                self.log_with_clock(
                    logging.INFO,
                    f"🔍 买入：用 {quote_amount} {quote} 买入 {base}"
                )

                # 获取参考价格
                # 注意：Gateway API 返回的 price 是 "base per quote"
                # 即：1 quote token 能换多少 base token
                temp_price = await self.connector.get_quote_price(
                    trading_pair=trading_pair,
                    is_buy=True,
                    amount=Decimal("1")
                )

                if temp_price is None:
                    self.log_with_clock(logging.ERROR, "❌ 获取价格失败")
                    return False

                # 计算 base token 数量
                # price 是 base/quote，所以：base_amount = quote_amount * price
                base_amount = quote_amount * temp_price

                self.log_with_clock(
                    logging.INFO,
                    f"📊 初步估算：{quote_amount} {quote} → {base_amount:.6f} {base} "
                    f"(汇率: {temp_price:.6f} {base}/{quote})"
                )
            else:
                # SELL: amount 直接就是 base 数量
                base_amount = amount
                self.log_with_clock(
                    logging.INFO,
                    f"🔍 卖出：{base_amount} {base} 换取 {quote}"
                )

            # 获取精确报价
            current_price = await self.connector.get_quote_price(
                trading_pair=trading_pair,
                is_buy=is_buy,
                amount=base_amount
            )

            if current_price is None:
                self.log_with_clock(logging.ERROR, "❌ 获取精确报价失败")
                return False

            # 计算最终价格
            # 注意：Gateway API 的 price 是 base/quote (base token per quote token)
            # 例如：44310 TOKEN/WBNB 意味着 1 WBNB = 44310 TOKEN
            if is_buy:
                # BUY: 愿意支付更多 quote（即接受更低的 base/quote 汇率）
                # 降低汇率 = 除以 (1 + 调整)
                final_price = current_price / ((Decimal("1") + slippage) * self.config.gas_buffer)
            else:
                # SELL: 愿意得到更少 quote（即接受更高的 base/quote 汇率）
                # 提高汇率 = 乘以 (1 + 调整)
                final_price = current_price * ((Decimal("1") + slippage) * self.config.gas_buffer)

            # 计算预估金额
            # price 是 base/quote，所以：quote = base / price
            estimated_quote = base_amount / final_price

            self.log_with_clock(
                logging.INFO,
                f"💰 价格信息\n"
                f"   市场汇率: {current_price:.6f} {base}/{quote}\n"
                f"   最终汇率: {final_price:.6f} {base}/{quote}\n"
                f"   {base} 数量: {base_amount:.6f}\n"
                f"   {'支付' if is_buy else '获得'} {quote}: {estimated_quote:.6f}\n"
                f"   滑点: {slippage * 100}%\n"
                f"   Gas Buffer: {self.config.gas_buffer}"
            )

            # 下单
            order_id = self.connector.place_order(
                is_buy=is_buy,
                trading_pair=trading_pair,
                amount=base_amount,
                price=final_price
            )

            if not order_id:
                self.log_with_clock(logging.ERROR, "❌ 下单失败：未返回 order_id")
                return False

            # 记录订单
            self.pending_orders[order_id] = {
                "side": side,
                "pair": trading_pair,
                "base_amount": base_amount,
                "quote_amount": estimated_quote,
                "input_amount": amount,
                "market_price": current_price,
                "final_price": final_price,
                "slippage": slippage,
                "timestamp": self.current_timestamp
            }

            self.log_with_clock(logging.INFO, f"⚡ 订单已提交: {order_id}")
            return True

        except Exception as e:
            self.log_with_clock(logging.ERROR, f"❌ 交易执行失败: {e}")
            return False

    # ========== 事件处理 ==========
    def did_fill_order(self, event: OrderFilledEvent):
        """订单成交事件"""
        order_id = event.order_id

        self.log_with_clock(
            logging.INFO,
            f"🎉 订单成交: {order_id}\n"
            f"   交易对: {event.trading_pair}\n"
            f"   数量: {event.amount:.6f}\n"
            f"   价格: {event.price:.6f}"
        )

        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
            self.stats["success"] += 1

    def did_fail_order(self, event: MarketOrderFailureEvent):
        """订单失败事件"""
        order_id = event.order_id
        self.log_with_clock(logging.ERROR, f"❌ 订单失败: {order_id}")

        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
            self.stats["failed"] += 1

    # ========== 状态显示 ==========
    def format_status(self) -> str:
        """格式化状态显示"""
        if not self.ready_to_trade:
            return "⚠️ 未就绪"

        lines = []
        lines.append("=" * 60)
        lines.append("BSC 新闻狙击 - 运行状态")
        lines.append("=" * 60)
        lines.append("配置信息:")
        lines.append(f"  Connector: {self.config.connector}")
        lines.append(f"  Trading Pair: {self.config.trading_pair}")
        lines.append(f"  Default Amount: {self.config.default_trade_amount}")
        lines.append(f"  Quote Token: {self.config.default_quote_token}")
        lines.append(f"  Gas Buffer: {self.config.gas_buffer}")
        lines.append(f"  Slippage: {self.config.slippage * 100}%")
        lines.append(f"  Max Retries: {self.config.max_retries}")
        lines.append("")
        lines.append("状态:")

        connector_status = "🟢 就绪" if self.connector.ready else "🔴 未就绪"
        lines.append(f"  Connector: {connector_status}")

        if self.mqtt_client:
            mqtt_status = "🟢 连接" if self.mqtt_client.is_connected() else "🔴 断开"
        else:
            mqtt_status = "⚪ 未启用"
        lines.append(f"  MQTT: {mqtt_status}")

        lines.append("")
        lines.append("📊 统计:")
        lines.append(f"  信号: {self.stats['signals']}")
        lines.append(f"  成功: {self.stats['success']}")
        lines.append(f"  失败: {self.stats['failed']}")
        lines.append(f"  重试: {self.stats['retries']}")
        lines.append(f"  待处理: {len(self.pending_orders)}")
        lines.append("=" * 60)

        return "\n".join(lines)

    # ========== 清理 ==========
    async def on_stop(self):
        """停止策略"""
        if self.mqtt_client:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            except Exception as e:
                self.log_with_clock(logging.WARNING, f"MQTT 断开时出错: {e}")

        self.log_with_clock(
            logging.INFO,
            f"🛑 策略已停止\n"
            f"   总信号: {self.stats['signals']}\n"
            f"   成功: {self.stats['success']}\n"
            f"   失败: {self.stats['failed']}\n"
            f"   重试: {self.stats['retries']}"
        )
