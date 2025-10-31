"""
CEX-DEX LP 套利策略

核心原理:
- 在 DEX 上被动做 LP Maker
- 在 CEX 上主动做 Taker 对冲
- 赚取价差 + LP 手续费

工作流程:
1. 监控 CEX 价格和 DEX 池子
2. 发现套利机会 → 在 DEX 开 LP 仓位
3. 等待 LP 被成交（被动）
4. LP 成交后 → 立即在 CEX 对冲
5. 计算利润并记录

参考:
- lp_manage_position.py (LP 管理)
- arbitrage_controller.py (套利逻辑)
- amm_trade_example.py (DEX 交易)
"""

import aiohttp
import asyncio
import json
import logging
import os
import redis.asyncio as aioredis
import ssl
import time
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional, Tuple, Union
from urllib.parse import urlencode

from pydantic import Field

from hummingbot.client.config.config_data_types import BaseClientModel
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.gateway.gateway_http_client import GatewayHttpClient
from hummingbot.core.rate_oracle.rate_oracle import RateOracle
from hummingbot.connector.gateway.common_types import ConnectorType, get_connector_type
from hummingbot.connector.gateway.gateway_lp import AMMPositionInfo, CLMMPositionInfo
from hummingbot.core.data_type.common import OrderType
from hummingbot.core.event.events import BuyOrderCompletedEvent, OrderFilledEvent, SellOrderCompletedEvent
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


# ========================================
# 配置类
# ========================================

class CexDexLpArbitrageConfig(BaseClientModel):
    """CEX-DEX LP 套利配置"""

    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))

    # ========== 交易所配置 ==========
    cex_exchange: str = Field(
        "binance",
        json_schema_extra={"prompt": "CEX 交易所名称（用于对冲）", "prompt_on_new": True}
    )

    dex_exchange: str = Field(
        "pancakeswap/clmm",
        json_schema_extra={"prompt": "DEX 交易所（格式: name/type，如 uniswap/clmm）", "prompt_on_new": True}
    )

    dex_network: str = Field(
        "bsc",
        json_schema_extra={"prompt": "DEX 网络（如 bsc, mainnet, arbitrum）", "prompt_on_new": True}
    )

    trading_pair: str = Field(
        "WETH-USDC",
        json_schema_extra={"prompt": "DEX 交易对（如 GIGGLE-WBNB）", "prompt_on_new": True}
    )

    # 注意：所有价格最终都会标准化为 USDT 计价
    # CEX 不需要配置，它只用于对冲，不用于价格发现

    # ========== LP 配置 ==========
    lp_token_amount: Decimal = Field(
        Decimal("0.1"),
        json_schema_extra={"prompt": "LP 单边 Token 数量", "prompt_on_new": True}
    )

    lp_spread_pct: Decimal = Field(
        Decimal("0.01"),
        json_schema_extra={"prompt": "LP 价格区间宽度（小数，如 0.01 = 1%）", "prompt_on_new": True}
    )

    lp_timeout_seconds: int = Field(
        300,
        json_schema_extra={"prompt": "LP 最长持有时间（秒）", "prompt_on_new": True}
    )

    # ========== 盈利目标 ==========
    target_profitability: Decimal = Field(
        Decimal("0.02"),
        json_schema_extra={"prompt": "目标利润率（小数，如 0.02 = 2%）", "prompt_on_new": True}
    )

    min_profitability: Decimal = Field(
        Decimal("0.005"),
        json_schema_extra={"prompt": "最低利润率（小数，止损线）", "prompt_on_new": True}
    )

    # ========== 费用估算 ==========
    cex_taker_fee_pct: Decimal = Field(
        Decimal("0.001"),
        json_schema_extra={"prompt": "CEX Taker 手续费率（小数）", "prompt_on_new": False}
    )

    dex_lp_fee_pct: Decimal = Field(
        Decimal("0.003"),
        json_schema_extra={"prompt": "DEX LP 手续费率（小数，这是收入）", "prompt_on_new": False}
    )

    gas_cost_quote: Decimal = Field(
        Decimal("5"),
        json_schema_extra={"prompt": "预估 Gas 成本（Quote Token 单位）", "prompt_on_new": False}
    )

    # ========== 策略配置 ==========
    enable_sell_side: bool = Field(
        True,
        json_schema_extra={"prompt": "启用卖方套利（DEX LP 卖出，CEX 买入）", "prompt_on_new": False}
    )

    enable_buy_side: bool = Field(
        False,
        json_schema_extra={"prompt": "启用买方套利（DEX LP 买入，CEX 卖出）", "prompt_on_new": False}
    )

    check_interval_seconds: int = Field(
        10,
        json_schema_extra={"prompt": "检查间隔（秒）", "prompt_on_new": False}
    )

    # ========== Redis 配置 ==========
    redis_host: str = Field(
        "localhost",
        json_schema_extra={"prompt": "Redis 主机地址", "prompt_on_new": False}
    )

    redis_port: int = Field(
        6379,
        json_schema_extra={"prompt": "Redis 端口", "prompt_on_new": False}
    )

    redis_db: int = Field(
        0,
        json_schema_extra={"prompt": "Redis 数据库编号", "prompt_on_new": False}
    )

    conversion_rate_ttl: int = Field(
        60,
        json_schema_extra={"prompt": "转换率缓存有效期（秒）", "prompt_on_new": False}
    )


# ========================================
# Redis 缓存管理器
# ========================================

class ConversionRateCache:
    """
    基于 Redis 的转换率缓存管理器

    支持：
    - 分布式缓存（多策略实例共享）
    - TTL 自动过期
    - 降级到 last_known 价格
    - 更新锁（防止多实例并发刷新）
    """

    def __init__(self, config: 'CexDexLpArbitrageConfig', logger):
        self.config = config
        self.logger = logger
        self.redis_client: Optional[aioredis.Redis] = None
        self.instance_id = str(uuid.uuid4())[:8]  # 实例标识
        self.fallback_cache: Dict[str, Tuple[Decimal, float]] = {}  # Redis 不可用时的降级

    async def connect(self):
        """连接 Redis"""
        try:
            self.redis_client = await aioredis.from_url(
                f"redis://{self.config.redis_host}:{self.config.redis_port}/{self.config.redis_db}",
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            # 测试连接
            await self.redis_client.ping()
            self.logger.info(f"✅ Redis 连接成功: {self.config.redis_host}:{self.config.redis_port}")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ Redis 连接失败: {e}，将使用内存缓存降级")
            self.redis_client = None
            return False

    async def close(self):
        """关闭 Redis 连接"""
        if self.redis_client:
            await self.redis_client.close()

    def _make_key(self, token: str, key_type: str = "main") -> str:
        """生成 Redis key"""
        if key_type == "main":
            return f"conv_rate:{token}:USDT"
        elif key_type == "last_known":
            return f"conv_rate:{token}:USDT:last_known"
        elif key_type == "lock":
            return f"conv_rate:{token}:USDT:lock"
        else:
            raise ValueError(f"Unknown key_type: {key_type}")

    async def get_rate(self, token: str) -> Optional[Tuple[Decimal, str]]:
        """
        获取转换率（优先从 Redis）

        Returns:
            (price, source) 或 None
            source: "redis_cache", "redis_last_known", "memory_fallback"
        """
        # 稳定币无需查询
        if token.upper() in ["USDT", "USDC", "BUSD", "DAI"]:
            return (Decimal("1"), "stablecoin")

        # 尝试从 Redis 主缓存获取
        if self.redis_client:
            try:
                key = self._make_key(token, "main")
                data_str = await self.redis_client.get(key)

                if data_str:
                    data = json.loads(data_str)
                    price = Decimal(data["price"])
                    self.logger.debug(
                        f"✅ Redis 缓存命中: {token}/USDT = {price} "
                        f"(source: {data.get('source', 'unknown')})"
                    )
                    return (price, "redis_cache")

                # 主缓存未命中，尝试 last_known
                last_known_key = self._make_key(token, "last_known")
                last_data_str = await self.redis_client.get(last_known_key)

                if last_data_str:
                    data = json.loads(last_data_str)
                    price = Decimal(data["price"])
                    age = time.time() - data.get("timestamp", 0)
                    self.logger.warning(
                        f"⚠️ 使用 last_known 价格: {token}/USDT = {price} "
                        f"(age: {age:.0f}s)"
                    )
                    return (price, "redis_last_known")

            except Exception as e:
                self.logger.debug(f"Redis 读取失败: {e}")

        # Redis 不可用，使用内存降级缓存
        if token in self.fallback_cache:
            price, cached_time = self.fallback_cache[token]
            age = time.time() - cached_time
            if age < self.config.conversion_rate_ttl:
                self.logger.debug(f"使用内存缓存: {token}/USDT = {price} (age: {age:.0f}s)")
                return (price, "memory_fallback")

        return None

    async def set_rate(
        self,
        token: str,
        price: Decimal,
        source: str,
        confidence: str = "high"
    ):
        """
        设置转换率到 Redis

        Args:
            token: Token 名称
            price: 价格
            source: 数据源 (cex, dex, oracle)
            confidence: 置信度 (high, medium, low)
        """
        data = {
            "price": str(price),
            "source": source,
            "confidence": confidence,
            "timestamp": time.time(),
            "instance_id": self.instance_id
        }
        data_str = json.dumps(data)

        # 写入 Redis
        if self.redis_client:
            try:
                # 主缓存
                key = self._make_key(token, "main")
                await self.redis_client.setex(
                    key,
                    self.config.conversion_rate_ttl,
                    data_str
                )

                # Last known（24小时）
                if confidence in ["high", "medium"]:
                    last_known_key = self._make_key(token, "last_known")
                    await self.redis_client.setex(
                        last_known_key,
                        86400,  # 24 hours
                        data_str
                    )

                self.logger.debug(
                    f"✅ 写入 Redis: {token}/USDT = {price} "
                    f"(source: {source}, confidence: {confidence})"
                )
            except Exception as e:
                self.logger.warning(f"⚠️ Redis 写入失败: {e}")

        # 同时写入内存降级缓存
        self.fallback_cache[token] = (price, time.time())

    async def try_acquire_lock(self, token: str, timeout: int = 5) -> bool:
        """
        尝试获取更新锁（防止多实例同时刷新）

        Returns:
            True if lock acquired
        """
        if not self.redis_client:
            return True  # Redis 不可用，直接允许

        try:
            lock_key = self._make_key(token, "lock")
            acquired = await self.redis_client.set(
                lock_key,
                self.instance_id,
                nx=True,  # 只在 key 不存在时设置
                ex=timeout
            )
            return bool(acquired)
        except Exception as e:
            self.logger.debug(f"获取锁失败: {e}")
            return True  # 出错时允许，避免阻塞

    async def release_lock(self, token: str):
        """释放更新锁"""
        if not self.redis_client:
            return

        try:
            lock_key = self._make_key(token, "lock")
            # 只删除自己的锁
            lock_owner = await self.redis_client.get(lock_key)
            if lock_owner == self.instance_id:
                await self.redis_client.delete(lock_key)
        except Exception as e:
            self.logger.debug(f"释放锁失败: {e}")


# ========================================
# 盈利计算器
# ========================================

class ProfitabilityCalculator:
    """盈利计算器"""

    def __init__(self, config: CexDexLpArbitrageConfig, logger):
        self.config = config
        self.logger = logger

    def estimate_total_fees_pct(self, trade_value: Decimal) -> Decimal:
        """
        估算总费用百分比

        包括:
        - CEX Taker 手续费
        - Gas 成本（转为百分比）
        - 滑点预留
        """
        # CEX 手续费
        cex_fee_pct = self.config.cex_taker_fee_pct

        # Gas 成本转为百分比（设置合理上限，避免小交易量导致巨大百分比）
        if trade_value > 0:
            gas_pct = self.config.gas_cost_quote / trade_value
            # Gas 成本不应超过 5%（如果超过，说明交易量太小）
            gas_pct = min(gas_pct, Decimal("0.05"))
        else:
            gas_pct = Decimal("0.01")

        # 滑点预留（1%）
        slippage_pct = Decimal("0.01")

        total_fees_pct = cex_fee_pct + gas_pct + slippage_pct

        return total_fees_pct

    def calculate_target_lp_price(
        self,
        cex_price: Decimal,
        is_sell_side: bool,
        trade_value: Decimal
    ) -> Decimal:
        """
        计算目标 LP 价格（开仓线）

        Args:
            cex_price: CEX 参考价格
            is_sell_side: 是否卖方套利
            trade_value: 交易价值（用于计算 gas 占比）

        Returns:
            目标 LP 价格
        """
        total_fees_pct = self.estimate_total_fees_pct(trade_value)

        # 减去 LP 手续费收入
        net_cost_pct = self.config.target_profitability + total_fees_pct - self.config.dex_lp_fee_pct

        if is_sell_side:
            # 卖方: LP 卖价 > CEX 买价 * (1 + 成本)
            target_price = cex_price * (Decimal("1") + net_cost_pct)
        else:
            # 买方: LP 买价 < CEX 卖价 * (1 - 成本)
            target_price = cex_price * (Decimal("1") - net_cost_pct)

        return target_price

    def calculate_min_lp_price(
        self,
        cex_price: Decimal,
        is_sell_side: bool,
        trade_value: Decimal
    ) -> Decimal:
        """
        计算最低 LP 价格（止损线）

        使用 min_profitability 而不是 target_profitability
        """
        total_fees_pct = self.estimate_total_fees_pct(trade_value)
        net_cost_pct = self.config.min_profitability + total_fees_pct - self.config.dex_lp_fee_pct

        if is_sell_side:
            min_price = cex_price * (Decimal("1") + net_cost_pct)
        else:
            min_price = cex_price * (Decimal("1") - net_cost_pct)

        return min_price


# ========================================
# LP 仓位管理器
# ========================================

class LpPositionManager:
    """LP 仓位管理器"""

    def __init__(
        self,
        strategy: ScriptStrategyBase,
        dex_connector: ConnectorBase,
        config: CexDexLpArbitrageConfig
    ):
        self.strategy = strategy
        self.dex_connector = dex_connector
        self.config = config
        self.connector_type = get_connector_type(config.dex_exchange)

        # 仓位信息
        self.position_info: Union[CLMMPositionInfo, AMMPositionInfo, None] = None
        self.position_opening = False
        self.position_closing = False
        self.open_order_id: Optional[str] = None
        self.close_order_id: Optional[str] = None

    def logger(self):
        return self.strategy.logger()

    async def open_lp_position(
        self,
        is_sell_side: bool,
        price_range: Tuple[Decimal, Decimal],
        token_amount: Decimal
    ) -> str:
        """
        开 LP 仓位

        Args:
            is_sell_side: True = 卖方（放入 base token），False = 买方（放入 quote token）
            price_range: (lower_bound, upper_bound)
            token_amount: Token 数量

        Returns:
            订单 ID
        """
        if self.position_opening or self.position_info:
            raise Exception("LP 仓位已存在或正在开仓")

        self.position_opening = True
        lower_bound, upper_bound = price_range
        center_price = (lower_bound + upper_bound) / 2

        try:
            # 计算 LP 参数
            if self.connector_type == ConnectorType.CLMM:
                # CLMM: 使用 width_pct
                spread_pct = float((upper_bound - center_price) / center_price)

                if is_sell_side:
                    # 卖方: 只放 base token
                    base_amount = float(token_amount)
                    quote_amount = 0.0
                else:
                    # 买方: 只放 quote token
                    base_amount = 0.0
                    quote_amount = float(token_amount)

                self.logger().info(
                    f"开 CLMM LP 仓位:\n"
                    f"   方向: {'SELL' if is_sell_side else 'BUY'}\n"
                    f"   中心价: {center_price}\n"
                    f"   价格区间: {lower_bound} - {upper_bound}\n"
                    f"   Base: {base_amount}, Quote: {quote_amount}"
                )

                order_id = self.dex_connector.add_liquidity(
                    trading_pair=self.config.trading_pair,
                    price=float(center_price),
                    upper_width_pct=spread_pct,
                    lower_width_pct=spread_pct,
                    base_token_amount=base_amount,
                    quote_token_amount=quote_amount
                )
            else:
                # AMM: 使用固定比例
                if is_sell_side:
                    base_amount = float(token_amount)
                    quote_amount = 0.0
                else:
                    base_amount = 0.0
                    quote_amount = float(token_amount)

                self.logger().info(
                    f"开 AMM LP 仓位:\n"
                    f"   方向: {'SELL' if is_sell_side else 'BUY'}\n"
                    f"   价格: {center_price}\n"
                    f"   Base: {base_amount}, Quote: {quote_amount}"
                )

                order_id = self.dex_connector.add_liquidity(
                    trading_pair=self.config.trading_pair,
                    price=float(center_price),
                    base_token_amount=base_amount,
                    quote_token_amount=quote_amount
                )

            self.open_order_id = order_id
            return order_id

        except Exception as e:
            self.logger().error(f"开 LP 仓位失败: {e}")
            self.position_opening = False
            raise

    async def close_lp_position(self) -> Optional[str]:
        """关闭 LP 仓位"""
        if not self.position_info or self.position_closing:
            return None

        self.position_closing = True

        try:
            if isinstance(self.position_info, CLMMPositionInfo):
                self.logger().info(f"关闭 CLMM LP: {self.position_info.address}")
                order_id = self.dex_connector.remove_liquidity(
                    trading_pair=self.config.trading_pair,
                    position_address=self.position_info.address
                )
            else:
                # AMM: 不需要 position_address
                self.logger().info("关闭 AMM LP")
                order_id = self.dex_connector.remove_liquidity(
                    trading_pair=self.config.trading_pair
                )

            self.close_order_id = order_id
            return order_id

        except Exception as e:
            self.logger().error(f"关闭 LP 仓位失败: {e}")
            self.position_closing = False
            raise

    async def update_position_info(self):
        """更新仓位信息"""
        if not self.position_info:
            return

        try:
            if isinstance(self.position_info, CLMMPositionInfo):
                self.position_info = await self.dex_connector.get_position_info(
                    trading_pair=self.config.trading_pair,
                    position_address=self.position_info.address
                )
            else:
                # AMM: 使用池子地址
                pool_address = await self.dex_connector.get_pool_address(self.config.trading_pair)
                if pool_address:
                    self.position_info = await self.dex_connector.get_position_info(
                        trading_pair=self.config.trading_pair,
                        position_address=pool_address
                    )
        except Exception as e:
            self.logger().error(f"更新仓位信息失败: {e}")


# ========================================
# CEX 对冲执行器
# ========================================

class CexHedgeExecutor:
    """CEX 对冲执行器"""

    def __init__(
        self,
        strategy: ScriptStrategyBase,
        cex_connector: ConnectorBase,
        config: CexDexLpArbitrageConfig
    ):
        self.strategy = strategy
        self.cex_connector = cex_connector
        self.config = config
        self.pending_order_id: Optional[str] = None

    def logger(self):
        return self.strategy.logger()

    async def hedge_lp_fill(
        self,
        is_buy: bool,
        amount: Decimal,
        price_limit: Decimal
    ) -> str:
        """
        执行 CEX 对冲

        Args:
            is_buy: True = 买入，False = 卖出
            amount: 数量
            price_limit: 限价

        Returns:
            订单 ID
        """
        try:
            # 量化参数
            amount_quantized = self.cex_connector.quantize_order_amount(
                self.config.trading_pair,
                amount
            )

            price_quantized = self.cex_connector.quantize_order_price(
                self.config.trading_pair,
                price_limit
            )

            self.logger().info(
                f"CEX 对冲:\n"
                f"   方向: {'BUY' if is_buy else 'SELL'}\n"
                f"   数量: {amount_quantized}\n"
                f"   限价: {price_quantized}"
            )

            # 下市价单（使用限价保护）
            if is_buy:
                order_id = self.strategy.buy(
                    connector_name=self.config.cex_exchange,
                    trading_pair=self.config.trading_pair,
                    amount=amount_quantized,
                    order_type=OrderType.MARKET,
                    price=price_quantized
                )
            else:
                order_id = self.strategy.sell(
                    connector_name=self.config.cex_exchange,
                    trading_pair=self.config.trading_pair,
                    amount=amount_quantized,
                    order_type=OrderType.MARKET,
                    price=price_quantized
                )

            self.pending_order_id = order_id
            return order_id

        except Exception as e:
            self.logger().error(f"CEX 对冲失败: {e}")
            raise


# ========================================
# 主策略
# ========================================

class CexDexLpArbitrageStrategy(ScriptStrategyBase):
    """CEX-DEX LP 套利策略"""

    @classmethod
    def init_markets(cls, config: CexDexLpArbitrageConfig):
        # 解析 DEX 交易对
        base_token, dex_quote_token = config.trading_pair.split("-")

        # CEX 使用 base_token-USDT 格式
        cex_trading_pair = f"{base_token}-USDT"

        # DEX 使用配置中的交易对
        dex_trading_pair = config.trading_pair

        cls.markets = {
            config.cex_exchange: {cex_trading_pair},  # 如 GIGGLE-USDT
            config.dex_exchange: {dex_trading_pair}   # 如 GIGGLE-WBNB
        }

    def __init__(self, connectors: Dict[str, ConnectorBase], config: CexDexLpArbitrageConfig):
        super().__init__(connectors)
        self.config = config

        # 连接器
        self.cex_connector = connectors[config.cex_exchange]
        self.dex_connector = connectors[config.dex_exchange]

        # Token 名称（从 DEX 交易对解析）
        self.base_token, self.dex_quote_token = config.trading_pair.split("-")

        # 所有价格统一标准化为 USDT（美元稳定币）
        self.standard_quote = "USDT"  # 标准计价单位

        # 价格转换缓存管理器（Redis + 内存降级）
        self.conversion_rate_cache = ConversionRateCache(config, self.logger())
        # 异步初始化 Redis 连接
        safe_ensure_future(self.conversion_rate_cache.connect())

        # 初始化模块
        self.profit_calculator = ProfitabilityCalculator(config, self.logger())
        self.lp_manager = LpPositionManager(self, self.dex_connector, config)
        self.hedge_executor = CexHedgeExecutor(self, self.cex_connector, config)

        # 状态跟踪
        self.lp_position_opened = False
        self.lp_position_info: Optional[dict] = None  # 自定义信息
        self.last_check_time = None

        # 统计
        self.stats = {
            "total_profit": Decimal("0"),
            "completed_cycles": 0,
            "lp_open_failures": 0,
            "hedge_failures": 0,
        }

        # 启动信息
        price_conversion_info = ""
        if self.dex_quote_token.upper() not in ["USDT", "USDC", "BUSD", "DAI"]:
            price_conversion_info = (
                f"\n   🔄 价格转换: {self.dex_quote_token} → {self.standard_quote} "
                f"(所有价格统一为美元计价)"
            )

        self.log_with_clock(
            logging.INFO,
            f"CEX-DEX LP 套利策略启动:\n"
            f"   CEX: {config.cex_exchange} (仅用于对冲)\n"
            f"   DEX: {config.dex_exchange} (价格发现)\n"
            f"   交易对: {config.trading_pair} (DEX 池子)\n"
            f"   LP 数量: {config.lp_token_amount} {self.base_token}\n"
            f"   目标利润: {config.target_profitability * 100:.2f}%\n"
            f"   最低利润: {config.min_profitability * 100:.2f}%\n"
            f"   卖方套利: {'启用' if config.enable_sell_side else '禁用'}\n"
            f"   买方套利: {'启用' if config.enable_buy_side else '禁用'}"
            f"{price_conversion_info}"
        )

    # ========================================
    # 主循环
    # ========================================

    def on_tick(self):
        """策略主循环"""
        current_time = datetime.now()

        # 检查间隔
        if self.last_check_time and (current_time - self.last_check_time).total_seconds() < self.config.check_interval_seconds:
            return

        self.last_check_time = current_time

        # 根据 LP 状态执行不同逻辑
        if self.lp_position_opened:
            # 持仓监控
            safe_ensure_future(self._monitor_lp_position())
        else:
            # 寻找开仓机会
            safe_ensure_future(self._check_opening_opportunity())

    # ========================================
    # 开仓逻辑
    # ========================================

    async def _check_opening_opportunity(self):
        """检查开仓机会"""
        try:
            # 获取市场数据
            cex_best_ask = await self._get_cex_best_ask()
            cex_best_bid = await self._get_cex_best_bid()
            dex_price = await self._get_dex_price()

            self.logger().info(
                f"市场数据:\n"
                f"   CEX 买价: {cex_best_ask}\n"
                f"   CEX 卖价: {cex_best_bid}\n"
                f"   DEX 价格: {dex_price}"
            )

            if not all([cex_best_ask, cex_best_bid, dex_price]):
                self.logger().warning("无法获取市场数据")
                return

            trade_value = self.config.lp_token_amount * cex_best_ask

            # 检查卖方机会
            if self.config.enable_sell_side:
                target_price = self.profit_calculator.calculate_target_lp_price(
                    cex_price=cex_best_ask,
                    is_sell_side=True,
                    trade_value=trade_value
                )

                if dex_price >= target_price:
                    self.logger().info(
                        f"发现卖方套利机会:\n"
                        f"   DEX 价格: {dex_price}\n"
                        f"   目标价格: {target_price}\n"
                        f"   CEX 买价: {cex_best_ask}"
                    )
                    await self._open_sell_side_position(target_price, cex_best_ask)
                    return

            # 检查买方机会
            if self.config.enable_buy_side:
                target_price = self.profit_calculator.calculate_target_lp_price(
                    cex_price=cex_best_bid,
                    is_sell_side=False,
                    trade_value=trade_value
                )

                if dex_price <= target_price:
                    self.logger().info(
                        f"发现买方套利机会:\n"
                        f"   DEX 价格: {dex_price}\n"
                        f"   目标价格: {target_price}\n"
                        f"   CEX 卖价: {cex_best_bid}"
                    )
                    await self._open_buy_side_position(target_price, cex_best_bid)
                    return

        except Exception as e:
            self.logger().error(f"检查开仓机会失败: {e}")

    async def _open_sell_side_position(self, target_price: Decimal, cex_price: Decimal):
        """开卖方 LP 仓位"""
        # 计算 LP 区间
        lower_bound = target_price
        upper_bound = target_price * (Decimal("1") + self.config.lp_spread_pct)

        try:
            order_id = await self.lp_manager.open_lp_position(
                is_sell_side=True,
                price_range=(lower_bound, upper_bound),
                token_amount=self.config.lp_token_amount
            )

            # 记录信息
            self.lp_position_opened = True
            self.lp_position_info = {
                "side": "SELL",
                "order_id": order_id,
                "price_range": (lower_bound, upper_bound),
                "token_amount": self.config.lp_token_amount,
                "open_time": time.time(),
                "open_cex_price": cex_price,
            }

        except Exception as e:
            self.logger().error(f"开卖方仓位失败: {e}")
            self.stats["lp_open_failures"] += 1

    async def _open_buy_side_position(self, target_price: Decimal, cex_price: Decimal):
        """开买方 LP 仓位"""
        # 计算 LP 区间
        upper_bound = target_price
        lower_bound = target_price * (Decimal("1") - self.config.lp_spread_pct)

        try:
            order_id = await self.lp_manager.open_lp_position(
                is_sell_side=False,
                price_range=(lower_bound, upper_bound),
                token_amount=self.config.lp_token_amount
            )

            self.lp_position_opened = True
            self.lp_position_info = {
                "side": "BUY",
                "order_id": order_id,
                "price_range": (lower_bound, upper_bound),
                "token_amount": self.config.lp_token_amount,
                "open_time": time.time(),
                "open_cex_price": cex_price,
            }

        except Exception as e:
            self.logger().error(f"开买方仓位失败: {e}")
            self.stats["lp_open_failures"] += 1

    # ========================================
    # 持仓监控
    # ========================================

    async def _monitor_lp_position(self):
        """监控 LP 仓位"""
        if not self.lp_position_info:
            return

        try:
            # 更新 LP 仓位信息
            await self.lp_manager.update_position_info()

            # 获取当前 CEX 价格
            side = self.lp_position_info["side"]

            if side == "SELL":
                current_cex_price = await self._get_cex_best_ask()
            else:
                current_cex_price = await self._get_cex_best_bid()

            if not current_cex_price:
                return

            # 计算止损价格
            lower_bound, upper_bound = self.lp_position_info["price_range"]
            avg_lp_price = (lower_bound + upper_bound) / 2
            trade_value = self.config.lp_token_amount * current_cex_price

            cutoff_price = self.profit_calculator.calculate_min_lp_price(
                cex_price=current_cex_price,
                is_sell_side=(side == "SELL"),
                trade_value=trade_value
            )

            # 检查止损
            if side == "SELL":
                if avg_lp_price < cutoff_price:
                    self.logger().warning(
                        f"触发止损:\n"
                        f"   LP 均价: {avg_lp_price}\n"
                        f"   止损线: {cutoff_price}\n"
                        f"   CEX 价格: {current_cex_price}"
                    )
                    await self._close_lp_position_with_reason("STOP_LOSS")
                    return
            else:
                if avg_lp_price > cutoff_price:
                    self.logger().warning(
                        f"触发止损:\n"
                        f"   LP 均价: {avg_lp_price}\n"
                        f"   止损线: {cutoff_price}\n"
                        f"   CEX 价格: {current_cex_price}"
                    )
                    await self._close_lp_position_with_reason("STOP_LOSS")
                    return

            # 检查超时
            elapsed = time.time() - self.lp_position_info["open_time"]
            if elapsed > self.config.lp_timeout_seconds:
                self.logger().info(f"LP 仓位超时 ({elapsed:.0f}秒)，关闭")
                await self._close_lp_position_with_reason("TIMEOUT")

        except Exception as e:
            self.logger().error(f"监控 LP 仓位失败: {e}")

    async def _close_lp_position_with_reason(self, reason: str):
        """关闭 LP 仓位"""
        try:
            await self.lp_manager.close_lp_position()
            self.logger().info(f"LP 仓位已关闭，原因: {reason}")
        except Exception as e:
            self.logger().error(f"关闭 LP 仓位失败: {e}")

    # ========================================
    # 事件处理
    # ========================================

    def did_fill_order(self, event: OrderFilledEvent):
        """
        订单成交事件（生产级优化）

        参考: mtqq_cex_webhook.py:338-435
        """
        # 注入价格到 RateOracle（在 MarketsRecorder 计算前完成）
        try:
            rate_oracle = RateOracle.get_instance()
            rate_oracle.set_price(event.trading_pair, Decimal(str(event.price)))
            self.logger().debug(
                f"📊 注入价格到 RateOracle: {event.trading_pair} = ${event.price:.6f}"
            )
        except Exception as oracle_err:
            self.logger().debug(f"⚠️ 注入 RateOracle 失败: {oracle_err}")

        # LP 开仓成交
        if hasattr(event, 'order_id') and event.order_id == self.lp_manager.open_order_id:
            self.logger().info(f"✅ LP 开仓成交: {event.order_id}")

            # 记录 Gas Price（EVM 链）
            safe_ensure_future(self._record_gas_price(event))

            self.lp_manager.position_opening = False
            # 需要异步获取 position_info
            safe_ensure_future(self._fetch_lp_position_info())

        # LP 关仓成交
        elif hasattr(event, 'order_id') and event.order_id == self.lp_manager.close_order_id:
            self.logger().info(f"✅ LP 关仓成交: {event.order_id}")

            # 记录 Gas Price（EVM 链）
            safe_ensure_future(self._record_gas_price(event))

            self.lp_manager.position_closing = False
            self.lp_manager.position_info = None
            self.lp_position_opened = False
            self.lp_position_info = None

        # CEX 对冲成交
        elif hasattr(event, 'order_id') and event.order_id == self.hedge_executor.pending_order_id:
            self.logger().info(f"✅ CEX 对冲成交: {event.order_id}")
            safe_ensure_future(self._handle_hedge_filled(event))

    async def _fetch_lp_position_info(self):
        """获取 LP 仓位信息"""
        try:
            await asyncio.sleep(2)  # 等待链上确认

            if self.lp_manager.connector_type == ConnectorType.CLMM:
                pool_address = await self.dex_connector.get_pool_address(self.config.trading_pair)
                positions = await self.dex_connector.get_user_positions(pool_address=pool_address)
                if positions:
                    self.lp_manager.position_info = positions[-1]  # 最新的
            else:
                pool_address = await self.dex_connector.get_pool_address(self.config.trading_pair)
                if pool_address:
                    self.lp_manager.position_info = await self.dex_connector.get_position_info(
                        trading_pair=self.config.trading_pair,
                        position_address=pool_address
                    )
        except Exception as e:
            self.logger().error(f"获取 LP 仓位信息失败: {e}")

    async def _record_gas_price(self, event: OrderFilledEvent):
        """
        记录 Gas Price（仅 EVM 链）

        参考: mtqq_cex_webhook.py:361-383
        """
        try:
            # 只对 DEX 交易记录 gas
            if not hasattr(self.dex_connector, 'in_flight_orders'):
                return

            in_flight_order = self.dex_connector.in_flight_orders.get(event.order_id)
            if not in_flight_order:
                return

            if not hasattr(in_flight_order, 'gas_price'):
                return

            gas_price_wei = float(in_flight_order.gas_price)
            if gas_price_wei <= 0:
                return

            # 转换 Wei -> Gwei
            gas_price_gwei = gas_price_wei / 1e9

            self.logger().info(
                f"⛽ Gas Price 记录:\n"
                f"   订单: {event.order_id}\n"
                f"   Gas: {gas_price_gwei:.2f} Gwei\n"
                f"   TX: {event.exchange_trade_id}"
            )

            # 可选：记录到统计信息
            if "total_gas_gwei" not in self.stats:
                self.stats["total_gas_gwei"] = 0
                self.stats["gas_tx_count"] = 0

            self.stats["total_gas_gwei"] += gas_price_gwei
            self.stats["gas_tx_count"] += 1

        except Exception as e:
            # Gas 记录失败不应影响主流程
            self.logger().debug(f"⚠️ 记录 Gas Price 失败: {e}")

    async def _handle_hedge_filled(self, event: OrderFilledEvent):
        """处理对冲成交"""
        # 这里可以计算实际利润
        self.logger().info(
            f"对冲完成:\n"
            f"   价格: {event.price}\n"
            f"   数量: {event.amount}\n"
            f"   手续费: {event.trade_fee}"
        )

        self.stats["completed_cycles"] += 1

    # ========================================
    # 辅助方法
    # ========================================

    def _get_cex_trading_pair(self) -> str:
        """
        构建 CEX 交易对

        DEX 交易对可能是 GIGGLE-WBNB，但 CEX 是 GIGGLE-USDT
        统一使用 base_token-USDT 格式
        """
        return f"{self.base_token}-{self.standard_quote}"

    async def _get_cex_best_ask(self) -> Optional[Decimal]:
        """获取 CEX 最佳卖价（我们的买入价）"""
        try:
            cex_pair = self._get_cex_trading_pair()

            self.logger().debug(f"从 CEX 获取 {cex_pair} Ask 价格...")

            price = await self.cex_connector.get_quote_price(
                trading_pair=cex_pair,
                is_buy=True,
                amount=self.config.lp_token_amount
            )

            if price and price > 0:
                self.logger().debug(f"✅ CEX Ask: {price} {self.standard_quote}")
                return Decimal(str(price))
            else:
                self.logger().warning(f"⚠️ CEX {cex_pair} 返回空价格")
                return None

        except Exception as e:
            self.logger().error(f"获取 CEX 买价失败 ({self._get_cex_trading_pair()}): {e}")
            return None

    async def _get_cex_best_bid(self) -> Optional[Decimal]:
        """获取 CEX 最佳买价（我们的卖出价）"""
        try:
            cex_pair = self._get_cex_trading_pair()

            self.logger().debug(f"从 CEX 获取 {cex_pair} Bid 价格...")

            price = await self.cex_connector.get_quote_price(
                trading_pair=cex_pair,
                is_buy=False,
                amount=self.config.lp_token_amount
            )

            if price and price > 0:
                self.logger().debug(f"✅ CEX Bid: {price} {self.standard_quote}")
                return Decimal(str(price))
            else:
                self.logger().warning(f"⚠️ CEX {cex_pair} 返回空价格")
                return None

        except Exception as e:
            self.logger().error(f"获取 CEX 卖价失败 ({self._get_cex_trading_pair()}): {e}")
            return None

    async def _gateway_request(self, method: str, path_url: str, params: Optional[dict] = None) -> Optional[dict]:
        """
        调用 Gateway REST API（使用官方 GatewayHttpClient）

        Args:
            method: HTTP 方法（get/post）
            path_url: API 路径（不带前导斜杠），例如 "connectors/pancakeswap_clmm/quote-swap"
            params: 请求参数字典

        Returns:
            响应字典，失败返回 None
        """
        try:
            # 使用官方 GatewayHttpClient
            gateway_client = GatewayHttpClient.get_instance()

            # 调用 api_request 方法
            # 参考: gateway_http_client.py:414-421
            response = await gateway_client.api_request(
                method=method.lower(),
                path_url=path_url,
                params=params or {},
                fail_silently=True
            )

            return response

        except Exception as e:
            self.logger().error(f"Gateway API 请求失败 [{method} {path_url}]: {e}")
            return None

    async def _get_dex_price_with_retry(self) -> Optional[Decimal]:
        """
        获取 DEX 价格（带重试逻辑）

        使用 Gateway quote-swap 端点，参考 mtqq_cex_webhook.py
        """
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                price = await self._fetch_dex_price_internal()
                if price:
                    return price
            except Exception as e:
                error_msg = str(e)

                # 检测暂时性错误（Division by zero 通常是 RPC 缓存问题）
                if "division by zero" in error_msg.lower():
                    if attempt < max_retries - 1:
                        self.logger().warning(
                            f"⚠️ 暂时性错误（尝试 {attempt + 1}/{max_retries}）: {error_msg[:50]} - "
                            f"等待 {retry_delay}s 后重试..."
                        )
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避
                        continue
                    else:
                        self.logger().error(f"❌ 重试 {max_retries} 次后仍失败: {error_msg}")
                        return None

                # 其他错误直接失败
                self.logger().error(f"❌ 获取 DEX 价格失败: {e}")
                return None

        return None

    async def _fetch_dex_price_internal(self) -> Optional[Decimal]:
        """
        内部方法：使用 Gateway quote-swap 端点获取 DEX 价格

        返回价格已标准化为 USDT 计价（如果 DEX 使用不同的 Quote Token）

        参考: mtqq_cex_webhook.py:3454-3606
        """
        try:
            # 解析交易对
            base, _ = self.config.trading_pair.split("-")

            # 使用实际的 DEX Quote Token（可能与交易对中的不同）
            dex_quote = self.dex_quote_token

            # 解析 DEX 配置
            exchange_parts = self.config.dex_exchange.split("/")
            exchange = exchange_parts[0]  # "pancakeswap"
            pool_type = exchange_parts[1] if len(exchange_parts) > 1 else "clmm"

            # 构建 Gateway API 路径（不带前导斜杠）
            # Gateway 路由格式: /connectors/{exchange}/{pool_type}/quote-swap
            # 参考: gateway/src/app.ts - prefix: '/connectors/pancakeswap/clmm'
            path_url = f"connectors/{exchange}/{pool_type}/quote-swap"

            # 构建查询参数
            price_params = {
                "network": self.config.dex_network,
                "baseToken": base,
                "quoteToken": dex_quote,  # 使用 DEX 的实际 Quote Token
                "amount": "1",  # 获取 1 个 base token 的价格
                "side": "SELL"
            }

            self.logger().debug(
                f"🔍 获取 DEX 价格: {exchange}/{pool_type} on {self.config.dex_network}, "
                f"{base}-{dex_quote}"
            )

            # 调用 Gateway API（GET 请求）
            response = await self._gateway_request("get", path_url, params=price_params)

            if not response:
                self.logger().warning("⚠️ Gateway 返回空响应")
                return None

            # 提取价格（DEX Quote Token 计价）
            dex_price_raw = None
            if "amountOut" in response:
                dex_price_raw = Decimal(str(response["amountOut"]))
            elif "price" in response:
                dex_price_raw = Decimal(str(response["price"]))
            elif "expectedAmount" in response:
                dex_price_raw = Decimal(str(response["expectedAmount"]))

            if not dex_price_raw:
                self.logger().warning(f"⚠️ 响应中没有价格数据: {response}")
                return None

            self.logger().debug(f"✅ DEX 原始价格: {dex_price_raw} {dex_quote}")

            # 如果 DEX Quote Token 不是稳定币，需要标准化为 USDT
            if dex_quote.upper() not in ["USDT", "USDC", "BUSD", "DAI"]:
                self.logger().debug(
                    f"🔄 需要价格标准化: {dex_quote} -> {self.standard_quote}"
                )
                normalized_price = await self._normalize_price_to_usdt(dex_price_raw, dex_quote)

                if not normalized_price:
                    self.logger().error("❌ 价格标准化失败")
                    return None

                self.logger().info(
                    f"✅ DEX 标准化价格: {dex_price_raw} {dex_quote} "
                    f"-> {normalized_price} {self.standard_quote}"
                )
                final_price = normalized_price
            else:
                # DEX 已经是稳定币计价，无需转换
                self.logger().debug(f"✅ DEX 已是稳定币计价，无需转换")
                final_price = dex_price_raw

            # 注入价格到 RateOracle（避免 "rate not found" 错误）
            try:
                rate_oracle = RateOracle.get_instance()
                rate_oracle.set_price(self.config.trading_pair, final_price)
                self.logger().debug(
                    f"📊 已注入价格到 RateOracle: {self.config.trading_pair} = ${final_price}"
                )
            except Exception as oracle_err:
                self.logger().debug(f"⚠️ 注入 RateOracle 失败: {oracle_err}")

            return final_price

        except Exception as e:
            self.logger().error(f"❌ 获取 DEX 价格失败: {e}", exc_info=True)
            raise

    async def _get_dex_price(self) -> Optional[Decimal]:
        """获取 DEX 价格（公开接口）"""
        return await self._get_dex_price_with_retry()

    # ========================================
    # 价格转换（跨 Quote Token 支持）
    # ========================================

    async def _get_conversion_rate_to_usdt(self, token: str) -> Optional[Decimal]:
        """
        获取 Token 相对 USDT 的转换率（Redis + 多层降级）

        Args:
            token: Token 名称（如 WBNB, BNB）

        Returns:
            转换率（1 token = ? USDT），失败返回 None

        示例:
            WBNB = 600 USDT -> 返回 Decimal("600")
        """
        # 1. 先从 Redis 缓存获取
        cached_result = await self.conversion_rate_cache.get_rate(token)
        if cached_result:
            price, source = cached_result
            self.logger().debug(f"✅ 缓存命中: {token}/USDT = {price} (source: {source})")
            return price

        # 2. 尝试获取分布式锁（防止多实例并发刷新）
        lock_acquired = await self.conversion_rate_cache.try_acquire_lock(token, timeout=5)

        if not lock_acquired:
            # 未获取锁，等待并重试一次缓存
            self.logger().debug(f"⏳ 等待其他实例更新 {token}/USDT...")
            await asyncio.sleep(0.5)
            cached_result = await self.conversion_rate_cache.get_rate(token)
            if cached_result:
                price, source = cached_result
                return price

        try:
            # 3. 依次尝试数据源获取转换率
            conversion_rate = None
            source = None
            confidence = "low"

            # 策略 1: CEX (最可靠，延迟 ~100ms)
            conversion_rate = await self._fetch_conversion_rate_from_cex(token)
            if conversion_rate:
                source, confidence = "cex", "high"
                self.logger().info(f"✅ CEX 获取成功: {token}/USDT = {conversion_rate}")

            # 策略 2: DEX (备用，延迟 ~500ms)
            if not conversion_rate:
                conversion_rate = await self._fetch_conversion_rate_from_dex(token)
                if conversion_rate:
                    source, confidence = "dex", "medium"
                    self.logger().info(f"✅ DEX 获取成功: {token}/USDT = {conversion_rate}")

            # 策略 3: RateOracle (历史价格)
            if not conversion_rate:
                conversion_rate = await self._fetch_conversion_rate_from_oracle(token)
                if conversion_rate:
                    source, confidence = "oracle", "low"
                    self.logger().warning(f"⚠️ 使用历史价格: {token}/USDT = {conversion_rate}")

            # 4. 写入 Redis 缓存
            if conversion_rate:
                await self.conversion_rate_cache.set_rate(
                    token, conversion_rate, source, confidence
                )
                return conversion_rate
            else:
                self.logger().error(f"❌ 所有数据源均失败: {token}/USDT")
                return None

        finally:
            # 5. 释放分布式锁
            if lock_acquired:
                await self.conversion_rate_cache.release_lock(token)

    async def _fetch_conversion_rate_from_cex(self, token: str) -> Optional[Decimal]:
        """从 CEX 获取 Token/USDT 价格"""
        try:
            # 构建交易对（如 WBNB -> BNB-USDT）
            # 去掉 "W" 前缀（WBNB -> BNB）
            base_token = token[1:] if token.startswith("W") and len(token) > 1 else token
            trading_pair = f"{base_token}-USDT"

            self.logger().debug(f"从 CEX 获取 {trading_pair} 价格...")

            # 使用 CEX connector 获取价格
            price = await self.cex_connector.get_quote_price(
                trading_pair=trading_pair,
                is_buy=True,  # 获取卖价（Ask）
                amount=Decimal("1")
            )

            if price and price > 0:
                return Decimal(str(price))
            else:
                return None

        except Exception as e:
            self.logger().debug(f"从 CEX 获取 {token}/USDT 失败: {e}")
            return None

    async def _fetch_conversion_rate_from_dex(self, token: str) -> Optional[Decimal]:
        """从 DEX 获取 Token/USDT 价格（通过 Gateway）"""
        try:
            # 解析 DEX 配置
            exchange_parts = self.config.dex_exchange.split("/")
            exchange = exchange_parts[0]
            pool_type = exchange_parts[1] if len(exchange_parts) > 1 else "clmm"

            path_url = f"connectors/{exchange}/{pool_type}/quote-swap"

            # 构建查询参数
            price_params = {
                "network": self.config.dex_network,
                "baseToken": token,
                "quoteToken": "USDT",
                "amount": "1",
                "side": "SELL"
            }

            self.logger().debug(f"从 DEX 获取 {token}/USDT 价格...")

            response = await self._gateway_request("get", path_url, params=price_params)

            if not response:
                return None

            # 提取价格
            price = None
            if "amountOut" in response:
                price = Decimal(str(response["amountOut"]))
            elif "price" in response:
                price = Decimal(str(response["price"]))
            elif "expectedAmount" in response:
                price = Decimal(str(response["expectedAmount"]))

            return price if price and price > 0 else None

        except Exception as e:
            self.logger().debug(f"从 DEX 获取 {token}/USDT 失败: {e}")
            return None

    async def _fetch_conversion_rate_from_oracle(self, token: str) -> Optional[Decimal]:
        """从 RateOracle 获取 Token/USDT 价格"""
        try:
            # 去掉 "W" 前缀
            base_token = token[1:] if token.startswith("W") and len(token) > 1 else token
            trading_pair = f"{base_token}-USDT"

            rate_oracle = RateOracle.get_instance()
            rate = rate_oracle.get_pair_rate(trading_pair)

            if rate and rate > 0:
                self.logger().debug(f"从 RateOracle 获取 {trading_pair} 价格: {rate}")
                return Decimal(str(rate))
            else:
                return None

        except Exception as e:
            self.logger().debug(f"从 RateOracle 获取 {token}/USDT 失败: {e}")
            return None

    async def _normalize_price_to_usdt(
        self,
        price: Decimal,
        quote_token: str
    ) -> Optional[Decimal]:
        """
        将价格统一转换为 USDT 计价

        Args:
            price: 原始价格
            quote_token: 原始 Quote Token（如 WBNB, USDT）

        Returns:
            USDT 计价的价格，失败返回 None

        示例:
            价格 0.00155 WBNB，WBNB = 600 USDT
            -> 返回 0.00155 * 600 = 0.93 USDT
        """
        if quote_token.upper() in ["USDT", "USDC", "BUSD", "DAI"]:
            # 已经是稳定币计价，无需转换
            return price

        # 获取转换率
        conversion_rate = await self._get_conversion_rate_to_usdt(quote_token)

        if not conversion_rate:
            self.logger().error(f"❌ 无法获取 {quote_token}/USDT 转换率，价格标准化失败")
            return None

        # 转换价格
        normalized_price = price * conversion_rate

        self.logger().debug(
            f"价格标准化: {price} {quote_token} × {conversion_rate} (USDT/{quote_token}) "
            f"= {normalized_price} USDT"
        )

        return normalized_price

    # ========================================
    # 状态显示
    # ========================================

    def format_status(self) -> str:
        """格式化状态显示（带详细市场数据）"""
        lines = []

        # ========== 标题 ==========
        lines.append("=" * 70)
        lines.append("CEX-DEX LP 套利策略".center(70))
        lines.append("=" * 70)
        lines.append(f"CEX: {self.config.cex_exchange:20} | DEX: {self.config.dex_exchange}")
        lines.append(f"交易对: {self.config.trading_pair:18} | LP 数量: {self.config.lp_token_amount}")

        # 价格转换提示
        if self.dex_quote_token.upper() not in ["USDT", "USDC", "BUSD", "DAI"]:
            lines.append(f"🔄 价格转换: {self.dex_quote_token} → {self.standard_quote} (所有价格统一为美元计价)")

        lines.append("-" * 70)

        # ========== 实时市场数据 ==========
        lines.append("")
        lines.append("📈 实时市场数据")
        lines.append("-" * 70)

        # 异步获取最新价格（非阻塞）
        try:
            import asyncio
            loop = asyncio.get_event_loop()

            # 创建异步任务获取价格
            async def get_prices():
                cex_ask = await self._get_cex_best_ask()
                cex_bid = await self._get_cex_best_bid()
                dex_price = await self._get_dex_price()
                return cex_ask, cex_bid, dex_price

            # 如果事件循环正在运行，创建任务
            if loop.is_running():
                # 使用缓存的价格（如果有的话）
                cex_ask = getattr(self, '_cached_cex_ask', None)
                cex_bid = getattr(self, '_cached_cex_bid', None)
                dex_price = getattr(self, '_cached_dex_price', None)

                # 启动后台任务更新缓存
                safe_ensure_future(self._update_price_cache())
            else:
                # 同步获取
                cex_ask, cex_bid, dex_price = loop.run_until_complete(get_prices())

        except Exception as e:
            lines.append(f"⚠️  无法获取实时价格: {e}")
            cex_ask = None
            cex_bid = None
            dex_price = None

        # 显示 CEX 价格
        if cex_ask and cex_bid:
            cex_mid = (cex_ask + cex_bid) / 2
            cex_spread = ((cex_ask - cex_bid) / cex_mid * 100) if cex_mid > 0 else 0

            lines.append(f"CEX ({self.config.cex_exchange}) - 用于对冲:")
            lines.append(f"   基准价:     {cex_mid:>12.6f} {self.standard_quote}")
            lines.append(f"   价差:       {cex_spread:>12.4f} %")
            lines.append(f"   (Ask: {cex_ask:.6f}, Bid: {cex_bid:.6f})")
        else:
            lines.append(f"CEX: ⚠️  无法获取价格")

        lines.append("")

        # 显示 DEX 价格（已标准化为 USDT）
        if dex_price:
            dex_price_label = f"DEX ({self.config.dex_exchange}) - 价格发现:"
            if self.dex_quote_token.upper() not in ["USDT", "USDC", "BUSD", "DAI"]:
                dex_price_label += f" [池子: {self.base_token}-{self.dex_quote_token}]"

            lines.append(dex_price_label)
            lines.append(f"   报价:       {dex_price:>12.6f} {self.standard_quote}")
        else:
            lines.append(f"DEX: ⚠️  无法获取价格")

        lines.append("")

        # ========== 价差分析 ==========
        if cex_ask and cex_bid and dex_price:

            lines.append("💰 套利机会分析")
            lines.append("-" * 70)

            # 卖方套利分析
            if self.config.enable_sell_side:
                lines.append("卖方套利 (DEX LP 卖出 → CEX 买入):")

                # 计算价差
                price_diff = dex_price - cex_ask
                price_diff_pct = (price_diff / cex_ask * 100) if cex_ask > 0 else 0

                # 计算目标价格
                trade_value = self.config.lp_token_amount * cex_ask
                target_price = self.profit_calculator.calculate_target_lp_price(
                    cex_price=cex_ask,
                    is_sell_side=True,
                    trade_value=trade_value
                )
                min_price = self.profit_calculator.calculate_min_lp_price(
                    cex_price=cex_ask,
                    is_sell_side=True,
                    trade_value=trade_value
                )

                # 计算费用
                total_fees_pct = self.profit_calculator.estimate_total_fees_pct(trade_value)

                # 计算各项费用（应用相同的上限逻辑）
                cex_fee_display = self.config.cex_taker_fee_pct * 100
                gas_pct_raw = self.config.gas_cost_quote / trade_value if trade_value > 0 else Decimal("0.01")
                gas_pct_display = min(gas_pct_raw, Decimal("0.05")) * 100  # 应用 5% 上限
                slippage_display = Decimal("1.0")

                lines.append(f"   DEX 价格:        {dex_price:>12.6f} {self.standard_quote}")
                lines.append(f"   CEX 买价:        {cex_ask:>12.6f} {self.standard_quote}")
                lines.append(f"   价差:            {price_diff:>+12.6f} {self.standard_quote}  ({price_diff_pct:+.2f}%)")
                lines.append(f"   ")
                lines.append(f"   目标开仓价:      {target_price:>12.6f} {self.standard_quote}  (需 {self.config.target_profitability*100:.1f}% 利润)")
                lines.append(f"   最低价格(止损):  {min_price:>12.6f} {self.standard_quote}  (需 {self.config.min_profitability*100:.1f}% 利润)")
                lines.append(f"   ")
                lines.append(f"   总费用率:        {total_fees_pct*100:>12.2f} %")
                lines.append(f"     - CEX 手续费:  {cex_fee_display:>12.2f} %")
                lines.append(f"     - Gas 成本:    {gas_pct_display:>12.2f} % (实际: {self.config.gas_cost_quote} {self.standard_quote})")
                lines.append(f"     - 滑点预留:    {slippage_display:>12.2f} %")
                lines.append(f"     - LP 费收入:   -{self.config.dex_lp_fee_pct*100:>12.2f} %")
                lines.append(f"   ")

                # 判断是否有机会
                if dex_price >= target_price:
                    expected_profit_pct = (dex_price - cex_ask) / cex_ask - total_fees_pct + self.config.dex_lp_fee_pct
                    expected_profit_amount = expected_profit_pct * trade_value
                    lines.append(f"   ✅ 有套利机会！")
                    lines.append(f"      预期利润:     {expected_profit_amount:>12.4f} {self.standard_quote}  ({expected_profit_pct*100:+.2f}%)")
                else:
                    gap = target_price - dex_price
                    gap_pct = (gap / dex_price * 100) if dex_price > 0 else 0
                    lines.append(f"   ❌ 暂无机会")
                    lines.append(f"      需要涨幅:     {gap:>12.6f} {self.standard_quote}  ({gap_pct:.2f}%)")

            lines.append("")

            # 买方套利分析
            if self.config.enable_buy_side:
                lines.append("买方套利 (DEX LP 买入 → CEX 卖出):")

                price_diff = cex_bid - dex_price
                price_diff_pct = (price_diff / dex_price * 100) if dex_price > 0 else 0

                trade_value = self.config.lp_token_amount * cex_bid
                target_price = self.profit_calculator.calculate_target_lp_price(
                    cex_price=cex_bid,
                    is_sell_side=False,
                    trade_value=trade_value
                )

                lines.append(f"   CEX 卖价:        {cex_bid:>12.6f} {self.standard_quote}")
                lines.append(f"   DEX 价格:        {dex_price:>12.6f} {self.standard_quote}")
                lines.append(f"   价差:            {price_diff:>+12.6f} {self.standard_quote}  ({price_diff_pct:+.2f}%)")
                lines.append(f"   目标开仓价:      {target_price:>12.6f} {self.standard_quote}")

                if dex_price <= target_price:
                    lines.append(f"   ✅ 有套利机会！")
                else:
                    gap = dex_price - target_price
                    gap_pct = (gap / dex_price * 100) if dex_price > 0 else 0
                    lines.append(f"   ❌ 暂无机会，需要跌幅: {gap:.6f} ({gap_pct:.2f}%)")

            lines.append("")

        # ========== LP 仓位状态 ==========
        if self.lp_position_opened and self.lp_position_info:
            lines.append("📊 LP 仓位状态")
            lines.append("-" * 70)

            side = self.lp_position_info["side"]
            lower, upper = self.lp_position_info["price_range"]
            avg_price = (lower + upper) / 2
            elapsed = time.time() - self.lp_position_info["open_time"]
            open_cex_price = self.lp_position_info["open_cex_price"]

            lines.append(f"方向:        {side}")
            lines.append(f"价格区间:    {lower:.6f} - {upper:.6f} {self.standard_quote}")
            lines.append(f"均价:        {avg_price:.6f} {self.standard_quote}")
            lines.append(f"数量:        {self.lp_position_info['token_amount']} {self.base_token}")
            lines.append(f"持仓时间:    {int(elapsed)}秒 / {self.config.lp_timeout_seconds}秒")
            lines.append(f"开仓CEX价:   {open_cex_price:.6f} {self.standard_quote}")

            # 当前 CEX 价格变化
            if cex_ask and side == "SELL":
                current_cex = cex_ask
                price_change = current_cex - open_cex_price
                price_change_pct = (price_change / open_cex_price * 100) if open_cex_price > 0 else 0
                lines.append(f"当前CEX价:   {current_cex:.6f} {self.standard_quote}  (变化: {price_change:+.6f}, {price_change_pct:+.2f}%)")

                # 预期盈亏
                if avg_price > current_cex:
                    expected_profit = (avg_price - current_cex) / current_cex * 100
                    lines.append(f"预期盈亏:    +{expected_profit:.2f}%  ✅")
                else:
                    expected_loss = (current_cex - avg_price) / avg_price * 100
                    lines.append(f"预期盈亏:    -{expected_loss:.2f}%  ⚠️")

            lines.append("")

        # ========== 统计信息 ==========
        lines.append("📊 统计信息")
        lines.append("-" * 70)
        lines.append(f"完成周期:    {self.stats['completed_cycles']}")
        lines.append(f"累计利润:    {self.stats['total_profit']:.4f} {self.standard_quote}")
        if self.stats['completed_cycles'] > 0:
            avg_profit = self.stats['total_profit'] / self.stats['completed_cycles']
            lines.append(f"平均利润:    {avg_profit:.4f} {self.standard_quote}")
        lines.append(f"LP开仓失败:  {self.stats['lp_open_failures']}")
        lines.append(f"对冲失败:    {self.stats['hedge_failures']}")

        # Gas 统计
        if "gas_tx_count" in self.stats and self.stats["gas_tx_count"] > 0:
            avg_gas = self.stats["total_gas_gwei"] / self.stats["gas_tx_count"]
            lines.append(f"Gas 交易数:  {self.stats['gas_tx_count']}")
            lines.append(f"平均 Gas:    {avg_gas:.2f} Gwei")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    async def _update_price_cache(self):
        """后台更新价格缓存"""
        try:
            self._cached_cex_ask = await self._get_cex_best_ask()
            self._cached_cex_bid = await self._get_cex_best_bid()
            self._cached_dex_price = await self._get_dex_price()
        except Exception as e:
            self.logger().debug(f"更新价格缓存失败: {e}")
