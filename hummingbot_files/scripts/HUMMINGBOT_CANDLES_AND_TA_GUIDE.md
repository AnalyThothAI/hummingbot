# Hummingbot K线数据与技术分析完整指南

## 一、Hummingbot K线数据系统架构

### 1.1 核心组件

```
┌────────────────────────────────────────────────────────────────┐
│                   MarketDataProvider                           │
│  • 统一数据接口                                                 │
│  • 管理所有 Candles Feed 和 Connectors                         │
│  • 提供价格、订单簿、K线数据                                    │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                    CandlesFactory                              │
│  • 工厂模式创建 Candle 实例                                     │
│  • 支持 20+ 交易所 (Binance, OKX, Bybit, etc.)                 │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                    CandlesBase                                 │
│  • 基础 K 线类                                                 │
│  • WebSocket 实时更新                                          │
│  • REST API 历史数据                                           │
│  • deque 存储 (高效滑动窗口)                                   │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│              具体交易所 Candle 实现                             │
│  BinanceSpotCandles, BinancePerpetualCandles, etc.             │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 支持的交易所

#### CEX (中心化交易所) ✅

| 交易所 | Spot | Perpetual | 备注 |
|--------|------|-----------|------|
| Binance | ✅ | ✅ | 主流，数据最可靠 |
| OKX | ✅ | ✅ | 流动性好 |
| Bybit | ✅ | ✅ | 合约专业 |
| Kucoin | ✅ | ✅ | 山寨币多 |
| Gate.io | ✅ | ✅ | 币种全 |
| MEXC | ✅ | ✅ | 新币快 |
| Kraken | ✅ | ❌ | 合规性强 |
| Hyperliquid | ✅ | ✅ | 链上合约 |
| Dexalot | ✅ | ❌ | Avalanche DEX |

#### DEX (去中心化交易所) ⚠️

**目前 CandlesFactory 不支持 Gateway DEX 的原生 K 线**

原因：
- DEX 没有中心化服务器提供 K 线 API
- 需要从链上交易事件聚合生成 K 线
- Hummingbot Gateway 目前专注于交易执行，未实现 K 线聚合

**解决方案（3种）：**

### 1.3 DEX K 线数据获取方案

#### 方案 1: 使用 CEX 数据代理 ✅ 推荐

**原理：**
- DEX 和 CEX 价格高度相关
- 用 Binance SOL-USDT K 线代替 Meteora SOL-USDC

**实现：**
```python
class TechnicalAnalysisDLMM(StrategyV2Base):
    def __init__(self, connectors, config):
        super().__init__(connectors, config)

        # 交易在 Meteora (DEX)
        self.trading_connector = "meteora"
        self.trading_pair = "SOL-USDC"

        # K 线数据从 Binance (CEX) 获取
        candles_config = CandlesConfig(
            connector="binance",  # CEX
            trading_pair="SOL-USDT",  # 相似交易对
            interval="5m",
            max_records=500
        )
        self.candles = CandlesFactory.get_candle(candles_config)
        self.candles.start()
```

**优势：**
- ✅ 实现简单
- ✅ 数据质量高（CEX 流动性好）
- ✅ 实时性强（WebSocket）
- ✅ 多时间框架（1s - 1M）

**劣势：**
- ⚠️ 价格可能有小幅偏差（1-2%）
- ⚠️ 极端情况下 DEX 和 CEX 价差可能大

**适用场景：**
- 主流币种 (SOL, ETH, BTC)
- 价格发现在 CEX 的代币

#### 方案 2: 从 The Graph 获取历史数据 🔧

**原理：**
- The Graph 索引链上数据
- Meteora 有 Subgraph 提供交易历史
- 自己聚合成 K 线

**实现：**
```python
import requests
from datetime import datetime

class MeteoraSubgraphCandles:
    """从 Meteora Subgraph 获取交易数据并聚合 K 线"""

    SUBGRAPH_URL = "https://api.thegraph.com/subgraphs/name/meteora/dlmm-mainnet"

    async def fetch_trades(self, pool_address: str, start_time: int, end_time: int):
        query = """
        {
          swaps(
            where: {
              pool: "%s"
              timestamp_gte: %d
              timestamp_lte: %d
            }
            orderBy: timestamp
            orderDirection: asc
            first: 1000
          ) {
            timestamp
            amountIn
            amountOut
            price
          }
        }
        """ % (pool_address, start_time, end_time)

        response = requests.post(self.SUBGRAPH_URL, json={"query": query})
        return response.json()["data"]["swaps"]

    def aggregate_to_candles(self, trades: list, interval: int = 300) -> pd.DataFrame:
        """
        将交易聚合为 K 线

        interval: K 线间隔（秒），300 = 5分钟
        """
        candles = []
        current_candle = None

        for trade in trades:
            timestamp = int(trade["timestamp"])
            price = float(trade["price"])
            volume = float(trade["amountIn"])

            # 计算当前交易属于哪个 K 线周期
            candle_time = (timestamp // interval) * interval

            if current_candle is None or current_candle["timestamp"] != candle_time:
                # 新 K 线
                if current_candle:
                    candles.append(current_candle)

                current_candle = {
                    "timestamp": candle_time,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": volume,
                    "n_trades": 1
                }
            else:
                # 更新当前 K 线
                current_candle["high"] = max(current_candle["high"], price)
                current_candle["low"] = min(current_candle["low"], price)
                current_candle["close"] = price
                current_candle["volume"] += volume
                current_candle["n_trades"] += 1

        if current_candle:
            candles.append(current_candle)

        return pd.DataFrame(candles)
```

**优势：**
- ✅ 真实 DEX 数据
- ✅ 无价差问题
- ✅ 去中心化，无单点故障

**劣势：**
- ❌ 实时性差（The Graph 延迟 1-5 分钟）
- ❌ 历史数据量大，查询慢
- ❌ 实现复杂
- ❌ Subgraph 可能不稳定

**适用场景：**
- 回测
- 低频策略 (> 5 分钟)
- 需要精确 DEX 数据的场景

#### 方案 3: 使用 Birdeye/DexScreener API 📊

**原理：**
- 第三方聚合器已经做好 K 线聚合
- 直接调用 API 获取

**Birdeye API 示例：**
```python
import aiohttp

class BirdeyeCandles:
    """Birdeye API K 线"""

    BASE_URL = "https://public-api.birdeye.so"
    API_KEY = "your_api_key"

    async def get_candles(
        self,
        pool_address: str,
        interval: str = "5m",  # 1m, 5m, 15m, 1h, 4h, 1d
        limit: int = 100
    ) -> pd.DataFrame:
        url = f"{self.BASE_URL}/defi/ohlcv"
        params = {
            "address": pool_address,
            "type": interval,
            "time_from": int(time.time()) - 86400,  # 最近 24 小时
            "time_to": int(time.time())
        }
        headers = {"X-API-KEY": self.API_KEY}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as resp:
                data = await resp.json()

        candles = []
        for item in data["data"]["items"]:
            candles.append({
                "timestamp": item["unixTime"],
                "open": item["o"],
                "high": item["h"],
                "low": item["l"],
                "close": item["c"],
                "volume": item["v"]
            })

        return pd.DataFrame(candles)
```

**DexScreener API 示例：**
```python
class DexScreenerCandles:
    """DexScreener API K 线"""

    BASE_URL = "https://api.dexscreener.com"

    async def get_pair_info(self, chain: str, pair_address: str):
        """获取交易对信息（包含实时价格）"""
        url = f"{self.BASE_URL}/latest/dex/pairs/{chain}/{pair_address}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                return await resp.json()

    # 注意：DexScreener 免费版不提供 K 线，仅提供实时价格
    # K 线需要订阅 Pro 版
```

**优势：**
- ✅ 真实 DEX 数据
- ✅ 实时性好 (< 1 分钟延迟)
- ✅ 实现简单
- ✅ 多 DEX 支持

**劣势：**
- ⚠️ 需要 API Key（免费额度有限）
- ⚠️ 依赖第三方服务
- ⚠️ 可能有费用（Birdeye Pro: $99/月）

**适用场景：**
- 生产环境
- 中高频策略
- 需要多 DEX 支持

### 1.4 推荐方案对比

| 方案 | 实时性 | 准确性 | 成本 | 难度 | 推荐度 |
|------|--------|--------|------|------|--------|
| **CEX 代理** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 免费 | ⭐ | ⭐⭐⭐⭐⭐ |
| **The Graph** | ⭐⭐ | ⭐⭐⭐⭐⭐ | 免费 | ⭐⭐⭐⭐ | ⭐⭐ |
| **Birdeye** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $99/月 | ⭐⭐ | ⭐⭐⭐⭐ |

**最终推荐：CEX 代理方案**

理由：
1. Meteora 主要交易对 (SOL-USDC) 在 Binance 有对应的 SOL-USDT
2. SOL 价格发现在 CEX，DEX 价格跟随 CEX
3. 价差通常 < 0.5%，对 TA 影响极小
4. 免费、实时、简单

---

## 二、Hummingbot K线数据使用详解

### 2.1 基础用法

#### 示例 1: ScriptStrategyBase 中使用

```python
from typing import Dict
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase

class MyStrategy(ScriptStrategyBase):
    """使用 K 线的策略示例"""

    markets = {"binance_paper_trade": {"ETH-USDT"}}

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)

        # 创建 K 线实例
        self.candles_5m = CandlesFactory.get_candle(
            CandlesConfig(
                connector="binance",
                trading_pair="ETH-USDT",
                interval="5m",
                max_records=200  # 最多存储 200 根 K 线
            )
        )

        # 启动 K 线数据流
        self.candles_5m.start()

    def on_tick(self):
        # 检查 K 线是否就绪
        if not self.candles_5m.ready:
            self.logger().info("等待 K 线数据...")
            return

        # 获取 DataFrame
        df = self.candles_5m.candles_df

        # 计算技术指标
        import pandas_ta as ta
        df.ta.rsi(length=14, append=True)
        df.ta.bbands(length=20, append=True)

        # 获取最新值
        latest_rsi = df["RSI_14"].iloc[-1]
        latest_bb_upper = df["BBU_20_2.0"].iloc[-1]

        self.logger().info(f"RSI: {latest_rsi:.2f}, BB Upper: {latest_bb_upper:.2f}")

    async def on_stop(self):
        # 停止策略时关闭 K 线流
        self.candles_5m.stop()
```

#### 示例 2: StrategyV2Base 中使用 (推荐)

```python
import os
from decimal import Decimal
from typing import Dict, List
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase

class MyStrategyConfig(StrategyV2ConfigBase):
    script_file_name: str = os.path.basename(__file__)

    # K 线配置
    candles_connector: str = "binance"
    candles_pair: str = "SOL-USDT"
    candles_interval: str = "5m"
    candles_length: int = 200

class MyStrategy(StrategyV2Base):
    """StrategyV2 会自动管理 K 线生命周期"""

    @classmethod
    def init_markets(cls, config: MyStrategyConfig):
        cls.markets = {"binance_paper_trade": {"SOL-USDT"}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: MyStrategyConfig):
        # 自动初始化 K 线配置
        if len(config.candles_config) == 0:
            config.candles_config.append(
                CandlesConfig(
                    connector=config.candles_connector,
                    trading_pair=config.candles_pair,
                    interval=config.candles_interval,
                    max_records=config.candles_length
                )
            )
        super().__init__(connectors, config)

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        # 通过 market_data_provider 获取 K 线
        candles_df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_pair,
            interval=self.config.candles_interval,
            max_records=self.config.candles_length
        )

        # 计算指标
        candles_df.ta.rsi(length=14, append=True)
        rsi = candles_df["RSI_14"].iloc[-1]

        # 根据 RSI 决策
        if rsi < 30:
            # 超卖，做多
            return [CreateExecutorAction(...)]
        elif rsi > 70:
            # 超买，做空
            return [CreateExecutorAction(...)]

        return []
```

### 2.2 多时间框架分析

```python
class MultiTimeframeStrategy(StrategyV2Base):
    """多时间框架策略"""

    def __init__(self, connectors, config):
        # 配置多个时间框架
        config.candles_config = [
            CandlesConfig(connector="binance", trading_pair="SOL-USDT", interval="1m", max_records=200),
            CandlesConfig(connector="binance", trading_pair="SOL-USDT", interval="5m", max_records=200),
            CandlesConfig(connector="binance", trading_pair="SOL-USDT", interval="1h", max_records=200),
        ]
        super().__init__(connectors, config)

    def get_multi_timeframe_signal(self) -> str:
        """多时间框架趋势确认"""

        # 1分钟 K 线
        df_1m = self.market_data_provider.get_candles_df("binance", "SOL-USDT", "1m", 200)
        df_1m.ta.ema(length=20, append=True)

        # 5分钟 K 线
        df_5m = self.market_data_provider.get_candles_df("binance", "SOL-USDT", "5m", 200)
        df_5m.ta.ema(length=20, append=True)

        # 1小时 K 线
        df_1h = self.market_data_provider.get_candles_df("binance", "SOL-USDT", "1h", 200)
        df_1h.ta.ema(length=20, append=True)

        # 获取当前价格
        price_1m = df_1m["close"].iloc[-1]
        price_5m = df_5m["close"].iloc[-1]
        price_1h = df_1h["close"].iloc[-1]

        ema_1m = df_1m["EMA_20"].iloc[-1]
        ema_5m = df_5m["EMA_20"].iloc[-1]
        ema_1h = df_1h["EMA_20"].iloc[-1]

        # 多时间框架确认
        if price_1m > ema_1m and price_5m > ema_5m and price_1h > ema_1h:
            return "bullish"  # 所有时间框架都上涨
        elif price_1m < ema_1m and price_5m < ema_5m and price_1h < ema_1h:
            return "bearish"  # 所有时间框架都下跌
        else:
            return "ranging"  # 时间框架不一致
```

### 2.3 K线数据结构

```python
# candles_df 列结构
columns = [
    "timestamp",           # Unix 时间戳（秒）
    "open",               # 开盘价
    "high",               # 最高价
    "low",                # 最低价
    "close",              # 收盘价
    "volume",             # 成交量 (Base Token)
    "quote_asset_volume", # 成交额 (Quote Token)
    "n_trades",           # 成交笔数
    "taker_buy_base_volume",  # 主动买入成交量 (Base)
    "taker_buy_quote_volume", # 主动买入成交额 (Quote)
]

# 示例数据
#  timestamp      open      high       low     close    volume  quote_asset_volume  n_trades
#  1698739200   95.50     95.80     95.20     95.70   1250.5          119532.5         523
#  1698739500   95.70     96.00     95.50     95.80   1100.2          105319.2         451
```

---

## 三、技术分析库集成

### 3.1 pandas_ta (推荐)

**安装：**
```bash
pip install pandas_ta
```

**支持指标：**
- 趋势指标：EMA, SMA, WMA, DEMA, TEMA, HMA, VWMA, etc.
- 动量指标：RSI, MACD, Stochastic, CCI, ROC, etc.
- 波动率指标：Bollinger Bands, ATR, Keltner Channels, etc.
- 成交量指标：OBV, AD, VWAP, etc.
- 130+ 指标

**使用示例：**
```python
import pandas_ta as ta

df = candles.candles_df

# 单个指标
df.ta.rsi(length=14, append=True)  # 添加 RSI_14 列

# 多个指标
df.ta.bbands(length=20, std=2, append=True)  # BBL_20_2.0, BBM_20_2.0, BBU_20_2.0

# 指标组合
df.ta.strategy("all")  # 计算所有指标（慢）

# 自定义策略
my_strategy = ta.Strategy(
    name="My TA",
    ta=[
        {"kind": "rsi", "length": 14},
        {"kind": "bbands", "length": 20},
        {"kind": "ema", "length": 50},
        {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
    ]
)
df.ta.strategy(my_strategy)
```

### 3.2 TA-Lib (可选)

**安装：**
```bash
# macOS
brew install ta-lib
pip install TA-Lib

# Ubuntu
sudo apt-get install ta-lib
pip install TA-Lib
```

**使用：**
```python
import talib

df = candles.candles_df

# 计算 RSI
df["RSI"] = talib.RSI(df["close"], timeperiod=14)

# 计算 MACD
df["MACD"], df["MACD_signal"], df["MACD_hist"] = talib.MACD(
    df["close"],
    fastperiod=12,
    slowperiod=26,
    signalperiod=9
)

# 计算 Bollinger Bands
df["BB_upper"], df["BB_middle"], df["BB_lower"] = talib.BBANDS(
    df["close"],
    timeperiod=20,
    nbdevup=2,
    nbdevdn=2
)
```

---

## 四、针对 DLMM LP 的技术分析策略

### 4.1 核心思路

**传统 LP 策略问题：**
- ❌ 固定区间 (如 ±10%)，不考虑市场结构
- ❌ 简单阈值触发，忽略趋势和支撑/压力位

**TA 驱动 DLMM LP 策略优势：**
- ✅ 用技术分析识别**真实的支撑/压力位**
- ✅ 在高概率震荡区间设置 LP
- ✅ 窄区间高频，类似 PMM
- ✅ 突破支撑/压力立即止损

### 4.2 关键技术指标

#### 指标 1: Pivot Points (枢轴点)

**作用：** 识别日内支撑/压力位

```python
def calculate_pivot_points(df: pd.DataFrame) -> dict:
    """
    计算枢轴点支撑/压力位

    使用前一日 (或前一根 K 线) 的高、低、收计算
    """
    prev = df.iloc[-2]  # 前一根 K 线

    high = prev["high"]
    low = prev["low"]
    close = prev["close"]

    # 枢轴点
    PP = (high + low + close) / 3

    # 支撑位
    S1 = 2 * PP - high
    S2 = PP - (high - low)
    S3 = low - 2 * (high - PP)

    # 压力位
    R1 = 2 * PP - low
    R2 = PP + (high - low)
    R3 = high + 2 * (PP - low)

    return {
        "PP": PP,
        "S1": S1, "S2": S2, "S3": S3,
        "R1": R1, "R2": R2, "R3": R3
    }
```

**LP 应用：**
```python
# 获取当前价格
current_price = df["close"].iloc[-1]

# 计算枢轴点
pivots = calculate_pivot_points(df)

# 确定 LP 区间
if pivots["S1"] < current_price < pivots["PP"]:
    # 价格在 S1 和 PP 之间
    lp_lower = pivots["S1"]
    lp_upper = pivots["PP"]
elif pivots["PP"] < current_price < pivots["R1"]:
    # 价格在 PP 和 R1 之间
    lp_lower = pivots["PP"]
    lp_upper = pivots["R1"]
```

#### 指标 2: Bollinger Bands

**作用：** 动态识别震荡区间

```python
df.ta.bbands(length=20, std=2, append=True)

# 获取布林带
bb_upper = df["BBU_20_2.0"].iloc[-1]
bb_middle = df["BBM_20_2.0"].iloc[-1]
bb_lower = df["BBL_20_2.0"].iloc[-1]

# LP 区间 = 布林带上下轨
lp_lower = bb_lower
lp_upper = bb_upper
```

#### 指标 3: ATR (波动率)

**作用：** 动态调整区间宽度

```python
df.ta.atr(length=14, append=True)
atr = df["ATRr_14"].iloc[-1]  # ATR 百分比

current_price = df["close"].iloc[-1]

# 根据 ATR 设置区间宽度
if atr < 0.02:  # 低波动 (<2%)
    range_width = 0.005  # 0.5% 窄区间
elif atr < 0.05:  # 中波动
    range_width = 0.015  # 1.5%
else:  # 高波动
    range_width = 0.025  # 2.5%

lp_lower = current_price * (1 - range_width)
lp_upper = current_price * (1 + range_width)
```

#### 指标 4: EMA Ribbon (趋势)

**作用：** 识别趋势，决定是否开仓/单边流动性

```python
df.ta.ema(length=10, append=True)
df.ta.ema(length=20, append=True)
df.ta.ema(length=50, append=True)

ema_10 = df["EMA_10"].iloc[-1]
ema_20 = df["EMA_20"].iloc[-1]
ema_50 = df["EMA_50"].iloc[-1]

if ema_10 > ema_20 > ema_50:
    trend = "bullish"
    # 上涨趋势：单边 USDC LP 或暂停
elif ema_10 < ema_20 < ema_50:
    trend = "bearish"
    # 下跌趋势：单边 USDC LP
else:
    trend = "ranging"
    # 震荡市：双边 LP
```

### 4.3 完整 TA 决策示例

```python
class TechnicalAnalysisEngine:
    """技术分析引擎 for DLMM LP"""

    def analyze(self, candles_df: pd.DataFrame) -> dict:
        """
        综合技术分析

        返回:
        {
            "support": 95.5,
            "resistance": 105.2,
            "trend": "ranging",  # bullish/bearish/ranging
            "volatility": "low",  # low/medium/high
            "confidence": 0.85,  # 信心度 (0-1)
            "lp_range": (96.0, 104.0),  # 建议 LP 区间
            "should_open": True  # 是否应该开仓
        }
        """

        # 1. 计算所有指标
        candles_df.ta.rsi(length=14, append=True)
        candles_df.ta.bbands(length=20, append=True)
        candles_df.ta.atr(length=14, append=True)
        candles_df.ta.ema(length=10, append=True)
        candles_df.ta.ema(length=20, append=True)
        candles_df.ta.ema(length=50, append=True)

        # 2. 获取最新值
        current = candles_df.iloc[-1]
        current_price = current["close"]

        # 3. 趋势判断
        trend = self._detect_trend(candles_df)

        # 4. 波动率
        atr_pct = current["ATRr_14"]
        if atr_pct < 0.02:
            volatility = "low"
        elif atr_pct < 0.05:
            volatility = "medium"
        else:
            volatility = "high"

        # 5. 支撑/压力位
        pivots = self._calculate_pivot_points(candles_df)
        bb_upper = current["BBU_20_2.0"]
        bb_lower = current["BBL_20_2.0"]

        # 多指标投票
        support = (pivots["S1"] * 0.5 + bb_lower * 0.5)
        resistance = (pivots["R1"] * 0.5 + bb_upper * 0.5)

        # 6. 信心度
        # 布林带宽度越窄，信心度越高（震荡市）
        bb_width = (bb_upper - bb_lower) / current_price
        if bb_width < 0.05:  # <5%
            confidence = 0.9
        elif bb_width < 0.10:
            confidence = 0.7
        else:
            confidence = 0.5

        # 7. LP 区间建议
        if trend == "ranging" and confidence > 0.75:
            # 震荡市 + 高信心：开仓
            lp_range = (support * 1.002, resistance * 0.998)  # 稍微收窄，避免立即触发止损
            should_open = True
        elif trend == "bullish" and volatility == "low":
            # 上涨趋势 + 低波动：单边 USDC
            lp_range = (current_price * 0.997, current_price * 1.010)  # 区间偏上
            should_open = True
        elif trend == "bearish":
            # 下跌趋势：暂停或单边 USDC
            lp_range = None
            should_open = False
        else:
            # 其他情况：观望
            lp_range = None
            should_open = False

        return {
            "support": support,
            "resistance": resistance,
            "trend": trend,
            "volatility": volatility,
            "confidence": confidence,
            "lp_range": lp_range,
            "should_open": should_open
        }

    def _detect_trend(self, df: pd.DataFrame) -> str:
        """检测趋势"""
        ema_10 = df["EMA_10"].iloc[-1]
        ema_20 = df["EMA_20"].iloc[-1]
        ema_50 = df["EMA_50"].iloc[-1]

        if ema_10 > ema_20 > ema_50:
            return "bullish"
        elif ema_10 < ema_20 < ema_50:
            return "bearish"
        else:
            return "ranging"

    def _calculate_pivot_points(self, df: pd.DataFrame) -> dict:
        """计算枢轴点"""
        prev = df.iloc[-2]
        high = prev["high"]
        low = prev["low"]
        close = prev["close"]

        PP = (high + low + close) / 3
        S1 = 2 * PP - high
        R1 = 2 * PP - low

        return {"PP": PP, "S1": S1, "R1": R1}
```

---

## 五、总结与建议

### 5.1 K线数据获取策略

| 交易场所 | 推荐方案 | 理由 |
|----------|----------|------|
| **Meteora DLMM (DEX)** | CEX 代理 (Binance SOL-USDT) | 免费、实时、简单 |
| **其他 Solana DEX** | CEX 代理 | 同上 |
| **EVM DEX** | CEX 代理 或 Birdeye API | 根据预算选择 |
| **回测** | The Graph | 真实 DEX 数据 |

### 5.2 技术分析框架

```
K 线数据 (CandlesFactory)
    ↓
技术指标计算 (pandas_ta)
    ↓
多指标综合评分
    ↓
趋势 + 支撑/压力 + 波动率
    ↓
LP 区间决策
    ↓
开仓/平仓/再平衡
```

### 5.3 最佳实践

1. **多时间框架确认** - 1m + 5m + 1h 三重确认
2. **高信心度开仓** - confidence > 75%
3. **窄区间高频** - 0.5%-2% 区间，类似 PMM
4. **严格止损** - 突破支撑/压力立即平仓
5. **趋势适应** - 震荡做双边，趋势做单边

---

**文档版本：1.0**
**日期：2025-11-02**
**作者：Claude (Anthropic)**
