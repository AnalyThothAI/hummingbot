# Redis 缓存管理器使用说明

## 概述

`redis_cache_manager.py` 是一个通用的 Redis 缓存工具类，可供所有 Hummingbot 策略复用。

## 特性

- ✅ 分布式缓存（多策略实例共享）
- ✅ TTL 自动过期
- ✅ 内存降级（Redis 不可用时）
- ✅ 分布式锁（防止并发刷新）
- ✅ 简单易用的 API

## 快速开始

### 1. 导入

```python
from redis_cache_manager import RedisCacheManager, RedisConfig
```

### 2. 创建实例

```python
# 配置 Redis 连接
redis_config = RedisConfig(
    host="localhost",
    port=6379,
    db=0,
    password=None  # 如果有密码，填写这里
)

# 创建缓存管理器
cache = RedisCacheManager(
    prefix="my_strategy",  # 策略名称作为前缀，避免键冲突
    config=redis_config,
    logger=self.logger(),
    enable_memory_fallback=True  # 启用内存降级
)

# 连接 Redis
await cache.connect()
```

### 3. 基本操作

```python
# 写入缓存（60秒 TTL）
await cache.set("price:BTC", "95000", ttl=60)

# 读取缓存
price = await cache.get("price:BTC")
if price:
    print(f"BTC 价格: {price}")

# 写入持久化缓存（24小时 TTL）
await cache.set_persistent("config:last_known", "some_value", ttl=86400)

# 删除缓存
await cache.delete("price:BTC")

# 检查是否存在
exists = await cache.exists("price:BTC")

# 获取剩余 TTL
ttl = await cache.get_ttl("price:BTC")
```

### 4. 分布式锁

防止多个策略实例同时执行相同的操作：

```python
# 尝试获取锁（5秒超时）
if await cache.acquire_lock("refresh_price", timeout=5):
    try:
        # 临界区代码
        price = await fetch_price_from_api()
        await cache.set("price:BTC", str(price), ttl=60)
    finally:
        # 释放锁
        await cache.release_lock("refresh_price")
else:
    # 锁被其他实例占用，等待一下再读取缓存
    await asyncio.sleep(0.5)
    price = await cache.get("price:BTC")
```

### 5. 关闭连接

```python
# 策略退出时关闭 Redis 连接
await cache.close()
```

## 完整示例

```python
from redis_cache_manager import RedisCacheManager, RedisConfig
import asyncio
from decimal import Decimal

class MyStrategy:
    def __init__(self):
        # 初始化缓存
        redis_config = RedisConfig(host="localhost", port=6379, db=0)
        self.cache = RedisCacheManager(
            prefix="my_strategy",
            config=redis_config,
            logger=self.logger(),
            enable_memory_fallback=True
        )

        # 异步连接 Redis
        asyncio.create_task(self.cache.connect())

    async def get_token_price(self, token: str) -> Decimal:
        """获取代币价格（带缓存）"""

        # 1. 检查缓存
        cached_price = await self.cache.get(f"price:{token}")
        if cached_price:
            return Decimal(cached_price)

        # 2. 尝试获取锁
        lock_acquired = await self.cache.acquire_lock(f"fetch:{token}", timeout=5)

        if not lock_acquired:
            # 其他实例正在刷新，等待后再读缓存
            await asyncio.sleep(0.5)
            cached_price = await self.cache.get(f"price:{token}")
            if cached_price:
                return Decimal(cached_price)

        try:
            # 3. 从 API 获取价格
            price = await self.fetch_from_api(token)

            # 4. 写入缓存（60秒 TTL）
            await self.cache.set(f"price:{token}", str(price), ttl=60)

            return price

        finally:
            if lock_acquired:
                await self.cache.release_lock(f"fetch:{token}")

    async def fetch_from_api(self, token: str) -> Decimal:
        """从 API 获取价格"""
        # 实际 API 调用
        pass
```

## 配置参数

### RedisConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `host` | str | "localhost" | Redis 主机地址 |
| `port` | int | 6379 | Redis 端口 |
| `db` | int | 0 | Redis 数据库编号 |
| `password` | str\|None | None | Redis 密码（可选） |
| `socket_timeout` | int | 5 | Socket 超时（秒） |
| `socket_connect_timeout` | int | 5 | 连接超时（秒） |

### RedisCacheManager

| 参数 | 类型 | 说明 |
|------|------|------|
| `prefix` | str | 缓存键前缀（用于区分不同策略） |
| `config` | RedisConfig | Redis 连接配置 |
| `logger` | logging.Logger | 日志记录器 |
| `enable_memory_fallback` | bool | 启用内存降级（Redis 不可用时） |

## 实际应用

### 在 CEX-DEX LP 套利策略中的应用

`cex_dex_lp_arbitrage.py` 中使用 `RedisCacheManager` 缓存汇率转换：

```python
from redis_cache_manager import RedisCacheManager, RedisConfig

class ConversionRateCache:
    def __init__(self, config, logger):
        # 使用通用 Redis 缓存管理器
        redis_config = RedisConfig(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db
        )
        self.cache_manager = RedisCacheManager(
            prefix="conv_rate",
            config=redis_config,
            logger=logger,
            enable_memory_fallback=True
        )

    async def get_rate(self, token: str):
        # 从缓存读取汇率
        data = await self.cache_manager.get(f"{token}:USDT")
        # ...

    async def set_rate(self, token: str, price: Decimal, source: str):
        # 写入缓存
        await self.cache_manager.set(f"{token}:USDT", data_str, ttl=60)
        # ...
```

## 注意事项

1. **键命名规范**：使用 `prefix:category:key` 格式，例如：
   - `my_strategy:price:BTC`
   - `conv_rate:WBNB:USDT`

2. **TTL 设置**：根据数据更新频率合理设置 TTL
   - 高频数据（如价格）: 30-60 秒
   - 中频数据（如配置）: 5-10 分钟
   - 低频数据（如历史记录）: 1-24 小时

3. **内存降级**：Redis 不可用时自动切换到内存缓存，确保策略不中断

4. **分布式锁**：防止多实例并发刷新相同数据，避免 API 请求浪费

## 依赖

```bash
pip install redis
```

或者使用异步版本：

```bash
pip install redis[hiredis]
```

## 许可证

MIT License
