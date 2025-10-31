"""
Redis 缓存管理工具类
用于 Hummingbot 策略的分布式缓存

特性:
- 支持多策略实例共享缓存
- TTL 自动过期
- 降级到 last_known 缓存
- 分布式锁（防止并发刷新）
- 内存降级（Redis 不可用时）
"""

import asyncio
import uuid
from typing import Optional, Tuple
from decimal import Decimal
import logging

try:
    import redis.asyncio as aioredis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None


class RedisConfig:
    """Redis 连接配置"""
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout


class RedisCacheManager:
    """
    Redis 缓存管理器

    用法:
    ```python
    cache = RedisCacheManager(
        prefix="strategy_name",
        config=RedisConfig(),
        logger=strategy.logger()
    )
    await cache.connect()

    # 读取缓存
    value = await cache.get("key")

    # 写入缓存（60秒 TTL）
    await cache.set("key", "value", ttl=60)

    # 写入持久化缓存（24小时 TTL）
    await cache.set_persistent("key", "value", ttl=86400)

    # 获取分布式锁
    if await cache.acquire_lock("lock_key", timeout=5):
        try:
            # 临界区代码
            pass
        finally:
            await cache.release_lock("lock_key")

    # 关闭连接
    await cache.close()
    ```
    """

    def __init__(
        self,
        prefix: str,
        config: RedisConfig,
        logger: logging.Logger,
        enable_memory_fallback: bool = True
    ):
        """
        初始化 Redis 缓存管理器

        Args:
            prefix: 缓存键前缀（用于区分不同策略）
            config: Redis 连接配置
            logger: 日志记录器
            enable_memory_fallback: 启用内存降级（Redis 不可用时）
        """
        self.prefix = prefix
        self.config = config
        self.logger = logger
        self.enable_memory_fallback = enable_memory_fallback

        # Redis 客户端
        self.redis_client: Optional[Redis] = None
        self.redis_available = False

        # 内存降级存储
        self.memory_cache: dict = {}

        # 锁管理
        self.instance_id = str(uuid.uuid4())[:8]
        self.owned_locks: set = set()

    async def connect(self) -> bool:
        """
        连接到 Redis

        Returns:
            是否成功连接
        """
        if not REDIS_AVAILABLE:
            self.logger.warning("⚠️ redis.asyncio 未安装，使用内存降级")
            return False

        try:
            self.redis_client = aioredis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=True
            )

            # 测试连接
            await self.redis_client.ping()
            self.redis_available = True
            self.logger.info(f"✅ Redis 连接成功: {self.config.host}:{self.config.port}")
            return True

        except Exception as e:
            self.logger.warning(f"⚠️ Redis 连接失败，使用内存降级: {e}")
            self.redis_available = False
            return False

    async def close(self):
        """关闭 Redis 连接"""
        if self.redis_client:
            try:
                # 释放所有持有的锁
                for lock_key in list(self.owned_locks):
                    await self.release_lock(lock_key)

                await self.redis_client.close()
                self.logger.info("Redis 连接已关闭")
            except Exception as e:
                self.logger.error(f"关闭 Redis 连接失败: {e}")

    def _make_key(self, key: str) -> str:
        """生成带前缀的缓存键"""
        return f"{self.prefix}:{key}"

    async def get(self, key: str) -> Optional[str]:
        """
        获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存值，不存在返回 None
        """
        full_key = self._make_key(key)

        # 优先从 Redis 读取
        if self.redis_available and self.redis_client:
            try:
                value = await self.redis_client.get(full_key)
                if value:
                    return value
            except Exception as e:
                self.logger.error(f"Redis 读取失败: {e}")

        # 降级到内存缓存
        if self.enable_memory_fallback:
            return self.memory_cache.get(full_key)

        return None

    async def set(self, key: str, value: str, ttl: int = 60) -> bool:
        """
        设置缓存值（带 TTL）

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）

        Returns:
            是否成功
        """
        full_key = self._make_key(key)

        # 优先写入 Redis
        if self.redis_available and self.redis_client:
            try:
                await self.redis_client.setex(full_key, ttl, value)

                # 同时写入内存缓存作为备份
                if self.enable_memory_fallback:
                    self.memory_cache[full_key] = value

                return True
            except Exception as e:
                self.logger.error(f"Redis 写入失败: {e}")

        # 降级到内存缓存
        if self.enable_memory_fallback:
            self.memory_cache[full_key] = value
            return True

        return False

    async def set_persistent(self, key: str, value: str, ttl: int = 86400) -> bool:
        """
        设置持久化缓存（默认 24 小时 TTL）

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒），默认 24 小时

        Returns:
            是否成功
        """
        return await self.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """
        删除缓存

        Args:
            key: 缓存键

        Returns:
            是否成功
        """
        full_key = self._make_key(key)

        success = False

        # 从 Redis 删除
        if self.redis_available and self.redis_client:
            try:
                await self.redis_client.delete(full_key)
                success = True
            except Exception as e:
                self.logger.error(f"Redis 删除失败: {e}")

        # 从内存缓存删除
        if self.enable_memory_fallback and full_key in self.memory_cache:
            del self.memory_cache[full_key]
            success = True

        return success

    async def acquire_lock(self, lock_key: str, timeout: int = 5) -> bool:
        """
        获取分布式锁

        Args:
            lock_key: 锁的键名
            timeout: 锁的超时时间（秒）

        Returns:
            是否成功获取锁
        """
        full_key = self._make_key(f"lock:{lock_key}")
        lock_value = f"{self.instance_id}:{asyncio.current_task().get_name() if asyncio.current_task() else 'unknown'}"

        # 优先使用 Redis 分布式锁
        if self.redis_available and self.redis_client:
            try:
                # SET NX EX（原子操作）
                acquired = await self.redis_client.set(
                    full_key,
                    lock_value,
                    nx=True,  # 仅当键不存在时设置
                    ex=timeout  # 过期时间
                )

                if acquired:
                    self.owned_locks.add(lock_key)
                    self.logger.debug(f"🔒 获取锁成功: {lock_key}")
                    return True
                else:
                    self.logger.debug(f"🔒 锁已被占用: {lock_key}")
                    return False

            except Exception as e:
                self.logger.error(f"Redis 获取锁失败: {e}")

        # 降级到内存锁（仅单实例有效）
        if self.enable_memory_fallback:
            if full_key not in self.memory_cache:
                self.memory_cache[full_key] = lock_value
                self.owned_locks.add(lock_key)
                self.logger.debug(f"🔒 获取内存锁成功: {lock_key}")
                return True
            else:
                self.logger.debug(f"🔒 内存锁已被占用: {lock_key}")
                return False

        return False

    async def release_lock(self, lock_key: str) -> bool:
        """
        释放分布式锁

        Args:
            lock_key: 锁的键名

        Returns:
            是否成功释放
        """
        if lock_key not in self.owned_locks:
            self.logger.warning(f"⚠️ 尝试释放未持有的锁: {lock_key}")
            return False

        full_key = self._make_key(f"lock:{lock_key}")
        lock_value = f"{self.instance_id}:{asyncio.current_task().get_name() if asyncio.current_task() else 'unknown'}"

        # 从 Redis 释放锁（仅释放自己持有的锁）
        if self.redis_available and self.redis_client:
            try:
                # Lua 脚本确保原子性：只删除自己持有的锁
                lua_script = """
                if redis.call("get", KEYS[1]) == ARGV[1] then
                    return redis.call("del", KEYS[1])
                else
                    return 0
                end
                """
                result = await self.redis_client.eval(lua_script, 1, full_key, lock_value)

                if result == 1:
                    self.owned_locks.discard(lock_key)
                    self.logger.debug(f"🔓 释放锁成功: {lock_key}")
                    return True
                else:
                    self.logger.warning(f"⚠️ 锁已被其他实例占用，无法释放: {lock_key}")
                    return False

            except Exception as e:
                self.logger.error(f"Redis 释放锁失败: {e}")

        # 从内存释放锁
        if self.enable_memory_fallback and full_key in self.memory_cache:
            if self.memory_cache[full_key] == lock_value:
                del self.memory_cache[full_key]
                self.owned_locks.discard(lock_key)
                self.logger.debug(f"🔓 释放内存锁成功: {lock_key}")
                return True

        return False

    async def exists(self, key: str) -> bool:
        """
        检查缓存是否存在

        Args:
            key: 缓存键

        Returns:
            是否存在
        """
        full_key = self._make_key(key)

        # 检查 Redis
        if self.redis_available and self.redis_client:
            try:
                return await self.redis_client.exists(full_key) > 0
            except Exception as e:
                self.logger.error(f"Redis exists 检查失败: {e}")

        # 检查内存缓存
        if self.enable_memory_fallback:
            return full_key in self.memory_cache

        return False

    async def get_ttl(self, key: str) -> int:
        """
        获取缓存剩余 TTL

        Args:
            key: 缓存键

        Returns:
            剩余秒数，-1 表示永久，-2 表示不存在
        """
        full_key = self._make_key(key)

        if self.redis_available and self.redis_client:
            try:
                return await self.redis_client.ttl(full_key)
            except Exception as e:
                self.logger.error(f"Redis TTL 检查失败: {e}")

        # 内存缓存不支持 TTL
        if self.enable_memory_fallback and full_key in self.memory_cache:
            return -1

        return -2
