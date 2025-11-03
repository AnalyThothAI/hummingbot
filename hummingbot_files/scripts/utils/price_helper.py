"""
价格获取和计算辅助函数

负责：
1. 获取当前价格（多重降级策略）
2. 获取池子信息
3. 注入价格到 RateOracle

简化主策略文件的价格获取逻辑
"""

from decimal import Decimal
from typing import Optional

from hummingbot.connector.gateway.gateway_lp import CLMMPoolInfo
from hummingbot.core.rate_oracle.rate_oracle import RateOracle


async def get_current_price(
    connector,
    trading_pair: str,
    logger
) -> Optional[Decimal]:
    """
    获取当前价格（多重降级策略）

    优先级：
    1. get_pool_info() - 最完整的信息
    2. get_quote_price() - 备用方案（swap 报价）

    Args:
        connector: Gateway connector 实例
        trading_pair: 交易对
        logger: 日志记录器

    Returns:
        当前价格（Decimal）或 None
    """
    try:
        # 方法 1: get_pool_info()（推荐）
        try:
            logger.debug(f"尝试获取池子信息: {trading_pair}")
            pool_info = await connector.get_pool_info(
                trading_pair=trading_pair
            )
            if pool_info and hasattr(pool_info, 'price') and pool_info.price > 0:
                price = Decimal(str(pool_info.price))
                logger.debug(f"✅ 池子价格: {price} (active_bin_id: {pool_info.active_bin_id})")

                # 注入价格到 RateOracle
                try:
                    rate_oracle = RateOracle.get_instance()
                    rate_oracle.set_price(trading_pair, price)
                except Exception as oracle_err:
                    logger.debug(f"RateOracle 注入失败: {oracle_err}")

                return price
            else:
                logger.warning(f"⚠️ 池子信息无效或价格为 0: {pool_info}")
        except Exception as e:
            logger.warning(f"⚠️ get_pool_info() 失败: {e}，尝试备用方案...")

        # 方法 2: get_quote_price()（备用）
        try:
            logger.debug(f"尝试使用 get_quote_price() 获取价格: {trading_pair}")
            quote_price = await connector.get_quote_price(
                trading_pair=trading_pair,
                is_buy=True,
                amount=Decimal("1")
            )

            if quote_price and quote_price > 0:
                price = Decimal(str(quote_price))
                logger.debug(f"✅ Quote 价格: {price}")

                # 注入价格到 RateOracle
                try:
                    rate_oracle = RateOracle.get_instance()
                    rate_oracle.set_price(trading_pair, price)
                except Exception as oracle_err:
                    logger.debug(f"RateOracle 注入失败: {oracle_err}")

                return price
            else:
                logger.warning(f"⚠️ Quote 价格无效: {quote_price}")
        except Exception as e:
            logger.warning(f"⚠️ get_quote_price() 失败: {e}")

        # 所有方法都失败
        logger.error("❌ 所有价格获取方法均失败")
        return None

    except Exception as e:
        logger.error(f"❌ 获取价格时发生严重错误: {e}", exc_info=True)
        return None


async def fetch_pool_info(
    connector,
    trading_pair: str,
    logger
) -> Optional[CLMMPoolInfo]:
    """
    获取池子信息

    Args:
        connector: Gateway connector 实例
        trading_pair: 交易对
        logger: 日志记录器

    Returns:
        池子信息（CLMMPoolInfo）或 None
    """
    try:
        pool_info = await connector.get_pool_info(
            trading_pair=trading_pair
        )

        # 注入价格到 RateOracle
        if pool_info:
            try:
                current_price = Decimal(str(pool_info.price))
                rate_oracle = RateOracle.get_instance()
                rate_oracle.set_price(trading_pair, current_price)
                logger.debug(f"注入池子价格: {current_price}")
            except Exception as oracle_err:
                logger.debug(f"RateOracle 注入失败: {oracle_err}")

        return pool_info
    except Exception as e:
        logger.error(f"获取池子信息失败: {e}")
        return None


async def get_pool_address(
    connector,
    trading_pair: str,
    pool_address_from_config: Optional[str],
    logger
) -> Optional[str]:
    """
    获取池子地址

    Args:
        connector: Gateway connector 实例
        trading_pair: 交易对
        pool_address_from_config: 配置文件中的池子地址
        logger: 日志记录器

    Returns:
        池子地址（str）或 None
    """
    if pool_address_from_config:
        return pool_address_from_config
    else:
        try:
            return await connector.get_pool_address(trading_pair)
        except Exception as e:
            logger.error(f"获取池子地址失败: {e}")
            return None
