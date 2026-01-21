from typing import Any

from hummingbot.strategy_v2.executors.gateway_swap_executor.data_types import GatewaySwapExecutorConfig

__all__ = ["GatewaySwapExecutor", "GatewaySwapExecutorConfig"]


def __getattr__(name: str) -> Any:
    if name == "GatewaySwapExecutor":
        from hummingbot.strategy_v2.executors.gateway_swap_executor.gateway_swap_executor import GatewaySwapExecutor
        return GatewaySwapExecutor
    raise AttributeError(f"module {__name__} has no attribute {name}")
