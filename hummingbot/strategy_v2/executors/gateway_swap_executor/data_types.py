from decimal import Decimal
from typing import Literal, Optional

from pydantic import BaseModel

from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy_v2.executors.data_types import ExecutorConfigBase


class GatewaySwapExecutorConfig(ExecutorConfigBase):
    type: Literal["gateway_swap_executor"] = "gateway_swap_executor"
    connector_name: str
    trading_pair: str
    side: TradeType
    amount: Decimal
    amount_in_is_quote: bool = False
    slippage_pct: Optional[Decimal] = None
    pool_address: Optional[str] = None
    timeout_sec: int = 120
    poll_interval_sec: Decimal = Decimal("2")
    max_retries: int = 0
    retry_delay_sec: Decimal = Decimal("1")
    level_id: Optional[str] = None
    budget_key: Optional[str] = None


class GatewaySwapExecutorStatus(BaseModel):
    state: str
    order_id: Optional[str] = None
    last_error: Optional[str] = None
