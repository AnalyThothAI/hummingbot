import hashlib
import random
import time
from decimal import Decimal
from typing import Literal, Optional

import base58
from pydantic import BaseModel, field_validator, model_validator

from hummingbot.client.settings import AllConnectorSettings
from hummingbot.core.data_type.common import TradeType


class ExecutorConfigBase(BaseModel):
    id: str = None  # Make ID optional
    type: Literal["position_executor", "dca_executor", "grid_executor", "order_executor",
                  "xemm_executor", "arbitrage_executor", "twap_executor", "lp_position_executor",
                  "gateway_swap_executor"]
    timestamp: Optional[float] = None
    controller_id: str = "main"

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, value: Optional[float]) -> float:
        if value is None:
            # Use current time if timestamp is not provided
            return time.time()
        return value

    @model_validator(mode="after")
    def set_id(self):
        if self.id is None:
            # Use timestamp from values if available, else current time
            unique_component = random.randint(0, 99999)
            raw_id = f"{self.timestamp}-{unique_component}"
            hashed_id = hashlib.sha256(raw_id.encode()).digest()  # Get bytes
            self.id = base58.b58encode(hashed_id).decode()  # Base58 encode
        return self


class ConnectorPair(BaseModel):
    connector_name: str
    trading_pair: str

    def is_amm_connector(self) -> bool:
        return self.connector_name in sorted(
            AllConnectorSettings.get_gateway_amm_connector_names()
        )

    class Config:
        frozen = True  # This makes the model immutable and thus hashable

    def __iter__(self):
        yield self.connector_name
        yield self.trading_pair


class PositionSummary(BaseModel):
    connector_name: str
    trading_pair: str
    volume_traded_quote: Decimal
    side: TradeType
    amount: Decimal
    breakeven_price: Decimal
    unrealized_pnl_quote: Decimal
    realized_pnl_quote: Decimal
    cum_fees_quote: Decimal

    @property
    def amount_quote(self) -> Decimal:
        return self.amount * self.breakeven_price

    @property
    def global_pnl_quote(self) -> Decimal:
        return self.unrealized_pnl_quote + self.realized_pnl_quote - self.cum_fees_quote


class LPPositionSummary(BaseModel):
    """Summary of an LP position for display in status."""
    connector_name: str
    trading_pair: str
    position_address: str
    side: str  # BUY, SELL, BOTH
    state: str  # IN_RANGE, OUT_OF_RANGE, OPENING, CLOSING, etc.
    # Price info
    current_price: Decimal
    lower_price: Decimal
    upper_price: Decimal
    # Token amounts
    base_amount: Decimal
    quote_amount: Decimal
    base_token: str
    quote_token: str
    # Fees collected
    base_fee: Decimal
    quote_fee: Decimal
    # P&L
    total_value_quote: Decimal
    unrealized_pnl_quote: Decimal
    # Timing
    out_of_range_since: Optional[float] = None

    @property
    def fees_quote(self) -> Decimal:
        """Total fees value in quote token."""
        if self.current_price > 0:
            return self.base_fee * self.current_price + self.quote_fee
        return self.quote_fee

    @property
    def is_in_range(self) -> bool:
        """Check if current price is within position bounds."""
        return self.lower_price <= self.current_price <= self.upper_price
