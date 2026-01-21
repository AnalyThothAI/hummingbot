import asyncio
import logging
import threading
import uuid
from collections import defaultdict
from decimal import Decimal
from typing import Dict, Optional, Tuple

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.logger import HummingbotLogger


class BudgetCoordinator:
    _logger: Optional[HummingbotLogger] = None

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    def __init__(self):
        self._lock = threading.Lock()
        self._locked: Dict[Tuple[str, str], Decimal] = defaultdict(lambda: Decimal("0"))
        self._reservations: Dict[str, Dict[Tuple[str, str], Decimal]] = {}
        self.action_lock = asyncio.Lock()

    def reserve(
        self,
        connector_name: str,
        connector: ConnectorBase,
        requirements: Dict[str, Decimal],
        native_token: Optional[str] = None,
        min_native_balance: Decimal = Decimal("0"),
    ) -> Optional[str]:
        filtered_requirements = {token: amount for token, amount in requirements.items() if amount > 0}
        if not filtered_requirements:
            return None

        with self._lock:
            for token, amount in filtered_requirements.items():
                available = self._get_available_balance(connector, token)
                available -= self._locked[(connector_name, token)]
                if native_token and token == native_token and min_native_balance > 0:
                    available -= min_native_balance
                if available < amount:
                    return None

            reservation_id = uuid.uuid4().hex
            reservation = {}
            for token, amount in filtered_requirements.items():
                key = (connector_name, token)
                self._locked[key] += amount
                reservation[key] = amount
            self._reservations[reservation_id] = reservation
            return reservation_id

    def release(self, reservation_id: Optional[str]):
        if not reservation_id:
            return
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if not reservation:
                return
            for key, amount in reservation.items():
                new_amount = self._locked[key] - amount
                if new_amount <= 0:
                    self._locked.pop(key, None)
                else:
                    self._locked[key] = new_amount

    @staticmethod
    def _get_available_balance(connector: ConnectorBase, token: str) -> Decimal:
        try:
            balance = connector.get_available_balance(token)
        except Exception:
            try:
                balance = connector.get_balance(token)
            except Exception:
                balance = Decimal("0")
        if balance is None:
            return Decimal("0")
        return Decimal(str(balance))


class BudgetCoordinatorRegistry:
    _lock = threading.Lock()
    _coordinators: Dict[str, BudgetCoordinator] = {}

    @classmethod
    def get(cls, key: str) -> BudgetCoordinator:
        if not key:
            key = "default"
        with cls._lock:
            coordinator = cls._coordinators.get(key)
            if coordinator is None:
                coordinator = BudgetCoordinator()
                cls._coordinators[key] = coordinator
            return coordinator
