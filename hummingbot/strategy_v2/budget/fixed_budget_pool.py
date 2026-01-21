import threading
import uuid
from decimal import Decimal
from typing import Dict, Optional


class FixedBudgetPool:
    def __init__(self, base_token: str, quote_token: str, base_budget: Decimal, quote_budget: Decimal):
        self._lock = threading.Lock()
        self.base_token = base_token
        self.quote_token = quote_token
        self._balances: Dict[str, Decimal] = {
            base_token: Decimal(str(base_budget)),
            quote_token: Decimal(str(quote_budget)),
        }
        self._locked: Dict[str, Decimal] = {
            base_token: Decimal("0"),
            quote_token: Decimal("0"),
        }
        self._reservations: Dict[str, Dict[str, Decimal]] = {}

    def available(self, token: str) -> Decimal:
        with self._lock:
            return self._balances.get(token, Decimal("0")) - self._locked.get(token, Decimal("0"))

    def reserve(self, requirements: Dict[str, Decimal]) -> Optional[str]:
        filtered = {token: amount for token, amount in requirements.items() if amount > 0}
        if not filtered:
            return None
        with self._lock:
            for token, amount in filtered.items():
                available = self._balances.get(token, Decimal("0")) - self._locked.get(token, Decimal("0"))
                if available < amount:
                    return None
            reservation_id = uuid.uuid4().hex
            for token, amount in filtered.items():
                self._locked[token] = self._locked.get(token, Decimal("0")) + amount
            self._reservations[reservation_id] = filtered
            return reservation_id

    def release(self, reservation_id: Optional[str]):
        if not reservation_id:
            return
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if not reservation:
                return
            for token, amount in reservation.items():
                new_amount = self._locked.get(token, Decimal("0")) - amount
                if new_amount <= 0:
                    self._locked.pop(token, None)
                else:
                    self._locked[token] = new_amount

    def settle(self, reservation_id: Optional[str], returned: Dict[str, Decimal]):
        if not reservation_id:
            return
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if not reservation:
                return
            for token, amount in reservation.items():
                self._locked[token] = self._locked.get(token, Decimal("0")) - amount
                self._balances[token] = self._balances.get(token, Decimal("0")) - amount
            for token, amount in returned.items():
                if amount <= 0:
                    continue
                self._balances[token] = self._balances.get(token, Decimal("0")) + amount

    def apply_swap(self, token_in: str, amount_in: Decimal, token_out: str, amount_out: Decimal) -> bool:
        if amount_in <= 0:
            return False
        with self._lock:
            available = self._balances.get(token_in, Decimal("0")) - self._locked.get(token_in, Decimal("0"))
            if available < amount_in:
                return False
            self._balances[token_in] = self._balances.get(token_in, Decimal("0")) - amount_in
            if amount_out > 0:
                self._balances[token_out] = self._balances.get(token_out, Decimal("0")) + amount_out
            return True

    def snapshot(self) -> Dict[str, Dict[str, Decimal]]:
        with self._lock:
            total = dict(self._balances)
            locked = dict(self._locked)
            available = {token: total.get(token, Decimal("0")) - locked.get(token, Decimal("0")) for token in total}
            return {
                "total": total,
                "locked": locked,
                "available": available,
            }


class FixedBudgetPoolRegistry:
    _lock = threading.Lock()
    _pools: Dict[str, FixedBudgetPool] = {}

    @classmethod
    def get(
        cls,
        key: str,
        base_token: str,
        quote_token: str,
        base_budget: Decimal,
        quote_budget: Decimal,
    ) -> FixedBudgetPool:
        if not key:
            key = "default"
        with cls._lock:
            pool = cls._pools.get(key)
            if pool is None:
                pool = FixedBudgetPool(
                    base_token=base_token,
                    quote_token=quote_token,
                    base_budget=base_budget,
                    quote_budget=quote_budget,
                )
                cls._pools[key] = pool
            return pool
