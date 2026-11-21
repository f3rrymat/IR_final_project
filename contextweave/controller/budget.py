"""Per-query budget accounting in cost units (1 unit = median full BM25 query time)."""
from __future__ import annotations


class Budget:
    def __init__(self, total: float, costs: dict[str, float]):
        self.total, self.costs, self.spent = float(total), costs, 0.0

    def cost(self, method: str) -> float:
        return float(self.costs.get(method, 1.0))

    def affordable(self, method: str) -> bool:
        return self.spent + self.cost(method) <= self.total + 1e-9

    def charge(self, method: str) -> float:
        c = self.cost(method)
        self.spent += c
        return c

    @property
    def left(self) -> float:
        return self.total - self.spent
