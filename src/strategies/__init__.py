"""
Registre des stratégies disponibles.

Pour ajouter une nouvelle stratégie :
  1. Créer src/strategies/ma_strategie.py avec une classe héritant de BaseStrategy
  2. L'importer ici et l'ajouter dans REGISTRY

Usage dans la CLI :
  python src/backtest.py --input data/... --strategy streak_rsi
  python src/backtest.py --input data/... --strategy ma_strategie
"""

from .base import BaseStrategy
from .streak_rsi import StreakRSIStrategy

REGISTRY: dict[str, type[BaseStrategy]] = {
    "streak_rsi": StreakRSIStrategy,
}


def get_strategy(name: str) -> BaseStrategy:
    if name not in REGISTRY:
        available = ", ".join(REGISTRY.keys())
        raise ValueError(f"Stratégie inconnue : '{name}'. Disponibles : {available}")
    return REGISTRY[name]()


def list_strategies() -> list[str]:
    return list(REGISTRY.keys())
