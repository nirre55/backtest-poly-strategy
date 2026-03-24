from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """
    Contrat que toute stratégie doit respecter.

    Le moteur de backtest (backtest.py) appelle :
      1. strategy.prepare(df)         → ajoute les indicateurs au DataFrame
      2. strategy.generate_signals()  → retourne un DataFrame de trades signalés

    Colonnes obligatoires dans le DataFrame retourné par generate_signals() :
      - signal_time            : Timestamp UTC de la bougie signal (bougie i)
      - entry_time             : Timestamp UTC de la bougie trade (bougie i+1)
      - direction              : "UP" ou "DOWN"
      - result                 : "win" ou "loss"
      - signal_hour_montreal   : heure locale (int) du signal
      - signal_weekday_montreal: jour de la semaine (str) du signal
      - next_candle_open       : open de la bougie i+1
      - next_candle_close      : close de la bougie i+1
    """

    name: str = "base"
    description: str = ""

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les indicateurs et colonnes nécessaires à la stratégie.
        Appelé une seule fois après le chargement des données.
        Par défaut : ne fait rien (les stratégies simples peuvent surcharger).
        """
        return df

    @abstractmethod
    def generate_signals(
        self,
        df: pd.DataFrame,
        timezone: str,
        use_time_filter: bool,
        time_filter_hours: set,
        params: dict,
    ) -> pd.DataFrame:
        """
        Génère les signaux sur le DataFrame préparé.

        Args:
            df               : DataFrame avec les colonnes calculées par prepare()
            timezone         : ex. "America/Montreal"
            use_time_filter  : True = Version B (filtre horaire actif)
            time_filter_hours: ensemble des heures autorisées si use_time_filter
            params           : paramètres libres passés par la CLI ou le YAML
                               (ex. rsi_up, streak_min, body_ratio_min…)

        Returns:
            DataFrame des trades signalés (voir colonnes obligatoires ci-dessus)
        """
        ...
