import numpy as np
import pandas as pd

from .base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    Stratégie MOMENTUM — Suivi de la force du marché.

    Génère beaucoup de trades avec un winrate correct (~54%).
    Peut enchaîner jusqu'à 24 victoires d'affilée en tendance forte.

    Signal UP  : bougie verte + variation > +0.2%  → parie que la suivante est verte
    Signal DOWN: bougie rouge + variation < -0.2%  → parie que la suivante est rouge

    Le trade est pris sur la bougie i+1 après le signal sur la bougie i.
    """

    name = "momentum"
    description = (
        "MOMENTUM : suit la force du marché. "
        "Signal UP si bougie verte avec variation > +0.2%, "
        "DOWN si bougie rouge avec variation < -0.2%. "
        "Génère beaucoup de trades, winrate ~54%."
    )

    # ------------------------------------------------------------------ #
    # Indicateurs (minimaliste — seulement open/close nécessaires)        #
    # ------------------------------------------------------------------ #

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].to_numpy(dtype=float)
        open_ = df["open"].to_numpy(dtype=float)

        # Variation en % : (close - open) / open
        with np.errstate(divide="ignore", invalid="ignore"):
            variation = np.where(open_ != 0, (close - open_) / open_ * 100.0, 0.0)

        df["variation_pct"] = variation
        return df

    # ------------------------------------------------------------------ #
    # Génération des signaux                                               #
    # ------------------------------------------------------------------ #

    def generate_signals(
        self,
        df: pd.DataFrame,
        timezone: str,
        use_time_filter: bool,
        time_filter_hours: set,
        params: dict,
    ) -> pd.DataFrame:
        # Seuil de variation (en %) — défaut : 0.2%
        threshold_pct = float(params.get("threshold_pct", 0.2))

        n       = len(df)
        records = []

        for i in range(n - 1):
            row = df.iloc[i]

            variation = row["variation_pct"]

            signal_local = row["dt_local"]
            signal_hour  = signal_local.hour

            if use_time_filter and signal_hour not in time_filter_hours:
                continue

            close_ = row["close"]
            open__ = row["open"]

            cond_up   = close_ > open__ and variation >  threshold_pct
            cond_down = close_ < open__ and variation < -threshold_pct

            if not cond_up and not cond_down:
                continue

            direction = "UP" if cond_up else "DOWN"

            next_row   = df.iloc[i + 1]
            next_open  = next_row["open"]
            next_close = next_row["close"]

            if next_close == next_open:
                continue

            result = (
                ("win" if next_close > next_open else "loss") if direction == "UP"
                else ("win" if next_close < next_open else "loss")
            )

            records.append({
                "signal_time":             row["dt_utc"],
                "entry_time":              next_row["dt_utc"],
                "direction":               direction,
                "variation_pct":           round(float(variation), 4),
                "signal_hour_montreal":    signal_hour,
                "signal_weekday_montreal": signal_local.strftime("%A"),
                "result":                  result,
                "next_candle_open":        next_open,
                "next_candle_close":       next_close,
            })

        trades = pd.DataFrame(records)
        if trades.empty:
            print("[WARN] Aucun signal généré.")
        else:
            print(f"[INFO] {len(trades)} signaux générés.")
        return trades
