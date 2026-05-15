import numpy as np
import pandas as pd

from .base import BaseStrategy


class HeikinAshiStrategy(BaseStrategy):
    """
    Stratégie HEIKIN ASHI — Suivi de tendance par bougies synthétiques.

    Principe :
    - Calcule les bougies Heikin Ashi (HA) à partir des OHLC réels.
    - Si la bougie HA courante (i) est verte → signal UP  → mise sur la bougie i+1.
    - Si la bougie HA courante (i) est rouge → signal DOWN → mise sur la bougie i+1.
    - Filtre optionnel : ignorer les bougies HA dont le corps est trop petit
      (body_ratio < body_ratio_min) pour éviter les signaux en indécision.

    Formules HA :
        HA_close = (open + high + low + close) / 4
        HA_open  = (prev_HA_open + prev_HA_close) / 2
        HA_high  = max(high, HA_open, HA_close)
        HA_low   = min(low,  HA_open, HA_close)

    Le trade est pris sur la bougie réelle i+1 (open/close réels).
    """

    name = "heikin_ashi"
    description = (
        "HEIKIN ASHI : suivi de tendance via bougies synthétiques. "
        "Signal UP si bougie HA verte, DOWN si rouge. "
        "Filtre de corps configurable (body_ratio_min)."
    )

    # ------------------------------------------------------------------ #
    # Indicateurs                                                          #
    # ------------------------------------------------------------------ #

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)

        open_  = df["open"].to_numpy(dtype=float)
        high   = df["high"].to_numpy(dtype=float)
        low    = df["low"].to_numpy(dtype=float)
        close  = df["close"].to_numpy(dtype=float)

        ha_close = (open_ + high + low + close) / 4.0

        ha_open = np.empty(n, dtype=float)
        ha_open[0] = (open_[0] + close[0]) / 2.0
        for i in range(1, n):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

        ha_high = np.maximum(high, np.maximum(ha_open, ha_close))
        ha_low  = np.minimum(low,  np.minimum(ha_open, ha_close))

        ha_body  = np.abs(ha_close - ha_open)
        ha_range = ha_high - ha_low

        with np.errstate(divide="ignore", invalid="ignore"):
            ha_body_ratio = np.where(ha_range > 0, ha_body / ha_range, 0.0)

        df["ha_open"]       = ha_open
        df["ha_close"]      = ha_close
        df["ha_body_ratio"] = ha_body_ratio
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
        body_ratio_min = float(params.get("body_ratio_min", 0.3))

        n       = len(df)
        records = []

        for i in range(n - 1):
            row = df.iloc[i]

            signal_local = row["dt_local"]
            signal_hour  = signal_local.hour

            if use_time_filter and signal_hour not in time_filter_hours:
                continue

            ha_open_val  = row["ha_open"]
            ha_close_val = row["ha_close"]
            body_ratio   = row["ha_body_ratio"]

            # Doji HA → ignorer
            if ha_close_val == ha_open_val:
                continue

            # Filtre corps minimum
            if body_ratio < body_ratio_min:
                continue

            direction = "UP" if ha_close_val > ha_open_val else "DOWN"

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
                "ha_body_ratio":           round(float(body_ratio), 4),
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
