import numpy as np
import pandas as pd

from .base import BaseStrategy


class SniperStrategy(BaseStrategy):
    """
    Stratégie SNIPER — Rejet Extrême.

    Vise un winrate maximum (~68%) en étant très sélective.
    Ne génère un signal que lorsque le volume est 4× la moyenne ET
    la mèche de rejet est 3× le corps.

    Signal UP  : bougie rouge + volume > vma20×4.0 + mèche basse > corps×3.0
    Signal DOWN: bougie verte + volume > vma20×4.0 + mèche haute > corps×3.0

    Le trade est pris sur la bougie i+1 après le signal sur la bougie i.
    """

    name = "sniper"
    description = (
        "SNIPER — Rejet Extrême : volume flash (>4× MA20) + rejet massif (mèche >3× corps). "
        "Très sélectif, vise un winrate élevé avec peu de signaux."
    )

    # ------------------------------------------------------------------ #
    # Indicateurs                                                          #
    # ------------------------------------------------------------------ #

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close  = df["close"].to_numpy(dtype=float)
        open_  = df["open"].to_numpy(dtype=float)
        high   = df["high"].to_numpy(dtype=float)
        low    = df["low"].to_numpy(dtype=float)
        volume = df["volume"].to_numpy(dtype=float)

        body      = np.abs(close - open_)
        # Mèche basse sur bougie rouge : open - low
        # Mèche haute sur bougie verte : high - close
        wick_low  = open_ - low          # pertinent uniquement si bougie rouge
        wick_high = high  - close        # pertinent uniquement si bougie verte

        df["body"]         = body
        df["wick_low"]     = wick_low
        df["wick_high"]    = wick_high
        df["range"]        = high - low
        df["valid_candle"] = (high - low) > 0

        # Volume MA20
        df["vma20"] = pd.Series(volume).rolling(20, min_periods=1).mean().to_numpy()

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
        vol_mult   = float(params.get("vol_mult",   4.0))
        wick_mult  = float(params.get("wick_mult",  3.0))

        n       = len(df)
        records = []

        for i in range(n - 1):
            row = df.iloc[i]

            if not row["valid_candle"]:
                continue

            signal_local = row["dt_local"]
            signal_hour  = signal_local.hour

            if use_time_filter and signal_hour not in time_filter_hours:
                continue

            close_ = row["close"]
            open__ = row["open"]
            body   = row["body"]
            volume = row["volume"]
            vma20  = row["vma20"]
            is_red   = close_ < open__
            is_green = close_ > open__

            vol_flash = volume > vma20 * vol_mult

            # Signal UP : bougie rouge + volume flash + mèche basse massive
            cond_up = (
                is_red
                and vol_flash
                and row["wick_low"] > body * wick_mult
            )

            # Signal DOWN : bougie verte + volume flash + mèche haute massive
            cond_down = (
                is_green
                and vol_flash
                and row["wick_high"] > body * wick_mult
            )

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
                "body":                    round(float(body), 6),
                "wick_low":                round(float(row["wick_low"]), 6),
                "wick_high":               round(float(row["wick_high"]), 6),
                "volume":                  round(float(volume), 2),
                "vma20":                   round(float(vma20), 2),
                "vol_ratio":               round(float(volume / vma20), 2) if vma20 > 0 else 0.0,
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
