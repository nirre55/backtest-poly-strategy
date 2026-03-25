import numpy as np
import pandas as pd

from .base import BaseStrategy


class WickVolumeReboundStrategy(BaseStrategy):
    """
    Stratégie Wick-Volume Rebound.

    Signal UP  : bougie rouge + RSI7 < 30 + mèche basse > body×1.5 + volume > vma20×1.25
    Signal DOWN: bougie verte + RSI7 > 70 + mèche haute > body×1.5 + volume > vma20×1.25

    Le trade est pris sur la bougie i+1 après le signal sur la bougie i.
    """

    name = "wick_volume_rebound"
    description = (
        "Wick-Volume Rebound : rejet de mèche (low/high) + RSI7 survendu/suracheté "
        "+ confirmation volume > MA20. Signal UP sur bougie rouge, DOWN sur bougie verte."
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
        wick_low  = np.minimum(open_, close) - low
        wick_high = high - np.maximum(open_, close)

        df["body"]      = body
        df["wick_low"]  = wick_low
        df["wick_high"] = wick_high
        df["range"]     = high - low
        df["valid_candle"] = df["range"] > 0

        # Volume MA20
        df["vma20"] = pd.Series(volume).rolling(20, min_periods=1).mean().to_numpy()

        # RSI7 (Wilder / EWM)
        delta    = pd.Series(close).diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        p        = 7
        avg_gain = gain.ewm(com=p - 1, min_periods=p, adjust=False).mean()
        avg_loss = loss.ewm(com=p - 1, min_periods=p, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi7"] = 100 - (100 / (1 + rs))

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
        rsi_oversold    = float(params.get("rsi_oversold",    30.0))
        rsi_overbought  = float(params.get("rsi_overbought",  70.0))
        wick_body_mult  = float(params.get("wick_body_mult",  1.5))
        vol_ma_mult     = float(params.get("vol_ma_mult",     1.25))

        n       = len(df)
        records = []

        for i in range(n - 1):
            row = df.iloc[i]

            if not row["valid_candle"]:
                continue
            if np.isnan(row["rsi7"]):
                continue

            signal_local = row["dt_local"]
            signal_hour  = signal_local.hour

            if use_time_filter and signal_hour not in time_filter_hours:
                continue

            close_ = row["close"]
            open__ = row["open"]
            body   = row["body"]
            vma20  = row["vma20"]
            volume = row["volume"]
            rsi7   = row["rsi7"]

            # Signal UP : bougie rouge + RSI survendu + mèche basse dominante + volume fort
            cond_up = (
                close_ < open__
                and rsi7 < rsi_oversold
                and row["wick_low"] > body * wick_body_mult
                and volume > vma20 * vol_ma_mult
            )

            # Signal DOWN : bougie verte + RSI suracheté + mèche haute dominante + volume fort
            cond_down = (
                close_ > open__
                and rsi7 > rsi_overbought
                and row["wick_high"] > body * wick_body_mult
                and volume > vma20 * vol_ma_mult
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
                "RSI7":                    round(float(rsi7), 4),
                "wick_low":                round(float(row["wick_low"]), 6),
                "wick_high":               round(float(row["wick_high"]), 6),
                "body":                    round(float(body), 6),
                "volume":                  round(float(volume), 2),
                "vma20":                   round(float(vma20), 2),
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
