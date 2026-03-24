import numpy as np
import pandas as pd

from .base import BaseStrategy


class StreakRSIStrategy(BaseStrategy):
    """
    Stratégie originale : streak de bougies consécutives + RSI7 + ATR14 + body ratio.

    Signal UP  : streak rouge ≥ streak_min, RSI7 ≤ rsi_up,  range ≥ mult×ATR14, body_ratio ≥ min
    Signal DOWN: streak vert  ≥ streak_min, RSI7 ≥ rsi_down, range ≥ mult×ATR14, body_ratio ≥ min

    Le trade est pris sur la bougie i+1 après le signal sur la bougie i.
    """

    name = "streak_rsi"
    description = (
        "Streak de bougies consécutives (≥3) + RSI7 + ATR14 + body ratio. "
        "Signal UP sur streak rouge, signal DOWN sur streak vert."
    )

    # ------------------------------------------------------------------ #
    # Indicateurs                                                          #
    # ------------------------------------------------------------------ #

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._compute_indicators(df)
        df = self._compute_streaks(df)
        return df

    @staticmethod
    def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        high  = df["high"].to_numpy(dtype=float)
        low   = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        open_ = df["open"].to_numpy(dtype=float)

        df["range"] = high - low
        df["valid_candle"] = df["range"] > 0
        df["body"] = np.abs(close - open_)
        df["body_ratio"] = np.where(df["range"] > 0, df["body"] / df["range"], 0.0)
        df["close_pos"] = np.where(df["range"] > 0, (close - low) / df["range"], 0.5)

        # ATR14 (Wilder)
        n = len(df)
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low  - np.roll(close, 1)),
            ),
        )
        tr[0] = high[0] - low[0]

        atr = np.zeros(n)
        period = 14
        if n >= period:
            atr[period - 1] = np.mean(tr[:period])
            for i in range(period, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        df["atr14"] = atr
        df["atr14"] = df["atr14"].replace(0, np.nan)

        # RSI7
        delta    = pd.Series(close).diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        p        = 7
        avg_gain = gain.ewm(com=p - 1, min_periods=p, adjust=False).mean()
        avg_loss = loss.ewm(com=p - 1, min_periods=p, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi7"] = 100 - (100 / (1 + rs))

        return df

    @staticmethod
    def _compute_streaks(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].to_numpy(dtype=float)
        open_ = df["open"].to_numpy(dtype=float)
        n     = len(df)

        color = np.zeros(n, dtype=int)
        color[close > open_] =  1
        color[close < open_] = -1

        streak_green = np.zeros(n, dtype=int)
        streak_red   = np.zeros(n, dtype=int)

        for i in range(n):
            if color[i] == 1:
                streak_green[i] = streak_green[i - 1] + 1 if i > 0 else 1
                streak_red[i]   = 0
            elif color[i] == -1:
                streak_red[i]   = streak_red[i - 1] + 1 if i > 0 else 1
                streak_green[i] = 0
            else:
                streak_green[i] = 0
                streak_red[i]   = 0

        df["color"]        = color
        df["streak_green"] = streak_green
        df["streak_red"]   = streak_red
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
        rsi_up         = float(params.get("rsi_up",         35.0))
        rsi_down       = float(params.get("rsi_down",       65.0))
        body_ratio_min = float(params.get("body_ratio_min", 0.60))
        range_atr_mult = float(params.get("range_atr_mult", 1.0))
        streak_min     = int(params.get("streak_min",       3))
        use_streak     = bool(params.get("use_streak",      True))
        use_rsi        = bool(params.get("use_rsi",         True))
        use_range      = bool(params.get("use_range",       True))
        use_body_ratio = bool(params.get("use_body_ratio",  True))

        n       = len(df)
        records = []

        for i in range(n - 1):
            row = df.iloc[i]

            if not row["valid_candle"]:
                continue
            if np.isnan(row["atr14"]) or np.isnan(row["rsi7"]):
                continue

            signal_local = row["dt_local"]
            signal_hour  = signal_local.hour

            if use_time_filter and signal_hour not in time_filter_hours:
                continue

            # Signal UP (bougie rouge répétée)
            cond_up = True
            if use_streak:
                cond_up = cond_up and (row["streak_red"] >= streak_min)
            if use_rsi:
                cond_up = cond_up and (row["rsi7"] <= rsi_up)
            if use_range:
                cond_up = cond_up and (row["range"] >= range_atr_mult * row["atr14"])
            if use_body_ratio:
                cond_up = cond_up and (row["body_ratio"] >= body_ratio_min)

            # Signal DOWN (bougie verte répétée)
            cond_down = True
            if use_streak:
                cond_down = cond_down and (row["streak_green"] >= streak_min)
            if use_rsi:
                cond_down = cond_down and (row["rsi7"] >= rsi_down)
            if use_range:
                cond_down = cond_down and (row["range"] >= range_atr_mult * row["atr14"])
            if use_body_ratio:
                cond_down = cond_down and (row["body_ratio"] >= body_ratio_min)

            if not cond_up and not cond_down:
                continue

            direction = "UP" if cond_up else "DOWN"

            next_row    = df.iloc[i + 1]
            next_open   = next_row["open"]
            next_close  = next_row["close"]

            if next_close == next_open:
                continue

            result = (
                ("win" if next_close > next_open else "loss") if direction == "UP"
                else ("win" if next_close < next_open else "loss")
            )

            records.append({
                "signal_time":            row["dt_utc"],
                "entry_time":             next_row["dt_utc"],
                "direction":              direction,
                "streak":                 int(row["streak_red"] if direction == "UP" else row["streak_green"]),
                "RSI7":                   round(float(row["rsi7"]), 4),
                "ATR14":                  round(float(row["atr14"]), 6),
                "range":                  round(float(row["range"]), 6),
                "body_ratio":             round(float(row["body_ratio"]), 4),
                "signal_hour_montreal":   signal_hour,
                "signal_weekday_montreal": signal_local.strftime("%A"),
                "result":                 result,
                "next_candle_open":       next_open,
                "next_candle_close":      next_close,
            })

        trades = pd.DataFrame(records)
        if trades.empty:
            print("[WARN] Aucun signal généré.")
        else:
            print(f"[INFO] {len(trades)} signaux générés.")
        return trades
