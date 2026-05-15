import numpy as np
import pandas as pd

from .base import BaseStrategy


class DonchZscoreStrategy(BaseStrategy):
    """
    Stratégie DONCH_ZSCORE — Contrarian Donchian + Body Sum + Z-Score.

    Source : feature_threshold_rule...c4add18818 (promising_prediction_strategies.md, #12)
    Score  : 66.35 | Backtest : 61.57% / 4359 trades | Test : 62.37% / 287 trades
    Direction : BOTH

    Signal UP :
        donch_low12 <= donch_thr          (close proche du plus bas 12 bougies)
        AND body_sum6 <= -body_sum_thr    (pression baissière récente)
        AND close_z24 <= -z_thr           (prix statistiquement très bas vs MA24)
        → rebond haussier anticipé (contrarian)

    Signal DOWN :
        donch_high12 >= -donch_thr        (close proche du plus haut 12 bougies)
        AND body_sum6 >= +body_sum_thr    (pression haussière récente)
        AND close_z24 >= +z_thr           (prix statistiquement très haut vs MA24)
        → retournement baissier anticipé (contrarian)

    Features :
        donch_low12  = (close - min(low,  12)) / close  ≥ 0
        donch_high12 = (close - max(high, 12)) / close  ≤ 0
        body_sum6    = sum((close - open) / close, 6 bougies)
        close_z24    = (close - mean(close, 24)) / std(close, 24)
    """

    name = "donch_zscore"
    description = (
        "DONCH_ZSCORE : contrarian Donchian + Body Sum + Z-Score (c4add18818). "
        "UP si prix proche bas12 + pression baissière + z24 <= -2.1. "
        "DOWN si prix proche haut12 + pression haussière + z24 >= +2.1."
    )

    # ------------------------------------------------------------------ #
    # Indicateurs                                                          #
    # ------------------------------------------------------------------ #

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].to_numpy(dtype=float)
        open_ = df["open"].to_numpy(dtype=float)
        low   = df["low"].to_numpy(dtype=float)
        high  = df["high"].to_numpy(dtype=float)

        # Donchian distance
        low_min12  = pd.Series(low).rolling(12).min().to_numpy()
        high_max12 = pd.Series(high).rolling(12).max().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            donch_low12  = np.where(close != 0, (close - low_min12)  / close, np.nan)
            donch_high12 = np.where(close != 0, (close - high_max12) / close, np.nan)

        # Body sum sur 6 bougies
        body_ratio = np.where(close != 0, (close - open_) / close, 0.0)
        body_sum6  = pd.Series(body_ratio).rolling(6).sum().to_numpy()

        # Z-score du close sur 24 bougies
        close_s   = pd.Series(close)
        mean_24   = close_s.rolling(24).mean().to_numpy()
        std_24    = close_s.rolling(24).std(ddof=0).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            close_z24 = np.where(std_24 > 0, (close - mean_24) / std_24, 0.0)

        df["donch_low12"]  = donch_low12
        df["donch_high12"] = donch_high12
        df["body_sum6"]    = body_sum6
        df["close_z24"]    = close_z24
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
        donch_thr    = float(params.get("donch_thr",    0.00035))
        body_sum_thr = float(params.get("body_sum_thr", 0.0045))
        z_thr        = float(params.get("z_thr",        2.1))
        horizon      = int(params.get("horizon",        1))

        n       = len(df)
        records = []

        for i in range(n - 1):
            row = df.iloc[i]

            signal_local = row["dt_local"]
            signal_hour  = signal_local.hour

            if use_time_filter and signal_hour not in time_filter_hours:
                continue

            dl = row["donch_low12"]
            dh = row["donch_high12"]
            bs = row["body_sum6"]
            cz = row["close_z24"]

            if pd.isna(dl) or pd.isna(dh) or pd.isna(bs) or pd.isna(cz):
                continue

            cond_up   = dl <= donch_thr and bs <= -body_sum_thr and cz <= -z_thr
            cond_down = dh >= -donch_thr and bs >= body_sum_thr and cz >= z_thr

            if not cond_up and not cond_down:
                continue

            direction = "UP" if cond_up else "DOWN"

            # Bet sur les `horizon` prochaines bougies dans la même direction
            for offset in range(1, horizon + 1):
                if i + offset >= n:
                    break

                target_row   = df.iloc[i + offset]
                next_open    = target_row["open"]
                next_close   = target_row["close"]

                if next_close == next_open:
                    continue

                result = (
                    ("win" if next_close > next_open else "loss") if direction == "UP"
                    else ("win" if next_close < next_open else "loss")
                )

                records.append({
                    "signal_time":             row["dt_utc"],
                    "entry_time":              target_row["dt_utc"],
                    "direction":               direction,
                    "horizon_offset":          offset,
                    "donch_low12":             round(float(dl), 8),
                    "donch_high12":            round(float(dh), 8),
                    "body_sum6":               round(float(bs), 6),
                    "close_z24":               round(float(cz), 4),
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
