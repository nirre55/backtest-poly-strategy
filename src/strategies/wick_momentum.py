import numpy as np
import pandas as pd

from .base import BaseStrategy


class WickMomentumStrategy(BaseStrategy):
    """
    Stratégie Wick-Momentum : deux chemins pour chaque direction.

    Signal UP :
      - Rejet   : bougie rouge + volume > vma20×1.5 + mèche basse > body×2
      - Momentum: bougie verte + volume > vma20×2.5 + body > range×0.8

    Signal DOWN :
      - Rejet   : bougie verte + volume > vma20×1.5 + mèche haute > body×2
      - Momentum: bougie rouge + volume > vma20×2.5 + body > range×0.8

    Le trade est pris sur la bougie i+1 après le signal sur la bougie i.
    """

    name = "wick_momentum"
    description = (
        "Wick-Momentum : rejet de mèche (volume fort + mèche dominante) "
        "OU momentum (volume extrême + corps plein). "
        "Signal UP sur rejet bas ou momentum haussier, DOWN sur rejet haut ou momentum baissier."
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
        range_    = high - low

        df["body"]         = body
        df["wick_low"]     = wick_low
        df["wick_high"]    = wick_high
        df["range"]        = range_
        df["valid_candle"] = range_ > 0

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
        # Paramètres Rejet
        rej_vol_mult   = float(params.get("rej_vol_mult",   1.5))
        rej_wick_mult  = float(params.get("rej_wick_mult",  2.0))
        # Paramètres Momentum
        mom_vol_mult   = float(params.get("mom_vol_mult",   2.5))
        mom_body_ratio = float(params.get("mom_body_ratio", 0.8))

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

            close_  = row["close"]
            open__  = row["open"]
            body    = row["body"]
            range_  = row["range"]
            volume  = row["volume"]
            vma20   = row["vma20"]
            is_red  = close_ < open__
            is_green = close_ > open__

            # ── Signal UP ─────────────────────────────────────────────────
            # Rejet bas : bougie rouge + volume fort + mèche basse immense
            up_rejet = (
                is_red
                and volume > vma20 * rej_vol_mult
                and row["wick_low"] > body * rej_wick_mult
            )
            # Momentum haussier : bougie verte + volume extrême + corps plein
            up_momentum = (
                is_green
                and volume > vma20 * mom_vol_mult
                and body > range_ * mom_body_ratio
            )
            cond_up = up_rejet or up_momentum

            # ── Signal DOWN ───────────────────────────────────────────────
            # Rejet haut : bougie verte + volume fort + mèche haute immense
            down_rejet = (
                is_green
                and volume > vma20 * rej_vol_mult
                and row["wick_high"] > body * rej_wick_mult
            )
            # Momentum baissier : bougie rouge + volume extrême + corps plein
            down_momentum = (
                is_red
                and volume > vma20 * mom_vol_mult
                and body > range_ * mom_body_ratio
            )
            cond_down = down_rejet or down_momentum

            # Un signal UP et DOWN simultané ne peut pas arriver
            # (up_rejet ↔ bougie rouge, down_rejet ↔ bougie verte, etc.)
            # mais on garde la priorité UP en cas de conflit logique résiduel
            if not cond_up and not cond_down:
                continue

            if cond_up:
                direction  = "UP"
                signal_type = "rejet" if up_rejet else "momentum"
            else:
                direction  = "DOWN"
                signal_type = "rejet" if down_rejet else "momentum"

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
                "signal_type":             signal_type,
                "body":                    round(float(body), 6),
                "wick_low":                round(float(row["wick_low"]), 6),
                "wick_high":               round(float(row["wick_high"]), 6),
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
            n_rejet    = (trades["signal_type"] == "rejet").sum()
            n_momentum = (trades["signal_type"] == "momentum").sum()
            print(f"[INFO] {len(trades)} signaux générés "
                  f"(rejet: {n_rejet}, momentum: {n_momentum}).")
        return trades
