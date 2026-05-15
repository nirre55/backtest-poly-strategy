import pandas as pd

from .base import BaseStrategy


class PatternStrategy(BaseStrategy):
    """
    Stratégie PATTERN — Suit un motif configurable de bougies avec reset sur win.

    Principe :
    - Un pattern configurable (ex: "RVRVRVR") définit la séquence de paris :
        * R = parier DOWN (bougie rouge)
        * V = parier UP (bougie verte)
    - On suit le pattern position par position.
    - Sur un WIN  → reset au début du pattern (position 0).
    - Sur un LOSS → avancer à la position suivante.
    - Si on atteint la fin du pattern → recommencer depuis le début.

    Le trade est évalué immédiatement sur la bougie courante (pas de décalage i+1).
    """

    name = "pattern"
    description = (
        "PATTERN : suit un motif configurable (ex: RVRVRVR). "
        "R = parier DOWN, V = parier UP. "
        "Reset au début du pattern après chaque win."
    )

    # ------------------------------------------------------------------ #
    # Prepare : rien à calculer                                            #
    # ------------------------------------------------------------------ #

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
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
        pattern_str = str(params.get("pattern", "RVRVRVR")).upper()
        pattern_len = len(pattern_str)

        if pattern_len == 0:
            print("[WARN] Pattern vide, aucun signal généré.")
            return pd.DataFrame()

        # Valider le pattern
        for c in pattern_str:
            if c not in ("R", "V"):
                raise ValueError(
                    f"Pattern invalide : '{pattern_str}'. "
                    f"Caractère '{c}' non reconnu. Utiliser R (rouge/DOWN) ou V (vert/UP)."
                )

        n = len(df)
        if n == 0:
            return pd.DataFrame()

        pos = 0  # position courante dans le pattern
        trade_num = 0
        wins_total = 0
        resets_total = 0
        records = []

        for i in range(n):
            row = df.iloc[i]

            signal_local = row["dt_local"]
            signal_hour = signal_local.hour

            if use_time_filter and signal_hour not in time_filter_hours:
                continue

            next_open = row["open"]
            next_close = row["close"]

            # Bougie neutre → ignorée
            if next_close == next_open:
                continue

            trade_num += 1

            # Direction selon le pattern
            char = pattern_str[pos]
            direction = "UP" if char == "V" else "DOWN"

            # Evaluation du trade
            actual_green = next_close > next_open
            if direction == "UP":
                result = "win" if actual_green else "loss"
            else:
                result = "win" if not actual_green else "loss"

            records.append({
                "signal_time":             row["dt_utc"],
                "entry_time":              row["dt_utc"],
                "direction":               direction,
                "trade_num":               trade_num,
                "pattern_pos":             pos,
                "signal_hour_montreal":    signal_hour,
                "signal_weekday_montreal": signal_local.strftime("%A"),
                "result":                  result,
                "next_candle_open":        next_open,
                "next_candle_close":       next_close,
            })

            # Mise à jour de la position
            if result == "win":
                pos = 0  # reset au début du pattern
                wins_total += 1
                resets_total += 1
            else:
                pos = (pos + 1) % pattern_len  # avancer, boucler si fin

        trades = pd.DataFrame(records)
        if trades.empty:
            print("[WARN] Aucun signal généré.")
        else:
            print(
                f"[INFO] {len(trades)} trades générés — "
                f"pattern '{pattern_str}' (longueur {pattern_len}), "
                f"resets sur win : {resets_total}."
            )
        return trades
