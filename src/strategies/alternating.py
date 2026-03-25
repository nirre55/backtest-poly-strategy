import pandas as pd

from .base import BaseStrategy


class AlternatingStrategy(BaseStrategy):
    """
    Stratégie ALTERNATING — Alternance de prédiction avec switch de phase.

    Principe :
    - On alterne les prédictions : VERTE, ROUGE, VERTE, ROUGE, ...
    - La phase de départ est déterminée par la première bougie du fichier :
        * première bougie ROUGE  → première prédiction VERTE
        * première bougie VERTE  → première prédiction ROUGE
    - Si on accumule `loss_streak_switch` pertes consécutives (défaut : 2),
      la phase est inversée (switch), puis l'alternance continue depuis la nouvelle phase.

    Le trade est évalué immédiatement sur la bougie courante (pas de décalage i+1) :
    la prédiction est faite avant que la bougie se ferme, et on observe son résultat.
    """

    name = "alternating"
    description = (
        "ALTERNATING : alterne les prédictions VERTE/ROUGE à chaque bougie. "
        "Dès que loss_streak_switch pertes consécutives sont atteintes, "
        "la phase est inversée et l'alternance repart depuis la nouvelle phase."
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
        loss_streak_switch      = int(params.get("loss_streak_switch", 2))
        use_loss_streak_switch  = bool(params.get("use_loss_streak_switch", True))

        n = len(df)
        if n == 0:
            return pd.DataFrame()

        # ── Phase initiale déterminée par la première bougie ──────────
        first_close = df.iloc[0]["close"]
        first_open  = df.iloc[0]["open"]

        if first_close < first_open:
            # première bougie ROUGE → première prédiction VERTE
            phase = 0   # phase 0 = prédire VERTE sur trade impair, ROUGE sur trade pair
        else:
            # première bougie VERTE (ou neutre) → première prédiction ROUGE
            phase = 1   # phase 1 = prédire ROUGE sur trade impair, VERTE sur trade pair

        # La séquence d'alternance selon la phase :
        #   phase 0 : trade 1=VERTE, 2=ROUGE, 3=VERTE, ...  → direction = UP si (trade_num % 2 == 1)
        #   phase 1 : trade 1=ROUGE, 2=VERTE, 3=ROUGE, ...  → direction = DOWN si (trade_num % 2 == 1)
        # Plus simple : on garde juste "prochain signal attendu" et on alterne.

        next_direction = "UP" if phase == 0 else "DOWN"
        consecutive_losses = 0
        trade_num = 0

        records = []

        for i in range(n):
            row = df.iloc[i]

            signal_local = row["dt_local"]
            signal_hour  = signal_local.hour

            if use_time_filter and signal_hour not in time_filter_hours:
                continue

            next_open  = row["open"]
            next_close = row["close"]

            # Bougie neutre → ignorée (ne compte pas dans l'alternance)
            if next_close == next_open:
                continue

            trade_num += 1
            direction = next_direction

            # Évaluation du trade
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
                "consecutive_losses":      consecutive_losses,
                "signal_hour_montreal":    signal_hour,
                "signal_weekday_montreal": signal_local.strftime("%A"),
                "result":                  result,
                "next_candle_open":        next_open,
                "next_candle_close":       next_close,
            })

            # ── Mise à jour de l'état ──────────────────────────────────
            if result == "loss" and use_loss_streak_switch:
                consecutive_losses += 1
                if consecutive_losses >= loss_streak_switch:
                    # Switch de phase : on inverse la prochaine direction
                    next_direction = "DOWN" if next_direction == "UP" else "UP"
                    consecutive_losses = 0
                else:
                    next_direction = "DOWN" if next_direction == "UP" else "UP"
            else:
                consecutive_losses = 0
                next_direction = "DOWN" if next_direction == "UP" else "UP"

        trades = pd.DataFrame(records)
        if trades.empty:
            print("[WARN] Aucun signal généré.")
        else:
            switches = len([r for r in records
                            if r["consecutive_losses"] + 1 >= loss_streak_switch
                            and r["result"] == "loss"])
            print(f"[INFO] {len(trades)} trades générés "
                  f"(switchs de phase détectés : ~{switches}).")
        return trades
