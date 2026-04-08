#!/usr/bin/env python3
"""
Backtest BTCUSDT M5 — moteur générique avec multiple money managements.

Usage:
    python src/backtest.py --input data/fichier.csv
    python src/backtest.py --input data/fichier.csv --strategy streak_rsi
"""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from strategies import get_strategy, list_strategies

warnings.filterwarnings("ignore")

# ============================================================
# CONSTANTES (valeurs par défaut si pas de config YAML)
# ============================================================
DEFAULT_INITIAL_CAPITAL = 100.0

DEFAULT_PAYOUTS = {
    "A": {"win": 0.9,  "loss": -1.0},
    "B": {"win": 0.95, "loss": -1.0},
    "C": {"win": 1.0,  "loss": -1.0},
}

MONEY_MANAGEMENTS = [
    "MM1", "MM2", "MM3", "MM4", "MM5",
    "MM6", "MM7", "MM8", "MM9", "MM10", "MM11",
]

CAPITAL_TARGETS = [500, 1000, 5000, 10000, 50000]

MONTREAL_TZ = "America/Montreal"

# Clé YAML → code MM (pour retrouver la config de chaque MM)
MM_KEY_MAP = {
    "MM1":  "MM1_flat_fixed",
    "MM2":  "MM2_fixed_1pct",
    "MM3":  "MM3_fixed_5pct",
    "MM4":  "MM4_martingale_classic",
    "MM5":  "MM5_martingale_linear",
    "MM6":  "MM6_martingale_limited",
    "MM7":  "MM7_anti_martingale",
    "MM8":  "MM8_reduction_after_losses",
    "MM9":  "MM9_pause_after_losses",
    "MM10": "MM10_combined",
    "MM11": "MM11_alternating",
}


# ============================================================
# CHARGEMENT DE LA CONFIG YAML
# ============================================================

def load_config(config_path: str | None) -> dict:
    """
    Charge le fichier YAML de configuration.
    Retourne un dict vide si aucun fichier fourni (les fonctions utiliseront
    leurs valeurs par défaut).
    """
    if config_path is None:
        # Chercher le fichier par défaut à côté du projet
        default = Path(__file__).parent.parent / "config" / "money_management.yaml"
        if default.exists():
            config_path = str(default)
        else:
            print("[INFO] Aucun fichier de config YAML trouvé — valeurs par défaut utilisées.")
            return {}

    path = Path(config_path)
    if not path.exists():
        sys.exit(f"[ERREUR] Fichier de config introuvable : {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"[INFO] Config chargée : {path.resolve()}")
    return cfg or {}


def get_initial_capital(cfg: dict) -> float:
    return float(cfg.get("general", {}).get("starting_capital", DEFAULT_INITIAL_CAPITAL))


def get_payouts(cfg: dict) -> dict:
    """Retourne les payouts activés depuis la config, sinon les payouts par défaut."""
    raw = cfg.get("general", {}).get("payouts", {})
    if not raw:
        return DEFAULT_PAYOUTS

    result = {}
    for key, vals in raw.items():
        if vals.get("enabled", True):
            result[key] = {
                "win":  float(vals.get("win_payout",  0.9)),
                "loss": float(vals.get("loss_payout", -1.0)),
            }
    return result if result else DEFAULT_PAYOUTS


def get_enabled_mms(cfg: dict) -> list[str]:
    """Retourne la liste des MM activés dans la config."""
    strategies = cfg.get("strategies", {})
    if not strategies:
        return MONEY_MANAGEMENTS

    enabled = []
    for mm_code, yaml_key in MM_KEY_MAP.items():
        mm_cfg = strategies.get(yaml_key, {})
        if mm_cfg.get("enabled", True):
            enabled.append(mm_code)
    return enabled if enabled else MONEY_MANAGEMENTS


def get_mm_cfg(cfg: dict, mm_name: str) -> dict:
    """
    Retourne le bloc de config d'un MM spécifique.
    Injecte min_bet et min_capital depuis general si non définis au niveau MM.
    """
    yaml_key  = MM_KEY_MAP.get(mm_name, "")
    mm_block  = dict(cfg.get("strategies", {}).get(yaml_key, {}))
    general   = cfg.get("general", {})
    # Héritage des valeurs globales si non surchargées au niveau MM
    for key in ("min_bet", "min_capital"):
        if key not in mm_block and key in general:
            mm_block[key] = general[key]
    return mm_block

# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Charge le CSV et retourne un DataFrame brut."""
    path = Path(filepath)
    if not path.exists():
        sys.exit(f"[ERREUR] Fichier introuvable : {filepath}")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        sys.exit(f"[ERREUR] Impossible de lire le CSV : {e}")

    if df.empty:
        sys.exit("[ERREUR] Le fichier CSV est vide.")

    print(f"[INFO] {len(df)} lignes chargées depuis {filepath}")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les noms de colonnes (casse, espaces)."""
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {}
    for col in df.columns:
        if col in ("open", "o"):
            rename_map[col] = "open"
        elif col in ("high", "h"):
            rename_map[col] = "high"
        elif col in ("low", "l"):
            rename_map[col] = "low"
        elif col in ("close", "c"):
            rename_map[col] = "close"
        elif col in ("volume", "vol", "v"):
            rename_map[col] = "volume"
        elif col in ("timestamp", "time", "date", "datetime", "open_time"):
            rename_map[col] = "timestamp"

    df = df.rename(columns=rename_map)

    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"[ERREUR] Colonnes manquantes après normalisation : {missing}")

    return df


def parse_timestamps(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Parse les timestamps et ajoute une colonne datetime UTC + timezone locale."""
    ts = df["timestamp"]

    # Tenter plusieurs formats
    try:
        if pd.api.types.is_numeric_dtype(ts):
            # Unix timestamp en ms ou s
            if ts.max() > 1e12:
                df["dt_utc"] = pd.to_datetime(ts, unit="ms", utc=True)
            else:
                df["dt_utc"] = pd.to_datetime(ts, unit="s", utc=True)
        else:
            df["dt_utc"] = pd.to_datetime(ts, utc=True)
    except Exception:
        try:
            df["dt_utc"] = pd.to_datetime(ts).dt.tz_localize("UTC")
        except Exception as e:
            sys.exit(f"[ERREUR] Impossible de parser les timestamps : {e}")

    df["dt_local"] = df["dt_utc"].dt.tz_convert(timezone)
    df = df.sort_values("dt_utc").reset_index(drop=True)

    print(f"[INFO] Plage de donnees : {df['dt_utc'].iloc[0]} -> {df['dt_utc'].iloc[-1]}")
    return df


# ============================================================
# 4. MONEY MANAGEMENT
# ============================================================

def apply_money_management(
    trades: pd.DataFrame,
    mm_name: str,
    payout_win: float,
    payout_loss: float,
    initial_capital: float = 100.0,
    mm_cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Applique un money management sur la liste des trades.
    Les paramètres sont lus depuis mm_cfg (issu du YAML) avec des valeurs par défaut.
    """
    if mm_cfg is None:
        mm_cfg = {}

    results = []
    capital    = initial_capital
    max_equity = initial_capital
    peak       = initial_capital

    win_streak         = 0
    loss_streak        = 0
    consecutive_losses = 0   # pour MM6
    pause_counter      = 0
    mm6_index          = 0
    ruine              = False
    ruine_reason       = ""
    n_trades           = len(trades)

    # ── Lecture des paramètres depuis le YAML ─────────────────
    # MM6
    mm6_sequence   = mm_cfg.get("sequence", [1, 1, 2, 4, 8, 16, 32])
    mm6_pause      = int(mm_cfg.get("pause_trades", 1))

    # MM7
    mm7_base_pct   = float(mm_cfg.get("base_fraction_pct", 5.0)) / 100
    mm7_max_pct    = float(mm_cfg.get("max_fraction_pct", 10.0)) / 100
    mm7_win_mults  = sorted(
        mm_cfg.get("win_multipliers", [
            {"min_win_streak": 2, "multiplier": 1.25},
            {"min_win_streak": 3, "multiplier": 1.50},
        ]),
        key=lambda x: x["min_win_streak"], reverse=True,
    )

    # MM8
    mm8_base_pct   = float(mm_cfg.get("base_fraction_pct", 5.0)) / 100
    mm8_loss_steps = sorted(
        mm_cfg.get("loss_steps", [
            {"min_loss_streak": 3, "fraction_pct": 2.5},
            {"min_loss_streak": 5, "fraction_pct": 1.0},
        ]),
        key=lambda x: x["min_loss_streak"], reverse=True,
    )

    # MM9
    mm9_base_pct      = float(mm_cfg.get("base_fraction_pct", 5.0)) / 100
    mm9_pause_trigger = int(mm_cfg.get("pause_after_n_losses", 5))
    mm9_pause_trades  = int(mm_cfg.get("pause_trades", 10))

    # MM10
    mm10_base_pct      = float(mm_cfg.get("base_fraction_pct", 5.0)) / 100
    mm10_max_pct       = float(mm_cfg.get("max_fraction_pct", 10.0)) / 100
    mm10_win_mults     = sorted(
        mm_cfg.get("win_multipliers", [
            {"min_win_streak": 2, "multiplier": 1.25},
            {"min_win_streak": 3, "multiplier": 1.50},
        ]),
        key=lambda x: x["min_win_streak"], reverse=True,
    )
    mm10_loss_steps    = sorted(
        mm_cfg.get("loss_steps", [
            {"min_loss_streak": 3, "fraction_pct": 2.5},
            {"min_loss_streak": 5, "fraction_pct": 1.0},
        ]),
        key=lambda x: x["min_loss_streak"], reverse=True,
    )
    mm10_pause_trigger = int(mm_cfg.get("pause_after_n_losses", 7))
    mm10_pause_trades  = int(mm_cfg.get("pause_trades", 1))

    # MM11
    mm11_base_pct = float(mm_cfg.get("base_fraction_pct",     1.0)) / 100
    mm11_odd_pct  = float(mm_cfg.get("odd_loss_fraction_pct",  2.5)) / 100
    mm11_even_pct = float(mm_cfg.get("even_loss_fraction_pct", 1.0)) / 100

    for idx, row in enumerate(trades.itertuples(index=False)):
        result       = row.result
        pause_active = False

        # ── Calcul de la mise ──────────────────────────────────
        if mm_name == "MM1":
            bet = float(mm_cfg.get("base_stake", 1.0))

        elif mm_name == "MM2":
            bet = capital * (float(mm_cfg.get("fraction_pct", 1.0)) / 100)

        elif mm_name == "MM3":
            bet = capital * (float(mm_cfg.get("fraction_pct", 5.0)) / 100)

        elif mm_name == "MM6":
            if pause_counter > 0:
                pause_counter -= 1
                results.append(_trade_row(
                    idx, row, capital, 0.0, "skip", 0.0, capital,
                    win_streak, loss_streak, True, peak, max_equity,
                ))
                continue
            bet = float(mm6_sequence[mm6_index])

        elif mm_name == "MM7":
            base_bet  = capital * mm7_base_pct
            multiplier = 1.0
            for rule in mm7_win_mults:
                if win_streak >= rule["min_win_streak"]:
                    multiplier = float(rule["multiplier"])
                    break
            bet = min(base_bet * multiplier, capital * mm7_max_pct)

        elif mm_name == "MM8":
            bet = capital * mm8_base_pct
            for step in mm8_loss_steps:
                if loss_streak >= step["min_loss_streak"]:
                    bet = capital * (float(step["fraction_pct"]) / 100)
                    break

        elif mm_name == "MM9":
            if pause_counter > 0:
                pause_counter -= 1
                results.append(_trade_row(
                    idx, row, capital, 0.0, "skip", 0.0, capital,
                    win_streak, loss_streak, True, peak, max_equity,
                ))
                continue
            bet = capital * mm9_base_pct

        elif mm_name == "MM10":
            if pause_counter > 0:
                pause_counter -= 1
                results.append(_trade_row(
                    idx, row, capital, 0.0, "skip", 0.0, capital,
                    win_streak, loss_streak, True, peak, max_equity,
                ))
                continue
            base_bet = capital * mm10_base_pct
            bet = base_bet
            # Réduction prioritaire sur win-boost
            stepped = False
            for step in mm10_loss_steps:
                if loss_streak >= step["min_loss_streak"]:
                    bet = capital * (float(step["fraction_pct"]) / 100)
                    stepped = True
                    break
            if not stepped:
                for rule in mm10_win_mults:
                    if win_streak >= rule["min_win_streak"]:
                        bet = base_bet * float(rule["multiplier"])
                        break
            bet = min(bet, capital * mm10_max_pct)

        elif mm_name == "MM11":
            if loss_streak == 0:
                bet = capital * mm11_base_pct
            elif loss_streak % 2 == 1:
                bet = capital * mm11_odd_pct
            else:
                bet = capital * mm11_even_pct

        else:
            bet = 1.0

        # ── Vérification capital ───────────────────────────────
        min_capital = float(mm_cfg.get("min_capital", 1.0))
        if capital < min_capital:
            ruine        = True
            ruine_reason = f"Capital liquidé ({capital:.6f}$) sous le seuil minimum ({min_capital}$)"
            results.append(_trade_row(
                idx, row, capital, 0.0, "ruine", 0.0, capital,
                win_streak, loss_streak, False, peak, max_equity,
            ))
            break

        if bet > capital:
            ruine        = True
            ruine_reason = f"Capital insuffisant ({capital:.2f}$) pour la mise ({bet:.2f}$)"
            results.append(_trade_row(
                idx, row, capital, bet, "ruine", 0.0, capital,
                win_streak, loss_streak, False, peak, max_equity,
            ))
            break

        min_bet        = float(mm_cfg.get("min_bet", 1.0))
        bet            = max(bet, min_bet)
        capital_before = capital

        # ── Résultat ───────────────────────────────────────────
        if result == "win":
            pnl          = bet * payout_win
            capital     += pnl
            win_streak  += 1
            loss_streak  = 0
        else:
            pnl          = bet * payout_loss
            capital     += pnl
            loss_streak += 1
            win_streak   = 0

        if capital > max_equity:
            max_equity = capital
        if capital > peak:
            peak = capital

        drawdown_pct = (peak - capital) / peak * 100 if peak > 0 else 0.0
        results.append(_trade_row(
            idx, row, capital_before, bet, result, pnl, capital,
            win_streak, loss_streak, False, peak, max_equity, drawdown_pct,
        ))

        # ── Mise à jour états fin de boucle ────────────────────
        if mm_name == "MM6":
            if result == "win":
                mm6_index          = 0
                consecutive_losses = 0
            else:
                consecutive_losses += 1
                if consecutive_losses >= len(mm6_sequence):
                    pause_counter      = mm6_pause
                    mm6_index          = 0
                    consecutive_losses = 0
                else:
                    mm6_index = min(mm6_index + 1, len(mm6_sequence) - 1)

        elif mm_name == "MM9":
            if loss_streak >= mm9_pause_trigger:
                pause_counter = mm9_pause_trades
                loss_streak   = 0

        elif mm_name == "MM10":
            if loss_streak >= mm10_pause_trigger:
                pause_counter = mm10_pause_trades
                loss_streak   = 0
                win_streak    = 0

    # ── Finalisation ───────────────────────────────────────────
    ruine_trade_idx = None
    ruine_time      = None
    if ruine:
        ruine_trade_idx = len(results) - 1 if results else 0
        if ruine_trade_idx < n_trades:
            ruine_time = str(trades.iloc[ruine_trade_idx]["entry_time"])

    df_result = pd.DataFrame(results)
    df_result.attrs["ruine"]           = ruine
    df_result.attrs["ruine_reason"]    = ruine_reason
    df_result.attrs["ruine_trade_idx"] = ruine_trade_idx
    df_result.attrs["ruine_time"]      = ruine_time
    df_result.attrs["max_equity"]      = max_equity
    return df_result


def _apply_mm4_mm5(
    trades: pd.DataFrame,
    mm_name: str,
    payout_win: float,
    payout_loss: float,
    initial_capital: float = 100.0,
    mm_cfg: dict | None = None,
) -> pd.DataFrame:
    """MM4 (martingale classique) et MM5 (linéaire) avec état persistant de la mise."""
    if mm_cfg is None:
        mm_cfg = {}

    base_stake       = float(mm_cfg.get("base_stake", 1.0))
    increment        = float(mm_cfg.get("increment",  1.0))   # pour MM5 seulement
    loss_multiplier  = float(mm_cfg.get("loss_multiplier", 2.0))  # pour MM4 seulement
    use_fraction_pct = bool(mm_cfg.get("use_fraction_pct", False))  # pour MM4 seulement
    fraction_pct     = float(mm_cfg.get("fraction_pct", 1.0))  # pour MM4 seulement

    def _mm4_base_bet(current_capital: float) -> float:
        if mm_name == "MM4" and use_fraction_pct:
            return current_capital * (fraction_pct / 100)
        return base_stake

    results    = []
    capital    = initial_capital
    max_equity = initial_capital
    peak       = initial_capital
    win_streak = 0
    loss_streak = 0
    ruine       = False
    ruine_reason = ""
    bet        = _mm4_base_bet(initial_capital)
    loss_count = 0  # pour MM5

    min_bet = float(mm_cfg.get("min_bet", 1.0))

    for idx, row in enumerate(trades.itertuples(index=False)):
        result       = row.result
        capital_before = capital
        bet_used     = max(bet, min_bet)

        min_capital = float(mm_cfg.get("min_capital", 1.0))
        if capital < min_capital:
            ruine        = True
            ruine_reason = f"Capital liquidé ({capital:.6f}$) sous le seuil minimum ({min_capital}$)"
            results.append(_trade_row(idx, row, capital, 0.0, "ruine", 0.0, capital,
                                      win_streak, loss_streak, False, peak, max_equity))
            break

        if bet_used > capital:
            ruine        = True
            ruine_reason = f"Capital insuffisant ({capital:.2f}$) pour la mise ({bet_used:.2f}$)"
            results.append(_trade_row(idx, row, capital, bet_used, "ruine", 0.0, capital,
                                      win_streak, loss_streak, False, peak, max_equity))
            break

        if result == "win":
            pnl          = bet_used * payout_win
            capital     += pnl
            win_streak  += 1
            loss_streak  = 0
            bet          = _mm4_base_bet(capital)
            loss_count   = 0
        else:
            pnl          = bet_used * payout_loss
            capital     += pnl
            loss_streak += 1
            win_streak   = 0
            if mm_name == "MM4":
                bet = bet_used * loss_multiplier
            else:  # MM5
                loss_count += 1
                bet = base_stake + loss_count * increment

        if capital > max_equity:
            max_equity = capital
        if capital > peak:
            peak = capital

        dd = (peak - capital) / peak * 100 if peak > 0 else 0.0
        results.append(_trade_row(idx, row, capital_before, bet_used,
                                  result, pnl, capital, win_streak, loss_streak,
                                  False, peak, max_equity, dd))

    df_result = pd.DataFrame(results)
    df_result.attrs["ruine"]           = ruine
    df_result.attrs["ruine_reason"]    = ruine_reason
    df_result.attrs["ruine_trade_idx"] = len(results) - 1 if ruine and results else None
    df_result.attrs["ruine_time"]      = None
    df_result.attrs["max_equity"]      = max_equity
    return df_result


def _trade_row(idx, row, capital_before, bet, result, pnl, capital_after,
               win_streak, loss_streak, pause_active, peak, max_equity, drawdown_pct=None):
    if drawdown_pct is None:
        drawdown_pct = (peak - capital_after) / peak * 100 if peak > 0 else 0.0
    return {
        "trade_number":   idx + 1,
        "time":           row.entry_time,
        "capital_before": round(capital_before, 6),
        "bet_size":       round(float(bet), 6),
        "result":         result,
        "pnl_trade":      round(float(pnl), 6),
        "capital_after":  round(float(capital_after), 6),
        "win_streak":     win_streak,
        "loss_streak":    loss_streak,
        "pause_active":   pause_active,
        "drawdown_pct":   round(float(drawdown_pct), 4),
    }


def invert_signals(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Inverse les signaux : UP -> DOWN, DOWN -> UP.
    Recalcule le résultat en conséquence (win/loss s'échangent).
    """
    trades = trades.copy()
    trades["direction"] = trades["direction"].map({"UP": "DOWN", "DOWN": "UP"})
    trades["result"]    = trades["result"].map({"win": "loss", "loss": "win"})
    return trades


def run_mm_simulation(
    trades: pd.DataFrame,
    mm_name: str,
    payout_win: float,
    payout_loss: float,
    initial_capital: float = 100.0,
    mm_cfg: dict | None = None,
) -> pd.DataFrame:
    """Dispatche vers la bonne implémentation MM en passant la config YAML."""
    if mm_name in ("MM4", "MM5"):
        return _apply_mm4_mm5(trades, mm_name, payout_win, payout_loss,
                              initial_capital, mm_cfg)
    return apply_money_management(trades, mm_name, payout_win, payout_loss,
                                  initial_capital, mm_cfg)


# ============================================================
# 5. STATISTIQUES
# ============================================================

def compute_streak_stats(results_col: pd.Series) -> dict:
    """Calcule les statistiques de séries consécutives win/loss."""
    wins  = []
    losses = []
    cur_w = cur_l = 0

    for r in results_col:
        if r == "win":
            cur_w += 1
            if cur_l > 0:
                losses.append(cur_l)
            cur_l = 0
        elif r == "loss":
            cur_l += 1
            if cur_w > 0:
                wins.append(cur_w)
            cur_w = 0

    if cur_w > 0:
        wins.append(cur_w)
    if cur_l > 0:
        losses.append(cur_l)

    def distribution(series):
        if not series:
            return {}
        from collections import Counter
        c = Counter(series)
        total = sum(series)  # total de trades dans ces séries
        dist = {}
        for length, count in sorted(c.items()):
            trades_in = length * count
            dist[length] = {
                "count": count,
                "pct_series": round(count / len(series) * 100, 2),
                "pct_trades": round(trades_in / total * 100, 2) if total > 0 else 0,
            }
        return dist

    return {
        "max_win_streak":    max(wins) if wins else 0,
        "max_loss_streak":   max(losses) if losses else 0,
        "win_streak_dist":   distribution(wins),
        "loss_streak_dist":  distribution(losses),
    }


def compute_drawdown_stats(sim_df: pd.DataFrame) -> dict:
    """Calcule les statistiques de drawdown."""
    if sim_df.empty:
        return {}

    equity = sim_df["capital_after"].to_numpy(dtype=float)
    n = len(equity)
    peak = np.maximum.accumulate(equity)
    dd_pct = (peak - equity) / np.where(peak > 0, peak, 1) * 100
    dd_abs = peak - equity

    max_dd_pct = float(np.max(dd_pct))
    max_dd_abs = float(np.max(dd_abs))

    # Durée max d'un drawdown (en nombre de trades)
    in_dd = False
    max_dur = cur_dur = 0
    for d in dd_pct:
        if d > 0:
            cur_dur += 1
            in_dd = True
        else:
            if in_dd:
                max_dur = max(max_dur, cur_dur)
                cur_dur = 0
            in_dd = False
    max_dur = max(max_dur, cur_dur)

    # Nombre de drawdowns dépassant des seuils
    def count_dd_above(threshold):
        # Trouver les épisodes de DD continus > threshold
        above = dd_pct > threshold
        count = 0
        prev = False
        for a in above:
            if a and not prev:
                count += 1
            prev = a
        return count

    return {
        "max_drawdown_pct":      round(max_dd_pct, 4),
        "max_drawdown_abs":      round(max_dd_abs, 4),
        "max_drawdown_duration": max_dur,
        "dd_above_10pct":        count_dd_above(10),
        "dd_above_20pct":        count_dd_above(20),
        "dd_above_30pct":        count_dd_above(30),
        "dd_above_50pct":        count_dd_above(50),
    }


def compute_time_to_targets(
    sim_df: pd.DataFrame,
    targets: list = CAPITAL_TARGETS,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> dict:
    """Calcule le temps pour atteindre chaque palier de capital."""
    if sim_df.empty:
        return {t: {"trades": "jamais atteint", "days": "jamais atteint", "trades_per_day": "N/A"} for t in targets}

    result = {}
    equity = sim_df["capital_after"].values
    times  = pd.to_datetime(sim_df["time"])
    start  = times.iloc[0] if not times.empty else None

    for target in targets:
        idx_arr = np.where(equity >= target)[0]
        if len(idx_arr) == 0:
            result[target] = {
                "trades": "jamais atteint",
                "days": "jamais atteint",
                "trades_per_day": "N/A",
            }
        else:
            first_idx = idx_arr[0]
            n_trades  = int(first_idx) + 1
            if start is not None and not pd.isna(times.iloc[first_idx]):
                delta_days = (times.iloc[first_idx] - start).days + 1
            else:
                delta_days = 0
            tpd = round(n_trades / delta_days, 2) if delta_days > 0 else 0
            result[target] = {
                "trades":         n_trades,
                "days":           delta_days,
                "trades_per_day": tpd,
            }

    return result


def compute_global_stats(
    trades: pd.DataFrame,
    sim_df: pd.DataFrame,
    payout_win: float,
    payout_loss: float,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> dict:
    """Calcule les statistiques globales d'une simulation."""
    if sim_df.empty:
        return {}

    executed = sim_df[sim_df["result"].isin(["win", "loss"])]
    n_total   = len(sim_df)
    n_exec    = len(executed)
    n_wins    = int((executed["result"] == "win").sum())
    n_losses  = int((executed["result"] == "loss").sum())
    n_skipped = int((sim_df["result"] == "skip").sum())
    winrate   = round(n_wins / n_exec * 100, 4) if n_exec > 0 else 0.0
    lossrate  = round(100 - winrate, 4)

    expectancy = round((winrate / 100) * payout_win + (lossrate / 100) * payout_loss, 6)

    capital_final = float(sim_df["capital_after"].iloc[-1])
    pnl_total     = round(capital_final - initial_capital, 4)
    max_equity    = float(sim_df.attrs.get("max_equity", sim_df["capital_after"].max()))
    rendement_pct = round((capital_final - initial_capital) / initial_capital * 100, 4)

    ruine         = sim_df.attrs.get("ruine", False)
    ruine_reason  = sim_df.attrs.get("ruine_reason", "")
    ruine_idx     = sim_df.attrs.get("ruine_trade_idx", None)
    ruine_time    = sim_df.attrs.get("ruine_time", None)

    return {
        "n_signals":       len(trades),
        "n_executed":      n_exec,
        "n_skipped":       n_skipped,
        "n_wins":          n_wins,
        "n_losses":        n_losses,
        "winrate":         winrate,
        "lossrate":        lossrate,
        "expectancy":      expectancy,
        "pnl_total":       pnl_total,
        "capital_final":   round(capital_final, 4),
        "max_equity":      round(max_equity, 4),
        "rendement_pct":   rendement_pct,
        "ruine":           ruine,
        "ruine_reason":    ruine_reason,
        "ruine_trade_idx": ruine_idx,
        "ruine_time":      ruine_time,
    }


def compute_time_stats(trades: pd.DataFrame) -> dict:
    """Calcule winrate par heure, jour et combinaison jour+heure."""
    if trades.empty:
        return {"by_hour": {}, "by_weekday": {}, "by_day_hour": {}}

    df = trades.copy()
    df["result_bin"] = (df["result"] == "win").astype(int)

    def agg(group_col):
        grouped = df.groupby(group_col)["result_bin"].agg(["sum", "count"])
        grouped.columns = ["wins", "trades"]
        grouped["winrate"] = (grouped["wins"] / grouped["trades"] * 100).round(4)
        return grouped.reset_index().to_dict("records")

    by_hour    = agg("signal_hour_montreal")
    by_weekday = agg("signal_weekday_montreal")

    df["day_hour"] = df["signal_weekday_montreal"].astype(str) + "_" + df["signal_hour_montreal"].astype(str).str.zfill(2)
    by_day_hour = agg("day_hour")

    by_day_hour_sorted = sorted(by_day_hour, key=lambda x: x["winrate"], reverse=True)

    return {
        "by_hour":    by_hour,
        "by_weekday": by_weekday,
        "by_day_hour": by_day_hour,
        "top20_best":  by_day_hour_sorted[:20],
        "top20_worst": by_day_hour_sorted[-20:],
    }


def compute_period_stats(
    trades: pd.DataFrame,
    sim_df: pd.DataFrame,
    payout_win: float,
    payout_loss: float,
    initial_capital: float,
    ref_date: pd.Timestamp | None = None,
) -> dict:
    """Calcule les statistiques pour différentes périodes récentes."""
    if ref_date is None:
        ref_date = pd.Timestamp.now(tz="UTC")

    periods = {
        "all":      None,
        "24m":      24,
        "12m":      12,
        "6m":       6,
        "3m":       3,
        "1m":       1,
    }

    result = {}
    for label, months in periods.items():
        if months is None:
            t_filt = trades.copy()
            s_filt = sim_df.copy()
        else:
            cutoff = ref_date - pd.DateOffset(months=months)
            t_filt = trades[pd.to_datetime(trades["entry_time"], utc=True) >= cutoff].copy()
            s_filt = sim_df[pd.to_datetime(sim_df["time"], utc=True) >= cutoff].copy()

        if t_filt.empty or s_filt.empty:
            result[label] = {}
            continue

        n_exec = len(s_filt[s_filt["result"].isin(["win", "loss"])])
        n_wins = int((s_filt["result"] == "win").sum())
        winrate = round(n_wins / n_exec * 100, 4) if n_exec > 0 else 0.0
        lossrate = round(100 - winrate, 4)
        expectancy = round((winrate / 100) * payout_win + (lossrate / 100) * payout_loss, 6)
        pnl = round(s_filt["pnl_trade"].sum(), 4)

        result[label] = {
            "n_executed": n_exec,
            "n_wins": n_wins,
            "n_losses": n_exec - n_wins,
            "winrate": winrate,
            "expectancy": expectancy,
            "pnl_total": pnl,
        }

    return result


# ============================================================
# 6. EXPORTS
# ============================================================

def export_reports(
    output_dir: Path,
    trades: pd.DataFrame,
    all_results: list,
    time_filter_hours: set,
    strategy_name: str = "",
    strategy_description: str = "",
    market_label: str = "",
):
    """Génère tous les fichiers de sortie."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. CSV des trades signalés (commun à version A et B)
    for version in ["A", "B"]:
        v_trades = [r for r in all_results if r["strategy_version"] == version]
        if not v_trades:
            continue
        # On prend le premier (les trades sont les mêmes peu importe MM/payout)
        first = v_trades[0]
        trade_csv = output_dir / f"trades_version_{version}.csv"
        first["trades"].to_csv(trade_csv, index=False)
        print(f"[EXPORT] {trade_csv}")

    # 2. CSV par simulation MM
    for r in all_results:
        fname = (f"sim_{r['strategy_version']}_"
                 f"payout{r['payout']}_"
                 f"{r['mm_name']}.csv")
        fpath = output_dir / fname
        r["sim_df"].to_csv(fpath, index=False)
        print(f"[EXPORT] {fpath}")

    # 3. CSV synthèse comparatif
    summary_rows = []
    for r in all_results:
        g = r["global_stats"]
        dd = r["drawdown_stats"]
        tt = r["time_to_targets"]
        sk = r["streak_stats"]

        def fmt_target(t):
            v = tt.get(t, {})
            return v.get("days", "jamais atteint")

        summary_rows.append({
            "strategy_version": r["strategy_version"],
            "payout":           r["payout"],
            "money_management": r["mm_name"],
            "trades_executed":  g.get("n_executed", 0),
            "wins":             g.get("n_wins", 0),
            "losses":           g.get("n_losses", 0),
            "winrate":          g.get("winrate", 0),
            "expectancy":       g.get("expectancy", 0),
            "pnl_total":        g.get("pnl_total", 0),
            "capital_final":    g.get("capital_final", 0),
            "max_drawdown_pct": dd.get("max_drawdown_pct", 0),
            "max_win_streak":   sk.get("max_win_streak", 0),
            "max_loss_streak":  sk.get("max_loss_streak", 0),
            "reached_500":      fmt_target(500),
            "reached_1000":     fmt_target(1000),
            "reached_5000":     fmt_target(5000),
            "reached_10000":    fmt_target(10000),
            "reached_50000":    fmt_target(50000),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary_all.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[EXPORT] {summary_path}")

    # 4. CSV stats temporelles (sur les trades version A, payout C, MM1 comme référence)
    ref = next((r for r in all_results
                if r["strategy_version"] == "A"
                and r["payout"] == "C"
                and r["mm_name"] == MM_KEY_MAP.get("MM1", "MM1")), None)
    if ref is None and all_results:
        ref = all_results[0]

    if ref:
        ts = ref["time_stats"]
        pd.DataFrame(ts.get("by_hour", [])).to_csv(output_dir / "stats_by_hour.csv", index=False)
        pd.DataFrame(ts.get("by_weekday", [])).to_csv(output_dir / "stats_by_weekday.csv", index=False)
        pd.DataFrame(ts.get("by_day_hour", [])).to_csv(output_dir / "stats_by_day_hour.csv", index=False)
        print(f"[EXPORT] stats_by_hour.csv / stats_by_weekday.csv / stats_by_day_hour.csv")

    # 5. JSON complet
    def _json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return str(obj)

    json_data = []
    for r in all_results:
        entry = {
            "strategy_version": r["strategy_version"],
            "payout":           r["payout"],
            "mm_name":          r["mm_name"],
            "global_stats":     r["global_stats"],
            "streak_stats":     r["streak_stats"],
            "drawdown_stats":   r["drawdown_stats"],
            "time_to_targets":  {str(k): v for k, v in r["time_to_targets"].items()},
            "period_stats":     r["period_stats"],
        }
        json_data.append(entry)

    json_path = output_dir / "all_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, default=_json_safe, ensure_ascii=False)
    print(f"[EXPORT] {json_path}")

    # 6. Rapport Markdown
    md_path = output_dir / "rapport.md"
    _write_markdown_report(md_path, all_results, summary_df, strategy_name, strategy_description, market_label)
    print(f"[EXPORT] {md_path}")


def _write_markdown_report(path: Path, all_results: list, summary_df: pd.DataFrame,
                           strategy_name: str = "", strategy_description: str = "",
                           market_label: str = ""):
    """Génère le rapport Markdown."""
    lines = []
    report_market_label = market_label or "Backtest"
    lines.append(f"# Rapport de Backtest - {report_market_label}\n")
    lines.append(f"*Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    lines.append("## Résumé de la stratégie\n")
    lines.append(f"- **Stratégie** : `{strategy_name}`")
    if strategy_description:
        lines.append(f"- {strategy_description}")
    lines.append("- **Version A** : sans filtre horaire")
    lines.append("- **Version B** : filtre horaire {04, 05, 06, 07, 08, 17} (America/Montreal)\n")

    lines.append("## Résumé des payouts\n")
    lines.append("| Payout | Win | Loss |")
    lines.append("|--------|-----|------|")
    for k, v in DEFAULT_PAYOUTS.items():
        lines.append(f"| {k} | +{v['win']} | {v['loss']} |")
    lines.append("")

    lines.append("## Tableau comparatif de toutes les gestions\n")
    cols = ["strategy_version", "payout", "money_management", "trades_executed",
            "wins", "losses", "winrate",
            "expectancy", "pnl_total", "capital_final", "max_drawdown_pct",
            "max_win_streak", "max_loss_streak", "reached_50000"]
    available = [c for c in cols if c in summary_df.columns]
    lines.append(summary_df[available].to_markdown(index=False))
    lines.append("")

    lines.append("## Distribution des séries consécutives\n")
    lines.append(
        "Nombre de fois où une série de exactement N trades consécutifs identiques s'est produite.\n"
    )
    lines.append(
        "> **Comment lire** : une série est comptée à sa longueur totale uniquement. "
        "Exemple : `loss, loss, loss, win` = **une** série de longueur 3, "
        "pas trois séries de longueurs 1, 2 et 3.\n"
    )

    def streak_table(dist: dict, label: str) -> list[str]:
        if not dist:
            return [f"*Aucune série de {label} enregistrée.*\n"]
        rows = ["| Longueur | Occurrences | % des séries | % des trades |",
                "|----------|-------------|--------------|--------------|"]
        for length, v in sorted(dist.items()):
            rows.append(
                f"| {length} | {v['count']} | {v['pct_series']:.1f}% | {v['pct_trades']:.1f}% |"
            )
        return rows + [""]

    # Une seule fois par version (A/B) — les trades sont identiques peu importe MM/payout
    seen_versions = set()
    for r in all_results:
        v = r["strategy_version"]
        if v in seen_versions:
            continue
        seen_versions.add(v)
        sk = r["streak_stats"]
        lines.append(f"### Version {v}\n")
        lines.append(f"**Séries de gains** (max : {sk.get('max_win_streak', 0)})\n")
        lines.extend(streak_table(sk.get("win_streak_dist", {}), "gains"))
        lines.append(f"**Séries de pertes** (max : {sk.get('max_loss_streak', 0)})\n")
        lines.extend(streak_table(sk.get("loss_streak_dist", {}), "pertes"))

    lines.append("## Classements\n")

    # Classement par capital final (non-ruine)
    valid = summary_df[summary_df["capital_final"] > 0].copy()

    lines.append("### 1. Par capital final (décroissant)\n")
    top_cap = valid.nlargest(10, "capital_final")[["strategy_version", "payout", "money_management", "capital_final"]]
    lines.append(top_cap.to_markdown(index=False))
    lines.append("")

    lines.append("### 2. Par expectancy (décroissant)\n")
    top_exp = valid.nlargest(10, "expectancy")[["strategy_version", "payout", "money_management", "expectancy"]]
    lines.append(top_exp.to_markdown(index=False))
    lines.append("")

    lines.append("### 3. Par max drawdown (croissant = meilleur)\n")
    top_dd = valid.nsmallest(10, "max_drawdown_pct")[["strategy_version", "payout", "money_management", "max_drawdown_pct"]]
    lines.append(top_dd.to_markdown(index=False))
    lines.append("")

    lines.append("### 4. Par temps pour atteindre 50k\n")
    tmp = valid[valid["reached_50000"] != "jamais atteint"].copy()
    if not tmp.empty:
        tmp["reached_50000_num"] = pd.to_numeric(tmp["reached_50000"], errors="coerce")
        top_50k = tmp.nsmallest(10, "reached_50000_num")[["strategy_version", "payout", "money_management", "reached_50000"]]
        lines.append(top_50k.to_markdown(index=False))
    else:
        lines.append("*Aucune simulation n'a atteint 50 000$.*")
    lines.append("")

    # Meilleur pour croissance
    lines.append("## Meilleur pour croissance maximale\n")
    if not valid.empty:
        best_growth = valid.loc[valid["capital_final"].idxmax()]
        lines.append(f"**{best_growth['money_management']}** — "
                     f"Version {best_growth['strategy_version']}, "
                     f"Payout {best_growth['payout']}, "
                     f"Capital final : {best_growth['capital_final']:.2f}$")
    lines.append("")

    lines.append("## Meilleur compromis survie / croissance\n")
    if not valid.empty:
        valid["score_surv"] = valid["capital_final"] / (valid["max_drawdown_pct"] + 1)
        best_surv = valid.loc[valid["score_surv"].idxmax()]
        lines.append(f"**{best_surv['money_management']}** — "
                     f"Version {best_surv['strategy_version']}, "
                     f"Payout {best_surv['payout']}, "
                     f"Score : {best_surv['score_surv']:.2f}")
    lines.append("")

    lines.append("## Pire stratégie de gestion\n")
    ruined = summary_df[summary_df["capital_final"] <= 0]
    if not ruined.empty:
        worst = ruined.iloc[0]
        lines.append(f"**{worst['money_management']}** — ruine (capital final ≤ 0$)")
    elif not valid.empty:
        worst = valid.loc[valid["capital_final"].idxmin()]
        lines.append(f"**{worst['money_management']}** — capital final le plus bas : {worst['capital_final']:.2f}$")
    lines.append("")

    lines.append("## Conclusion\n")
    lines.append(f"Ce rapport presente les resultats du backtest sur {report_market_label}. "
                 "Les performances passées ne garantissent pas les performances futures. "
                 "Choisissez votre money management en fonction de votre tolérance au risque : "
                 "les martingales offrent une croissance agressive mais comportent un risque de ruine élevé ; "
                 "les stratégies à pourcentage fixe ou réduit après pertes offrent un meilleur compromis. "
                 "La version B (filtre horaire) peut améliorer le winrate en ciblant les heures les plus actives.")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 7. AFFICHAGE CONSOLE
# ============================================================

def infer_market_context(input_path: str, df: pd.DataFrame | None = None) -> tuple[str, str]:
    """Infere le symbole et le timeframe a partir du nom de fichier, avec fallback sur les timestamps."""
    stem = Path(input_path).stem
    parts = stem.split("_")

    symbol = parts[0].upper() if parts else stem.upper()
    timeframe = ""

    if len(parts) > 1:
        candidate = parts[1].lower()
        match = re.fullmatch(r"(\d+)([mhdw])", candidate)
        if match:
            value, unit = match.groups()
            unit_map = {"m": "M", "h": "H", "d": "D", "w": "W"}
            timeframe = f"{value}{unit_map[unit]}"

    if not timeframe and df is not None and len(df) >= 2 and "dt_utc" in df.columns:
        delta = df["dt_utc"].sort_values().diff().dropna()
        if not delta.empty:
            seconds = int(delta.mode().iloc[0].total_seconds())
            if seconds % 604800 == 0:
                timeframe = f"{seconds // 604800}W"
            elif seconds % 86400 == 0:
                timeframe = f"{seconds // 86400}D"
            elif seconds % 3600 == 0:
                timeframe = f"{seconds // 3600}H"
            elif seconds % 60 == 0:
                timeframe = f"{seconds // 60}M"

    return symbol, timeframe


def print_summary(
    strategy_version: str,
    payout: str,
    mm_name: str,
    global_stats: dict,
    drawdown_stats: dict,
    time_to_targets: dict,
    streak_stats: dict,
):
    """Affiche un résumé lisible en console."""
    g  = global_stats
    dd = drawdown_stats
    sk = streak_stats

    def fmt_target(t):
        v = time_to_targets.get(t, {})
        if v.get("trades") == "jamais atteint":
            return "jamais atteint"
        return f"{v.get('trades')} trades / {v.get('days')} jours"

    print("\n" + "=" * 60)
    print(f"  Stratégie : Version {strategy_version} | Payout {payout} | {mm_name}")
    print("=" * 60)
    print(f"  Trades exécutés  : {g.get('n_executed', 0)}")
    print(f"  Wins             : {g.get('n_wins', 0)}")
    print(f"  Losses           : {g.get('n_losses', 0)}")
    print(f"  Winrate          : {g.get('winrate', 0):.2f}%")
    print(f"  Expectancy/trade : {g.get('expectancy', 0):.4f}")
    print(f"  PnL total        : {g.get('pnl_total', 0):.2f}$")
    print(f"  Capital final    : {g.get('capital_final', 0):.2f}$")
    print(f"  Max drawdown     : {dd.get('max_drawdown_pct', 0):.2f}%  ({dd.get('max_drawdown_abs', 0):.2f}$)")
    def fmt_streak_dist(dist: dict) -> str:
        if not dist:
            return "—"
        return "  ".join(f"x{length}={v['count']}" for length, v in sorted(dist.items()))

    print(f"  Max win streak   : {sk.get('max_win_streak', 0)}")
    print(f"  Dist. wins       : {fmt_streak_dist(sk.get('win_streak_dist', {}))}")
    print(f"  Max loss streak  : {sk.get('max_loss_streak', 0)}")
    print(f"  Dist. losses     : {fmt_streak_dist(sk.get('loss_streak_dist', {}))}")
    print(f"  -> 500$  : {fmt_target(500)}")
    print(f"  -> 1k$   : {fmt_target(1000)}")
    print(f"  -> 5k$   : {fmt_target(5000)}")
    print(f"  -> 10k$  : {fmt_target(10000)}")
    print(f"  -> 50k$  : {fmt_target(50000)}")
    if g.get("ruine"):
        print(f"  *** RUINE *** : {g.get('ruine_reason', '')} (trade #{g.get('ruine_trade_idx', '?')})")
    print("=" * 60)


# ============================================================
# 8. MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Backtest BTCUSDT M5 — moteur générique + money management (style Polymarket)"
    )
    parser.add_argument("--input",        required=True,  help="Chemin vers le fichier CSV")
    parser.add_argument("--strategy",     default="streak_rsi",
                        help=f"Stratégie à utiliser (défaut: streak_rsi). Disponibles : {list_strategies()}")
    parser.add_argument("--config",       default=None,   help="Chemin vers le YAML de config MM (défaut: config/money_management.yaml)")
    parser.add_argument("--timezone",     default=MONTREAL_TZ, help="Timezone locale (défaut: America/Montreal)")
    parser.add_argument("--time-filter",  action="store_true",  help="Activer le filtre horaire (Version B)")
    parser.add_argument("--payout",       choices=["A", "B", "C", "all"], default="all",
                        help="Payout à tester (défaut: all — ou selon YAML si non spécifié)")
    parser.add_argument("--mm",           default="all",
                        help="Money management à tester (ex: MM1 ou all — ou selon YAML)")
    parser.add_argument("--output-dir",   default=None,   help="Répertoire de sortie (défaut: YAML general.output_dir ou 'output')")

    # Paramètres de la stratégie streak_rsi (ignorés par les autres stratégies)
    parser.add_argument("--rsi-up",       type=float, default=35.0,  help="[streak_rsi] Seuil RSI7 pour signal UP")
    parser.add_argument("--rsi-down",     type=float, default=65.0,  help="[streak_rsi] Seuil RSI7 pour signal DOWN")
    parser.add_argument("--body-ratio",   type=float, default=0.60,  help="[streak_rsi] Seuil body_ratio minimum")
    parser.add_argument("--range-mult",   type=float, default=1.0,   help="[streak_rsi] Multiplicateur ATR pour seuil range")
    parser.add_argument("--streak-min",   type=int,   default=3,     help="[streak_rsi] Streak minimum pour le signal")
    parser.add_argument("--no-streak",    action="store_true", help="[streak_rsi] Désactiver filtre streak")
    parser.add_argument("--no-rsi",       action="store_true", help="[streak_rsi] Désactiver filtre RSI")
    parser.add_argument("--no-range",     action="store_true", help="[streak_rsi] Désactiver filtre range/ATR")
    parser.add_argument("--no-body-ratio",action="store_true", help="[streak_rsi] Désactiver filtre body_ratio")

    # Walk-forward / split
    parser.add_argument("--walk-forward", action="store_true", help="Mode walk-forward chronologique")
    parser.add_argument("--split",        type=float, default=None,
                        help="Fraction train (ex: 0.7 pour 70/30 split)")

    # Inverse
    parser.add_argument("--inverse", action="store_true",
                        help="Inverser les signaux : UP -> DOWN et DOWN -> UP")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Chargement de la config YAML ──────────────────────
    cfg             = load_config(args.config)
    initial_capital = get_initial_capital(cfg)
    payouts_cfg     = get_payouts(cfg)

    # Versions à tester : CLI > YAML > défaut [A, B]
    general_cfg = cfg.get("general", {})
    versions_cfg = general_cfg.get("versions", {})

    if args.time_filter:
        versions_to_run = ["B"]
    elif versions_cfg:
        versions_to_run = [v for v, vcfg in versions_cfg.items() if vcfg.get("enabled", True)]
    else:
        versions_to_run = ["A", "B"]

    if not versions_to_run:
        sys.exit("[ERREUR] Aucune version activée dans la config (versions.A.enabled / versions.B.enabled).")

    # Heures du filtre Version B : YAML > défaut
    time_filter_hours = set(
        versions_cfg.get("B", {}).get("time_filter_hours", [4, 5, 6, 7, 8, 17])
    )

    # Payouts à tester : CLI filtre les payouts actifs du YAML
    if args.payout == "all":
        payouts_to_run = list(payouts_cfg.keys())
    else:
        payouts_to_run = [args.payout] if args.payout in payouts_cfg else list(payouts_cfg.keys())

    # MM à tester : CLI > YAML enabled
    if args.mm == "all":
        mms_to_run = get_enabled_mms(cfg)
    else:
        mms_to_run = [args.mm.upper()]

    print(f"[INFO] Capital initial   : {initial_capital}$")
    print(f"[INFO] Payouts actifs    : {payouts_to_run}")
    print(f"[INFO] MM actifs         : {mms_to_run}")

    # ── Chargement des données ─────────────────────────────
    df_raw = load_data(args.input)
    df = normalize_columns(df_raw)
    df = parse_timestamps(df, args.timezone)

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    # ── Stratégie ──────────────────────────────────────────
    strategy = get_strategy(args.strategy)
    print(f"[INFO] Stratégie        : {strategy.name}")
    # Contexte de marche extrait du nom du fichier (ex: BTCUSDT_5m_... -> BTCUSDT / 5M)
    _symbol, _timeframe = infer_market_context(args.input, df)
    _market_label = f"{_symbol} {_timeframe}".strip()

    # Output dir : CLI > YAML > défaut, sous-dossier par symbole puis stratégie
    _strategy_output_name = f"{strategy.name}_inverse" if args.inverse else strategy.name
    _base_output = (args.output_dir
                    or cfg.get("general", {}).get("output_dir", "output"))
    output_dir_str = str(Path(_base_output) / _symbol / _strategy_output_name)

    # ── Indicateurs (délégués à la stratégie) ──────────────
    df = strategy.prepare(df)

    # ── Split si demandé ───────────────────────────────────
    if args.split is not None:
        split_idx = int(len(df) * args.split)
        df_test = df.iloc[split_idx:].copy().reset_index(drop=True)
        print(f"[INFO] Split : train={split_idx} lignes, test={len(df_test)} lignes")
        df = df_test

    if args.walk_forward:
        print("[INFO] Mode walk-forward active (donnees deja triees chronologiquement).")

    # Paramètres transmis à la stratégie :
    # YAML signal_strategies.<name> → base, CLI → surcharge si valeur non-défaut
    _yaml_sparams = (
        cfg.get("signal_strategies", {}).get(strategy.name, {})
    )

    # Valeurs CLI (les argparse defaults sont les mêmes que le code → on surcharge
    # seulement si l'utilisateur a explicitement passé une valeur différente du défaut)
    _cli_defaults = {
        "rsi_up": 35.0, "rsi_down": 65.0, "body_ratio": 0.60,
        "range_mult": 1.0, "streak_min": 3,
    }

    def _pick(cli_val, cli_default, yaml_key, yaml_default=None):
        """Retourne la valeur CLI si elle diffère du défaut, sinon la valeur YAML."""
        if cli_val != cli_default:
            return cli_val
        return _yaml_sparams.get(yaml_key, cli_val if yaml_default is None else yaml_default)

    strategy_params = {
        # streak_rsi
        "rsi_up":         _pick(args.rsi_up,    35.0, "rsi_up",         35.0),
        "rsi_down":       _pick(args.rsi_down,  65.0, "rsi_down",       65.0),
        "body_ratio_min": _pick(args.body_ratio, 0.60, "body_ratio_min", 0.60),
        "range_atr_mult": _pick(args.range_mult, 1.0,  "range_atr_mult", 1.0),
        "streak_min":     _pick(args.streak_min, 3,    "streak_min",     3),
        "use_streak":     _yaml_sparams.get("use_streak",     not args.no_streak),
        "use_rsi":        _yaml_sparams.get("use_rsi",        not args.no_rsi),
        "use_range":      _yaml_sparams.get("use_range",      not args.no_range),
        "use_body_ratio": _yaml_sparams.get("use_body_ratio", not args.no_body_ratio),
        # wick_volume_rebound
        "rsi_oversold":   _yaml_sparams.get("rsi_oversold",   30.0),
        "rsi_overbought": _yaml_sparams.get("rsi_overbought", 70.0),
        "wick_body_mult": _yaml_sparams.get("wick_body_mult", 1.5),
        "vol_ma_mult":    _yaml_sparams.get("vol_ma_mult",    1.25),
        # wick_momentum
        "rej_vol_mult":   _yaml_sparams.get("rej_vol_mult",   1.5),
        "rej_wick_mult":  _yaml_sparams.get("rej_wick_mult",  2.0),
        "mom_vol_mult":   _yaml_sparams.get("mom_vol_mult",   2.5),
        "mom_body_ratio": _yaml_sparams.get("mom_body_ratio", 0.8),
        # sniper
        "vol_mult":       _yaml_sparams.get("vol_mult",  4.0),
        "wick_mult":      _yaml_sparams.get("wick_mult", 3.0),
        # momentum
        "threshold_pct":  _yaml_sparams.get("threshold_pct", 0.2),
        # alternating
        "use_loss_streak_switch": _yaml_sparams.get("use_loss_streak_switch", True),
        "loss_streak_switch":     _yaml_sparams.get("loss_streak_switch", 2),
    }

    # ── Génération des signaux par version ─────────────────
    signals_cache = {}
    for version in versions_to_run:
        sig = strategy.generate_signals(
            df,
            timezone=args.timezone,
            use_time_filter=(version == "B"),
            time_filter_hours=time_filter_hours,
            params=strategy_params,
        )
        if args.inverse and not sig.empty:
            sig = invert_signals(sig)
        signals_cache[version] = sig

    if args.inverse:
        print("[INFO] Mode INVERSE actif — signaux UP/DOWN retournés.")

    # ── Simulations ────────────────────────────────────────
    all_results = []
    ref_date    = pd.Timestamp.now(tz="UTC")

    for version in versions_to_run:
        trades = signals_cache[version]
        if trades.empty:
            print(f"[WARN] Version {version} : aucun signal, on passe.")
            continue

        time_stats = compute_time_stats(trades)

        for payout_key in payouts_to_run:
            pw = payouts_cfg[payout_key]["win"]
            pl = payouts_cfg[payout_key]["loss"]

            for mm_name in mms_to_run:
                mm_label      = MM_KEY_MAP.get(mm_name, mm_name)  # nom complet pour l'affichage
                mm_cfg_block  = get_mm_cfg(cfg, mm_name)
                sim_df = run_mm_simulation(trades, mm_name, pw, pl,
                                           initial_capital, mm_cfg_block)

                global_stats    = compute_global_stats(trades, sim_df, pw, pl, initial_capital)
                streak_stats    = compute_streak_stats(
                    sim_df[sim_df["result"].isin(["win", "loss"])]["result"]
                )
                drawdown_stats  = compute_drawdown_stats(sim_df)
                time_to_targets = compute_time_to_targets(sim_df, CAPITAL_TARGETS, initial_capital)
                period_stats    = compute_period_stats(trades, sim_df, pw, pl, initial_capital, ref_date)

                all_results.append({
                    "strategy_version": version,
                    "payout":           payout_key,
                    "mm_name":          mm_label,
                    "trades":           trades,
                    "sim_df":           sim_df,
                    "global_stats":     global_stats,
                    "streak_stats":     streak_stats,
                    "drawdown_stats":   drawdown_stats,
                    "time_to_targets":  time_to_targets,
                    "period_stats":     period_stats,
                    "time_stats":       time_stats,
                })

                print_summary(version, payout_key, mm_label,
                              global_stats, drawdown_stats,
                              time_to_targets, streak_stats)

    # ── Exports ────────────────────────────────────────────
    if all_results:
        export_reports(Path(output_dir_str), trades, all_results, time_filter_hours,
                       strategy_name=strategy.name,
                       strategy_description=strategy.description,
                       market_label=_market_label)
    else:
        print("[WARN] Aucun resultat a exporter.")


if __name__ == "__main__":
    main()
