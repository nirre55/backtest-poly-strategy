"""
Stratégie MICRO_ENSEMBLE — Ensemble de micro-règles booléennes avec vote majoritaire.

Variantes disponibles :
  btc_5m_rules_90_min_votes_1 : 90 règles BTC/5m  — 65.04% / 8971 | Test 68.74% / 643
  eth_5m_rules_25_min_votes_1 : 25 règles ETH/5m  — 67.97% / 2891 | Test 72.80% / 250

Paramètres :
  variant   : nom du jeu de règles (default: btc_5m_rules_90_min_votes_1)
  min_votes : votes totaux minimum pour générer un signal (default: 1)
"""

import numpy as np
import pandas as pd

from .base import BaseStrategy


# ================================================================== #
# Helpers indicateurs                                                  #
# ================================================================== #

def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """RSI avec lissage de Wilder (EWM alpha=1/period)."""
    s = pd.Series(close)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = (100.0 - 100.0 / (1.0 + rs)).to_numpy().copy()
    rsi[avg_loss.to_numpy() == 0.0] = 100.0
    return rsi


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """ATR avec lissage de Wilder."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    alpha = 1.0 / period
    return pd.Series(tr).ewm(alpha=alpha, min_periods=period, adjust=False).mean().to_numpy()


def _stoch_k(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Stochastique %K."""
    h = pd.Series(high).rolling(period).max().to_numpy()
    l = pd.Series(low).rolling(period).min().to_numpy()
    rng = h - l
    k = np.where(rng > 0, 100.0 * (close - l) / rng, 50.0)
    k[:period - 1] = np.nan
    return k


def _cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3.0
    tp_s = pd.Series(tp)
    sma = tp_s.rolling(period).mean().to_numpy()
    mad = tp_s.rolling(period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    ).to_numpy()
    cci = np.where(mad > 0, (tp - sma) / (0.015 * mad), 0.0)
    cci[:period - 1] = np.nan
    return cci


def _mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         volume: np.ndarray, period: int) -> np.ndarray:
    """Money Flow Index."""
    tp   = (high + low + close) / 3.0
    mf   = pd.Series(tp * volume)
    tp_s = pd.Series(tp)
    up   = (tp_s > tp_s.shift(1)).astype(float)
    down = (tp_s < tp_s.shift(1)).astype(float)
    pmf  = (mf * up).rolling(period).sum()
    nmf  = (mf * down).rolling(period).sum()
    mfi  = 100.0 - 100.0 / (1.0 + pmf / nmf.replace(0.0, np.nan))
    mfi[nmf == 0.0] = 100.0
    return mfi.to_numpy()


def _macd_hist_pct(close: np.ndarray) -> np.ndarray:
    """Histogramme MACD (12-26-9) normalisé par le close."""
    s    = pd.Series(close)
    macd = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    hist = (macd - macd.ewm(span=9, adjust=False).mean()).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(close != 0, hist / close, np.nan)


def _heikin_ashi(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                 close: np.ndarray):
    """Calcule les bougies Heikin-Ashi (ha_open, ha_close, ha_high, ha_low)."""
    n = len(close)
    ha_close = (open_ + high + low + close) / 4.0
    ha_open  = np.empty(n, dtype=float)
    ha_open[0] = (open_[0] + close[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    ha_high = np.maximum(high, np.maximum(ha_open, ha_close))
    ha_low  = np.minimum(low,  np.minimum(ha_open, ha_close))
    return ha_open, ha_close, ha_high, ha_low


def _streak_count(is_event: np.ndarray) -> np.ndarray:
    """Compte les séries consécutives de True (reset à 0 dès que False)."""
    result = np.zeros(len(is_event), dtype=float)
    for i in range(len(is_event)):
        if is_event[i]:
            result[i] = result[i - 1] + 1 if i > 0 else 1
    return result


# ================================================================== #
# Jeux de règles                                                       #
# ================================================================== #
# Format : (vote, [(feature, op, valeur), ...])
# op : "le" = <=,  "ge" = >=,  "eq" = ==,  "in" = np.isin

_RULE_SETS: dict[str, list] = {}

# ------------------------------------------------------------------ #
# btc_5m_rules_90_min_votes_1                                         #
# Source : micro_ensemble_combined_90_min_votes_1                     #
# Backtest : 65.04% / 8971 | Test : 68.74% / 643                     #
# ------------------------------------------------------------------ #
_RULE_SETS["btc_5m_rules_90_min_votes_1"] = [
    # 1
    ("RED",   [("stoch_k12",    "ge", 98.87542722),
               ("ret12",        "ge", 0.02486548257),
               ("lower_wick",   "le", 0.001638794532)]),
    # 2
    ("GREEN", [("bb_pctb",      "le", -0.107140425),
               ("hour",         "eq", 13),
               ("volume_z96",   "le", 0.7579850134)]),
    # 3
    ("GREEN", [("cci12",        "le", -239.1832969),
               ("atr72_pct",    "le", 0.0006406964493),
               ("rsi8",         "ge", 16.98439155)]),
    # 4
    ("GREEN", [("donch_low144", "le", 0.001457840018),
               ("body_sum6",    "le", -0.01817602056),
               ("volume_z96",   "ge", 3.587853962)]),
    # 5
    ("GREEN", [("close_z48",    "le", -2.44769017),
               ("atr72_pct",    "le", 0.0006406964493),
               ("body_sum12",   "le", -0.004124811926)]),
    # 6
    ("GREEN", [("close_z24",    "le", -2.774117242),
               ("atr72_pct",    "le", 0.0006406964493),
               ("stoch_k24",    "ge", 10.74126157)]),
    # 7
    ("GREEN", [("rsi21",        "le", 33.28245704),
               ("ret24",        "ge", -0.005787322997),
               ("dist_sma24",   "le", -0.005476785129)]),
    # 8
    ("GREEN", [("bb_pctb",      "le", -0.2340435963),
               ("hour",         "eq", 13),
               ("dist_sma24",   "ge", -0.007134307442)]),
    # 9
    ("GREEN", [("close_z24",    "le", -2.774117242),
               ("atr72_pct",    "le", 0.0006406964493),
               ("bb_pctb",      "ge", -0.2340435963)]),
    # 10
    ("GREEN", [("donch_low72",  "le", 0.0008266398454),
               ("body_sum12",   "le", -0.01970911953),
               ("volume_z96",   "ge", 2.919374564)]),
    # 11
    ("GREEN", [("donch_low144", "le", 0.0006404179203),
               ("hour",         "eq", 14),
               ("ret72",        "le", -0.01475596055)]),
    # 12
    ("RED",   [("donch_high12", "ge", -0.000239825686),
               ("ret12",        "ge", 0.01939351557),
               ("lower_wick",   "le", 0.001638794532)]),
    # 13
    ("GREEN", [("cci12",        "le", -239.1832969),
               ("close_z24",    "ge", -2.058231615),
               ("lower_wick",   "le", 0.001092315747)]),
    # 14
    ("RED",   [("macd_hist_pct","ge", 0.001865948161),
               ("stoch_k12",    "ge", 97.6614989),
               ("close_z24",    "ge", 2.292740233)]),
    # 15
    ("GREEN", [("stoch_k24",    "le", 2.898900732),
               ("donch_high12", "le", -0.02994953395),
               ("mfi8",         "le", 14.51065973)]),
    # 16
    ("GREEN", [("bb_pctb",      "le", -0.107140425),
               ("atr72_pct",    "le", 0.0004654084234),
               ("atr14_pct",    "ge", 0.0002988690878)]),
    # 17
    ("GREEN", [("bb_pctb",      "le", -0.173645982),
               ("hour",         "eq", 13),
               ("mfi8",         "ge", 25.84167143)]),
    # 18
    ("GREEN", [("cci12",        "le", -209.9115581),
               ("atr72_pct",    "le", 0.0007461976822),
               ("atr14_pct",    "ge", 0.0006619818413)]),
    # 19
    ("GREEN", [("close_z24",    "le", -2.774117242),
               ("atr72_pct",    "le", 0.0007461976822),
               ("cci72",        "le", -130.1004463)]),
    # 20
    ("GREEN", [("cci12",        "le", -239.1832969),
               ("atr72_pct",    "le", 0.0007461976822),
               ("stoch_k24",    "ge", 7.498316732)]),
    # 21
    ("GREEN", [("close_z24",    "le", -2.487372513),
               ("atr72_pct",    "le", 0.0004654084234),
               ("cci24",        "ge", -192.094143)]),
    # 22
    ("GREEN", [("close_z24",    "le", -2.774117242),
               ("atr14_pct",    "le", 0.0007499679689),
               ("close_z48",    "le", -3.032681751)]),
    # 23
    ("RED",   [("stoch_k12",    "ge", 95.36043284),
               ("macd_hist_pct","ge", 0.002366750995),
               ("close_z24",    "ge", 2.046146229)]),
    # 24
    ("RED",   [("macd_hist_pct","ge", 0.001865948161),
               ("stoch_k12",    "ge", 95.36043284),
               ("mfi21",        "ge", 73.95404425)]),
    # 25
    ("RED",   [("mfi8",         "ge", 94.13387702),
               ("donch_high12", "ge", -0.0001214530509),
               ("stoch_k24",    "le", 99.45945562)]),
    # 26
    ("RED",   [("macd_hist_pct","ge", 0.001865948161),
               ("stoch_k12",    "ge", 98.87542722),
               ("ret72",        "le", 0.03681466689)]),
    # 27
    ("GREEN", [("bb_pctb",      "le", -0.2340435963),
               ("dist_sma24",   "ge", -0.004431553752),
               ("cci72",        "le", -153.295298)]),
    # 28
    ("GREEN", [("bb_pctb",      "le", -0.2340435963),
               ("dist_sma24",   "ge", -0.004431553752),
               ("rsi14",        "le", 27.52356397)]),
    # 29
    ("GREEN", [("bb_pctb",      "le", -0.107140425),
               ("atr72_pct",    "le", 0.0006406964493),
               ("donch_low72",  "ge", 0.001652921292)]),
    # 30
    ("GREEN", [("close_z24",    "le", -2.774117242),
               ("atr72_pct",    "le", 0.0006406964493),
               ("mfi8",         "le", 18.35512826)]),
    # 31
    ("RED",   [("stoch_k12",    "ge", 95.36043284),
               ("body_sum12",   "ge", 0.01911639022),
               ("donch_high72", "le", -0.001380130085)]),
    # 32
    ("GREEN", [("bb_pctb",      "le", -0.2340435963),
               ("dist_sma24",   "ge", -0.004431553752),
               ("mfi14",        "le", 27.45320025)]),
    # 33
    ("GREEN", [("close_z24",    "le", -2.774117242),
               ("atr72_pct",    "le", 0.0006406964493),
               ("green_count6", "ge", 2)]),
    # 34
    ("GREEN", [("close_z48",    "le", -2.668322486),
               ("atr72_pct",    "le", 0.0004654084234),
               ("mfi14",        "le", 31.37307417)]),
    # 35
    ("GREEN", [("donch_low144", "le", 0.0006404179203),
               ("donch_high12", "le", -0.02387268949),
               ("green_count6", "le", 1)]),
    # 36
    ("RED",   [("rsi8",         "ge", 79.78754453),
               ("hour",         "eq", 21),
               ("red_count6",   "ge", 2)]),
    # 37
    ("RED",   [("stoch_k12",    "ge", 95.36043284),
               ("body_sum12",   "ge", 0.01911639022),
               ("cci72",        "ge", 301.1917591)]),
    # 38
    ("GREEN", [("donch_low72",  "le", 0.0008266398454),
               ("ret24",        "le", -0.03454574655),
               ("upper_wick",   "ge", 0.001054938882)]),
    # 39
    ("RED",   [("stoch_k12",    "ge", 95.36043284),
               ("atr14_pct",    "ge", 0.005489115066),
               ("atr72_pct",    "le", 0.003517992609)]),
    # 40
    ("RED",   [("macd_hist_pct","ge", 0.001865948161),
               ("stoch_k12",    "ge", 95.36043284),
               ("stoch_k24",    "le", 96.70846245)]),
    # 41
    ("GREEN", [("bb_pctb",      "le", -0.107140425),
               ("atr72_pct",    "le", 0.0006406964493),
               ("stoch_k24",    "ge", 4.53517561)]),
    # 42
    ("GREEN", [("stoch_k24",    "le", 1.091431392),
               ("body_sum6",    "le", -0.01392502169),
               ("green_count6", "le", 1)]),
    # 43
    ("RED",   [("stoch_k12",    "ge", 95.36043284),
               ("body_sum6",    "ge", 0.01753783257),
               ("lower_wick",   "le", 0.001638794532)]),
    # 44
    ("GREEN", [("close_z24",    "le", -2.487372513),
               ("hour",         "eq", 13),
               ("volume_z96",   "le", 1.219114982)]),
    # 45
    ("RED",   [("stoch_k12",    "ge", 95.36043284),
               ("body_sum12",   "ge", 0.02432417868),
               ("lower_wick",   "le", 0.001638794532)]),
    # 46
    ("RED",   [("stoch_k12",    "ge", 95.36043284),
               ("ret12",        "ge", 0.01939351557),
               ("weekday",      "eq", 4)]),
    # 47
    ("GREEN", [("close_z24",    "le", -2.774117242),
               ("dist_sma24",   "ge", -0.004431553752),
               ("close_z48",    "le", -3.385888687)]),
    # 48
    ("GREEN", [("bb_pctb",      "le", -0.06443813881),
               ("atr72_pct",    "le", 0.0004654084234),
               ("rsi21",        "ge", 39.3931585)]),
    # 49
    ("GREEN", [("close_z24",    "le", -2.774117242),
               ("donch_low72",  "le", 0.001228070749),
               ("hour",         "eq", 6)]),
    # 50
    ("GREEN", [("donch_low144", "le", 0.001457840018),
               ("close_z48",    "le", -3.385888687),
               ("donch_high12", "ge", -0.007190350995)]),
    # 51
    ("RED",   [("body_sum6",    "ge", 0.008390907843),
               ("donch_high12", "ge", -0.0003773777571),
               ("lower_wick",   "le", 0.0)]),
    # 52
    ("GREEN", [("donch_low144", "le", 0.001457840018),
               ("close_z48",    "le", -3.385888687),
               ("dist_sma24",   "le", -0.0156322462)]),
    # 53
    ("GREEN", [("close_z24",    "le", -2.304024546),
               ("hour",         "eq", 22),
               ("upper_wick",   "le", 9.223945209e-08)]),
    # 54
    ("RED",   [("close_z24",    "ge", 2.783849801),
               ("hour",         "eq", 9),
               ("close_z48",    "le", 3.429563387)]),
    # 55
    ("GREEN", [("close_z24",    "le", -2.774117242),
               ("body_sum12",   "ge", -0.005758468918),
               ("body_sum6",    "le", -0.004965357046)]),
    # 56
    ("GREEN", [("close_z24",    "le", -2.774117242),
               ("body_sum12",   "ge", -0.005758468918),
               ("hour",         "eq", 13)]),
    # 57
    ("RED",   [("donch_high12", "ge", -0.000509590257),
               ("donch_low72",  "le", 0.0005943956918),
               ("lower_wick",   "le", 3.702010378e-07)]),
    # 58
    ("GREEN", [("rsi8",         "le", 31.93496681),
               ("ret72",        "ge", 0.0242546669),
               ("body_sum12",   "ge", -0.007039969776)]),
    # 59
    ("GREEN", [("donch_low72",  "le", 0.0008266398454),
               ("dist_sma24",   "le", -0.0156322462),
               ("ret72",        "ge", -0.02315688552)]),
    # 60
    ("RED",   [("rsi8",         "ge", 79.78754453),
               ("hour",         "eq", 21),
               ("rsi14",        "le", 73.34429789)]),
    # 61
    ("GREEN", [("close_z24",    "le", -2.774117242),
               ("ret12",        "ge", -0.005712097907),
               ("rsi21",        "le", 31.37459303)]),
    # 62
    ("GREEN", [("rsi21",        "le", 33.28245704),
               ("ret24",        "ge", -0.005787322997),
               ("bb_pctb",      "le", -0.173645982)]),
    # 63
    ("GREEN", [("bb_pctb",      "le", -0.2340435963),
               ("dist_sma24",   "ge", -0.004431553752),
               ("body_sum6",    "le", -0.004083019831)]),
    # 64
    ("GREEN", [("close_z24",    "le", -2.774117242),
               ("rsi21",        "ge", 31.37459303),
               ("donch_low72",  "le", 0.0001481980796)]),
    # 65
    ("RED",   [("close_z24",    "ge", 3.068414947),
               ("weekday",      "eq", 5),
               ("close_z48",    "ge", 3.429563387)]),
    # 66
    ("RED",   [("donch_high72", "ge", -0.000212151614),
               ("hour",         "eq", 12),
               ("bb_pctb",      "le", 1.061948757)]),
    # 67
    ("RED",   [("body_sum6",    "ge", 0.01364096683),
               ("lower_wick",   "le", 0.0),
               ("upper_wick",   "le", 0.001347937908)]),
    # 68
    ("GREEN", [("close_z24",    "le", -2.487372513),
               ("hour",         "eq", 11),
               ("body_sum6",    "ge", -0.004965357046)]),
    # 69
    ("GREEN", [("bb_pctb",      "le", -0.06443813881),
               ("hour",         "eq", 13),
               ("weekday",      "eq", 5)]),
    # 70
    ("GREEN", [("rsi8",         "le", 23.24758078),
               ("donch_low144", "ge", 0.03033879208),
               ("close_z24",    "le", -2.304024546)]),
    # 71
    ("GREEN", [("rsi8",         "le", 31.93496681),
               ("donch_low72",  "ge", 0.03468189691),
               ("donch_high72", "ge", -0.02252162313)]),
    # 72
    ("GREEN", [("close_z24",    "le", -2.058231615),
               ("donch_low72",  "ge", 0.02558085724),
               ("donch_high72", "ge", -0.02252162313)]),
    # 73
    ("GREEN", [("bb_pctb",      "le", -0.2340435963),
               ("donch_low72",  "le", 0.001228070749),
               ("weekday",      "eq", 1)]),
    # 74
    ("GREEN", [("close_z24",    "le", -2.774117242),
               ("body_sum6",    "ge", -0.002956606273),
               ("rsi21",        "le", 35.7929937)]),
    # 75
    ("RED",   [("close_z24",    "ge", 3.068414947),
               ("weekday",      "eq", 5),
               ("ret12",        "le", 0.004280476703)]),
    # 76
    ("RED",   [("close_z24",    "ge", 3.068414947),
               ("close_z48",    "le", 1.903776951),
               ("donch_low144", "le", 0.03033879208)]),
    # 77
    ("GREEN", [("close_z24",    "le", -2.487372513),
               ("hour",         "eq", 22),
               ("volume_z96",   "le", 0.7579850134)]),
    # 78
    ("GREEN", [("bb_pctb",      "le", -0.2340435963),
               ("dist_sma24",   "ge", -0.004431553752),
               ("donch_low144", "le", 0.001091384768)]),
    # 79
    ("RED",   [("rsi8",         "ge", 79.78754453),
               ("weekday",      "eq", 5),
               ("rsi21",        "le", 64.95549342)]),
    # 80
    ("RED",   [("close_z24",    "ge", 2.783849801),
               ("weekday",      "eq", 5),
               ("donch_high72", "le", -0.001004677022)]),
    # 81
    ("GREEN", [("bb_pctb",      "le", -0.2340435963),
               ("rsi14",        "ge", 35.01130025),
               ("dist_sma24",   "le", -0.003142104413)]),
    # 82
    ("GREEN", [("bb_pctb",      "le", -0.005731108736),
               ("donch_low72",  "ge", 0.02923394723),
               ("donch_low144", "ge", 0.04226528076)]),
    # 83
    ("GREEN", [("bb_pctb",      "le", -0.107140425),
               ("hour",         "eq", 22),
               ("ret24",        "ge", -0.005787322997)]),
    # 84
    ("GREEN", [("close_z24",    "le", -2.774117242),
               ("body_sum12",   "ge", -0.005758468918),
               ("donch_low144", "le", 0.001091384768)]),
    # 85
    ("GREEN", [("close_z24",    "le", -2.487372513),
               ("hour",         "eq", 13),
               ("body_sum6",    "ge", -0.004083019831)]),
    # 86
    ("RED",   [("rsi14",        "ge", 77.03278368),
               ("hour",         "eq", 11),
               ("volume_z96",   "le", 2.919374564)]),
    # 87
    ("RED",   [("rsi8",         "ge", 73.82299439),
               ("ret72",        "le", -0.01475596055),
               ("close_z48",    "le", 1.450800747)]),
    # 88
    ("RED",   [("body_sum6",    "ge", 0.01364096683),
               ("donch_high12", "ge", -0.000509590257),
               ("ret24",        "le", 0.0173612306)]),
    # 89
    ("RED",   [("close_z24",    "ge", 2.783849801),
               ("close_z48",    "le", 1.450800747),
               ("ret12",        "ge", 0.004280476703)]),
    # 90
    ("GREEN", [("bb_pctb",      "le", -0.107140425),
               ("hour",         "eq", 11),
               ("donch_low144", "le", 0.004305280217)]),
]

# ------------------------------------------------------------------ #
# eth_5m_rules_25_min_votes_1                                         #
# Source : micro_deduped_implementation_details.md                    #
# Backtest : 67.97% / 2891 | Test : 72.80% / 250                     #
# ------------------------------------------------------------------ #
_RULE_SETS["eth_5m_rules_25_min_votes_1"] = [
    # 1
    ("GREEN", [("stoch_k24",      "le", 0.5443385043),
               ("hour",           "eq", 5),
               ("rsi21",          "le", 39.11398072)]),
    # 2
    ("RED",   [("close_z24",      "ge", 3.082851148),
               ("weekday",        "eq", 3),
               ("mfi21",          "le", 66.52204642)]),
    # 3
    ("GREEN", [("donch_low72",    "le", 0.0005704541149),
               ("bb_pctb",        "le", -0.24090522),
               ("close_z48",      "ge", -2.696757558)]),
    # 4
    ("GREEN", [("stoch_k12",      "le", 2.325581395),
               ("bb_pctb",        "le", -0.24090522),
               ("body_abs_pct",   "le", 0.003473998504)]),
    # 5
    ("GREEN", [("donch_low72",    "le", 0.0005704541149),
               ("cci12",          "le", -213.8206725),
               ("stoch_k72",      "ge", 2.887028121)]),
    # 6
    ("RED",   [("rsi8",           "ge", 84.02165584),
               ("atr72_pct",      "le", 0.0009239513225),
               ("lower_wick_body","ge", 0.01797752809)]),
    # 7
    ("GREEN", [("stoch_k24",      "le", 5.132606156),
               ("ret72",          "ge", 0.02599541236),
               ("rsi7",           "le", 24.46140344)]),
    # 8
    ("GREEN", [("donch_low144",   "le", 0.0006709289818),
               ("bb_pctb",        "le", -0.24090522),
               ("volume_z96",     "le", 2.912906413)]),
    # 9
    ("GREEN", [("donch_low72",    "le", 0.0005704541149),
               ("bb_pctb",        "le", -0.24090522),
               ("body_sum6",      "ge", -0.005522189783)]),
    # 10
    ("RED",   [("bb_pctb",        "ge", 1.242274122),
               ("weekday",        "eq", 3),
               ("volume_z96",     "le", 2.098463339)]),
    # 11
    ("RED",   [("rsi8",           "ge", 80.23066448),
               ("close_z48",      "le", 1.456773235),
               ("mfi14",          "le", 73.70646328)]),
    # 12
    ("GREEN", [("mfi8",           "le", 7.65525868),
               ("body_abs_pct",   "ge", 0.01206610733),
               ("stoch_k72",      "le", 10.166951)]),
    # 13
    ("RED",   [("close_z48",      "ge", 3.429940156),
               ("body_sum12",     "le", 0.005753340807),
               ("rsi21",          "ge", 67.95174536)]),
    # 14
    ("RED",   [("rsi8",           "ge", 80.23066448),
               ("atr14_pct",      "le", 0.000953452966),
               ("mfi21",          "ge", 78.27879154)]),
    # 15
    ("GREEN", [("donch_high12",   "le", -0.03797357864),
               ("mfi21",          "le", 13.8331558),
               ("close_z48",      "le", -3.063615586)]),
    # 16
    ("GREEN", [("cci12",          "le", -243.4867158),
               ("atr14_pct",      "le", 0.001277921698),
               ("mfi21",          "le", 33.22794653)]),
    # 17
    ("GREEN", [("stoch_k12",      "le", 2.325581395),
               ("hour",           "eq", 11),
               ("williams_r12",   "ge", -99.31370042)]),
    # 18
    ("GREEN", [("cci12",          "le", -243.4867158),
               ("atr72_pct",      "le", 0.001411663091),
               ("body_abs_pct",   "ge", 0.003473998504)]),
    # 19
    ("GREEN", [("mfi8",           "le", 13.48098558),
               ("body_abs_pct",   "ge", 0.01206610733),
               ("atr72_pct",      "le", 0.00646353357)]),
    # 20
    ("GREEN", [("donch_low72",    "le", 0.0005704541149),
               ("upper_wick_body","ge", 5.117647059),
               ("lower_wick",     "ge", 1.612549971e-05)]),
    # 21
    ("GREEN", [("stoch_k12",      "le", 0.6862995766),
               ("close_z24",      "le", -2.808854839),
               ("body_abs_pct",   "le", 0.003473998504)]),
    # 22
    ("GREEN", [("stoch_k72",      "le", 6.986747793),
               ("body_ratio",     "le", 0.03305785124),
               ("cci12",          "ge", -89.88475125)]),
    # 23
    ("GREEN", [("ret12",          "le", -0.03224343338),
               ("stoch_k12",      "le", 3.779328959),
               ("cci24",          "le", -175.7548746)]),
    # 24
    ("GREEN", [("stoch_k24",      "le", 1.644587669),
               ("bb_pctb",        "le", -0.24090522),
               ("lower_wick_body","ge", 0.01797752809)]),
    # 25
    ("GREEN", [("cci12",          "le", -243.4867158),
               ("atr14_pct",      "le", 0.001072510421),
               ("close_z24",      "le", -2.519107999)]),
]


# ------------------------------------------------------------------ #
# btc_5m_rules_23_min_votes_1                                         #
# Source : btcusdt_5m_best_implementation_details.md                  #
# ------------------------------------------------------------------ #
_RULE_SETS["btc_5m_rules_23_min_votes_1"] = [
    # 1
    ("RED",   [("stoch_k12",      "ge", 98.87542775),
               ("ret12",          "ge", 0.02486548978),
               ("lower_wick",     "le", 0.001638796436)]),
    # 2
    ("GREEN", [("close_z48",      "le", -2.447691672),
               ("atr72_pct",      "le", 0.0006406963614),
               ("body_sum12",     "le", -0.004124824725)]),
    # 3
    ("GREEN", [("donch_low144",   "le", 0.0006404171868),
               ("ha_body",        "le", -0.007640254912),
               ("body_abs_pct",   "ge", 0.007312429007)]),
    # 4
    ("GREEN", [("cci12",          "le", -239.1833565),
               ("atr72_pct",      "le", 0.0006406963614),
               ("rsi8",           "ge", 16.98438998)]),
    # 5
    ("GREEN", [("donch_low144",   "le", 0.001091384106),
               ("body_sum6",      "le", -0.01817603669),
               ("volume_z96",     "ge", 2.919388313)]),
    # 6
    ("GREEN", [("stoch_k24",      "le", 2.898892702),
               ("macd_hist_pct",  "le", -0.002344176743),
               ("range_atr14",    "ge", 1.403339542)]),
    # 7
    ("GREEN", [("bb_pctb",        "le", -0.2340438348),
               ("hour",           "eq", 13),
               ("dist_sma24",     "ge", -0.007134313431)]),
    # 8
    ("GREEN", [("close_z24",      "le", -2.487374544),
               ("atr72_pct",      "le", 0.0004654080978),
               ("cci24",          "ge", -192.0944316)]),
    # 9
    ("GREEN", [("donch_low72",    "le", 0.0008266367569),
               ("body_abs_pct",   "ge", 0.007312429007),
               ("mfi8",           "le", 14.51065799)]),
    # 10
    ("RED",   [("macd_hist_pct",  "ge", 0.00186594868),
               ("stoch_k12",      "ge", 97.66150155),
               ("green_streak",   "ge", 4)]),
    # 11
    ("RED",   [("stoch_k24",      "ge", 98.04361321),
               ("body_sum12",     "ge", 0.02432418065),
               ("green_streak",   "ge", 3)]),
    # 12
    ("GREEN", [("cci12",          "le", -239.1833565),
               ("close_z24",      "ge", -2.058232069),
               ("lower_wick",     "le", 0.001092316795)]),
    # 13
    ("GREEN", [("stoch_k24",      "le", 2.898892702),
               ("donch_high12",   "le", -0.02994954907),
               ("mfi8",           "le", 14.51065799)]),
    # 14
    ("RED",   [("macd_hist_pct",  "ge", 0.00186594868),
               ("stoch_k12",      "ge", 95.36034773),
               ("rsi8",           "ge", 86.81303658)]),
    # 15
    ("GREEN", [("cci12",          "le", -209.9116877),
               ("atr72_pct",      "le", 0.0007461972567),
               ("atr14_pct",      "ge", 0.0006619825044)]),
    # 16
    ("GREEN", [("body",           "le", 0),
               ("rsi7",           "le", 25),
               ("lower_wick_body","ge", 4),
               ("volume_ratio20", "ge", 2),
               ("weekday",        "eq", 2)]),
    # 17
    ("RED",   [("green_streak",   "ge", 5),
               ("rsi7",           "ge", 75),
               ("range_atr14",    "ge", 1.5),
               ("body_ratio",     "ge", 0.75),
               ("weekday",        "eq", 3)]),
    # 18
    ("RED",   [("green_streak",   "ge", 4),
               ("rsi7",           "ge", 75),
               ("range_atr14",    "ge", 1),
               ("body_ratio",     "ge", 0.75),
               ("hour",           "eq", 1)]),
    # 19
    ("RED",   [("green_streak",   "ge", 4),
               ("rsi7",           "ge", 75),
               ("range_atr14",    "ge", 0.8),
               ("body_ratio",     "ge", 0.75),
               ("hour",           "eq", 11)]),
    # 20
    ("GREEN", [("red_streak",     "ge", 3),
               ("rsi7",           "le", 30),
               ("range_atr14",    "ge", 1.5),
               ("body_ratio",     "ge", 0.75),
               ("hour",           "eq", 21)]),
    # 21
    ("RED",   [("green_streak",   "ge", 6),
               ("rsi7",           "ge", 70),
               ("range_atr14",    "ge", 0.8),
               ("body_ratio",     "ge", 0.75),
               ("weekday",        "eq", 5)]),
    # 22
    ("GREEN", [("red_streak",     "ge", 5),
               ("rsi7",           "le", 30),
               ("range_atr14",    "ge", 1.5),
               ("body_ratio",     "ge", 0.75),
               ("hour",           "in", [21, 22, 23])]),
    # 23
    ("GREEN", [("body",           "le", 0),
               ("rsi7",           "le", 30),
               ("lower_wick_body","ge", 1.5),
               ("volume_ratio20", "ge", 1.5),
               ("hour",           "eq", 22)]),
]

# ------------------------------------------------------------------ #
# btc_15m_rules_18_min_votes_1                                        #
# Source : btcusdt_15m_micro_deduped_implementation_details.md        #
# ------------------------------------------------------------------ #
_RULE_SETS["btc_15m_rules_18_min_votes_1"] = [
    # 1
    ("GREEN", [("close_z24",      "le", -2.346311048),
               ("lower_wick",     "le", 2.645309807e-05),
               ("body_sum12",     "ge", -0.007253030737)]),
    # 2
    ("GREEN", [("stoch_k12",      "le", 0.5066458518),
               ("weekday",        "eq", 5),
               ("lower_wick",     "ge", 9.290936623e-08)]),
    # 3
    ("GREEN", [("bb_pctb",        "le", -0.01125914436),
               ("body_ratio",     "ge", 0.9656401664),
               ("range_atr14",    "le", 1.586225659)]),
    # 4
    ("GREEN", [("cci12",          "le", -168.9532813),
               ("lower_wick_body","le", 0.01349188119),
               ("dist_sma24",     "ge", -0.005575910157)]),
    # 5
    ("GREEN", [("close_z24",      "le", -2.547097817),
               ("atr14_pct",      "le", 0.001795740443),
               ("weekday",        "eq", 5)]),
    # 6
    ("GREEN", [("cci12",          "le", -145.0194062),
               ("lower_wick_body","le", 0.01349188119),
               ("rsi8",           "ge", 29.60116771)]),
    # 7
    ("GREEN", [("cci12",          "le", -145.0194062),
               ("lower_wick_body","le", 0.01349188119),
               ("upper_wick",     "le", 8.848352749e-05)]),
    # 8
    ("GREEN", [("stoch_k12",      "le", 3.592157413),
               ("weekday",        "eq", 5),
               ("cci12",          "le", -145.0194062)]),
    # 9
    ("GREEN", [("stoch_k12",      "le", 1.679463493),
               ("body_abs_pct",   "ge", 0.008140445126),
               ("macd_pct",       "le", -0.00345019142)]),
    # 10
    ("GREEN", [("close_z24",      "le", -2.346311048),
               ("body_ratio",     "ge", 0.9656401664),
               ("ret24",          "ge", -0.01042349892)]),
    # 11
    ("GREEN", [("bb_pctb",        "le", -0.01125914436),
               ("close_position", "le", 0.002359360549),
               ("rsi7",           "ge", 28.1055143)]),
    # 12
    ("GREEN", [("bb_pctb",        "le", -0.01125914436),
               ("body_ratio",     "ge", 0.9656401664),
               ("upper_wick",     "le", 8.429477069e-08)]),
    # 13
    ("GREEN", [("stoch_k24",      "le", 2.359680944),
               ("atr14_pct",      "le", 0.001795740443),
               ("bb_pctb",        "le", -0.07487622772)]),
    # 14
    ("GREEN", [("cci12",          "le", -145.0194062),
               ("lower_wick",     "le", 2.645309807e-05),
               ("lower_wick_body","ge", 5.001062838e-05)]),
    # 15
    ("GREEN", [("stoch_k12",      "le", 3.592157413),
               ("weekday",        "eq", 5),
               ("mfi8",           "le", 18.94457115)]),
    # 16
    ("GREEN", [("stoch_k12",      "le", 1.679463493),
               ("cci12",          "le", -145.0194062),
               ("atr14_pct",      "le", 0.001522731461)]),
    # 17
    ("RED",   [("bb_pctb",        "ge", 1.202478662),
               ("volume_ratio20", "le", 1.758262671),
               ("volume_z96",     "le", 0.7249823529)]),
    # 18
    ("GREEN", [("bb_pctb",        "le", 0.04489019383),
               ("body_ratio",     "ge", 0.9987638412),
               ("cci24",          "ge", -201.1674785)]),
]

# ------------------------------------------------------------------ #
# btc_h1_rules_15_min_votes_1                                         #
# Source : btcusdt_h1_best_implementation_details.md                  #
# ------------------------------------------------------------------ #
_RULE_SETS["btc_h1_rules_15_min_votes_1"] = [
    # 1
    ("GREEN", [("stoch_k12",      "le", 13.40911567),
               ("hour",           "eq", 19),
               ("body_abs_pct",   "le", 0.007181889216)]),
    # 2
    ("GREEN", [("stoch_k12",      "le", 13.40911567),
               ("volume_z96",     "le", -0.8641092984),
               ("lower_wick_body","ge", 0.1106917761)]),
    # 3
    ("GREEN", [("stoch_k24",      "le", 7.662631226),
               ("donch_high12",   "le", -0.04360964035),
               ("atr14_pct",      "le", 0.01278625059)]),
    # 4
    ("GREEN", [("stoch_k72",      "le", 3.661654758),
               ("rsi7",           "le", 19.79139536),
               ("rsi21",          "ge", 26.77185052)]),
    # 5
    ("GREEN", [("rsi14",          "le", 27.69290095),
               ("stoch_k24",      "le", 3.478803314),
               ("red_streak",     "le", 3)]),
    # 6
    ("GREEN", [("macd_hist_pct",  "le", -0.002650404386),
               ("stoch_k12",      "le", 5.152344313),
               ("stoch_k24",      "le", 3.478803314)]),
    # 7
    ("GREEN", [("stoch_k72",      "le", 3.661654758),
               ("mfi8",           "le", 15.33077261),
               ("ret24",          "ge", -0.05828397287)]),
    # 8
    ("GREEN", [("body",           "le", 0),
               ("rsi7",           "le", 35),
               ("lower_wick_body","ge", 1.25),
               ("volume_ratio20", "ge", 1),
               ("hour",           "in", [19, 20, 21, 22, 23])]),
    # 9
    ("GREEN", [("red_streak",     "ge", 2),
               ("rsi7",           "le", 35),
               ("range_atr14",    "ge", 0.8),
               ("body_ratio",     "ge", 0.6),
               ("hour",           "in", [19, 20, 21, 22, 23])]),
    # 10
    ("GREEN", [("red_streak",     "ge", 3),
               ("rsi7",           "le", 25),
               ("range_atr14",    "ge", 0.8),
               ("body_ratio",     "ge", 0.45),
               ("weekday",        "eq", 3)]),
    # 11
    ("RED",   [("green_streak",   "ge", 2),
               ("rsi7",           "ge", 60),
               ("range_atr14",    "ge", 1.2),
               ("body_ratio",     "ge", 0.6),
               ("weekday",        "eq", 6)]),
    # 12
    ("GREEN", [("red_streak",     "ge", 2),
               ("rsi7",           "le", 25),
               ("range_atr14",    "ge", 1.2),
               ("body_ratio",     "ge", 0.45),
               ("weekday",        "eq", 6)]),
    # 13
    ("RED",   [("green_streak",   "ge", 2),
               ("rsi7",           "ge", 65),
               ("range_atr14",    "ge", 0.8),
               ("body_ratio",     "ge", 0.6),
               ("weekday",        "eq", 3)]),
    # 14
    ("GREEN", [("red_streak",     "ge", 2),
               ("rsi7",           "le", 35),
               ("range_atr14",    "ge", 1.5),
               ("body_ratio",     "ge", 0.6),
               ("hour",           "in", [0, 1, 2, 3, 4, 5, 6, 7])]),
    # 15
    ("RED",   [("green_streak",   "ge", 2),
               ("rsi7",           "ge", 70),
               ("range_atr14",    "ge", 0.8),
               ("body_ratio",     "ge", 0.75),
               ("hour",           "in", [7, 8, 9, 10, 11, 12])]),
]

# ------------------------------------------------------------------ #
# eth_15m_rules_24_min_votes_1                                        #
# Source : ethusdt_15m_best_implementation_details.md                 #
# ------------------------------------------------------------------ #
_RULE_SETS["eth_15m_rules_24_min_votes_1"] = [
    # 1
    ("GREEN", [("donch_low72",    "le", 0.002929379759),
               ("upper_wick",     "ge", 0.005230358453),
               ("range_atr14",    "ge", 1.386145597)]),
    # 2
    ("GREEN", [("rsi7",           "le", 18.00845307),
               ("ret24",          "ge", -0.01422341075),
               ("volume_z96",     "le", 0.678389517)]),
    # 3
    ("GREEN", [("rsi7",           "le", 20.53343598),
               ("lower_wick",     "le", 0.0001510480269),
               ("close_z48",      "ge", -2.194643073)]),
    # 4
    ("GREEN", [("rsi7",           "le", 18.00845307),
               ("donch_high12",   "ge", -0.01631766978),
               ("body_ratio",     "ge", 0.761334494)]),
    # 5
    ("GREEN", [("stoch_k72",      "le", 6.338975885),
               ("atr14_pct",      "le", 0.002139583407),
               ("donch_low144",   "le", 0.008900804981)]),
    # 6
    ("GREEN", [("stoch_k12",      "le", 2.190740935),
               ("ha_body_ratio",  "ge", 0.7392377051),
               ("donch_low144",   "ge", 0.01135054861)]),
    # 7
    ("GREEN", [("rsi8",           "le", 22.26951785),
               ("body_sum12",     "ge", -0.009912424083),
               ("body_ratio",     "ge", 0.761334494)]),
    # 8
    ("GREEN", [("rsi8",           "le", 22.26951785),
               ("atr72_pct",      "le", 0.002457878466),
               ("stoch_k12",      "le", 12.12790869)]),
    # 9
    ("GREEN", [("bb_pctb",        "le", -0.1302923821),
               ("atr14_pct",      "le", 0.002139583407),
               ("rsi14",          "le", 34.24698515)]),
    # 10
    ("GREEN", [("rsi8",           "le", 22.26951785),
               ("lower_wick",     "le", 0.0001510480269),
               ("cci72",          "ge", -111.0492288)]),
    # 11
    ("GREEN", [("rsi7",           "le", 20.53343598),
               ("atr14_pct",      "le", 0.002139583407),
               ("stoch_k24",      "le", 16.25327588)]),
    # 12
    ("GREEN", [("donch_low72",    "le", 0.0015569182),
               ("rsi21",          "le", 30.19044475),
               ("macd_hist_pct",  "ge", -0.00105807534)]),
    # 13
    ("GREEN", [("bb_pctb",        "le", -0.1302923821),
               ("atr14_pct",      "le", 0.002139583407),
               ("weekday",        "eq", 6)]),
    # 14
    ("GREEN", [("rsi7",           "le", 18.00845307),
               ("donch_high12",   "ge", -0.01631766978),
               ("donch_low72",    "le", 0.0015569182)]),
    # 15
    ("GREEN", [("rsi7",           "le", 18.00845307),
               ("stoch_k12",      "le", 2.190740935),
               ("donch_high12",   "ge", -0.02357829884)]),
    # 16
    ("GREEN", [("stoch_k72",      "le", 8.202131158),
               ("hour",           "eq", 12),
               ("rsi21",          "le", 32.08032065)]),
    # 17
    ("GREEN", [("rsi8",           "le", 22.26951785),
               ("atr14_pct",      "le", 0.002521277008),
               ("cci12",          "le", -191.4788279)]),
    # 18
    ("GREEN", [("close_z24",      "le", -2.581548268),
               ("atr14_pct",      "le", 0.002521277008),
               ("donch_low144",   "le", 0.006105932389)]),
    # 19
    ("GREEN", [("rsi7",           "le", 18.00845307),
               ("dist_sma24",     "ge", -0.01345743525),
               ("close_position", "le", 0.1384615385)]),
    # 20
    ("GREEN", [("red_streak",     "ge", 5),
               ("rsi7",           "le", 40),
               ("range_atr14",    "ge", 0.8),
               ("body_ratio",     "ge", 0.75),
               ("hour",           "in", [0, 1, 2, 3, 4, 5, 6, 7])]),
    # 21
    ("GREEN", [("red_streak",     "ge", 2),
               ("rsi7",           "le", 30),
               ("range_atr14",    "ge", 0.8),
               ("body_ratio",     "ge", 0.75),
               ("hour",           "eq", 6)]),
    # 22
    ("GREEN", [("red_streak",     "ge", 4),
               ("rsi7",           "le", 30),
               ("range_atr14",    "ge", 1.5),
               ("body_ratio",     "ge", 0.6),
               ("weekday",        "eq", 5)]),
    # 23
    ("GREEN", [("red_streak",     "ge", 3),
               ("rsi7",           "le", 30),
               ("range_atr14",    "ge", 0.8),
               ("body_ratio",     "ge", 0.75),
               ("weekday",        "eq", 5)]),
    # 24
    ("GREEN", [("body",           "le", 0),
               ("rsi7",           "le", 30),
               ("lower_wick_body","ge", 2),
               ("volume_ratio20", "ge", 1),
               ("weekday",        "eq", 5)]),
]

# ------------------------------------------------------------------ #
# eth_h1_rules_17_min_votes_1                                         #
# Source : ethusdt_h1_micro_deduped_implementation_details.md         #
# ------------------------------------------------------------------ #
_RULE_SETS["eth_h1_rules_17_min_votes_1"] = [
    # 1
    ("GREEN", [("donch_low72",    "le", 0.005558113318),
               ("macd_hist_pct",  "le", -0.002259446914),
               ("bb_pctb",        "ge", 0.04033662233)]),
    # 2
    ("GREEN", [("donch_low72",    "le", 0.006837778067),
               ("rsi7",           "le", 17.03206178),
               ("ha_body",        "ge", -0.01017634652)]),
    # 3
    ("GREEN", [("mfi21",          "le", 20.65922058),
               ("weekday",        "eq", 2),
               ("rsi7",           "le", 33.60114385)]),
    # 4
    ("GREEN", [("donch_low144",   "le", 0.005548156292),
               ("body_sum12",     "le", -0.03657074613),
               ("mfi14",          "le", 21.44107346)]),
    # 5
    ("GREEN", [("mfi21",          "le", 20.65922058),
               ("stoch_k72",      "le", 8.138635),
               ("donch_high12",   "ge", -0.04766910986)]),
    # 6
    ("GREEN", [("rsi21",          "le", 29.1932855),
               ("close_position", "le", 0.1724014402),
               ("ha_close_position","ge", 0.3840913291)]),
    # 7
    ("GREEN", [("donch_low72",    "le", 0.006837778067),
               ("mfi14",          "le", 16.05947179),
               ("cci12",          "ge", -112.7701187)]),
    # 8
    ("GREEN", [("stoch_k24",      "le", 7.980198437),
               ("mfi8",           "le", 18.38030246),
               ("rsi8",           "le", 12.50217521)]),
    # 9
    ("GREEN", [("stoch_k24",      "le", 7.980198437),
               ("macd_pct",       "le", -0.01229544773),
               ("mfi14",          "le", 18.18947095)]),
    # 10
    ("GREEN", [("donch_low72",    "le", 0.005558113318),
               ("macd_hist_pct",  "le", -0.003023911405),
               ("close_z48",      "ge", -2.500795018)]),
    # 11
    ("GREEN", [("mfi14",          "le", 21.44107346),
               ("close_position", "le", 0.1223925466),
               ("lower_wick_body","ge", 0.06465758156)]),
    # 12
    ("GREEN", [("donch_low72",    "le", 0.003842653373),
               ("rsi8",           "le", 24.91006406),
               ("close_z24",      "ge", -2.099858976)]),
    # 13
    ("GREEN", [("donch_low72",    "le", 0.006837778067),
               ("rsi8",           "le", 18.79011376),
               ("close_z24",      "ge", -2.099858976)]),
    # 14
    ("GREEN", [("donch_low72",    "le", 0.006837778067),
               ("rsi7",           "le", 23.29241515),
               ("bb_pctb",        "ge", 0.08661248075)]),
    # 15
    ("GREEN", [("donch_low144",   "le", 0.005548156292),
               ("rsi7",           "le", 17.03206178),
               ("ha_body_ratio",  "le", 0.6571549533)]),
    # 16
    ("GREEN", [("donch_low72",    "le", 0.006837778067),
               ("rsi7",           "le", 13.5929322),
               ("close_z24",      "ge", -2.584346456)]),
    # 17
    ("GREEN", [("mfi21",          "le", 20.65922058),
               ("stoch_k72",      "le", 8.138635),
               ("donch_high72",   "le", -0.1118632156)]),
]

# ------------------------------------------------------------------ #
# doge_5m_rules_11_min_votes_1                                        #
# Source : dogeusdt_m5_best_implementation_details.md                 #
# ------------------------------------------------------------------ #
_RULE_SETS["doge_5m_rules_11_min_votes_1"] = [
    # 1
    ("GREEN", [("ret24",          "le", -0.05666612446),
               ("rsi21",          "le", 28.5686663),
               ("cci24",          "le", -264.6708728)]),
    # 2
    ("GREEN", [("red_streak",     "ge", 4),
               ("rsi7",           "le", 35),
               ("range_atr14",    "ge", 1.2),
               ("body_ratio",     "ge", 0.75),
               ("hour",           "eq", 12)]),
    # 3
    ("RED",   [("green_streak",   "ge", 6),
               ("rsi7",           "ge", 70),
               ("range_atr14",    "ge", 1.5),
               ("body_ratio",     "ge", 0.75),
               ("hour",           "in", [11, 12, 13, 14, 15])]),
    # 4
    ("GREEN", [("red_streak",     "ge", 5),
               ("rsi7",           "le", 25),
               ("range_atr14",    "ge", 1.5),
               ("body_ratio",     "ge", 0.75),
               ("weekday",        "eq", 0)]),
    # 5
    ("GREEN", [("red_streak",     "ge", 4),
               ("rsi7",           "le", 25),
               ("range_atr14",    "ge", 0.8),
               ("body_ratio",     "ge", 0.75),
               ("hour",           "eq", 13)]),
    # 6
    ("GREEN", [("red_streak",     "ge", 6),
               ("rsi7",           "le", 30),
               ("range_atr14",    "ge", 1.5),
               ("body_ratio",     "ge", 0.6),
               ("weekday",        "eq", 1)]),
    # 7
    ("RED",   [("body",           "ge", 0),
               ("rsi7",           "ge", 70),
               ("upper_wick_body","ge", 3),
               ("volume_ratio20", "ge", 2),
               ("weekday",        "eq", 5)]),
    # 8
    ("RED",   [("green_streak",   "ge", 3),
               ("rsi7",           "ge", 65),
               ("range_atr14",    "ge", 1.5),
               ("body_ratio",     "ge", 0.75),
               ("hour",           "eq", 19)]),
    # 9
    ("GREEN", [("body",           "le", 0),
               ("rsi7",           "le", 35),
               ("lower_wick_body","ge", 2),
               ("volume_ratio20", "ge", 4),
               ("hour",           "in", [7, 8, 9, 10, 11, 12])]),
    # 10
    ("RED",   [("body",           "ge", 0),
               ("rsi7",           "ge", 70),
               ("upper_wick_body","ge", 3),
               ("volume_ratio20", "ge", 1),
               ("hour",           "eq", 9)]),
    # 11
    ("GREEN", [("body",           "le", 0),
               ("rsi7",           "le", 35),
               ("lower_wick_body","ge", 4),
               ("volume_ratio20", "ge", 1),
               ("hour",           "eq", 17)]),
]

# ------------------------------------------------------------------ #
# doge_15m_rules_16_min_votes_1                                       #
# Source : dogeusdt_m15_best_implementation_details.md                #
# ------------------------------------------------------------------ #
_RULE_SETS["doge_15m_rules_16_min_votes_1"] = [
    # 1
    ("GREEN", [("donch_low144",   "le", 0.005733286026),
               ("donch_high72",   "le", -0.1023903561),
               ("rsi14",          "le", 27.14980207)]),
    # 2
    ("GREEN", [("body_sum12",     "le", -0.07163992181),
               ("stoch_k12",      "le", 8.601601931),
               ("mfi8",           "le", 20.08710485)]),
    # 3
    ("GREEN", [("donch_high12",   "le", -0.08452491994),
               ("rsi7",           "le", 21.15267346),
               ("mfi14",          "le", 23.62586231)]),
    # 4
    ("GREEN", [("donch_low144",   "le", 0.005733286026),
               ("body_abs_pct",   "ge", 0.01649142991),
               ("macd_pct",       "le", -0.006858402107)]),
    # 5
    ("GREEN", [("donch_high12",   "le", -0.08452491994),
               ("red_streak",     "ge", 3),
               ("mfi8",           "le", 24.92378681)]),
    # 6
    ("GREEN", [("dist_sma24",     "le", -0.05304158274),
               ("stoch_k24",      "le", 6.967853725),
               ("rsi8",           "le", 20.42990831)]),
    # 7
    ("GREEN", [("ret24",          "le", -0.08953947387),
               ("rsi8",           "le", 26.33446173),
               ("stoch_k24",      "le", 6.967853725)]),
    # 8
    ("GREEN", [("dist_sma24",     "le", -0.05304158274),
               ("range_atr14",    "ge", 1.70326348),
               ("upper_wick",     "le", 0.00464566553)]),
    # 9
    ("GREEN", [("red_streak",     "ge", 4),
               ("rsi7",           "le", 40),
               ("range_atr14",    "ge", 1),
               ("body_ratio",     "ge", 0.6),
               ("hour",           "eq", 11)]),
    # 10
    ("RED",   [("green_streak",   "ge", 4),
               ("rsi7",           "ge", 65),
               ("range_atr14",    "ge", 1),
               ("body_ratio",     "ge", 0.6),
               ("hour",           "eq", 12)]),
    # 11
    ("GREEN", [("red_streak",     "ge", 4),
               ("rsi7",           "le", 25),
               ("range_atr14",    "ge", 1),
               ("body_ratio",     "ge", 0.6),
               ("weekday",        "eq", 5)]),
    # 12
    ("RED",   [("green_streak",   "ge", 6),
               ("rsi7",           "ge", 60),
               ("range_atr14",    "ge", 1.2),
               ("body_ratio",     "ge", 0.45),
               ("hour",           "in", [7, 8, 9, 10, 11, 12])]),
    # 13
    ("RED",   [("green_streak",   "ge", 5),
               ("rsi7",           "ge", 65),
               ("range_atr14",    "ge", 0.8),
               ("body_ratio",     "ge", 0.6),
               ("weekday",        "eq", 6)]),
    # 14
    ("GREEN", [("red_streak",     "ge", 3),
               ("rsi7",           "le", 25),
               ("range_atr14",    "ge", 1.2),
               ("body_ratio",     "ge", 0.45),
               ("hour",           "eq", 21)]),
    # 15
    ("RED",   [("green_streak",   "ge", 5),
               ("rsi7",           "ge", 75),
               ("range_atr14",    "ge", 0.8),
               ("body_ratio",     "ge", 0.6),
               ("weekday",        "eq", 5)]),
    # 16
    ("GREEN", [("red_streak",     "ge", 5),
               ("rsi7",           "le", 25),
               ("range_atr14",    "ge", 1),
               ("body_ratio",     "ge", 0.6),
               ("weekday",        "eq", 2)]),
]

# ------------------------------------------------------------------ #
# doge_h1_rules_19_min_votes_1                                        #
# Source : dogeusdt_h1_best_implementation_details.md                 #
# ------------------------------------------------------------------ #
_RULE_SETS["doge_h1_rules_19_min_votes_1"] = [
    # 1
    ("GREEN", [("ret24",          "le", -0.08795256767),
               ("green_count6",   "le", 1),
               ("donch_low72",    "ge", 0.01591942922)]),
    # 2
    ("GREEN", [("stoch_k12",      "le", 7.732799668),
               ("upper_wick",     "ge", 0.006536133456),
               ("mfi21",          "ge", 32.90566257)]),
    # 3
    ("GREEN", [("stoch_k12",      "le", 10.86781099),
               ("upper_wick",     "ge", 0.008313539192),
               ("cci12",          "le", -91.40704478)]),
    # 4
    ("GREEN", [("close_z24",      "le", -2.112495107),
               ("macd_pct",       "le", -0.02174086147),
               ("volume_z96",     "le", 3.553424213)]),
    # 5
    ("GREEN", [("close_z24",      "le", -2.112495107),
               ("atr14_pct",      "ge", 0.02158908929),
               ("stoch_k12",      "le", 10.86781099)]),
    # 6
    ("GREEN", [("close_z24",      "le", -2.112495107),
               ("atr14_pct",      "ge", 0.02158908929),
               ("red_streak",     "ge", 3)]),
    # 7
    ("GREEN", [("bb_pctb",        "le", -0.1365937141),
               ("volume_ratio20", "le", 1.87190469),
               ("rsi21",          "ge", 38.01092239)]),
    # 8
    ("GREEN", [("stoch_k12",      "le", 6.12998497),
               ("ret24",          "le", -0.05080321547),
               ("rsi14",          "ge", 28.21874247)]),
    # 9
    ("GREEN", [("ret24",          "le", -0.08795256767),
               ("red_streak",     "ge", 3),
               ("upper_wick",     "ge", 0.006536133456)]),
    # 10
    ("GREEN", [("close_z24",      "le", -2.112495107),
               ("donch_high12",   "le", -0.09779320525),
               ("cci12",          "ge", -175.5808578)]),
    # 11
    ("GREEN", [("stoch_k12",      "le", 6.12998497),
               ("macd_hist_pct",  "le", -0.005251646692),
               ("cci12",          "ge", -150.9935977)]),
    # 12
    ("GREEN", [("red_streak",     "ge", 3),
               ("rsi7",           "le", 25),
               ("range_atr14",    "ge", 1.2),
               ("body_ratio",     "ge", 0.6),
               ("weekday",        "eq", 0)]),
    # 13
    ("GREEN", [("red_streak",     "ge", 3),
               ("rsi7",           "le", 40),
               ("range_atr14",    "ge", 0.8),
               ("body_ratio",     "ge", 0.75),
               ("hour",           "in", [19, 20, 21, 22, 23])]),
    # 14
    ("RED",   [("green_streak",   "ge", 2),
               ("rsi7",           "ge", 70),
               ("range_atr14",    "ge", 0.8),
               ("body_ratio",     "ge", 0.45),
               ("hour",           "eq", 6)]),
    # 15
    ("GREEN", [("red_streak",     "ge", 3),
               ("rsi7",           "le", 25),
               ("range_atr14",    "ge", 1),
               ("body_ratio",     "ge", 0.45),
               ("hour",           "in", [21, 22, 23])]),
    # 16
    ("RED",   [("green_streak",   "ge", 2),
               ("rsi7",           "ge", 60),
               ("range_atr14",    "ge", 0.8),
               ("body_ratio",     "ge", 0.6),
               ("hour",           "eq", 2)]),
    # 17
    ("RED",   [("green_streak",   "ge", 2),
               ("rsi7",           "ge", 75),
               ("range_atr14",    "ge", 1),
               ("body_ratio",     "ge", 0.6),
               ("weekday",        "eq", 1)]),
    # 18
    ("RED",   [("green_streak",   "ge", 3),
               ("rsi7",           "ge", 75),
               ("range_atr14",    "ge", 0.8),
               ("body_ratio",     "ge", 0.45),
               ("weekday",        "eq", 4)]),
    # 19
    ("GREEN", [("red_streak",     "ge", 2),
               ("rsi7",           "le", 40),
               ("range_atr14",    "ge", 1.2),
               ("body_ratio",     "ge", 0.75),
               ("weekday",        "eq", 2)]),
]


# ================================================================== #
# Stratégie                                                            #
# ================================================================== #

class MicroEnsembleStrategy(BaseStrategy):

    name = "micro_ensemble"
    description = (
        "MICRO_ENSEMBLE : vote majoritaire de micro-règles booléennes. "
        "Variantes : btc_5m_rules_90_min_votes_1, btc_5m_rules_23_min_votes_1, "
        "btc_15m_rules_18_min_votes_1, btc_h1_rules_15_min_votes_1, "
        "eth_5m_rules_25_min_votes_1, eth_15m_rules_24_min_votes_1, "
        "eth_h1_rules_17_min_votes_1, doge_5m_rules_11_min_votes_1, "
        "doge_15m_rules_16_min_votes_1, doge_h1_rules_19_min_votes_1. "
        "Paramètres : variant, min_votes."
    )

    # ------------------------------------------------------------------ #
    # Indicateurs                                                          #
    # ------------------------------------------------------------------ #

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close  = df["close"].to_numpy(dtype=float)
        open_  = df["open"].to_numpy(dtype=float)
        high   = df["high"].to_numpy(dtype=float)
        low    = df["low"].to_numpy(dtype=float)
        volume = df["volume"].to_numpy(dtype=float) if "volume" in df.columns else np.ones(len(df))

        # ── Z-scores du close ───────────────────────────────────────────
        def _close_z(period):
            s = pd.Series(close)
            m = s.rolling(period).mean()
            d = s.rolling(period).std(ddof=0)
            return ((s - m) / d.replace(0, np.nan)).to_numpy()

        df["close_z24"] = _close_z(24)
        df["close_z48"] = _close_z(48)

        # ── Donchian distances ──────────────────────────────────────────
        # donch_low  = close / rolling_min_low(N) - 1  (>= 0)
        # donch_high = close / rolling_max_high(N) - 1 (<= 0)
        for n_win in (12, 72, 144):
            low_min  = pd.Series(low).rolling(n_win).min().to_numpy()
            high_max = pd.Series(high).rolling(n_win).max().to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                dl = np.where(low_min  > 0, close / low_min  - 1, np.nan)
                dh = np.where(high_max > 0, close / high_max - 1, np.nan)
            df[f"donch_low{n_win}"] = dl
            if n_win in (12, 72):
                df[f"donch_high{n_win}"] = dh

        # ── Bollinger Bands %B ─────────────────────────────────────────
        s  = pd.Series(close)
        bb_mean = s.rolling(20).mean()
        bb_std  = s.rolling(20).std(ddof=0)
        upper   = bb_mean + 2 * bb_std
        lower   = bb_mean - 2 * bb_std
        band_rng = (upper - lower).replace(0, np.nan)
        df["bb_pctb"] = ((s - lower) / band_rng).to_numpy()

        # ── ATR ────────────────────────────────────────────────────────
        atr14 = _atr(high, low, close, 14)
        atr72 = _atr(high, low, close, 72)
        with np.errstate(divide="ignore", invalid="ignore"):
            df["atr14_pct"] = np.where(close > 0, atr14 / close, np.nan)
            df["atr72_pct"] = np.where(close > 0, atr72 / close, np.nan)

        # ── Body sums ──────────────────────────────────────────────────
        with np.errstate(divide="ignore", invalid="ignore"):
            body_ratio_raw = np.where(close != 0, (close - open_) / close, 0.0)
        br = pd.Series(body_ratio_raw)
        df["body_sum6"]  = br.rolling(6).sum().to_numpy()
        df["body_sum12"] = br.rolling(12).sum().to_numpy()

        # ── Heure et jour UTC ──────────────────────────────────────────
        dt_utc = pd.to_datetime(df["dt_utc"], utc=True)
        df["hour"]    = dt_utc.dt.hour.to_numpy()
        df["weekday"] = dt_utc.dt.weekday.to_numpy()  # 0=lundi … 6=dimanche

        # ── RSI ────────────────────────────────────────────────────────
        df["rsi7"]  = _rsi(close, 7)
        df["rsi8"]  = _rsi(close, 8)
        df["rsi14"] = _rsi(close, 14)
        df["rsi21"] = _rsi(close, 21)

        # ── Stochastique %K ───────────────────────────────────────────
        df["stoch_k12"] = _stoch_k(high, low, close, 12)
        df["stoch_k24"] = _stoch_k(high, low, close, 24)
        df["stoch_k72"] = _stoch_k(high, low, close, 72)

        # ── Rendements ────────────────────────────────────────────────
        cs = pd.Series(close)
        with np.errstate(divide="ignore", invalid="ignore"):
            for k in (12, 24, 72):
                df[f"ret{k}"] = (cs / cs.shift(k) - 1).to_numpy()

        # ── Distance à la SMA24 ───────────────────────────────────────
        sma24 = cs.rolling(24).mean().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            df["dist_sma24"] = np.where(sma24 > 0, close / sma24 - 1, np.nan)

        # ── CCI ───────────────────────────────────────────────────────
        df["cci12"] = _cci(high, low, close, 12)
        df["cci24"] = _cci(high, low, close, 24)
        df["cci72"] = _cci(high, low, close, 72)

        # ── Mèches ────────────────────────────────────────────────────
        with np.errstate(divide="ignore", invalid="ignore"):
            lower_wick_arr = np.where(
                close > 0, (np.minimum(open_, close) - low) / close, np.nan
            )
            upper_wick_arr = np.where(
                close > 0, (high - np.maximum(open_, close)) / close, np.nan
            )
        df["lower_wick"] = lower_wick_arr
        df["upper_wick"] = upper_wick_arr

        # ── Features corps ────────────────────────────────────────────
        body_abs = np.abs(close - open_)
        with np.errstate(divide="ignore", invalid="ignore"):
            body_abs_pct_arr = np.where(close > 0, body_abs / close, np.nan)
            df["body_abs_pct"] = body_abs_pct_arr
            df["body_ratio"]   = np.where(
                (high - low) > 0, body_abs / (high - low), np.nan
            )
            df["lower_wick_body"] = np.where(
                body_abs_pct_arr > 0,
                lower_wick_arr / body_abs_pct_arr,
                np.nan,
            )
            df["upper_wick_body"] = np.where(
                body_abs_pct_arr > 0,
                upper_wick_arr / body_abs_pct_arr,
                np.nan,
            )

        # ── Williams %R 12 ────────────────────────────────────────────
        h12_max = pd.Series(high).rolling(12).max().to_numpy()
        l12_min = pd.Series(low).rolling(12).min().to_numpy()
        hl_range = h12_max - l12_min
        with np.errstate(divide="ignore", invalid="ignore"):
            df["williams_r12"] = np.where(
                hl_range > 0, (h12_max - close) / hl_range * -100.0, np.nan
            )

        # ── MFI ───────────────────────────────────────────────────────
        df["mfi8"]  = _mfi(high, low, close, volume, 8)
        df["mfi14"] = _mfi(high, low, close, volume, 14)
        df["mfi21"] = _mfi(high, low, close, volume, 21)

        # ── Volume Z-score ────────────────────────────────────────────
        vs = pd.Series(volume)
        vm = vs.rolling(96).mean()
        vd = vs.rolling(96).std(ddof=0)
        df["volume_z96"] = ((vs - vm) / vd.replace(0, np.nan)).to_numpy()

        # ── MACD histogram % ─────────────────────────────────────────
        df["macd_hist_pct"] = _macd_hist_pct(close)

        # ── Green / Red counts ────────────────────────────────────────
        is_green_arr = (close > open_)
        is_red_arr   = (close < open_)
        is_green_s = pd.Series(is_green_arr.astype(float))
        is_red_s   = pd.Series(is_red_arr.astype(float))
        df["green_count6"] = is_green_s.rolling(6).sum().to_numpy()
        df["red_count6"]   = is_red_s.rolling(6).sum().to_numpy()

        # ── Streaks consécutifs ──────────────────────────────────────
        df["green_streak"] = _streak_count(is_green_arr)
        df["red_streak"]   = _streak_count(is_red_arr)

        # ── Corps signé ──────────────────────────────────────────────
        with np.errstate(divide="ignore", invalid="ignore"):
            df["body"] = np.where(close > 0, (close - open_) / close, np.nan)

        # ── Position du close dans la bougie ─────────────────────────
        candle_range = high - low
        with np.errstate(divide="ignore", invalid="ignore"):
            df["close_position"] = np.where(
                candle_range > 0, (close - low) / candle_range, np.nan
            )

        # ── ATR range ratio ───────────────────────────────────────────
        with np.errstate(divide="ignore", invalid="ignore"):
            df["range_atr14"] = np.where(atr14 > 0, (high - low) / atr14, np.nan)

        # ── Volume ratio 20 ───────────────────────────────────────────
        vs20 = pd.Series(volume).rolling(20).mean().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            df["volume_ratio20"] = np.where(vs20 > 0, volume / vs20, np.nan)

        # ── MACD line % ───────────────────────────────────────────────
        cs2 = pd.Series(close)
        macd_line = (cs2.ewm(span=12, adjust=False).mean()
                     - cs2.ewm(span=26, adjust=False).mean()).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            df["macd_pct"] = np.where(close != 0, macd_line / close, np.nan)

        # ── Heikin-Ashi features ──────────────────────────────────────
        ha_open_arr, ha_close_arr, ha_high_arr, ha_low_arr = _heikin_ashi(
            open_, high, low, close
        )
        ha_range_arr = ha_high_arr - ha_low_arr
        with np.errstate(divide="ignore", invalid="ignore"):
            df["ha_body"]           = np.where(
                close > 0, (ha_close_arr - ha_open_arr) / close, np.nan
            )
            df["ha_body_ratio"]     = np.where(
                ha_range_arr > 0,
                (ha_close_arr - ha_open_arr) / ha_range_arr,
                np.nan,
            )
            df["ha_close_position"] = np.where(
                ha_range_arr > 0,
                (ha_close_arr - ha_low_arr) / ha_range_arr,
                np.nan,
            )

        return df

    # ------------------------------------------------------------------ #
    # Génération des signaux — évaluation vectorisée des règles           #
    # ------------------------------------------------------------------ #

    def generate_signals(
        self,
        df: pd.DataFrame,
        timezone: str,
        use_time_filter: bool,
        time_filter_hours: set,
        params: dict,
    ) -> pd.DataFrame:
        variant   = params.get("variant",   "btc_5m_rules_90_min_votes_1")
        min_votes = int(params.get("min_votes", 1))

        rules = _RULE_SETS.get(variant)
        if rules is None:
            known = list(_RULE_SETS.keys())
            raise ValueError(
                f"Variant micro_ensemble '{variant}' inconnu. "
                f"Variants disponibles : {known}"
            )

        n = len(df)

        # Pré-charger toutes les features en arrays numpy
        feat_arrays: dict[str, np.ndarray] = {}
        for col in df.columns:
            if col not in ("dt_utc", "dt_local", "open", "high", "low", "close", "volume"):
                try:
                    feat_arrays[col] = df[col].to_numpy(dtype=float)
                except (ValueError, TypeError):
                    pass

        # Compteurs de votes par bougie (vectorisés)
        green_votes = np.zeros(n, dtype=float)
        red_votes   = np.zeros(n, dtype=float)

        for vote, conditions in rules:
            mask = np.ones(n, dtype=bool)
            for feat, op, val in conditions:
                arr = feat_arrays.get(feat)
                if arr is None:
                    mask[:] = False
                    break
                nan_mask = np.isnan(arr)
                if op == "le":
                    cond = arr <= val
                elif op == "ge":
                    cond = arr >= val
                elif op == "in":
                    cond = np.isin(arr, val)
                else:  # "eq"
                    cond = arr == val
                mask &= cond & ~nan_mask

            if vote == "GREEN":
                green_votes += mask
            else:
                red_votes += mask

        # Filtre horaire (version B)
        if use_time_filter:
            hour_arr  = feat_arrays.get("hour", np.full(n, -1))
            time_mask = np.array([h in time_filter_hours for h in hour_arr])
        else:
            time_mask = np.ones(n, dtype=bool)

        total_votes = green_votes + red_votes
        sig_up   = (green_votes > red_votes) & (total_votes >= min_votes) & time_mask
        sig_down = (red_votes > green_votes) & (total_votes >= min_votes) & time_mask

        records = []
        for i in range(n - 1):
            if not sig_up[i] and not sig_down[i]:
                continue

            direction = "UP" if sig_up[i] else "DOWN"
            row       = df.iloc[i]
            next_row  = df.iloc[i + 1]

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
                "green_votes":             int(green_votes[i]),
                "red_votes":               int(red_votes[i]),
                "signal_hour_montreal":    row["dt_local"].hour,
                "signal_weekday_montreal": row["dt_local"].strftime("%A"),
                "result":                  result,
                "next_candle_open":        next_open,
                "next_candle_close":       next_close,
            })

        trades = pd.DataFrame(records)
        n_green = sum(1 for v, _ in rules if v == "GREEN")
        n_red   = sum(1 for v, _ in rules if v == "RED")
        if trades.empty:
            print("[WARN] Aucun signal généré.")
        else:
            print(f"[INFO] {len(trades)} signaux générés "
                  f"(variant={variant}, min_votes={min_votes}, "
                  f"{n_green} GREEN / {n_red} RED rules).")
        return trades
