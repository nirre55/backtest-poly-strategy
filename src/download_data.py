#!/usr/bin/env python3
"""
Téléchargement de données OHLCV depuis l'API publique Binance.

Usage:
    python download_data.py --start 2020-01-01 --end 2025-01-01
    python download_data.py --start 2020-01-01 --end 2025-01-01 --symbol ETHUSDT --interval 15m
    python download_data.py --start 2020-01-01 --end 2025-01-01 --output ../data/BTCUSDT_M5.csv
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ============================================================
# CONSTANTES
# ============================================================
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_CANDLES_PER_REQUEST = 1000

INTERVAL_MS = {
    "1m":  60_000,
    "3m":  3 * 60_000,
    "5m":  5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h":  3_600_000,
    "4h":  4 * 3_600_000,
    "1d":  86_400_000,
}

COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

EXPORT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


# ============================================================
# FONCTIONS
# ============================================================

def parse_date(date_str: str) -> int:
    """Convertit une date 'YYYY-MM-DD' en timestamp Unix milliseconds UTC."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_chunk(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    retries: int = 3,
) -> list:
    """
    Télécharge jusqu'à 1000 bougies depuis Binance entre start_ms et end_ms.
    Relance automatiquement en cas d'erreur réseau.
    """
    params = {
        "symbol":    symbol,
        "interval":  interval,
        "startTime": start_ms,
        "endTime":   end_ms,
        "limit":     MAX_CANDLES_PER_REQUEST,
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            status = resp.status_code if resp else "?"
            if status == 429:
                # Rate limit : attendre avant de réessayer
                wait = 60
                print(f"  [WARN] Rate limit (429). Attente {wait}s...")
                time.sleep(wait)
            elif attempt < retries:
                print(f"  [WARN] Erreur HTTP {status}, tentative {attempt}/{retries}...")
                time.sleep(2 ** attempt)
            else:
                sys.exit(f"[ERREUR] HTTP {status} : {e}")
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                print(f"  [WARN] Erreur réseau, tentative {attempt}/{retries} : {e}")
                time.sleep(2 ** attempt)
            else:
                sys.exit(f"[ERREUR] Impossible de contacter Binance : {e}")

    return []


def download(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    pause_ms: float = 0.3,
) -> list:
    """
    Télécharge toutes les bougies entre start_ms et end_ms par chunks de 1000.
    Affiche une barre de progression simple.
    """
    candle_ms = INTERVAL_MS.get(interval)
    if candle_ms is None:
        sys.exit(f"[ERREUR] Intervalle non supporté : {interval}. "
                 f"Choix : {list(INTERVAL_MS.keys())}")

    total_candles_est = (end_ms - start_ms) // candle_ms
    all_rows = []
    cursor = start_ms
    chunk_num = 0

    print(f"[INFO] Téléchargement {symbol} {interval} | "
          f"~{total_candles_est:,} bougies estimées")
    print(f"[INFO] De {_ms_to_str(start_ms)} à {_ms_to_str(end_ms)}")

    while cursor < end_ms:
        chunk_end = min(cursor + candle_ms * MAX_CANDLES_PER_REQUEST - 1, end_ms)
        data = fetch_chunk(symbol, interval, cursor, chunk_end)

        if not data:
            break

        all_rows.extend(data)
        chunk_num += 1
        last_ts = data[-1][0]
        progress = min((last_ts - start_ms) / (end_ms - start_ms) * 100, 100)

        print(f"  chunk #{chunk_num:4d} | {len(data):4d} bougies | "
              f"jusqu'au {_ms_to_str(last_ts)} | {progress:.1f}%",
              end="\r")

        # Avancer le curseur après la dernière bougie reçue
        cursor = last_ts + candle_ms

        # Pause pour éviter le rate limit Binance
        time.sleep(pause_ms)

    print()  # newline après la barre \r
    print(f"[INFO] {len(all_rows):,} bougies téléchargées.")
    return all_rows


def build_dataframe(raw: list) -> pd.DataFrame:
    """Convertit les données brutes Binance en DataFrame propre."""
    df = pd.DataFrame(raw, columns=COLUMNS)

    # Garder uniquement les colonnes utiles
    df = df[EXPORT_COLUMNS].copy()

    # Convertir les types
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    # Supprimer les doublons et trier
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    return df


def _ms_to_str(ms: int) -> str:
    """Formate un timestamp ms en string lisible."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Téléchargement de données OHLCV depuis Binance"
    )
    parser.add_argument(
        "--start", required=True,
        help="Date de début (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", required=True,
        help="Date de fin (YYYY-MM-DD, exclusif)"
    )
    parser.add_argument(
        "--symbol", default="BTCUSDT",
        help="Paire de trading (défaut: BTCUSDT)"
    )
    parser.add_argument(
        "--interval", default="5m",
        choices=list(INTERVAL_MS.keys()),
        help="Timeframe (défaut: 5m)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Chemin du CSV de sortie (défaut: data/<SYMBOL>_<INTERVAL>.csv)"
    )
    parser.add_argument(
        "--pause", type=float, default=0.3,
        help="Pause en secondes entre chaque requête (défaut: 0.3)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    start_ms = parse_date(args.start)
    end_ms   = parse_date(args.end)

    if start_ms >= end_ms:
        sys.exit("[ERREUR] La date de début doit être strictement antérieure à la date de fin.")

    # Chemin de sortie
    if args.output:
        output_path = Path(args.output)
    else:
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{args.symbol}_{args.interval}_{args.start}_{args.end}.csv"
        output_path = data_dir / fname

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Téléchargement
    raw = download(args.symbol, args.interval, start_ms, end_ms, pause_ms=args.pause)

    if len(raw) == 0:
        sys.exit("[ERREUR] Aucune donnée téléchargée.")

    df = build_dataframe(raw)

    # Informations finales
    print(f"[INFO] Plage effective : {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
    print(f"[INFO] Bougies totales : {len(df):,}")
    print(f"[INFO] Colonnes        : {list(df.columns)}")

    # Export CSV
    df.to_csv(output_path, index=False)
    print(f"[OK]   Fichier sauvegardé : {output_path.resolve()}")
    print()
    print("Commande backtest :")
    print(f"  python src/backtest.py --input {output_path.resolve()}")


if __name__ == "__main__":
    main()
