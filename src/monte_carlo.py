#!/usr/bin/env python3
"""
Monte Carlo — mélange aléatoire des données OHLCV.

Principe :
  - Les timestamps restent dans l'ordre chronologique original.
  - Toutes les autres colonnes (open, high, low, close, volume) sont mélangées :
    on redistribue aléatoirement les bougies existantes sur les timestamps.
    Chaque bougie reste cohérente (high >= low, etc.) car on mélange des LIGNES entières,
    pas des colonnes séparément.
  - Résultat : même distribution statistique des bougies, mais sans aucune dépendance
    temporelle. Si ta stratégie performe aussi bien sur ces données mélangées,
    son edge est probablement dû au hasard.

Usage :
    # 1 fichier mélangé (seed aléatoire)
    python src/monte_carlo.py --input data/BTCUSDT_5m_2020-01-01_2026-01-01.csv

    # 50 fichiers mélangés dans output/monte_carlo/
    python src/monte_carlo.py --input data/BTCUSDT_5m_2020-01-01_2026-01-01.csv --n 50

    # Seed fixe pour reproduire le même mélange
    python src/monte_carlo.py --input data/... --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Monte Carlo — mélange aléatoire des bougies OHLCV"
    )
    parser.add_argument("--input",      required=True,
                        help="Fichier CSV source (OHLCV)")
    parser.add_argument("--n",          type=int, default=1,
                        help="Nombre de fichiers mélangés à générer (défaut: 1)")
    parser.add_argument("--seed",       type=int, default=None,
                        help="Seed aléatoire pour reproductibilité (optionnel)")
    parser.add_argument("--output-dir", default=None,
                        help="Dossier de sortie (défaut: output/monte_carlo/)")
    return parser.parse_args()


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"[ERREUR] Fichier introuvable : {path}")
    df = pd.read_csv(path)
    if df.empty:
        sys.exit("[ERREUR] Le fichier CSV est vide.")
    print(f"[INFO] {len(df)} lignes chargees depuis {path}")
    return df


def detect_timestamp_col(df: pd.DataFrame) -> str:
    """Retourne le nom de la colonne timestamp."""
    candidates = ["timestamp", "time", "date", "datetime", "open_time"]
    for c in df.columns:
        if c.strip().lower() in candidates:
            return c
    sys.exit("[ERREUR] Colonne timestamp introuvable. Colonnes disponibles : "
             + ", ".join(df.columns.tolist()))


def shuffle_ohlcv(df: pd.DataFrame, ts_col: str, rng: np.random.Generator) -> pd.DataFrame:
    """
    Garde la colonne timestamp dans l'ordre original.
    Mélange toutes les autres colonnes en redistribuant les lignes aléatoirement.
    """
    ohlcv_cols = [c for c in df.columns if c != ts_col]
    ohlcv_data = df[ohlcv_cols].copy()

    # Mélange des indices → redistribution des bougies sur les timestamps
    shuffled_idx = rng.permutation(len(ohlcv_data))
    ohlcv_shuffled = ohlcv_data.iloc[shuffled_idx].reset_index(drop=True)

    result = df[[ts_col]].reset_index(drop=True).copy()
    for col in ohlcv_cols:
        result[col] = ohlcv_shuffled[col].values

    return result


def main():
    args = parse_args()

    output_dir = Path(args.output_dir or "output/monte_carlo")
    output_dir.mkdir(parents=True, exist_ok=True)

    df     = load_csv(args.input)
    ts_col = detect_timestamp_col(df)
    stem   = Path(args.input).stem   # ex: "BTCUSDT_5m_2020-01-01_2026-01-01"

    print(f"[INFO] Colonne timestamp  : {ts_col}")
    print(f"[INFO] Colonnes melangees : {[c for c in df.columns if c != ts_col]}")
    print(f"[INFO] Iterations         : {args.n}")
    print(f"[INFO] Seed               : {args.seed if args.seed is not None else 'aleatoire'}")
    print()

    # Seed global : chaque itération dérive du seed de base
    base_seed = args.seed if args.seed is not None else None
    rng_master = np.random.default_rng(base_seed)

    for i in range(1, args.n + 1):
        # Seed dérivé pour cette itération (reproductible si --seed fourni)
        iter_seed = int(rng_master.integers(0, 2**31))
        rng       = np.random.default_rng(iter_seed)

        shuffled  = shuffle_ohlcv(df, ts_col, rng)

        suffix    = f"_mc{i:04d}" if args.n > 1 else "_mc"
        out_path  = output_dir / f"{stem}{suffix}.csv"
        shuffled.to_csv(out_path, index=False)

        print(f"[{i:>{len(str(args.n))}}/{args.n}] seed={iter_seed:<12} -> {out_path}")

    print(f"\n[INFO] {args.n} fichier(s) genere(s) dans {output_dir}/")
    if args.n > 1:
        print(f"[INFO] Pour comparer, lance le backtest sur chaque fichier :")
        print(f"       .venv\\Scripts\\python src/backtest.py --input output/monte_carlo/{stem}_mc0001.csv")


if __name__ == "__main__":
    main()
