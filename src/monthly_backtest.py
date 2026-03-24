#!/usr/bin/env python3
"""
Backtest mensuel — découpe un CSV par mois et backteste chaque mois indépendamment.
Le capital est remis à zéro à chaque mois.

Usage:
    python src/monthly_backtest.py --input data/BTCUSDT_5m_2020-01-01_2026-01-01.csv
"""

import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from backtest import (
    load_config, get_initial_capital, get_payouts, get_enabled_mms, get_mm_cfg,
    load_data, normalize_columns, parse_timestamps,
    run_mm_simulation, compute_global_stats, compute_streak_stats,
    compute_drawdown_stats, compute_time_to_targets, compute_time_stats,
    compute_period_stats, export_reports,
    MM_KEY_MAP, CAPITAL_TARGETS, MONTREAL_TZ,
)
from strategies import get_strategy, list_strategies

warnings.filterwarnings("ignore")

MIN_CANDLES_PER_MONTH = 500   # ignorer les mois trop courts (données incomplètes)


# ============================================================
# BACKTEST D'UN SEUL MOIS
# ============================================================

def run_month(
    df_month: pd.DataFrame,
    strategy,
    versions_to_run: list,
    time_filter_hours: set,
    payouts_cfg: dict,
    payouts_to_run: list,
    mms_to_run: list,
    initial_capital: float,
    cfg: dict,
    timezone: str,
    strategy_params: dict,
) -> list:
    """
    Lance le backtest complet sur un mois.
    Retourne all_results (liste de dicts, un par version/payout/MM).
    """
    df = strategy.prepare(df_month.copy())
    ref_date = pd.Timestamp.now(tz="UTC")

    signals_cache = {}
    for version in versions_to_run:
        sig = strategy.generate_signals(
            df,
            timezone=timezone,
            use_time_filter=(version == "B"),
            time_filter_hours=time_filter_hours,
            params=strategy_params,
        )
        signals_cache[version] = sig

    all_results = []
    for version in versions_to_run:
        trades = signals_cache[version]
        if trades.empty:
            continue

        time_stats = compute_time_stats(trades)

        for payout_key in payouts_to_run:
            pw = payouts_cfg[payout_key]["win"]
            pl = payouts_cfg[payout_key]["loss"]

            for mm_name in mms_to_run:
                mm_label     = MM_KEY_MAP.get(mm_name, mm_name)
                mm_cfg_block = get_mm_cfg(cfg, mm_name)
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

    return all_results


# ============================================================
# RAPPORT GLOBAL
# ============================================================

def build_global_summary(monthly_data: list, initial_capital: float) -> str:
    """
    monthly_data : liste de dicts { "month": "2020-01", "results": all_results }
    Retourne le contenu Markdown du rapport global.
    """
    # Agrégation par stratégie (version + payout + MM)
    stats: dict[str, dict] = {}

    for entry in monthly_data:
        month = entry["month"]
        for r in entry["results"]:
            key = f"V{r['strategy_version']} | {r['payout']} | {r['mm_name']}"
            g   = r["global_stats"]
            if key not in stats:
                stats[key] = {
                    "months_tested":     0,
                    "months_profitable": 0,
                    "months_list":       [],
                    "capital_finals":    [],
                    "winrates":          [],
                    "expectancies":      [],
                    "pnl_totals":        [],
                }
            s = stats[key]
            s["months_tested"]     += 1
            s["capital_finals"].append(g.get("capital_final", initial_capital))
            s["winrates"].append(g.get("winrate", 0))
            s["expectancies"].append(g.get("expectancy", 0))
            s["pnl_totals"].append(g.get("pnl_total", 0))

            if g.get("capital_final", initial_capital) > initial_capital:
                s["months_profitable"] += 1
                s["months_list"].append(f"✓ {month}")
            else:
                s["months_list"].append(f"✗ {month}")

    if not stats:
        return "# Rapport global\n\n*Aucune donnée.*\n"

    # Tri par mois rentables décroissant, puis capital moyen décroissant
    rows = []
    for key, s in stats.items():
        n = s["months_tested"]
        rows.append({
            "strategie":          key,
            "mois_testes":        n,
            "mois_rentables":     s["months_profitable"],
            "pct_rentables":      round(s["months_profitable"] / n * 100, 1) if n else 0,
            "capital_moyen":      round(sum(s["capital_finals"]) / n, 2) if n else 0,
            "pnl_total_cumule":   round(sum(s["pnl_totals"]), 2),
            "winrate_moyen":      round(sum(s["winrates"]) / n, 2) if n else 0,
            "expectancy_moyenne": round(sum(s["expectancies"]) / n, 6) if n else 0,
            "detail_mois":        "  ".join(s["months_list"]),
        })

    rows.sort(key=lambda x: (-x["mois_rentables"], -x["capital_moyen"]))

    lines = []
    lines.append("# Rapport global — Backtest mensuel\n")
    lines.append(f"*Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    lines.append(f"- Capital initial par mois : **{initial_capital}$**")
    lines.append(f"- Mois analysés : **{len(monthly_data)}**\n")

    lines.append("## Classement des stratégies\n")
    lines.append(
        "| Stratégie | Mois testés | Mois rentables | % rentable | "
        "Capital moyen | PnL cumulé | Winrate moy. | Expectancy moy. |"
    )
    lines.append(
        "|-----------|-------------|----------------|------------|"
        "---------------|------------|--------------|-----------------|"
    )
    for r in rows:
        lines.append(
            f"| {r['strategie']} "
            f"| {r['mois_testes']} "
            f"| {r['mois_rentables']} "
            f"| {r['pct_rentables']:.1f}% "
            f"| {r['capital_moyen']:.2f}$ "
            f"| {r['pnl_total_cumule']:.2f}$ "
            f"| {r['winrate_moyen']:.2f}% "
            f"| {r['expectancy_moyenne']:.4f} |"
        )
    lines.append("")

    lines.append("## Détail mois par mois\n")
    for r in rows:
        lines.append(f"### {r['strategie']}\n")
        lines.append(r["detail_mois"] + "\n")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Backtest mensuel — un backtest par mois, capital remis à zéro"
    )
    parser.add_argument("--input",      required=True, help="Fichier CSV de données")
    parser.add_argument("--strategy",   default="streak_rsi",
                        help=f"Stratégie (défaut: streak_rsi). Disponibles : {list_strategies()}")
    parser.add_argument("--config",     default=None,  help="Fichier YAML de config MM")
    parser.add_argument("--timezone",   default=MONTREAL_TZ)
    parser.add_argument("--output-dir", default=None,
                        help="Dossier de sortie (défaut: output/monthly)")

    # Paramètres streak_rsi
    parser.add_argument("--rsi-up",        type=float, default=35.0)
    parser.add_argument("--rsi-down",      type=float, default=65.0)
    parser.add_argument("--body-ratio",    type=float, default=0.60)
    parser.add_argument("--range-mult",    type=float, default=1.0)
    parser.add_argument("--streak-min",    type=int,   default=3)
    parser.add_argument("--no-streak",     action="store_true")
    parser.add_argument("--no-rsi",        action="store_true")
    parser.add_argument("--no-range",      action="store_true")
    parser.add_argument("--no-body-ratio", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    cfg             = load_config(args.config)
    initial_capital = get_initial_capital(cfg)
    payouts_cfg     = get_payouts(cfg)
    payouts_to_run  = list(payouts_cfg.keys())
    mms_to_run      = get_enabled_mms(cfg)

    general_cfg   = cfg.get("general", {})
    versions_cfg  = general_cfg.get("versions", {})
    versions_to_run = (
        [v for v, vc in versions_cfg.items() if vc.get("enabled", True)]
        if versions_cfg else ["A", "B"]
    )
    time_filter_hours = set(
        versions_cfg.get("B", {}).get("time_filter_hours", [4, 5, 6, 7, 8, 17])
    )

    strategy = get_strategy(args.strategy)

    _base_output = Path(args.output_dir or general_cfg.get("output_dir", "output"))
    output_base = _base_output / strategy.name / "monthly"
    output_base.mkdir(parents=True, exist_ok=True)

    strategy_params = {
        "rsi_up":         args.rsi_up,
        "rsi_down":       args.rsi_down,
        "body_ratio_min": args.body_ratio,
        "range_atr_mult": args.range_mult,
        "streak_min":     args.streak_min,
        "use_streak":     not args.no_streak,
        "use_rsi":        not args.no_rsi,
        "use_range":      not args.no_range,
        "use_body_ratio": not args.no_body_ratio,
    }

    print(f"[INFO] Stratégie        : {strategy.name}")
    print(f"[INFO] Capital/mois     : {initial_capital}$")
    print(f"[INFO] Versions actives : {versions_to_run}")
    print(f"[INFO] Payouts actifs   : {payouts_to_run}")
    print(f"[INFO] MM actifs        : {[MM_KEY_MAP.get(m, m) for m in mms_to_run]}")

    # ── Chargement des données ──────────────────────────────
    df_raw = load_data(args.input)
    df     = normalize_columns(df_raw)
    df     = parse_timestamps(df, args.timezone)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    # ── Découpage par mois ─────────────────────────────────
    df["_month"] = df["dt_utc"].dt.to_period("M")
    months       = sorted(df["_month"].unique())
    print(f"[INFO] {len(months)} mois détectés\n")

    monthly_data = []

    for period in months:
        month_str  = str(period)          # "2020-01"
        df_month   = df[df["_month"] == period].drop(columns=["_month"]).reset_index(drop=True)

        if len(df_month) < MIN_CANDLES_PER_MONTH:
            print(f"[SKIP] {month_str} — seulement {len(df_month)} bougies (< {MIN_CANDLES_PER_MONTH})")
            continue

        print(f"[INFO] {month_str} — {len(df_month)} bougies ...")

        all_results = run_month(
            df_month, strategy,
            versions_to_run, time_filter_hours,
            payouts_cfg, payouts_to_run, mms_to_run,
            initial_capital, cfg, args.timezone, strategy_params,
        )

        if not all_results:
            print(f"[WARN] {month_str} — aucun signal.")
            continue

        # Export du mois
        month_dir = output_base / month_str
        export_reports(
            month_dir,
            all_results[0]["trades"],
            all_results,
            time_filter_hours,
            strategy_name=strategy.name,
            strategy_description=strategy.description,
        )
        print(f"       -> {month_dir}/rapport.md")

        monthly_data.append({"month": month_str, "results": all_results})

    # ── Rapport global ─────────────────────────────────────
    if not monthly_data:
        print("[WARN] Aucun mois traité.")
        return

    global_md   = build_global_summary(monthly_data, initial_capital)
    global_path = output_base / "summary_global.md"
    global_path.write_text(global_md, encoding="utf-8")
    print(f"\n[EXPORT] {global_path}")
    print(f"[INFO] {len(monthly_data)} mois traités.")


if __name__ == "__main__":
    main()
