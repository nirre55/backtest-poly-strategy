# backtest-poly-strategy

Backtest d'une stratégie de signal sur BTCUSDT M5, simulant des paris binaires style **Polymarket** (UP / DOWN sur la prochaine bougie).

> **Modèle de payout :**
> - Tu mises 1$ → bougie gagnante = **+0.90$** de profit net
> - Tu mises 1$ → bougie perdante = **-1.00$** de perte nette
> - Identique à un marché de prédiction : tu ne récupères pas ta mise, tu gagnes ou tu perds.

---

## Structure du projet

```
backtest-poly-strategy/
├── config/
│   └── money_management.yaml     # Paramètres de tous les MM (éditable)
├── data/                          # Données CSV téléchargées
├── output/                        # Résultats et exports
│   ├── monthly/                   # Résultats du backtest mensuel
│   └── monte_carlo/               # Fichiers CSV mélangés (Monte Carlo)
├── src/
│   ├── backtest.py                # Moteur de backtest principal
│   ├── monthly_backtest.py        # Backtest mois par mois
│   ├── monte_carlo.py             # Générateur de données Monte Carlo
│   ├── download_data.py           # Téléchargement des données Binance
│   └── strategies/
│       ├── base.py                # Classe abstraite BaseStrategy
│       └── streak_rsi.py          # Stratégie : streak + RSI + ATR + body ratio
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Créer l'environnement virtuel

```bash
# Avec uv (recommandé)
uv venv .venv

# Ou avec Python standard
python -m venv .venv
```

### 2. Activer l'environnement

```bash
# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

> **Important** : toujours utiliser le Python du venv pour lancer les scripts :
> `.venv\Scripts\python src/...` (Windows) ou `.venv/bin/python src/...` (Mac/Linux)

---

## Scripts

### `download_data.py` — Télécharger les données

Télécharge les bougies depuis l'**API publique Binance** (gratuit, sans clé API).

```bash
# 5 ans de données BTCUSDT M5
.venv\Scripts\python src/download_data.py --start 2020-01-01 --end 2025-01-01

# Résultat : data/BTCUSDT_5m_2020-01-01_2025-01-01.csv
```

| Option | Défaut | Description |
|--------|--------|-------------|
| `--start` | *requis* | Date de début `YYYY-MM-DD` |
| `--end` | *requis* | Date de fin `YYYY-MM-DD` (exclusif) |
| `--symbol` | `BTCUSDT` | Paire de trading |
| `--interval` | `5m` | Timeframe : `1m` `3m` `5m` `15m` `30m` `1h` `4h` `1d` |
| `--output` | `data/<SYMBOL>_<INTERVAL>_<start>_<end>.csv` | Chemin de sortie |
| `--pause` | `0.3` | Secondes entre chaque requête (évite le rate limit) |

> Durée estimée pour 5 ans de M5 : ~3–5 minutes (~525 000 bougies)

---

### `backtest.py` — Backtest principal

Lance le backtest complet sur un fichier CSV. Teste toutes les combinaisons de versions (A/B), payouts et money managements activés dans le YAML.

```bash
# Commande minimale — teste tout ce qui est activé dans le YAML
.venv\Scripts\python src/backtest.py --input data/BTCUSDT_5m_2020-01-01_2025-01-01.csv
```

#### Options principales

| Option | Défaut | Description |
|--------|--------|-------------|
| `--input` | *requis* | Fichier CSV de données |
| `--strategy` | `streak_rsi` | Stratégie de signal à utiliser |
| `--config` | `config/money_management.yaml` | Fichier de configuration MM |
| `--payout` | `all` | Payout à tester : `A`, `B`, `C` ou `all` |
| `--mm` | `all` | Money management à tester : `MM1`…`MM11` ou `all` |
| `--output-dir` | `output/` | Dossier de sortie |
| `--split` | — | Fraction train ex: `0.7` → teste sur les 30% les plus récents |
| `--time-filter` | — | Force la Version B uniquement (filtre horaire) |
| `--inverse` | — | Inverse les signaux : UP → DOWN et DOWN → UP |

#### Ajuster les seuils du signal

| Option | Défaut | Description |
|--------|--------|-------------|
| `--rsi-up` | `35.0` | Seuil RSI7 pour signal UP |
| `--rsi-down` | `65.0` | Seuil RSI7 pour signal DOWN |
| `--streak-min` | `3` | Streak minimum |
| `--body-ratio` | `0.60` | Body ratio minimum |
| `--range-mult` | `1.0` | Multiplicateur ATR pour le filtre range |
| `--no-rsi` | — | Désactiver le filtre RSI |
| `--no-streak` | — | Désactiver le filtre streak |
| `--no-range` | — | Désactiver le filtre range/ATR |
| `--no-body-ratio` | — | Désactiver le filtre body ratio |

#### Exemples

```bash
# Tester un seul MM et un seul payout
.venv\Scripts\python src/backtest.py --input data/... --mm MM7 --payout A

# Split train/test 70/30
.venv\Scripts\python src/backtest.py --input data/... --split 0.7

# Seuils personnalisés
.venv\Scripts\python src/backtest.py --input data/... --rsi-up 30 --rsi-down 70 --streak-min 4

# Dossier de sortie personnalisé
.venv\Scripts\python src/backtest.py --input data/... --output-dir output_strict/

# Inverser les signaux (tester la stratégie à l'envers)
.venv\Scripts\python src/backtest.py --input data/... --inverse
```

#### Fichiers générés dans `output/`

```
output/
├── trades_version_A.csv       # Tous les trades signalés (version A)
├── trades_version_B.csv       # Tous les trades signalés (version B)
├── sim_A_payoutA_MM1_flat_fixed.csv   # Simulation trade par trade
├── summary_all.csv            # Tableau comparatif de toutes les combinaisons
├── stats_by_hour.csv          # Winrate par heure (Montréal)
├── stats_by_weekday.csv       # Winrate par jour de la semaine
├── stats_by_day_hour.csv      # Winrate par combinaison jour + heure
├── all_metrics.json           # Toutes les métriques en JSON
└── rapport.md                 # Rapport comparatif complet en Markdown
```

---

### `monthly_backtest.py` — Backtest mensuel

Découpe automatiquement le CSV par mois et lance un backtest indépendant sur chaque mois. Le capital repart à zéro à chaque mois. Génère un rapport global qui classe les stratégies selon leur rentabilité sur l'ensemble des mois.

```bash
# Lancer sur 5 ans de données
.venv\Scripts\python src/monthly_backtest.py --input data/BTCUSDT_5m_2020-01-01_2026-01-01.csv
```

| Option | Défaut | Description |
|--------|--------|-------------|
| `--input` | *requis* | Fichier CSV de données |
| `--strategy` | `streak_rsi` | Stratégie de signal à utiliser |
| `--config` | `config/money_management.yaml` | Fichier de configuration MM |
| `--output-dir` | `output/monthly/` | Dossier de sortie |

> Accepte les mêmes options de seuils que `backtest.py` (`--rsi-up`, `--streak-min`, etc.)

#### Fichiers générés dans `output/monthly/`

```
output/monthly/
├── 2020-01/
│   ├── rapport.md             # Rapport du mois
│   ├── summary_all.csv        # Tableau comparatif du mois
│   └── sim_*.csv              # Simulations du mois
├── 2020-02/
│   └── ...
└── summary_global.md          # Classement global sur tous les mois
```

#### Lire `summary_global.md`

Le tableau de classement indique pour chaque stratégie :

| Colonne | Description |
|---------|-------------|
| `Mois rentables` | Nombre de mois où le capital final > capital initial |
| `% rentable` | Pourcentage de mois rentables |
| `Capital moyen` | Capital moyen en fin de mois |
| `PnL cumulé` | Somme des PnL sur tous les mois |
| `Winrate moy.` | Winrate moyen sur tous les mois |

La section **Détail mois par mois** montre ✓ (rentable) ou ✗ (perte) pour chaque mois.

---

### `monte_carlo.py` — Simulation Monte Carlo

Génère des fichiers CSV avec les timestamps dans l'ordre chronologique original mais les bougies OHLCV redistribuées aléatoirement. Chaque bougie reste cohérente (ses valeurs open/high/low/close/volume restent ensemble), seul l'ordre est mélangé.

**Utilité** : si ta stratégie performe aussi bien sur les données mélangées que sur les données réelles, son edge est probablement dû au hasard.

```bash
# Générer 1 fichier mélangé
.venv\Scripts\python src/monte_carlo.py --input data/BTCUSDT_5m_2020-01-01_2026-01-01.csv

# Générer 100 fichiers mélangés
.venv\Scripts\python src/monte_carlo.py --input data/BTCUSDT_5m_2020-01-01_2026-01-01.csv --n 100

# Seed fixe (résultat identique à chaque exécution)
.venv\Scripts\python src/monte_carlo.py --input data/... --n 100 --seed 42
```

| Option | Défaut | Description |
|--------|--------|-------------|
| `--input` | *requis* | Fichier CSV source (OHLCV) |
| `--n` | `1` | Nombre de fichiers mélangés à générer |
| `--seed` | aléatoire | Seed pour reproductibilité |
| `--output-dir` | `output/monte_carlo/` | Dossier de sortie |

> Chaque itération repart du **fichier original** — ce n'est pas un mélange de mélange.

#### Workflow Monte Carlo complet

```bash
# 1. Générer 50 fichiers mélangés
.venv\Scripts\python src/monte_carlo.py --input data/BTCUSDT_5m_2020-01-01_2026-01-01.csv --n 50

# 2. Lancer le backtest sur chaque fichier mélangé
.venv\Scripts\python src/backtest.py --input output/monte_carlo/BTCUSDT_5m_2020-01-01_2026-01-01_mc0001.csv
.venv\Scripts\python src/backtest.py --input output/monte_carlo/BTCUSDT_5m_2020-01-01_2026-01-01_mc0002.csv
# ...

# 3. Comparer les résultats avec le backtest original
.venv\Scripts\python src/backtest.py --input data/BTCUSDT_5m_2020-01-01_2026-01-01.csv
```

---

## Configuration — `config/money_management.yaml`

### Paramètres des stratégies de signal

Chaque stratégie a sa propre section dans le YAML. Les valeurs du YAML servent de base — le CLI les surcharge si tu passes une valeur différente du défaut.

```yaml
signal_strategies:

  streak_rsi:
    rsi_up: 35.0          # RSI7 ≤ seuil → signal UP
    rsi_down: 65.0        # RSI7 ≥ seuil → signal DOWN
    streak_min: 3         # bougies consécutives minimum
    body_ratio_min: 0.60  # corps / range minimum
    range_atr_mult: 1.0   # range ≥ mult × ATR14
    use_streak: true
    use_rsi: true
    use_range: true
    use_body_ratio: true

  wick_volume_rebound:
    rsi_oversold: 30.0    # RSI7 < seuil → condition UP
    rsi_overbought: 70.0  # RSI7 > seuil → condition DOWN
    wick_body_mult: 1.5   # mèche ≥ mult × corps
    vol_ma_mult: 1.25     # volume ≥ mult × vma20

  wick_momentum:
    rej_vol_mult: 1.5     # volume min Rejet (× vma20)
    rej_wick_mult: 2.0    # mèche min Rejet (× corps)
    mom_vol_mult: 2.5     # volume min Momentum (× vma20)
    mom_body_ratio: 0.8   # corps min Momentum (× range)

  sniper:
    vol_mult: 4.0         # volume ≥ mult × vma20
    wick_mult: 3.0        # mèche ≥ mult × corps

  momentum:
    threshold_pct: 0.2    # variation minimum en % pour déclencher un signal
```

> **Priorité :** CLI > YAML > valeur par défaut du code.
> Exemple : `--rsi-up 30` surcharge le `rsi_up: 35.0` du YAML.

---

### Versions A et B

```yaml
general:
  versions:
    A:
      enabled: true   # tous les signaux, sans filtre horaire
    B:
      enabled: true   # signaux filtrés selon time_filter_hours
      time_filter_hours: [4, 5, 6, 7, 8, 17]   # heures Montréal autorisées
```

- **Version A** : tous les signaux pris, 24h/24
- **Version B** : seulement les signaux dont l'heure (timezone Montréal) est dans la liste

### Payouts

```yaml
  payouts:
    A:
      enabled: true
      win_payout:  0.90
      loss_payout: -1.00
    B:
      enabled: false    # désactivé
      win_payout:  0.95
      loss_payout: -1.00
```

### Money managements

| MM | Nom complet | Paramètres clés |
|----|-------------|-----------------|
| MM1 | `MM1_flat_fixed` | `base_stake` |
| MM2 | `MM2_fixed_1pct` | `fraction_pct` |
| MM3 | `MM3_fixed_5pct` | `fraction_pct` |
| MM4 | `MM4_martingale_classic` | `base_stake` |
| MM5 | `MM5_martingale_linear` | `base_stake`, `increment` |
| MM6 | `MM6_martingale_limited` | `sequence`, `pause_trades` |
| MM7 | `MM7_anti_martingale` | `base_fraction_pct`, `max_fraction_pct`, `win_multipliers` |
| MM8 | `MM8_reduction_after_losses` | `base_fraction_pct`, `loss_steps` |
| MM9 | `MM9_pause_after_losses` | `base_fraction_pct`, `pause_after_n_losses`, `pause_trades` |
| MM10 | `MM10_combined` | tout ce qui précède combiné |
| MM11 | `MM11_alternating` | `base_fraction_pct`, `odd_loss_fraction_pct`, `even_loss_fraction_pct` |

---

## Stratégies disponibles

Le signal est toujours calculé sur la **bougie fermée i**, le trade pris sur la **bougie i+1**.
- Bougie i+1 verte + signal UP → **win**
- Bougie i+1 rouge + signal DOWN → **win**
- Bougie neutre (open == close) → **ignorée**

---

### `streak_rsi` — Streak + RSI + ATR + Body ratio

Détecte un épuisement directionnel : X bougies consécutives dans le même sens, confirmé par un RSI tendu, un range actif et un corps solide.

| Signal | Conditions |
|--------|-----------|
| UP | streak rouge ≥ 3 · RSI7 ≤ 35 · range ≥ 1×ATR14 · body ratio ≥ 0.60 |
| DOWN | streak vert ≥ 3 · RSI7 ≥ 65 · range ≥ 1×ATR14 · body ratio ≥ 0.60 |

```bash
.venv\Scripts\python src/backtest.py --input data/... --strategy streak_rsi
# Options : --rsi-up, --rsi-down, --streak-min, --body-ratio, --range-mult
#           --no-rsi, --no-streak, --no-range, --no-body-ratio
```

---

### `wick_volume_rebound` — Mèche + RSI + Volume

Détecte un rejet de prix violent : mèche longue (au moins 1.5× le corps) sur une bougie survendue/surachetée avec un volume supérieur à la normale.

| Signal | Conditions |
|--------|-----------|
| UP | bougie rouge · RSI7 < 30 · mèche basse > body×1.5 · volume > vma20×1.25 |
| DOWN | bougie verte · RSI7 > 70 · mèche haute > body×1.5 · volume > vma20×1.25 |

```bash
.venv\Scripts\python src/backtest.py --input data/... --strategy wick_volume_rebound
```

---

### `wick_momentum` — Rejet de mèche OU Momentum volume

Deux chemins par direction. Le **Rejet** cible un renversement après un rejet violent (mèche ≥ 2× body, volume ≥ 1.5×). Le **Momentum** cible une continuation explosive (corps ≥ 80% du range, volume ≥ 2.5×).

| Signal | Chemin | Conditions |
|--------|--------|-----------|
| UP | Rejet | bougie rouge · volume > vma20×1.5 · mèche basse > body×2 |
| UP | Momentum | bougie verte · volume > vma20×2.5 · body > range×0.8 |
| DOWN | Rejet | bougie verte · volume > vma20×1.5 · mèche haute > body×2 |
| DOWN | Momentum | bougie rouge · volume > vma20×2.5 · body > range×0.8 |

La colonne `signal_type` dans les CSV de trades indique `rejet` ou `momentum` pour chaque signal.

```bash
.venv\Scripts\python src/backtest.py --input data/... --strategy wick_momentum
```

---

### `sniper` — Rejet Extrême (volume flash + mèche massive)

Stratégie très sélective visant un winrate maximum (~68%). Ne se déclenche que sur des configurations rares : volume 4× la normale ET mèche de rejet 3× le corps. Génère peu de signaux mais de haute conviction.

| Signal | Conditions |
|--------|-----------|
| UP | bougie rouge · volume > vma20×4.0 · mèche basse > corps×3.0 |
| DOWN | bougie verte · volume > vma20×4.0 · mèche haute > corps×3.0 |

La colonne `vol_ratio` dans les CSV de trades indique le ratio volume/vma20 de chaque signal.

```bash
.venv\Scripts\python src/backtest.py --input data/... --strategy sniper
```

---

### `momentum` — Suivi de tendance

Stratégie simple qui suit la force du marché : si la bougie actuelle est suffisamment forte dans un sens, on parie que la suivante continue dans la même direction. Génère beaucoup de trades, winrate attendu ~54%.

| Signal | Conditions |
|--------|-----------|
| UP | bougie verte · variation > +0.2% |
| DOWN | bougie rouge · variation < -0.2% |

La colonne `variation_pct` dans les CSV de trades indique le % de variation de la bougie signal.

```bash
.venv\Scripts\python src/backtest.py --input data/... --strategy momentum
# Ajuster le seuil de variation (défaut 0.2%) :
# paramètre threshold_pct dans strategy_params
```

---

### `alternating` — Alternance VERTE/ROUGE avec switch de phase

Stratégie systématique sans indicateur : on alterne les prédictions VERTE/ROUGE à chaque bougie. La phase de départ est déterminée par la première bougie du fichier (opposé de sa couleur). Dès que `loss_streak_switch` pertes consécutives sont atteintes (défaut : 2), la phase s'inverse et l'alternance repart depuis la nouvelle phase.

| Paramètre | Défaut | Description |
|---|---|---|
| `use_loss_streak_switch` | `true` | `false` = alternance pure, sans jamais switcher |
| `loss_streak_switch` | `2` | Nombre de pertes consécutives avant d'inverser la phase |

```bash
.venv\Scripts\python src/backtest.py --input data/... --strategy alternating
```

> **Note :** cette stratégie évalue chaque bougie à sa fermeture (pas de décalage i+1). Elle couvre 100% des bougies non-neutres du fichier.

---

## Ajouter une nouvelle stratégie

1. Créer `src/strategies/ma_strategie.py` :

```python
from .base import BaseStrategy

class MaStrategie(BaseStrategy):
    name = "ma_strategie"
    description = "..."

    def prepare(self, df):
        # Calculer les indicateurs nécessaires
        return df

    def generate_signals(self, df, timezone, use_time_filter, time_filter_hours, params):
        # Retourner un DataFrame avec les colonnes obligatoires :
        # signal_time, entry_time, direction, result,
        # signal_hour_montreal, signal_weekday_montreal,
        # next_candle_open, next_candle_close
        ...
```

2. L'enregistrer dans `src/strategies/__init__.py` :

```python
from .ma_strategie import MaStrategie
REGISTRY["ma_strategie"] = MaStrategie
```

3. L'utiliser :

```bash
.venv\Scripts\python src/backtest.py --input data/... --strategy ma_strategie
```

---

## Interpréter les résultats

### Expectancy — la métrique clé

```
expectancy = winrate × win_payout + lossrate × loss_payout
```

- `> 0` → stratégie profitable sur le long terme
- `= 0` → point mort
- `< 0` → stratégie perdante

**Exemple avec payout A (+0.90 / -1.00) :**
- Seuil de rentabilité : winrate > **52.6%**
- À 57% de winrate : expectancy ≈ +0.083$ par trade

### Distribution des séries

Les séries sont comptées à leur longueur totale. Exemple : `loss, loss, loss, win` = **une** série de longueur 3, pas trois séries de longueurs 1, 2 et 3.

### Choisir son money management

| Objectif | MM recommandé |
|----------|--------------|
| Croissance maximale (risque élevé) | MM4, MM6 |
| Croissance régulière (risque modéré) | MM3, MM7, MM10 |
| Survie avant tout | MM8, MM9, MM11 |
| Référence neutre | MM1 (mise fixe 1$) |

> MM4 (martingale) : la ruine arrive quand la mise dépasse le capital disponible, pas automatiquement après N pertes. Avec suffisamment de capital accumulé, on peut survivre à 13 pertes consécutives.

### Seuil de liquidation (`min_capital`)

Par défaut, toute simulation est arrêtée dès que le capital tombe **sous 1$** — la stratégie est considérée comme liquidée. Ce seuil est configurable par MM dans le YAML :

```yaml
MM11_alternating:
  enabled: true
  min_capital: 5.0   # arrêt si capital < 5$
  ...
```

Ce comportement s'applique à tous les MM, y compris ceux en pourcentage fixe qui sinon continueraient à trader avec des fractions infinitésimales de capital.
