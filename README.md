# 🏀 March Machine Learning Mania 2026

> **Kaggle Competition** · [Competition Page](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)  
> **Metric**: Log Loss (lower = better)  
> **Task**: Predict P(Team1 beats Team2) for every possible NCAA tournament matchup

---

## 📋 Table of Contents
1. [Competition Overview](#competition-overview)
2. [Strategy & Architecture](#strategy--architecture)
3. [Feature Engineering](#feature-engineering)
4. [Models & Hyperparameters](#models--hyperparameters)
5. [Experiment Log](#experiment-log)
6. [How to Run](#how-to-run)
7. [Results & Leaderboard](#results--leaderboard)
8. [Repository Structure](#repository-structure)
9. [Improving Future Notebooks](#improving-future-notebooks)

---

## Competition Overview

The March Machine Learning Mania competition asks competitors to predict the outcome of **every possible NCAA tournament game** (both Men's and Women's brackets) before the tournament starts.

- **Input**: Historical game data, team stats, seeds, Elo ratings, Massey rankings
- **Output**: Probability that the lower-ID team beats the higher-ID team  
- **Submission**: `ID` = `Season_Team1_Team2`, `Pred` = float in [0, 1]
- **Stage 1** (before tournament): 519,144 rows  
- **Stage 2** (live predictions): 132,133 rows  
- **Data path on Kaggle**: `/kaggle/input/competitions/march-machine-learning-mania-2026`

---

## Strategy & Architecture

```
Raw Data
  │
  ├─ Feature Engineering ──────────────────────────────────────────┐
  │    ├─ Team Season Stats (25+ metrics: SOS, SRS, AdjO/D, etc.)  │
  │    ├─ Elo Rating System (K=20, HFA=100, MoV multiplier)        │
  │    ├─ Seed Features + Tournament History                        │
  │    ├─ Recent Form (last 10 games)                               │
  │    ├─ Massey Ordinal Rankings (avg of systems ≥ day 110)        │
  │    └─ Head-to-Head History                                      │
  │                                                                  │
  └─ 5-Model Ensemble ────────────────────────────────────────────┘
       ├─ Random Forest      (Optuna 30 trials, CPU n_jobs=-1)
       ├─ XGBoost            (Optuna 100 trials, GPU hist)
       ├─ CatBoost           (Optuna 50 trials, GPU)
       ├─ LightGBM           (Optuna 100 trials, GPU)
       └─ PyTorch TabularNN  (Residual 512→256→128→64, SiLU, BN)
              │
       Ensemble (Stacking + Weight Optimization + Rank Avg)
              │
       Calibration (Isotonic / Platt, clip [0.025, 0.975])
              │
       submission.csv
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| 5-fold StratifiedKFold CV | Robust OOF estimates, prevents leakage |
| Symmetric training rows (2× per game) | Forces model to learn P(A>B) = 1 - P(B>A) |
| Elo with Margin-of-Victory | Better captures team strength than simple W/L |
| Massey ordinals (late season) | Uses consensus of 100+ ranking systems |
| Calibration after ensemble | Prevents extreme probabilities damaging log loss |
| Clip to [0.025, 0.975] | Protects against overconfident predictions |

---

## Feature Engineering

### Team Season Statistics (per team per season)
| Feature | Description |
|---|---|
| `win_pct` | Win percentage |
| `avg_score_diff` | Average point differential |
| `adj_offensive_eff` | Adjusted offensive efficiency |
| `adj_defensive_eff` | Adjusted defensive efficiency |
| `sos` | Strength of Schedule |
| `srs` | Simple Rating System |
| `fg_pct`, `fg3_pct`, `ft_pct` | Shooting percentages |
| `reb_margin`, `ast_per_game` | Rebounding & assists |
| `to_per_game`, `blk_per_game`, `stl_per_game` | Turnovers, blocks, steals |
| `pace` | Possessions per game |
| `home_win_pct`, `away_win_pct` | Location-based win rates |
| `last_10_win_pct` | Recent form (last 10 games) |
| `conf_win_pct` | Conference win rate |
| `sos_adjusted_margin` | SOS-weighted point margin |

### Elo System
- Initial rating: **1500** for all teams
- **K-factor**: 20
- **Home Field Advantage**: 100 Elo points
- **Margin of Victory multiplier**: `ln(|score_diff| + 1) × (2.2 / (winner_Elo_diff × 0.001 + 2.2))`
- Win probability: Logistic `1 / (1 + 10^(-Δelo/400))`

### Seed Features
- `seed_num_T1`, `seed_num_T2` — numeric seed (1–16)
- `seed_diff`, `seed_ratio` — differential and ratio
- Historical P(lower_seed_wins) by seed matchup

---

## Models & Hyperparameters

Each model is tuned with **Optuna TPESampler**. Best hyperparameters are logged in `experiments/`.

### Random Forest
| Parameter | Search Space | Best |
|---|---|---|
| `n_estimators` | [200, 2000] | TBD |
| `max_depth` | [5, 30] | TBD |
| `min_samples_split` | [2, 20] | TBD |
| `max_features` | [0.3, 1.0] | TBD |
| `n_jobs` | -1 (all CPUs) | -1 |

### XGBoost
| Parameter | Search Space | Best |
|---|---|---|
| `n_estimators` | [300, 2000] | TBD |
| `max_depth` | [3, 10] | TBD |
| `learning_rate` | [0.005, 0.3] | TBD |
| `subsample` | [0.5, 1.0] | TBD |
| `colsample_bytree` | [0.5, 1.0] | TBD |
| `reg_alpha` | [0, 10] | TBD |
| `reg_lambda` | [0, 10] | TBD |
| `device` | cuda/cpu | cuda |

### CatBoost
| Parameter | Search Space | Best |
|---|---|---|
| `iterations` | [300, 2000] | TBD |
| `depth` | [4, 10] | TBD |
| `learning_rate` | [0.01, 0.3] | TBD |
| `l2_leaf_reg` | [1, 10] | TBD |
| `task_type` | GPU | GPU |

### LightGBM
| Parameter | Search Space | Best |
|---|---|---|
| `n_estimators` | [300, 2000] | TBD |
| `num_leaves` | [20, 200] | TBD |
| `learning_rate` | [0.005, 0.3] | TBD |
| `min_child_samples` | [10, 100] | TBD |
| `device` | gpu/cpu | gpu |

### PyTorch TabularNN
| Component | Spec |
|---|---|
| Architecture | Residual blocks: 512→256→128→64 |
| Activation | SiLU (Swish) |
| Regularization | BatchNorm + Dropout(0.3) |
| Optimizer | AdamW |
| Scheduler | OneCycleLR |
| Epochs | 200 (early stopping patience=15) |
| Device | CUDA / MPS / CPU |

---

## Experiment Log

> Experiments are tracked in `experiments/experiment_log.md`. Each run records:
> - Notebook version / run date
> - CV Log Loss (OOF)
> - Public LB Score
> - Best hyperparameters per model
> - Ensemble weights
> - Key changes from previous run

See → **[experiments/experiment_log.md](experiments/experiment_log.md)**

---

## How to Run

### On Kaggle (Recommended)
1. Open the competition: [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
2. Create a new notebook → **Import this notebook** from GitHub or upload `march_mania_2026_rank1.ipynb`
3. Set **Accelerator** → `GPU T4 x2` (Settings panel)
4. Run All cells (`Cell → Run All`)
5. Download `submission.csv` from `/kaggle/working/`

### Locally
```bash
# 1. Clone repo
git clone https://github.com/BadalSharma007/march-machine-learning-mania-2026.git
cd march-machine-learning-mania-2026

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place data files in ./data/ (download from Kaggle)
# https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data

# 4. Run notebook
jupyter notebook march_mania_2026_rank1.ipynb
```

### Data Path
The notebook **auto-detects** the environment:
- **Kaggle**: `/kaggle/input/competitions/march-machine-learning-mania-2026` (primary)
- **Kaggle fallback**: `/kaggle/input/march-machine-learning-mania-2026`
- **Local**: `./data/` (relative to notebook)

---

## Results & Leaderboard

| Version | Run Date | CV Log Loss | Public LB | Private LB | Key Change |
|---|---|---|---|---|---|
| v1.0 | 2026-03-11 | TBD | TBD | TBD | Initial submission — baseline 5-model ensemble |

*Updated after each training run.*

---

## Repository Structure

```
march-machine-learning-mania-2026/
│
├── march_mania_2026_rank1.ipynb    # Main competition notebook
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Excludes data, models, cache
│
├── experiments/
│   ├── experiment_log.md            # Full experiment history (scores, params)
│   └── v1.0_2026-03-11/
│       └── params.json              # Best hyperparameters from first run
│
└── data/                            # ⚠️ Not committed (add your own)
    ├── MRegularSeasonDetailedResults.csv
    ├── MNCAATourneySeeds.csv
    └── ... (all Kaggle data files)
```

---

## Improving Future Notebooks

This repo is designed so that **each trained notebook informs the next**. Here's how:

### After Each Training Run:
1. **Update `experiments/experiment_log.md`** with:
   - OOF CV log loss per model
   - Final ensemble log loss
   - Public LB score (after submission)
   - Best Optuna hyperparameters (auto-saved to `experiments/vX.X_DATE/params.json`)
   - Feature importances (top 20)
   - Calibration method used

2. **Push the trained notebook** (`march_mania_2026_rank1.ipynb` with outputs) to GitHub

3. **Read the experiment log** before starting the next version to:
   - Identify which model contributed most to the ensemble
   - Spot underperforming features to remove or improve
   - Use the best hyperparameters as Optuna starting points (via `add_trial`)
   - Adjust ensemble weights based on OOF performance

### Improvement Checklist
- [ ] Add `MTeamCoaches.csv` tenure/experience features
- [ ] Tune Elo K-factor per era (pre/post 2010 → different basketball styles)
- [ ] Add ExtraTreesClassifier as 6th ensemble model
- [ ] Increase Optuna trials: RF 30→50, XGB/LGB 100→200
- [ ] Add `VotingClassifier` as alternative meta-learner
- [ ] Experiment with temperature scaling for calibration
- [ ] Add conference tournament results as features
- [ ] Incorporate KenPom-style adjusted efficiency margins
- [ ] Try stratified split by both season AND gender for CV

---

## License

This repository is for educational and competition purposes.  
Competition data © Kaggle / NCAA. 

---

*Maintained by [@BadalSharma007](https://github.com/BadalSharma007)*
