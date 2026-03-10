# 📊 Experiment Log — March Machine Learning Mania 2026

This file tracks every training run. Read this before starting a new notebook version to build on previous results.

---

## How to Use This Log

1. **Before training**: Read the latest entry to understand what worked and what to try next.
2. **After training**: Fill in the `TBD` fields with actual values from your notebook output.
3. **Push trained notebook**: `git add march_mania_2026_rank1.ipynb && git commit -m "v1.0: trained notebook with outputs"`

---

## Template (copy for each new run)

```
---
### vX.X — YYYY-MM-DD
**Notebook**: `march_mania_2026_rank1.ipynb` (vX.X)
**Git Commit**: (commit hash)
**Kaggle GPU**: T4 x2 / P100 / None

#### Scores
| Metric | Value |
|---|---|
| CV OOF Log Loss (overall) | |
| RF OOF Log Loss | |
| XGBoost OOF Log Loss | |
| CatBoost OOF Log Loss | |
| LightGBM OOF Log Loss | |
| PyTorch NN OOF Log Loss | |
| Ensemble OOF Log Loss | |
| Calibrated OOF Log Loss | |
| Public LB Log Loss | |
| Private LB Log Loss | |

#### Best Hyperparameters
See `experiments/vX.X_YYYY-MM-DD/params.json`

#### Top 10 Feature Importances (XGBoost)
| Rank | Feature | Importance |
|---|---|---|
| 1 | | |

#### Ensemble Weights
| Model | Weight |
|---|---|
| RF | |
| XGBoost | |
| CatBoost | |
| LightGBM | |
| PyTorch NN | |

#### Calibration
- Method used: Isotonic / Platt
- Pre-calibration log loss: 
- Post-calibration log loss: 
- Clipping range: [0.025, 0.975]

#### Key Changes from Previous Version
- (describe changes)

#### What to Try Next
- (ideas based on this run's results)
```

---

## Run History

---

### v1.0 — 2026-03-11
**Notebook**: `march_mania_2026_rank1.ipynb` (v1.0)  
**Git Commit**: TBD (push after training)  
**Kaggle GPU**: T4 x2 (recommended)  
**Status**: 🟡 Submitted — awaiting training results

#### Scores
| Metric | Value |
|---|---|
| CV OOF Log Loss (overall) | TBD |
| RF OOF Log Loss | TBD |
| XGBoost OOF Log Loss | TBD |
| CatBoost OOF Log Loss | TBD |
| LightGBM OOF Log Loss | TBD |
| PyTorch NN OOF Log Loss | TBD |
| Ensemble OOF Log Loss | TBD |
| Calibrated OOF Log Loss | TBD |
| Public LB Log Loss | TBD |
| Private LB Log Loss | TBD |

#### Model Configuration
| Model | Optuna Trials | GPU | Notes |
|---|---|---|---|
| Random Forest | 30 | CPU (n_jobs=-1) | scikit-learn, CPU-only |
| XGBoost | 100 | CUDA (gpu_hist) | device=cuda |
| CatBoost | 50 | GPU | task_type=GPU |
| LightGBM | 100 | GPU | device=gpu |
| PyTorch NN | N/A | CUDA/MPS | Residual 512→256→128→64 |

#### Feature Groups
| Group | # Features | Key Signals |
|---|---|---|
| Team Season Stats | ~25 per team | win_pct, adj_eff, sos, pace, shooting% |
| Elo | 3 | elo_T1, elo_T2, elo_win_prob |
| Seed | 4 | seed_T1, seed_T2, seed_diff, seed_ratio |
| Recent Form | 2 | last_10_win_pct per team |
| Massey Rankings | 2 | avg_massey_rank per team |
| Head-to-Head | 2 | h2h_wins, h2h_games |
| Tournament History | 2 | tourney_win_pct per team |
| Differentials | ~25 | diff_ prefix for all team stats |

#### Architecture (PyTorch NN)
```
Input → Linear(n_features, 512) → BN → SiLU → Dropout(0.3)
     → ResBlock(512→256) → ResBlock(256→128) → ResBlock(128→64)
     → Linear(64, 1) → Sigmoid
```

#### Training Notes
- 5-fold StratifiedKFold CV
- Symmetric training: each game appears twice (team1/team2 swapped)
- Season split: last 2 seasons held out for validation
- Ensemble: Stacking (LR meta) + Scipy weight optimization + rank averaging → best selected

#### What to Try Next (v2.0)
- [ ] Fill in after seeing actual training results
- [ ] Add `MTeamCoaches.csv` features (coaching tenure, tournament experience)
- [ ] Tune Elo K-factor separately for pre/post 2010 era
- [ ] Increase Optuna trials to RF→50, XGB/LGB→200
- [ ] Try ExtraTreesClassifier as 6th model
- [ ] Experiment with KenPom-style AdjEM features
- [ ] Try VotingClassifier as alternative meta-learner

---

*Next entry will be added after training the v1.0 notebook on Kaggle.*
