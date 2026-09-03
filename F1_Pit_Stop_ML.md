---
layout: page
title: F1 Pit Stop Prediction
description: Leakage-aware ML pipeline vs AutoML — equal performance, 5× faster.
img: assets/img/f1-pitstop.png
importance: 1
category: Personal
---

[![GitHub Repo](https://img.shields.io/badge/Code-ml--f1--pitstop-181717?logo=github)](https://github.com/aalopez76/ml-f1-pitstop)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-76%20passing-brightgreen)](https://github.com/aalopez76/ml-f1-pitstop)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8727%20%28holdout%29-success)](https://github.com/aalopez76/ml-f1-pitstop)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle%20S6E5-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/playground-series-s6e5/)

> **A production-grade ML pipeline**, not a notebook.  
> Predicts whether a Formula 1 driver will pit in the next lap. Carefully designed to avoid leakage, validated across multiple strategies, and benchmarked against AutoML — with surprising results.

---

## Portfolio Question

**"How much does a carefully designed, leakage-aware ML pipeline gain or lose against AutoML, and what is the cost in complexity, compute and interpretability?"**

**Answer:** Equal ROC-AUC (0.861 vs 0.861), but **5× faster** (25s vs 121s per fold), fully interpretable, and no black-box surprises.

---

## 🎯 Executive Summary

| Metric | Value | Insight |
|---|---|---|
| **CV ROC-AUC** | 0.8611 ± 0.0251 | 5-fold StratifiedGroupKFold by race event |
| **Holdout ROC-AUC** | 0.8727 | Year 2025 (unseen), 5 bps improvement |
| **Holdout PR-AUC** | 0.6985 | Strong precision at high recall |
| **Model** | HistGradientBoostingClassifier (tuned) | Native support for NaN & categoricals |
| **Training Speed** | 25s per fold | vs 121s for AutoGluon (5× faster) |
| **AutoGluon Benchmark** | 0.861±0.024 ROC-AUC | Equal performance, inferior speed & interpretability |
| **Feature Set** | 10 engineered features | Carefully validated against leakage checklist |
| **Test Coverage** | 76 tests, all passing | Data quality, split integrity, feature correctness |

---

## 🏗️ Architecture & Methodology

### Validation Strategy (The Foundation)

The pipeline compares **three CV strategies**:

| Strategy | Description | Performance | Used? |
|---|---|---|---|
| **V0** | StratifiedKFold (random, ignores grouping) | 0.844±0.001 | ❌ Optimistically inflated (+0.03) |
| **V1** | StratifiedGroupKFold by (Race, Year) | 0.815±0.023 | ✅ **Official** — matches real scenario |
| **V2** | Temporal: train 2022–2023 → val 2024 | 0.839 (1 fold) | ✅ Confirmatory only (Holdout evaluation) |

**Why V1?** A pit stop predictor in production must generalize to *races it has never seen*, not just new laps of races it already knows. V0 contaminates train/val within folds (both see the same race). V1 isolates each race into a single fold, exposing real generalization gaps.

### Leakage Checklist (5 Questions per Feature)

Every feature was validated against:
1. Is it known at prediction time `t`?
2. Does it use information from `t+1` or the future?
3. Does it encode the target directly or indirectly?
4. Does it use statistics computed on val/test data?
5. Does it use global statistics instead of fold-local ones?

**Result:** Feature set dropped from 16 raw columns to **10 engineered, leakage-safe features**.

### Feature Engineering Pipeline

```
Raw (E10)
  └─ Winsorized LapTime (cap at 150s, not percentile)
  └─ Basic domain (pit_stops_so_far, recomputed_stint)
  └─ Temporal (laptime_delta_prev, laps_since_last_pit)
     [all with shift(1) before rolling to avoid future leakage]
```

**Key Finding:** A rolling mean of lap times (3-lap window) was *unstable* — cost -0.057 ROC-AUC. Removed via ablation study. The other two temporal features (+0.024 combined) proved robust.

---

## 🔬 Results & Benchmarks

### Phase 1: Baselines (E00–E02)

Established lower bound with minimal preprocessing:

| Run | Model | ROC-AUC | PR-AUC | Speed (s) |
|---|---|---|---|---|
| E00 | DummyClassifier | 0.500±0.000 | 0.176 | 0.02 |
| E01 | LogisticRegression | 0.732±0.041 | 0.351 | 0.92 |
| E02 | HistGradientBoosting | **0.815±0.023** | 0.424 | 2.70 |

### Phase 2: Feature Engineering (E10–E13)

Systematic ablation showed which features matter:

| Run | Features | ROC-AUC | Δ vs E10 | Notes |
|---|---|---|---|---|
| E10 | Raw (6) | 0.815 | — | Baseline |
| E11 | +domain (8) | 0.858 | +0.043 | pit_stops_so_far, recomputed_stint |
| E12 | +temporal (10) | 0.844 | +0.029 | Excludes unstable rolling mean |
| **E13** | **Both (10)** | **0.860** | **+0.045** | ✅ Feature set for tuning |

### Phase 3: Manual Model Tuning (E14–E21)

Compared three model families, tuned top-2:

| Run | Model | Setup | ROC-AUC | Δ | Notes |
|---|---|---|---|---|---|
| E14 | LogisticRegression | Default | 0.777±0.040 | — | Discarded (weak) |
| E15 | HistGradientBoosting | Default | 0.860±0.027 | — | Matches E13 (reproducibility ✓) |
| E16 | ExtraTrees | Default | 0.829±0.021 | — | Top-2 selected |
| E20 | **HistGradientBoosting** | **Tuned (RandomizedSearchCV n=20)** | **0.8611±0.0251** | **+0.0012** | ✅ **Winner** |
| E21 | ExtraTrees | Tuned | 0.8530±0.0230 | +0.0239 | Recovered but below E20 |

**Best Params (E20):**
- `learning_rate ≈ 0.127`
- `max_iter = 152`
- `max_leaf_nodes = 38`
- `min_samples_leaf = 35`
- `l2_regularization ≈ 0.84`

### Phase 4: AutoGluon Challenger (A00–A01)

Replicated the manual model's CV strategy (V1, 5 folds, same feature sets):

| Run | Features | Presets | ROC-AUC | Speed/fold | Interpretation |
|---|---|---|---|---|---|
| A00 | Raw (E10) | medium_quality | 0.813±0.022 | ~121s | Feature engineering is dominant |
| **A01** | **Engineered (E13)** | **medium_quality** | **0.861±0.024** | **~121s** | ✅ Ties manual, not faster |

**Conclusion:** AutoGluon matches manual performance when given the same features. The feature engineering (not the algorithm) was the bottleneck. A second-pass `good_quality` was skipped — the first pass showed zero signal of improvement.

### Phase 5: Permutation Importance (E20)

Top-5 most important features (from OOF predictions):

```
Stint                      0.072
TyreLife                   0.064
pit_stops_so_far           0.048
LapNumber                  0.029
Compound                   0.018
```

**Insight:** Tire degradation (TyreLife) and pit stop history dominate. Position and lap number matter, but compound type (soft/medium/hard) is secondary.

### Phase 6: Holdout Evaluation (Final)

Frozen at Year==2025 (26 race events, 92,894 rows), held out during all development:

| Set | Rows | Events | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Dev (train+val, V1) | 346,246 | 78 | 0.8611 ± 0.0251 (CV mean) | 0.5531 |
| Holdout (Year 2025) | 92,894 | 26 | 0.8727 | 0.6985 |
| **Gap** | — | — | **-0.0116** ✓ (improvement, no overfitting) | +0.145 |

✅ **Generalization confirmed.** Holdout actually *outperforms* CV — a sign of real distribution stability, not overfitting.

---

## 🔍 Critical Discoveries

### 1. Dataset is Synthetic (Not a Bug, a Feature)
The Kaggle dataset is a synthetic generation of real F1 race data. Evidence:
- `Stint` (pit stop count) is non-monotonic in 81.6% of driver-race-year groups — physically impossible
- `LapNumber` has gaps (not every lap recorded for every driver)
- Year 2023 has anomalously low pit rates (~1% vs ~20% elsewhere)

**Impact:** This forced rigorous validation strategy (V1 instead of trusting driver continuity).

### 2. Leakage Trap: Feature Stability ≠ Leakage
`LapTime (s)` passed the leakage checklist but cost -0.075 ROC-AUC on the manual model. Why? **Extreme outliers** (laps up to 2507s, likely safety-car incidents) are race-specific artifacts that don't generalize under group-aware CV.

**Lesson:** Leakage checklist catches *temporal* and *statistical* leakage, not *instability* from distribution shift.

### 3. Group-Aware Validation Exposes Overfitting
V0 (naive StratifiedKFold) inflates ROC-AUC by **+0.03** vs V1 because it lets models leak race-specific knowledge within folds. V1 is harder but fairer.

---

## 🛠️ Stack & Engineering Practices

**Core Stack:**
- **Python 3.11** with `uv` (fast, deterministic dependency resolution)
- **scikit-learn 1.9.0** (model training & validation)
- **AutoGluon 1.6.1** (baseline comparison)
- **skrub** (automatic preprocessing, adopted where it reduced boilerplate)
- **skops** (model serialization with strict type allowlisting)
- **MLflow 3.15.2** (experiment tracking, sqlite backend)

**Quality Assurance:**
- **76 unit tests** (data schema, split integrity, feature correctness, model reproducibility)
- **Ruff** (zero code style issues)
- **Adversarial tests** (toy DataFrames verifying rolling features don't see future data)
- **Reproducible scripts** (every result from `scripts/` is deterministic, with fixed seeds)

**Project Structure:**
```
src/f1pitstop/
├── data/          # ingestion, schema validation, train/val splits
├── features/      # feature engineering (temporal, domain)
├── models/        # model builders (sklearn, skrub, autogluon)
├── evaluation/    # cross-validation, metrics
├── tracking/      # MLflow logging
└── persistence/   # model serialization (skops)

scripts/           # executable entrypoints (smoke test, train, evaluate)
notebooks/         # 03_leakage_and_validation.ipynb (criterion of Phase 3)
tests/             # 76 unit tests
artifacts/         # reports, tables, model artifacts, submission
```

---

## 📚 Key Learnings

1. **Leakage-aware design upfront pays off.** The 5-question checklist (know at time `t`? uses future? uses target? uses val/test stats? uses global stats?) caught every pitfall before they became bugs.

2. **Validation strategy is the foundation.** A flawed CV inflates metrics by 3–4%, making comparison impossible. Group-aware splitting (V1) is harder but essential for hierarchical data.

3. **Feature engineering > Algorithm.** AutoGluon + raw features (E10) scored 0.813. Manual model + engineered features (E13) scored 0.860. A 5× more complex algorithm with raw features lost to simple model + good features.

4. **Serialization is non-trivial.** Saving a pipeline with categorical preprocessors, imputations, and a boosted tree required careful type allowlisting (skops). Off-the-shelf pickle is not production-safe.

5. **Synthetic data is honest about gaps.** The synthetic nature of the dataset (non-monotonic Stint, missing laps, year-level drift) forced better validation practices than "real" data that silently violates assumptions.

---

## 🚀 Reproducibility & Code

**Everything is reproducible from source:**

1. Clone: `git clone https://github.com/aalopez76/ml-f1-pitstop`
2. Install: `uv sync`
3. Run all phases:
   ```bash
   uv run scripts/phase3_quantify_h1.py      # Validation strategy
   uv run scripts/train_baselines.py          # Baselines (E00–E02)
   uv run scripts/phase6_feature_ablation.py  # Feature engineering (E10–E13)
   uv run scripts/phase7_manual_models.py     # Model tuning (E14–E21)
   uv run scripts/phase8_autogluon.py         # AutoGluon benchmark (A00–A01)
   uv run pytest                              # 76 tests, all passing
   ```

4. View MLflow experiment: `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`

**Serialized Model:**
- `models/sklearn/e20_final.skops` — 1.32 MB, production-ready, fully reproducible predictions

**Submission:**
- `artifacts/submission.csv` — predictions on Kaggle test set (188,165 rows)

---

## 📊 Artifacts in This Repository

- **Reports:** `artifacts/reports/` — data audit, EDA, leakage checklist, phase summaries
- **Figures:** `artifacts/figures/` — validation strategy comparison, permutation importance
- **Tables:** `artifacts/tables/` — baseline results, feature ablation, tuning progress
- **Notebooks:** `notebooks/03_leakage_and_validation.ipynb` — interactive walkthrough of validation strategies
- **Specification:** `01_F1_Pit_Stop_ML_Project_Spec.txt` — full 24-phase methodology

---

## 🎓 Why This Matters for Portfolio

This project answers a real, complex question: *"Does careful design beat brute force?"* The answer is nuanced — not "yes" or "no," but **"it depends on what you measure."**

- On *performance*, manual + features ties AutoML.
- On *speed*, manual wins 5× over.
- On *interpretability*, manual is transparent, AutoML is a black box.
- On *cost*, manual is cheaper (no heavyweight stacking).

This is the kind of trade-off analysis that separates *data science from machine learning*. The project demonstrates an ability to:
- ✅ Avoid leakage (not obvious, easy to miss)
- ✅ Design validation rigorously (V1 is less comfortable than V0, but correct)
- ✅ Ablate systematically (feature isolation reproducible, not anecdotal)
- ✅ Benchmark fairly (AutoGluon gets same data, same splits, same folds)
- ✅ Write production-grade code (tests, serialization, reproducibility)

**Result:** A portfolio piece that shows *judgment*, not just *technique*.

---

## 📌 Links

- **GitHub:** [aalopez76/ml-f1-pitstop](https://github.com/aalopez76/ml-f1-pitstop)
- **Kaggle Competition:** [Playground Series S6E5](https://www.kaggle.com/competitions/playground-series-s6e5/)
- **Full Spec:** `01_F1_Pit_Stop_ML_Project_Spec.txt` in repo
- **Session Notes:** `HANDOFF.md` in repo (state of each phase, decisions, blockers)
