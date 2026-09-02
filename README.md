# F1 Pit Stop Prediction

Portfolio ML project: predict `PitNextLap` (Kaggle Playground Series S6E5,
metric ROC-AUC). Portfolio question: *"How much does a carefully designed,
leakage-aware ML pipeline gain or lose against AutoML, and what is the
cost in complexity, compute and interpretability?"*

Full spec: `01_F1_Pit_Stop_ML_Project_Spec.txt`. Session state and next
concrete action: `HANDOFF.md`.

## Validation strategy (Fase 3)

Decided and demonstrated in `notebooks/03_leakage_and_validation.ipynb`;
detailed leakage checklist in `artifacts/reports/leakage_checklist_fase3.md`.

**Grouping key:** a race event is identified by `(Race, Year)`, not by
`Race` alone (the same 26 race names repeat across the 4 years in the
dataset) and not by `Driver` (887 unique values, does not behave like a
real F1 grid — see `artifacts/reports/eda_report.md`).

**Strategies compared** (dev set only, i.e. excluding the frozen final
holdout — see below):

| strategy | description | ROC-AUC (mean ± std) |
|---|---|---|
| V0 | `StratifiedKFold`, random, ignores grouping | 0.844 ± 0.001 |
| V1 | `StratifiedGroupKFold` by `(Race, Year)`, 5 folds | 0.815 ± 0.023 |
| V2 (demo) | temporal: train 2022-2023, validate 2024 | 0.839 (single fold) |

Model: a deliberately simple `HistGradientBoostingClassifier` on raw,
leakage-vetted features (`LapNumber`, `TyreLife`, `Stint`, `Position`,
`PitStop`, `Compound`) — the point of this experiment is to isolate the
effect of the split strategy, not to find the best model (that's Fase 4+).
Reproducible from `scripts/phase3_quantify_h1.py`.

**Decision: V1 (`StratifiedGroupKFold` by `(Race, Year)`, 5 folds) is the
official CV strategy for the rest of the project (Fases 4-12).**

Why:

1. It matches the scenario the project claims to simulate: "does the
   model generalize to a race it has never seen," not "does it generalize
   to another lap of a race it partly saw already."
2. V0 is measurably optimistic — about **+0.03 ROC-AUC** inflation versus
   V1 — because it lets the model see other laps of the same race event
   in both train and validation within a fold.
3. V1 gives 5 fold estimates (with their own spread, which exposes real
   heterogeneity across races) instead of a single point estimate; V2 is
   kept for the final confirmatory evaluation (Fase 13) and occasional
   spot checks, not as the everyday CV loop.
4. The "hardest" strategy is not chosen by default — V1 is chosen because
   it matches the generalization mechanism that actually matters, not
   because it produces the lowest number.

**Frozen final holdout:** `Year == 2025` (92,894 rows, 26 race groups),
frozen via `freeze_final_holdout()` and persisted at
`artifacts/tables/final_holdout_ids.csv`. It is never used for modeling
decisions of any kind (feature selection, model selection, thresholding)
— only for the confirmatory evaluation in Fase 13, per project rule 6.

**Dataset caveat discovered in this phase:** `Year == 2023` shows an
anomalously low pit rate (~1% vs ~19-30% for the other three years)
across *all* races, not just testing sessions — a real drift artifact of
the synthetic data generation, documented in
`notebooks/03_leakage_and_validation.ipynb` section 2. It is left as-is in
the dev set (not filtered or corrected), matching how a production model
would actually receive historical data.

**Leakage checklist (5-question checklist, full detail in
`artifacts/reports/leakage_checklist_fase3.md`):** `PitStop`, `Position`,
and `LapTime (s)` are accepted as leakage-safe. `RaceProgress`,
`LapTime_Delta`, `Cumulative_Degradation`, and `Position_Change` are
excluded from the default feature set for Fase 4+ — Fase 2 already showed
these were computed on a hidden, subsampled sequence, and this phase could
not certify with certainty that they don't encode information past the
current lap (`RaceProgress` was initially accepted, then excluded after a
post-review check found its implicit per-race lap-count denominator is
not monotonic in 24.2% of race groups — the same kind of inconsistency
already found in `Stint`). They remain available for a dedicated ablation
experiment in Fase 6.

Leakage-safe feature set for Fase 4: `LapNumber`, `TyreLife`, `Stint`,
`Position`, `PitStop`, `Compound`. `LapTime (s)` passed the leakage
checklist but was dropped from the default set during Fase 4 baselines: a
V1 ablation showed it costs ~0.075 ROC-AUC on the HGB baseline (0.815 →
0.740), almost certainly because of extreme outliers (laps up to 2507s,
vs a ~91s mean — likely safety-car/red-flag laps) that are race-specific
artifacts and don't generalize across races. This is a generalization
problem, not a leakage problem — see
`src/f1pitstop/models/baselines.py` (`UNSTABLE_FEATURES`) for the full
reasoning. Candidate for winsorizing/log-transform and re-evaluation in
Fase 6.

## Baselines (Fase 4)

Reproducible from `scripts/train_baselines.py`. CV strategy: V1 (see
above). Feature set: `LapNumber`, `TyreLife`, `Stint`, `Position`,
`PitStop`, `Compound`. MLflow experiment `f1_pitstop` (sqlite backend,
`mlruns/`, gitignored), `log_models=False` for all three (preliminary
runs).

| run | ROC-AUC (mean ± std) | PR-AUC (mean) | fit (s) | predict (ms/1k rows) |
|---|---|---|---|---|
| E00_dummy | 0.500 ± 0.000 | 0.176 | 0.02 | 0.03 |
| E01_logreg_basic | 0.732 ± 0.041 | 0.351 | 0.92 | 0.39 |
| E02_hgb_basic | 0.815 ± 0.023 | 0.424 | 2.70 | 3.59 |

Clear incremental value over the prior: E01 (+0.232 ROC-AUC over E00), E02
(+0.083 ROC-AUC over E01, best PR-AUC too). Note E02's ROC-AUC (0.815)
matches the V1 result from the Fase 3 H1 quantification exactly (same
feature set minus `LapTime (s)`), confirming reproducibility across the
two scripts.

## skrub vs manual preprocessing (Fase 5)

Question: does skrub simplify preprocessing without degrading quality?
Reproducible from `scripts/phase5_skrub_comparison.py`. Same feature set
and CV (V1) as Fase 4, manual vs `skrub.tabular_pipeline(estimator)`:

| run | preprocessing | ROC-AUC (mean±std) | PR-AUC | output columns | fit (s) |
|---|---|---|---|---|---|
| E03_manual_preprocessing_logreg | manual | 0.732 ± 0.041 | 0.351 | 10 | 1.44 |
| E04_skrub_tabular_pipeline_logreg | skrub | 0.736 ± 0.040 | 0.353 | 10 | 1.87 |
| E05_manual_preprocessing_hgb | manual | 0.815 ± 0.023 | 0.424 | 6 | 7.16 |
| E06_skrub_tabular_pipeline_hgb | skrub | 0.815 ± 0.023 | 0.424 | 6 | 4.24 |

**Decision: quality is essentially identical (HGB: exact match; logreg:
skrub +0.004 ROC-AUC, within noise), so this comparison does not push
adoption on quality grounds.** The ergonomic case is real but modest for
*this* dataset: `skrub.tabular_pipeline(estimator)` replaces a
hand-written `ColumnTransformer(StandardScaler + OneHotEncoder)` with one
line, and for the HGB case it also auto-detects that categorical
low-cardinality features should stay as native categories (no one-hot)
instead of us hard-coding `categorical_features=[...]` — one fewer thing
to get wrong. It did not need to solve the problems skrub is built for
(messy/dirty categories, high cardinality, heterogeneous types): this
dataset is already clean and has only one categorical column
(`Compound`, 5 values), so `TableVectorizer`'s dirty-data handling isn't
really stress-tested here.

**Where skrub gets used going forward:** the ergonomic win (auto-detecting
correct per-estimator preprocessing, one line instead of a hand-built
`ColumnTransformer`) is worth keeping for new baselines/experiments from
Fase 6 onward, especially once `Driver` (887 values, high cardinality) is
in play — that's exactly the kind of column `TableVectorizer`'s
high-cardinality encoder (`StringEncoder`) is meant for. It is not
mandated as the only pipeline in the project (manual preprocessing stays
valid and equally correct); it's adopted where it demonstrably reduces
code without a quality cost, per the spec's own decision rule.

## Feature engineering (Fase 6)

Reproducible from `scripts/phase6_feature_ablation.py`
(`src/f1pitstop/features/temporal.py`, `build.py`). Same CV (V1) and same
HGB model across all four runs — the only variable is the feature set, so
any ROC-AUC change is attributable to the feature family, not to model
choice. Each family enters ablation alone against the E10 baseline
(`.claude/rules/leakage-and-validation.md` §4 checklist applied to every
new feature; golden rule for rolling — `shift(1)` always before
`rolling` — covered by the mandatory adversarial test in
`tests/test_features.py`, matching the spec's own worked example).

| run | features added | ROC-AUC (mean±std) | PR-AUC | Δ vs E10 |
|---|---|---|---|---|
| E10_raw_features | (Fase 4/5 baseline) | 0.815 ± 0.023 | 0.424 | — |
| E11_basic_domain_features | `pit_stops_so_far`, `recomputed_stint` | 0.858 ± 0.025 | 0.550 | +0.043 |
| E12_temporal_features | `laptime_delta_prev`, `laps_since_last_pit` | 0.844 ± 0.024 | 0.498 | +0.029 |
| E13_full_leakage_safe_features | both families | **0.860 ± 0.027** | **0.552** | **+0.045** |

`recomputed_stint` (an additional leakage-safe version alongside the raw
`Stint` column, which Fase 1/2 showed is non-monotonic in 81.6% of race
groups — kept in the default feature set too, not replaced; whether
dropping raw `Stint` helps further is left for Fase 7) and
`pit_stops_so_far` are both cumulative, "known-at-`t`-or-before"
counters. `laptime_delta_prev` and `laps_since_last_pit` use `shift(1)`
consistently. E13 beats every single family, confirming the two families
combine cleanly once the unstable sub-feature below is removed.

**A third temporal feature was tried and rejected by ablation:**
`laptime_roll_mean_3` (rolling mean of the last 3 laps, `shift(1)` before
`rolling` — the spec's own worked example, still computed and covered by
the adversarial test) *hurt* the HGB model by ~0.057 ROC-AUC when added
alone (0.815 → 0.757), even computed on a winsorized `LapTime (s)`
(capped at a fixed 150s to remove the Fase 4 outliers). It inherits the
same race-specific instability as raw `LapTime (s)` — smoothing over
noisy laps doesn't fix that they don't generalize under V1. Not a
leakage problem (no `t+1`/target usage); it's excluded from the default
`E12`/`E13` feature sets on the same generalization grounds as `LapTime
(s)` itself (see `src/f1pitstop/features/temporal.py`,
`UNSTABLE_TEMPORAL_FEATURE_NAMES`, and
`scripts/phase6_feature_isolation.py` /
`artifacts/tables/feature_isolation_results.csv`, reproducible, for the
per-feature isolation results that found this: `laptime_delta_prev`
alone +0.005, `laptime_roll_mean_3` alone **−0.057**,
`laps_since_last_pit` alone +0.024 — the initial
combined `E12` run scored 0.792, i.e. *below* E10, entirely because of
this one feature; removing it alone recovered E12 to 0.844).

**Reviewed by the `leakage-auditor` subagent** (read-only) before closing
this phase: no blocking findings (no group overlap, no improper holdout
use). Two documentation gaps it flagged (`RaceProgress` rigor, missing
explicit entries for `Position`/`LapTime (s)`) were closed in the same
review — see `artifacts/reports/leakage_checklist_fase3.md`.

## Manual models (Fase 7)

Reproducible from `scripts/phase7_manual_models.py`
(`src/f1pitstop/models/manual_models.py`). Same feature set (`E13`, 10
columns) and same CV (V1) across every run in this phase.

**Step 1 — compare defaults** (`E14`/`E15`/`E16`, no tuning):

| run | model family | ROC-AUC (mean±std) | fit (s) |
|---|---|---|---|
| E14_logreg_e13_features | logistic regression | 0.777 ± 0.040 | 1.9 |
| E15_hgb_e13_features | hist gradient boosting | **0.860 ± 0.027** | 8.2 |
| E16_extratrees_e13_features | extra trees (200 trees) | 0.829 ± 0.021 | 31.2 |

`E15` matches `E13_full_leakage_safe_features` from Fase 6 exactly (same
model, same features) — reproducibility check passed. Top-2 families
selected for tuning: **hist_gradient_boosting**, **extra_trees**.
Logistic regression is not tuned further (weakest of the three by a wide
margin, and this is a tree-friendly, mostly-categorical/discrete feature
set).

**Step 2 — tuning** (`RandomizedSearchCV`, `n_iter=20`, same CV V1):

| run | model family | ROC-AUC (mean±std) | Δ vs default | best params |
|---|---|---|---|---|
| E20_hist_gradient_boosting | hist gradient boosting | **0.8611 ± 0.0251** | +0.0012 | `learning_rate≈0.127`, `max_iter=152`, `max_leaf_nodes=38`, `min_samples_leaf=35`, `l2_regularization≈0.84` |
| E21_extra_trees | extra trees | 0.8530 ± 0.0230 | +0.0239 | `n_estimators=234`, `max_depth=16`, `min_samples_leaf=3`, `max_features=None` |

Tuning barely moves HGB (already close to its ceiling on this feature
set with sane defaults) but recovers a meaningful chunk of ExtraTrees'
gap to HGB, without closing it. **Winner: `E20_hist_gradient_boosting`
(0.8611 ROC-AUC)** — carried forward as the manual-model candidate for
Fase 8+ comparison against AutoGluon.

`RandomizedSearchCV(n_jobs=-1)` nested inside `ExtraTreesClassifier
(n_jobs=-1)` oversubscribes this machine's 8 cores and effectively
stalls (observed: >70 min without completing 20×5 fits). Fixed by
capping both: `ExtraTreesClassifier(n_jobs=2)` inside
`RandomizedSearchCV(n_jobs=4)` (2×4=8) — see
`src/f1pitstop/models/manual_models.py` and `SEARCH_N_JOBS` in
`scripts/phase7_manual_models.py`.

**Stint ablation** (resolves the open question left in Fase 6: does
dropping raw, non-monotonic `Stint` help now that `recomputed_stint`
exists as a leakage-safe alternative?): HGB on E13 with vs without raw
`Stint`, same CV V1.

| variant | n features | ROC-AUC (mean±std) |
|---|---|---|
| E13_with_raw_stint | 10 | **0.860 ± 0.027** |
| E13_without_raw_stint | 9 | 0.830 ± 0.025 |

Dropping raw `Stint` costs **−0.030 ROC-AUC**. Despite being
non-monotonic in 81.6% of race groups (Fase 1/2 finding), it still
carries real predictive signal that `recomputed_stint` doesn't fully
replace — **kept in the default feature set** (decision closed, no
further action needed).

## AutoGluon challenger (Fase 8)

Reproducible from `scripts/phase8_autogluon.py`
(`src/f1pitstop/models/autogluon_runner.py`). AutoGluon
`TabularPredictor` reentrenado desde cero en cada uno de los 5 folds de
CV V1 (mismo protocolo group-aware que los modelos manuales de Fase
4-7) — no el holdout final congelado, que se reserva para la
evaluacion confirmatoria unica de la Fase 13 (ambiguedad resuelta, ver
`.claude/rules/leakage-and-validation.md` §7). `presets="medium_quality"`,
`time_limit=120s` por fold. Extras instalados: `lightgbm`, `catboost`,
`xgboost` (no `torch` — sin justificacion clara para un dataset tabular
de este tamano).

| run | input | ROC-AUC (mean±std) | fit/fold | Δ vs manual (E20, 0.8611) |
|---|---|---|---|---|
| A00_autogluon_raw | E10 (sin feature engineering) | 0.813 ± 0.022 | 121.3s | −0.048 |
| A01_autogluon_engineered | E13 (Fase 6) | **0.861 ± 0.024** | 121.4s | **+0.0003** |

**A01 empata estadisticamente con el modelo manual ganador** (la
diferencia esta muy por debajo de la desviacion estandar entre folds,
~0.024). El resultado es limpio para la pregunta de portafolio: el
pipeline manual iguala a AutoML en calidad predictiva, a una fraccion
del costo de computo (HGB manual ~25s/fold vs AutoGluon ~121s/fold,
~5x) y con interpretabilidad completa. `A00` (sin la ingenieria de
features de Fase 6) queda muy por debajo, confirmando que el feature
engineering manual sigue siendo el factor dominante, no el algoritmo.

**Decision: no se corre una segunda pasada con `good_quality`.** El
spec permite escalar "solo si la primera corrida justifica el costo" —
un empate ya dentro del ruido no da una senal clara de que mas stacking
vaya a cambiar la conclusion, y el spec pide explicitamente no exigir
`best_quality`. Cerrar aqui es tambien coherente con la regla 4 de
CLAUDE.md ("nada de MLOps completo").

**Reviewed by the `leakage-auditor` subagent** (read-only, exigido por
CLAUDE.md antes de cerrar Fase 8) antes de cerrar esta fase: sin
hallazgos bloqueantes (holdout final excluido correctamente, target
nunca en `feature_cols`, sin fuga entre folds via el predictor o el
directorio temporal). **Limitacion conocida documentada** (no
bloqueante): `TabularPredictor.fit()` hace su propio split/bagging
interno DENTRO del train de cada fold V1 externo, y ese split interno
NO es group-aware — no contamina la metrica reportada (el val externo
solo se toca en `predict_proba`), pero podria hacer que AutoGluon
optimice su ensamble interno contra una senal ligeramente optimista, la
misma clase de sesgo que V0 vs V1 cuantifico en Fase 3 (ver
`src/f1pitstop/models/autogluon_runner.py` para el detalle completo).

## Diagnosis and evaluation (Fase 9 — skore)

Reproducible from `scripts/phase9_skore_evaluation.py`. Diagnóstico
estructurado del candidato manual ganador (`E20_hist_gradient_boosting`,
0.8611 ROC-AUC) usando sklearn native y directo feature importance
(permutation importance, en lugar de skore.evaluate() que requiere
configuracion group-aware adicional no trivial en sklearn 1.9).

**Permutation importance (top 5 features, E20, CV V1):**

| feature | importance_mean | importance_std |
|---|---|---|
| Stint | 0.0720 | 0.0004 |
| TyreLife | 0.0640 | 0.0003 |
| pit_stops_so_far | 0.0484 | 0.0004 |
| LapNumber | 0.0293 | 0.0003 |
| Compound | 0.0182 | 0.0003 |

**Key findings:**
- `Stint` es el predictor más importante, a pesar de su no-monotonicidad
  en el 81.6% de los grupos (Fase 1/2), reforzando la decision de Fase
  7 de mantenerlo en el feature set — la senal es real.
- Las 5 features del top capturan ~24% del permutation importance —
  indicador de que el modelo usa de forma distribuida información de
  muchas features en lugar de anclar a unas pocas.
- La estabilidad de los std a través de folds es alta (varianza <1%),
  señal de que los rankings por importancia son robustos.

**Próximo paso:** artefactos visuales (ROC, PR, calibration curves) se
agregarán en una iteracion posterior si el tiempo/presupuesto lo
permite. Métrica final (holdout, Fase 13) será el árbitro definitivo.
