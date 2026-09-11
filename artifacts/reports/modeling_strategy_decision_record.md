# Modeling Strategy Decision Record

**Scope:** solution-class selection under project constraints (compute,
reproducibility, interpretability, maintainability). Deployment, serving,
monitoring and retraining policy are out of scope — they are named as
boundaries, not addressed.

**Status:** accepted. Revised 2026-09-10 after a documentation audit that
removed claims the code did not support.

---

## 1. Context

The task is to predict `PitNextLap` on a tabular dataset of ~439k rows
(Kaggle Playground Series S6E5, ROC-AUC). The question this record answers is
not *which model scores highest in the abstract*, but *which class of solution
is appropriate given the constraints actually in force here*.

### Constraints

| Constraint | Value | Consequence |
|---|---|---|
| Compute | CPU, 8 cores, 16 GB RAM | Rules out approaches whose practical use assumes GPU |
| Reproducibility | `uv run <script>` must reproduce reported metrics | Rules out artifacts too large or too stateful to version |
| Interpretability | Every feature must be explainable and auditable | Rules out opaque feature-generation procedures |
| Auditability | Every feature passes a 5-question leakage checklist | Rules out feature families that cannot be certified |
| Maintenance | Single maintainer | Weighs against multi-model serving paths |

These constraints are the premise of every decision below. A different set of
constraints would justify a different decision, and that is the point of
recording them.

---

## 2. Decision 1 — Group-aware cross-validation (V1)

**Chosen:** `StratifiedGroupKFold` with 5 folds, grouped by the race event
`(Race, Year)`.

The grouping key is built explicitly and passed to `split()`:

```python
groups = make_group_key(df)          # "Race|Year", see src/f1pitstop/data/split.py
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in cv.split(X, y, groups):
    ...
```

`Race` alone is not a valid key (the same 26 race names recur across the four
years) and `Driver` is not a reliable identifier in this synthetic dataset
(887 unique values; see `eda_report.md`).

**Alternatives considered:**

| Strategy | ROC-AUC (dev) | Why not chosen |
|---|---|---|
| V0 — random `StratifiedKFold` | 0.844 +/- 0.001 | Measurably optimistic: lets the model see other laps of the same race in both train and validation |
| **V1 — group-aware (chosen)** | **0.815 +/- 0.023** | Matches the generalization question the project claims to answer |
| V2 — temporal (2022-23 to 2024) | 0.839 (single fold) | Conceptually right but yields one estimate, no spread |

**Consequence:** roughly 0.03 ROC-AUC lower than V0 on the same data. That
difference is the measurement of V0's optimism, not a loss of model quality.
V1 was not chosen for being the harshest number — it was chosen because the
question is *"does this generalize to a race never seen?"*, not *"to another
lap of a race already partly seen?"*

---

## 3. Decision 2 — Manual feature engineering (10 features)

**Chosen:** a small, hand-built feature set where every column answers the
5-question leakage checklist (`.claude/rules/leakage-and-validation.md`).
Rolling and lag features apply `shift(1)` before any window, enforced by an
adversarial test in `tests/test_features.py`.

**Alternatives considered:**

| Approach | Effect on audit surface | Why not chosen |
|---|---|---|
| Target encoding (multi-fold, n-gram interactions) | Substantially increases the leakage-audit surface and implementation complexity | An out-of-fold target encoder *is* auditable; the objection is the cost of certifying each column, not impossibility |
| Polynomial / automatic interactions | Feature count grows faster than audit capacity | Same reason |
| AutoML-driven feature selection | Selection procedure not inspectable | Incompatible with the auditability constraint |

**Consequence:** the marginal expected gain from these families does not
justify the additional complexity under this project's constraints. The
measured effect of the manual families is documented in Phase 6
(`feature_ablation_results.csv`): E13 reaches 0.860 against 0.815 for the raw
feature set.

**Evidence that features dominate the algorithm choice:** AutoGluon on raw
features (A00) scores 0.813; on the same engineered features (A01) it reaches
0.861 — matching the manual model. The engineering, not the learner, carries
the difference.

---

## 4. Decision 3 — `HistGradientBoostingClassifier`, tuned

**Chosen:** E20, a single tuned HGB model. CV ROC-AUC 0.8611 +/- 0.0251.

| Candidate | ROC-AUC | Fit time | Why not chosen |
|---|---|---|---|
| **E20 HGB (tuned)** | **0.8611 +/- 0.0251** | ~25 s/fold | — |
| E21 ExtraTrees (tuned) | 0.8530 +/- 0.0230 | ~65 s/fold | Lower and slower |
| E01 Logistic Regression | 0.732 +/- 0.041 | ~1 s/fold | Far below on this feature set |

HGB also handles NaN natively, which removes an imputation step from the
pipeline (`laptime_delta_prev` is missing on each group's first visible lap
by construction).

### Interpretability, stated precisely

`HistGradientBoostingClassifier` does **not** expose `feature_importances_`.
Global interpretability comes from model-agnostic **permutation importance**
(`sklearn.inspection.permutation_importance`, used in Phase 9). Local
explanation — why a *particular* row received a particular probability —
would require SHAP or an equivalent method and **is not implemented in this
project**.

Top features by permutation importance (Phase 9, on `dev`): `Stint` 0.072,
`TyreLife` 0.064, `pit_stops_so_far` 0.048, `LapNumber` 0.029,
`Compound` 0.018.

---

## 5. Decision 4 — AutoML evaluated, not adopted

AutoGluon was run under the same protocol as the manual models: 5-fold
group-aware CV on `dev`, full retrain per fold
(`scripts/phase8_autogluon.py`).

| | E20 manual | A01 AutoGluon |
|---|---|---|
| ROC-AUC | 0.8611 +/- 0.0251 | 0.861 +/- 0.024 |
| Fit time | ~25 s/fold | ~121 s/fold |
| Artifacts | one serialized model | per-fold predictor directories, discarded |
| Explanation path | permutation importance over 10 named features | internal ensemble of multiple learners |

**Decision:** quality is a tie within noise; the compute cost is roughly 5x.
Under this project's constraints the automated path offers no net advantage.

**What this does *not* claim.** A01 was evaluated without group-aware internal
bagging: AutoGluon's own internal split within each outer fold was not
group-disjoint. This is a limitation of *this experiment*, not a statement
about the tool's capabilities. It does not contaminate the reported metric —
the outer folds remain group-disjoint — but it is a real asymmetry against the
manual model, which performs no internal split of its own.

---

## 6. Decision 5 — Higher-capacity alternatives not pursued

Tabular neural architectures (deep tabular models, MLP variants,
attention-based tabular learners) and large heterogeneous ensembles were
considered and not pursued.

**Reasons, in order of weight:**

1. **Compute constraint.** Their practical use assumes GPU availability. This
   project runs on CPU; that is a fixed premise, not a preference.
2. **Reproducibility constraint.** Their benefit depends on tuning budgets
   that cannot be reproduced by `uv run <script>` on a single machine.
3. **Interpretability constraint.** They do not offer an explanation path
   compatible with the auditability requirement.

**What is deliberately not claimed:** no estimate is given of how much ROC-AUC
these approaches would have added here. No such experiment was run, and
figures quoted from other settings do not transfer. The decision rests on the
constraints, not on a prediction of the foregone gain.

---

## 7. Choosing between these classes

The tree below starts from the objective, not from the tools, so it does not
resolve to this project by construction.

```
What is the primary objective?

├── Maximum score under a competition protocol
│   └── High-compute heterogeneous ensemble / higher-capacity tabular models
│       (assumes GPU budget; interpretability and single-artifact serving
│        are not requirements)
│
├── A working baseline fast, with limited engineering capacity
│   └── AutoML
│       (accepts an opaque internal ensemble in exchange for less hand-built
│        preprocessing and model selection)
│
└── An auditable model someone must maintain
    ├── strict CPU / latency / single-artifact constraints
    │   └── Compact GBDT pipeline with hand-audited features   <- this project
    │
    └── relaxed compute constraints
        └── Compare compact GBDT against AutoML on equal protocol,
            decide on total cost rather than on ROC-AUC alone
```

---

## 8. Consequences

**Accepted:**

- **Deployment complexity: low relative to the evaluated alternatives.** One
  serialized artifact (1.32 MB), one model path, no GPU in the dependency
  chain, and a smaller dependency surface than the AutoML or multi-model
  alternatives. This is a statement about relative complexity, not a claim
  that deployment has been demonstrated — deployment is out of scope.
- Inference at 3.59 ms per 1k rows, measured on the same hardware as training.
- Every feature traceable to a documented leakage decision.

**Given up:**

- Whatever ROC-AUC the unexplored higher-capacity approaches might have added.
  The magnitude is unknown because it was not measured.
- Automatic adaptation to new feature types, which an AutoML pipeline provides
  and this one does not.

**Reproducibility, stated precisely:** reproducible within a pinned software
environment, with deterministic seeds and documented platform assumptions.
Not "identical on any machine" — determinism also depends on Python, NumPy,
scikit-learn, BLAS/OpenMP threading and platform floating-point behaviour.

---

## 9. Known Limitations and Failure Modes

Where this solution is expected to degrade, independently of how carefully it
was built:

- **Season-to-season distribution shift.** `Year == 2023` shows a pit rate near
  1% against 19-30% in the other years, across *all* races of that season. The
  dataset is synthetic, so the cause is not established; whatever it is, a
  model trained across seasons inherits it. Phase 10 measured AUC 0.6668 on
  that segment against 0.8620 overall. In an operating system this is the
  failure mode that would call for drift detection, recalibration, a
  retraining trigger and per-segment monitoring — none of which are
  implemented here.
- **Circuit heterogeneity.** The +/- 0.025 spread across group-aware folds is
  real variation between race events, not only estimation noise. Per-race
  performance is not uniform.
- **Cold start.** New drivers, teams or regulation changes are not represented.
  Features such as `pit_stops_so_far` are uninformative on a group's first laps.
- **Calibration.** The model is selected on ROC-AUC, which is rank-based.
  Probability calibration was not optimized and should not be assumed.
- **Rare-event instability.** Mid-race phases are measurably harder
  (Phase 10: AUC 0.8137); low-prevalence segments are the least stable.
- **No true temporal validation in the operating sense.** V2 exists as a
  demonstration, but the project's decisions rest on V1. A deployment across a
  genuine season boundary would need a fresh temporal holdout.

---

## 10. Evaluation quantities, kept separate

Three numbers appear in this project. They come from different protocols and
are **never subtracted from one another**:

| Quantity | Value | Protocol |
|---|---|---|
| Group-aware CV | 0.8611 +/- 0.0251 | 5 folds over `dev`, grouped by `(Race, Year)` |
| Reserved internal holdout | 0.8727 | `Year == 2025`, evaluated exactly once (Phase 13) |
| Kaggle late-submission score | see `kaggle_late_submission.md` | Competition test set, row-level split, external reference only |

The holdout result falls within the variability observed during group-aware
CV, which provides **no evidence of a material generalization gap**. It does
not demonstrate the absence of overfitting — that is a stronger claim than the
evidence supports.

---

## References

- Validation strategy: `notebooks/03_leakage_and_validation.ipynb`, `src/f1pitstop/data/split.py`
- Leakage checklist: `artifacts/reports/leakage_checklist_fase3.md`, `.claude/rules/leakage-and-validation.md`
- Phase 7 model selection: `artifacts/tables/phase7_tuning_results.csv`
- Phase 8 AutoML: `artifacts/tables/phase8_autogluon_results.csv`
- Incumbent-challenge procedure: `artifacts/reports/model_selection_framework.md`
