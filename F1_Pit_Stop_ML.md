---
layout: page
title: F1 Pit Stop Prediction
description: A rigor-first ML pipeline vs AutoML — equal performance, a fifth of the compute, full explainability.
img: assets/img/f1-pitstop.png
importance: 1
category: Personal
---

[![GitHub Repo](https://img.shields.io/badge/Code-ml--f1--pitstop-181717?logo=github)](https://github.com/aalopez76/ml-f1-pitstop)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-101%20passing-brightgreen)](https://github.com/aalopez76/ml-f1-pitstop)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8727%20%28holdout%29-success)](https://github.com/aalopez76/ml-f1-pitstop)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle%20S6E5-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/playground-series-s6e5/)

> An auditable, explainable ML pipeline built to answer one question with evidence: under real constraints — CPU-only compute, reproducibility, a maintainable feature set — what does a carefully engineered model cost or gain against AutoML, and how do you *know* it's trustworthy enough to hand to someone who'd act on it?

---

## I. The problem, in its own terms

This isn't "a Kaggle dataset." It's a proxy for a real strategic decision.

In Formula 1, a pit stop call is one of the highest-leverage moments in a race. Call it a lap too early and you burn a fresh set of tires nobody needed yet. Call it a lap too late and a rival "undercuts" you — pits first, posts faster laps on new rubber while you're still on worn tires, and comes out ahead once you finally stop. The tactical battle between drivers is fought, in large part, lap by lap, around the question this project models: *is this driver about to pit next lap?*

The dataset (Kaggle Playground Series S6E5, ~439,140 training rows) is synthetic, but it's explicitly inspired by a real F1 telemetry dataset. Kaggle documents that a column called `Normalized_TyreLife` was intentionally removed from the original because it made the prediction task trivial — a detail worth keeping in mind, because it means the *difficulty* of this problem is itself a designed property of the dataset, not an accident.

The link to the original real-world dataset is broken on Kaggle's page as of this writing. That's a limitation worth naming rather than working around silently — H4 in the project's hypothesis list explicitly flags "synthetic vs. original distribution mismatch" as an open question that can't be fully resolved without that source.

## II. The data, before touching a model

Before any model gets built, the data audit and EDA (Phases 1–2 of the project) surface three structural facts that shape every decision after this point.

**`Driver` doesn't behave like a real grid.** 887 unique values across the dataset, 414 distinct values within a single race — nothing close to the ~20 drivers on a real F1 grid. Treating it as a stable driver identifier would be a mistake baked into every downstream feature.

**`Race` alone doesn't identify an event.** The same 26 race names recur across four seasons. Grouping by `Race` without `Year` silently merges four different editions of, say, the Bahrain Grand Prix into one unit — exactly the kind of subtle grouping bug that produces an optimistic cross-validation score without anyone noticing.

**`Stint` is non-monotonic in 81.6% of driver-race groups — and this is provably not a subsampling artifact.** This is worth walking through because the reasoning, not just the conclusion, is the interesting part. The dataset's rows are a *subsample* of a hidden, more complete lap sequence (confirmed by cross-checking `Position_Change` against the visible-row difference in `Position`, which don't match — the derived columns were computed on the full hidden sequence, then rows were dropped). A natural hypothesis is: "maybe `Stint` only looks non-monotonic because we're missing rows." But that hypothesis is falsifiable, and it's false: removing rows from a strictly non-decreasing sequence can *never* produce a visible decrease — you can only ever see fewer of the same non-decreasing values, never a drop. Yet decreases as large as −5 appear in 81.6% of a 3,000-group sample. That's mathematically impossible under the subsampling-only hypothesis, which means it's a genuine inconsistency in the synthetic generation process, not a byproduct of how the CSV was constructed.

This matters concretely for Phase 6: any feature that treats `Stint` as a reliable monotonic counter needs a leakage-safe alternative computed independently (`recomputed_stint`), and any rolling/lag feature needs `LapNumber`, not row order, as its notion of "how far back."

**One season is a distribution-shift outlier, left in on purpose.** `Year == 2023` shows a pit rate near 1% against 19–30% in the other three years — across *all* races that season, not just a subset. This is documented and left untouched in the dev set (not filtered, not corrected), because a production model would actually receive this kind of historical anomaly, and "cleaning" it away would hide a failure mode instead of confronting it. It resurfaces later, in both error analysis and explainability, as the concrete anchor for what "the model breaks here" looks like in practice.

## III. Validation: the decision that carries the most weight

Everything downstream depends on getting this right, so it's worth making the reasoning explicit rather than asserting the conclusion.

Three splitting strategies were compared on identical data with an identical model (`HistGradientBoostingClassifier` on the raw leakage-vetted feature set):

| strategy | description | ROC-AUC (mean ± std) |
|---|---|---|
| V0 | `StratifiedKFold`, random, ignores grouping | 0.844 ± 0.001 |
| **V1** | `StratifiedGroupKFold` by `(Race, Year)`, 5 folds | **0.815 ± 0.023** |
| V2 (demo) | temporal: train 2022–2023, validate 2024 | 0.839 (single fold) |

V0 scores *higher*. That's the trap: a random split lets rows from the same race event land in both train and validation within a fold, so the model partially memorizes race-specific patterns instead of learning to generalize to a race it's never seen. The +0.03 gap between V0 and V1 is the *measurement* of that optimism — not something to be proud of, something that would have silently inflated every subsequent number if left uncorrected.

**V1 was chosen not because it's the hardest number, but because it matches the actual question.** The scenario this project claims to simulate is "does the model generalize to a race it has never seen," not "does it generalize to another lap of a race it partly saw already." That's the criterion — matching the deployment scenario — not "pick whichever split produces the lowest score to look rigorous." V2 (a genuine temporal holdout) is conceptually closer to a real deployment boundary, but gives only one fold estimate with no visibility into cross-race variance, so it's kept for confirmatory checks (Phase 13) rather than the everyday CV loop.

**The frozen final holdout** (`Year == 2025`, 92,894 rows, 26 race groups) is set aside before any tuning and never touched for modeling decisions of any kind — not feature selection, not model selection, not thresholding. It gets evaluated exactly once, in Phase 13. This rule ends up mattering again much later, in a way that's genuinely one of the better decisions in this project (see Section VI).

## IV. The pipeline — the core of this project

Ten features, each one required to answer five explicit leakage questions before being accepted: is it known at time `t`? Does it use information from `t+1` or later? Does it use the target directly or indirectly? Does it use aggregates computed on validation/test data? Does it use global statistics that should be fold-local?

The two feature families that survive this checklist:

```python
# Every rolling/lag feature applies shift(1) BEFORE the window —
# enforced by a mandatory adversarial test, not just a code comment.
laptime_delta_prev      # current lap time minus the *previous* lap's time
laps_since_last_pit     # laps elapsed since the last recorded pit stop
pit_stops_so_far        # cumulative count, known-at-t-or-before by construction
recomputed_stint        # leakage-safe stint counter, independent of raw Stint
```

| run | features added | ROC-AUC (mean±std) | Δ vs E10 |
|---|---|---|---|
| E10_raw_features | (baseline) | 0.815 ± 0.023 | — |
| E11_basic_domain_features | `pit_stops_so_far`, `recomputed_stint` | 0.858 ± 0.025 | +0.043 |
| E12_temporal_features | `laptime_delta_prev`, `laps_since_last_pit` | 0.844 ± 0.024 | +0.029 |
| **E13_full_leakage_safe_features** | both families | **0.860 ± 0.027** | **+0.045** |

**What's equally worth showing is what got rejected.** A third temporal feature, `laptime_roll_mean_3` — a rolling mean of the last three laps' times, `shift(1)` applied correctly before the window, exactly the worked example in the project's own spec — *hurt* the model by 0.057 ROC-AUC when isolated (0.815 → 0.757). It's not a leakage bug (no future information is used); it inherits the same race-specific instability as raw lap times themselves, driven by extreme outliers (laps up to 2,507 seconds against a ~91-second mean, almost certainly safety-car periods) that don't generalize across races. Correctly implemented and still rejected — a generalization problem, not a correctness problem, and the ablation table exists precisely to catch this instead of attributing an aggregate feature-family gain to a feature that's actually net-negative.

The tuning phase surfaces a mundane but real engineering bug worth naming: `RandomizedSearchCV(n_jobs=-1)` nested inside `ExtraTreesClassifier(n_jobs=-1)` oversubscribes an 8-core machine and effectively stalls — observed at over 70 minutes without completing a 20×5 search. Fixed by explicitly capping both (`n_jobs=2` inside, `n_jobs=4` outside, 2×4=8). Small, but it's the kind of failure that silently wastes hours if you don't notice the process is technically "running" while making no progress.

Final tuned candidate: `HistGradientBoostingClassifier`, **0.8611 ± 0.0251 ROC-AUC**, `learning_rate≈0.127`, `max_iter=152`, `max_leaf_nodes=38`, `min_samples_leaf=35`, `l2_regularization≈0.84`. Notably, tuning barely moves it (+0.0012 over sane defaults) — a sign the feature engineering already did the heavy lifting, which the next section confirms independently.

## V. Manual vs. AutoML, with the detail a shorter write-up skips

AutoGluon's `TabularPredictor` was retrained from scratch on each of the same five V1 folds — identical protocol to the manual model, `presets="medium_quality"`, `time_limit=120s` per fold.

| run | input | ROC-AUC (mean±std) | fit/fold | Δ vs manual |
|---|---|---|---|---|
| A00_autogluon_raw | raw features, no engineering | 0.813 ± 0.022 | 121.3s | −0.048 |
| **A01_autogluon_engineered** | same E13 features as the manual model | **0.861 ± 0.024** | 121.4s | **+0.0003** |

The tie is real — 0.0003 is nowhere close to the ~0.025 fold-to-fold standard deviation. But **A00 is the number that actually settles the "features vs. algorithm" question**: give AutoGluon raw features and it lands *below* the manual pipeline with engineering, at 0.813. Feature engineering, done by a human who understands the leakage constraints, contributed more to the final score than the choice of learning algorithm — automated or not.

**A limitation found in AutoGluon during this comparison, documented rather than glossed over:** `TabularPredictor.fit()` performs its own internal train/validation bagging *within* each outer V1 fold, and that internal split is not group-aware — the same class of optimism V0 demonstrated against V1 in Section III, just one level deeper inside AutoGluon's own ensembling. This doesn't contaminate the reported metric (the outer validation fold is never touched during fitting), but it's a real asymmetry: AutoGluon may be optimizing its internal ensemble weights against a slightly optimistic internal signal, while the manual model performs no internal split of its own. Reviewed by a read-only leakage-auditor subagent before closing this phase — no blocking findings, but this asymmetry is recorded as a known, non-blocking limitation rather than left implicit.

The practical conclusion: under CPU-only constraints, a single reproducible artifact requirement, and an interpretability requirement, the manual pipeline reaches AutoML's ceiling at roughly a fifth of the training cost (25s vs. 121s per fold) and produces one 1.3MB file instead of an ensembled predictor directory. Higher-capacity approaches — larger ensembles, deep tabular architectures, the kind of infrastructure published competition write-ups in this dataset's source competition describe — were considered and set aside for the same reason: their practical use assumes GPU budgets and tuning cycles outside this project's stated constraints. That's a constraint-driven decision, stated as such, not an implicit claim that they wouldn't outperform on an unconstrained budget.

## VI. Phase 14 as a case study in process, not just outcome

This is the part of the project that most demonstrates what "rigor" actually costs to practice, because it includes a real mistake that was caught and corrected in the open rather than quietly fixed.

Three cheap, default-configuration challengers — XGBoost, CatBoost, LightGBM, on the identical E13 feature set and V1 folds as the incumbent — were screened against the tuned `HistGradientBoostingClassifier`. A logit-stacking ensemble of all four was also tried.

The original comparison method compared a challenger's mean delta against the *standard deviation between folds* — which sounds reasonable and is statistically wrong. Fold-to-fold standard deviation measures how much one model's own score varies across different race groupings; it says nothing about the *uncertainty of the difference* between two models measured on the *same* folds. Since every candidate shares exactly the same five folds, the correct comparison is a **paired delta per fold**:

```
d_i = ROC-AUC(challenger, fold_i) − ROC-AUC(incumbent, fold_i),   i = 1..5
```

| run | mean paired delta | median | range [min, max] | folds favoring challenger |
|---|---|---|---|---|
| XGBoost | −0.0021 | −0.0010 | [−0.0062, +0.0013] | 1/5 |
| CatBoost | −0.0005 | −0.0003 | [−0.0043, +0.0054] | 2/5 |
| LightGBM | −0.0018 | −0.0015 | [−0.0060, +0.0010] | 1/5 |
| Ensemble (logit-stack) | +0.0004 | +0.0009 | [−0.0007, +0.0010] | 4/5 |

The ensemble looked like the strongest candidate — small but consistent gain, 4 of 5 folds favorable, the tightest range of the table. **It's also the one that turned out to be invalid evidence.** Its out-of-fold predictions and the stacker's own cross-validation reused the *same* five folds: when fold 1 served as the stacker's validation fold, the stacker had been trained on rows whose meta-features came from base models that had already seen fold 1 during their own fits. That's leakage between stacking levels — the evaluation was never properly nested. Worse, the code contained a comment explaining that reusing the same seed for both the OOF generation and the stacker's evaluation meant there was *no* leakage. The reasoning was exactly backwards: sharing the same folds across both stages is precisely what produces the contamination, not what prevents it.

It didn't get patched with a nested-CV rebuild — that's expensive, and the observed gain (+0.0004) was already small enough that investing more compute to measure it *correctly* would have meant optimizing something that was likely to be discarded regardless. It got relabeled instead: marked `exploratory_contaminated` with `promotion_evaluation_valid = false` directly in the results CSV, so the caveat travels with the data itself rather than depending on someone remembering to re-read a paragraph in a report six months later.

**The structural lesson that outlasts this specific bug:** no minimum practical-effect threshold had been decided *before* running these experiments. The decision of what counted as "good enough" was made after seeing the numbers — which is exactly the kind of process gap that lets a contaminated +0.0004 look tempting. Rather than retroactively inventing a threshold that conveniently fits, the project declares this phase's decision **retrospective**, and introduces a **Challenger Acceptance Policy** — pre-registered thresholds (`configs/experiments/<id>.yaml`), committed to version control *before* an experiment runs, not after — effective from the next phase onward. The commit history itself becomes the proof that the bar was set before the result was known.

**Final outcome: the incumbent stands.** Not because it's provably optimal, but because none of the four candidates produced valid evidence to displace it.

## VII. Explainability — SHAP, with the technical detail that mattered

A model handed to a strategist needs to say *why*, not just *what*. The project already had permutation importance (Phase 9); adding SHAP meant reconsidering an earlier explicit decision to leave local explanation out of scope.

Two real engineering problems surfaced during this work, worth documenting because they're exactly the kind of thing a smoke test is supposed to catch before hours are sunk into the wrong architecture.

**First: the newest resolvable version of `shap` doesn't install on this Python version.** `uv add shap` resolved to `shap==0.51.0`, which pins `numba==0.53.1`, which in turn requires `llvmlite==0.36.0` — a build that only supports Python `<3.10`. This project runs Python 3.11. The fix was a one-line exclusion (`shap!=0.51.0` in `pyproject.toml`), which resolved cleanly to `shap==0.50.0` with a modern, compatible `numba`/`llvmlite` pair. Small, but the kind of thing that would have silently broken a CI pipeline months later if left unpinned.

**Second, and more interesting: `shap.TreeExplainer` doesn't support `HistGradientBoostingClassifier`'s native categorical handling.** The model treats `Compound` (tire compound) as a native pandas `"category"` dtype, using scikit-learn's built-in `categorical_features=[...]` support rather than one-hot encoding it manually. `TreeExplainer` — the fast, tree-structure-aware path — internally tries to cast the entire input array to `float` and fails outright on that column. This isn't a hypothetical; it was confirmed with an isolated smoke test *before* building anything else, following the same "prove compatibility before committing architecture" discipline the project's own Phase 0 established for its core dependency stack.

The fix: a wrapper (`build_onehot_wrapper`) that exposes `model.predict_proba` over a fully numeric one-hot-encoded space, reconstructing the original categorical row internally before calling the *real* model — not a substitute model trained differently just to make it explainable. Before trusting a single SHAP value, the wrapper's output is checked to be bit-for-bit identical to the model's own `predict_proba` on the same rows. This unlocks the generic, model-agnostic `shap.Explainer` (`PermutationExplainer`), at a real cost: ~400ms per row, measured on the actual trained model, versus the near-instant native `TreeExplainer` path that isn't available here. That's why the global importance calculation runs on a stratified sample of 500 rows rather than the full 346,246-row dev set — at that rate, the full set would take upward of 38 hours. 500 rows is a deliberate cost/coverage trade-off for a readable summary plot, not a resource limitation left unstated.

**Do the two independent methods agree?**

| feature | permutation importance | SHAP `mean(|value|)` |
|---|---|---|
| `Stint` | 0.0719 | 0.1034 |
| `TyreLife` | 0.0640 | 0.0822 |
| `pit_stops_so_far` | 0.0484 | 0.0819 |
| `LapNumber` | 0.0293 | 0.0340 |
| `Compound` (aggregated) | 0.0182 | 0.0175 |

The top four features rank identically across both methods — a meaningful cross-check, since permutation importance measures metric degradation under perturbation while SHAP computes additive per-prediction attribution; two mechanistically unrelated methods converging on the same ranking is stronger evidence than either alone. A second, unplanned cross-check: SHAP decomposes `Compound` into five one-hot categories that sum to 0.0175 — almost exactly matching permutation importance's 0.0182 for the same feature treated as a single column.

**One local example is deliberately not a success story.** Rather than only showcasing a high-confidence correct prediction (`y_true=1`, `predict_proba=0.90` — included, but not the interesting one), the second example is a real false negative pulled from the `Year == 2023` drift segment already identified in error analysis (AUC 0.667 there against 0.862 globally): `y_true=1`, `predict_proba=0.105` — the model was confident this driver would *not* pit, and was wrong. SHAP shows exactly which features pushed that prediction down for this specific row, connecting *where* the model struggles (already known from error analysis) to *why*, in one concrete instance.

## VIII. What this project deliberately doesn't do

No API. No serving layer. No live monitoring. That's not an oversight — it's a scope boundary, stated as such rather than implied by absence.

What a production version of this system would actually need, named without being built:

- **A monitoring plan anchored to a real, already-observed failure mode.** The 2023 drift isn't hypothetical — it's a measured 0.667 AUC segment sitting inside data this model already trained on. A real deployment needs a detector for exactly this kind of shift, with a defined threshold and a defined action, not just "monitor the model" as an aspiration.
- **A model promotion policy, which already exists in embryonic form.** The Challenger Acceptance Policy from Phase 14 — pre-registered thresholds, committed before results are known — is functionally a model registry promotion gate. It needs a name and a home in a real MLOps pipeline; the logic is already built.
- **A serving contract.** Input schema, output schema, versioning, error handling — none implemented, but the one-artifact, no-GPU, low-dependency-surface design of the current pipeline (Section V) is exactly the kind of shape that makes this cheap to add later, rather than requiring a rearchitecture.

That gap — from "a rigorously validated, explainable model" to "an operated production system" — is real, and it's the subject of a separate project in this portfolio built specifically to demonstrate that end-to-end MLOps discipline. This project's job was narrower and, I'd argue, more foundational: showing that the modeling and validation rigor underneath a production system can be demonstrated on its own, with evidence, before a single line of deployment infrastructure exists.

---

*Full code, tests, and every document referenced above:
[github.com/aalopez76/ml-f1-pitstop](https://github.com/aalopez76/ml-f1-pitstop).*
