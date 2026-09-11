# Medium Article Draft

> **Nota de uso:** este artículo reemplaza la necesidad del post existente en Medium
> (`https://medium.com/@aalpzp/how-i-built-an-f1-pit-stop-predictor-andtl-dr-68c3fe2b1cb6`).
> Publicar este texto como reemplazo completo del post, no como addendum al final del
> anterior.
>
> Escrito en inglés para consistencia con la audiencia técnica de Medium.
> Regla editorial vigente: nunca nombrar competidores de la competición de origen del
> dataset; hablar de "published write-ups report..."; nunca restar el score público del
> leaderboard contra la CV o el holdout internos — son protocolos distintos.

---

# Predicting F1 Pit Stops: A Rigor-First Approach to the Manual-vs-AutoML Question

## Why pit stop timing matters

A pit stop call in Formula 1 is one of the highest-leverage decisions a race strategist makes. Call it a lap too early and you lose track position on fresher tires nobody needed yet. Call it a lap too late and a rival executes an "undercut" — pits first, sets faster laps on new tires while you're still on worn ones, and comes out ahead once you finally stop. Predicting *when a driver is about to pit* is a proxy for a much harder, much more valuable problem: understanding the signal a strategist actually watches lap by lap.

This project builds a model for that signal — not to win a leaderboard, but to demonstrate how you'd build one carefully enough to trust in a real strategy tool. The dataset (~439k laps, Kaggle Playground Series S6E5) is synthetic but inspired by real F1 telemetry, which means the modeling problem is real even though the specific numbers are generated.

## The data tells you the shape of the problem before you touch a model

The dataset looks clean at a glance. It isn't, in ways that matter for validation:

- The same 26 race names repeat across four seasons — grouping by `Race` alone silently merges different years of the same track into one "unit."
- `Stint` (which tire stint a driver is on) is *non-monotonic* in 81.6% of driver-race groups — physically impossible in a real race, a synthetic-generation artifact.
- One season, 2023, shows a pit rate near 1% against 19–30% everywhere else — a real distribution shift baked into the data, not a bug to code around.

None of this is decoration. It's the reason a naive random split *lies* to you about how well your model will generalize — and the entire next section exists because of it.

## The validation decision that changes everything downstream

Split the data randomly (`StratifiedKFold`) and you get 0.844 ROC-AUC. Split it so that every lap from the same race event stays together in either train or validation (`StratifiedGroupKFold` on `Race + Year`) and you get 0.815. That 0.03 gap isn't noise — it's the random split letting the model see other laps of the *same race* in both training and validation, which no real deployment would ever get. The group-aware number is lower and it is the honest one, because the question this model needs to answer is "does this generalize to a race it's never seen," not "does it generalize to another lap of a race it partly saw already."

Every result from here on is measured under that group-aware protocol.

## Two paths, evaluated on equal footing: a hand-built pipeline and AutoML

The core of this project is a controlled comparison, not a competition. A manually engineered pipeline — 10 leakage-audited features, a tuned `HistGradientBoostingClassifier` — is evaluated under the exact same cross-validation protocol as AutoGluon, an AutoML system given the same data and the same folds.

| | Manual pipeline | AutoGluon (AutoML) |
|---|---|---|
| ROC-AUC | 0.8611 ± 0.0251 | 0.861 ± 0.024 |
| Training time/fold | ~25s | ~121s (~5x) |
| Artifact | one 1.3MB file | multiple ensembled models |
| Explanation | permutation importance + SHAP, by name | internal ensemble |

They tie, within noise. That tie is the actual finding — not "manual wins" or "AutoML wins," but that under CPU-only, single-artifact, auditable-feature constraints, the tuned pipeline reaches AutoML's ceiling at a fifth of the compute cost and with a model someone can actually explain to a stakeholder. Feature engineering, not algorithm choice, did almost all of the work: AutoGluon on raw features alone scores 0.813 — *below* the manual pipeline with engineered features, and below its own score once the same engineering is added.

Other high-capacity approaches — larger ensembles, deep tabular architectures, the kind of infrastructure published competition write-ups describe using — were considered and set aside, not because they wouldn't work, but because their practical use assumes GPU budgets and tuning cycles this project's constraints (CPU-only, reproducible from a single command, one maintainer) don't allow. That's a constraint-driven decision, not a capability gap, and the project says so explicitly instead of pretending the comparison never came up.

## Deciding whether a cheaper model should replace the incumbent — with a real process, not a vibe check

After the manual-vs-AutoML result, three cheap default-configuration challengers (XGBoost, CatBoost, LightGBM) were screened against the tuned incumbent under identical folds. All three underperform. A stacking ensemble looked promising (+0.0004) until an audit of its own evaluation found the ensemble's cross-validation reused the same folds that generated its input predictions — a classic leakage-between-layers bug that made its score uninterpretable as evidence. It got reclassified, not patched to "still count."

The lesson that outlasts this specific project: **no minimum improvement threshold had been set before running these experiments.** That's a retrospective decision, and the project says so instead of hiding it. Going forward, a formal acceptance policy — thresholds defined and committed to version control *before* running an experiment — replaces ad hoc judgment calls made after seeing the number.

## Explaining the model that won — two independent methods, checked against each other

A model that's comfortable being put in front of a strategist needs to say *why*, not just *what*. Two explainability methods were computed — permutation importance and SHAP — and they agree on the top of the ranking: `Stint`, tire age, and cumulative pit-stop count dominate, in that order, across both methods. Agreement between two mechanistically unrelated methods (one measures performance degradation from shuffling a feature, the other computes additive attribution per prediction) is a stronger signal than either alone.

One SHAP example is deliberately not a cherry-picked success case: it's a real false negative from the 2023 drift segment, where the model's overall confidence collapses (AUC 0.667 there versus 0.862 everywhere else) — showing not just *that* the model struggles there, but *what specifically* confused it in one concrete row.

## What this project is not: a production system

No API, no serving infrastructure, no monitoring pipeline exists here, and that's deliberate — this project answers a modeling and process-rigor question, not a deployment one. What *would* be needed to take this further — a monitoring plan (the 2023 drift is exactly the kind of shift a real system would need to detect), a promotion policy for future model versions (the acceptance policy above is a first draft of exactly that), a serving contract — is named as the next step, not glossed over as solved.

---

*Full audit trail, corrected documents, and reproducible code:
[github.com/aalopez76/ml-f1-pitstop](https://github.com/aalopez76/ml-f1-pitstop). Start with
`artifacts/reports/modeling_strategy_decision_record.md`,
`artifacts/reports/model_selection_framework.md`, and
`artifacts/reports/explainability_report.md`.*
