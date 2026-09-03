"""Fase 14 — Model Selection Framework.

Contexto (ver `HANDOFF.md`, seccion "Analisis Comparativo: Tu Proyecto vs
Kaggle Top 2", y `artifacts/reports/model_selection_framework.md`): el
proyecto estaba cerrado en 13 fases con buen resultado (holdout 0.8727,
empata con AutoML). Analizar los writeups del 1er y 2do lugar de la
competencia Kaggle real (186 y 218 modelos respectivamente, ninguno
documenta CUANDO parar de optimizar) motivo reencuadrar el eje del
proyecto: no es "el mejor score posible", es "un framework reproducible
para decidir cuando un modelo es lo suficientemente bueno".

Esta fase implementa la parte con evidencia empirica de ese framework:

- **Tier 1 (14a):** 3 candidatos boosting adicionales (E22 XGBoost, E23
  CatBoost, E24 LightGBM) sobre EXACTAMENTE el mismo feature set E13 y CV
  V1 que E20 — sin tuning individual (regla de esta fase). Se comparan
  contra E20 YA TUNEADO (Fase 7), no contra un E20 sin tunear: la pregunta
  real es "¿un candidato nuevo por defecto supera al que ya esta
  optimizado y en produccion?", no "¿le ganamos a un E20 en desventaja?".
  Ademas E25: ensemble simple (logit-stack) de los 4 modelos via
  predicciones OOF, generadas con el MISMO split V1 (mismo seed) para que
  el stacking no tenga leakage entre modelos base y meta-modelo.

- **Tier 2 (14b):** 2 features candidatas (`laptime_roll_mean_5`,
  `pit_stops_rate_last3`) agregadas UNA A LA VEZ sobre E13 (mismo patron
  de `scripts/phase6_feature_isolation.py`), cada una ya paso el checklist
  de 5 preguntas y el test adversarial obligatorio
  (`tests/test_features.py`) antes de llegar aqui.

**Restriccion central de esta fase (leakage-and-validation.md seccion 9):**
NO se toca el holdout congelado en ningun punto de este script. La
decision de que candidato es mejor se toma enteramente sobre CV V1 en
`dev` — el holdout ya se evaluo una unica vez en la Fase 13 y esa
evaluacion no se repite ni se extiende a candidatos posteriores.

Uso: `uv run python scripts/phase14_model_selection_framework.py`
Loguea a MLflow (`stage=tuning` para 14a, `stage=features` para 14b) y
escribe `artifacts/tables/phase14_*.csv`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from f1pitstop.data.ingest import load_raw
from f1pitstop.data.split import load_frozen_holdout_ids, make_group_key, v1_group_stratified_kfold
from f1pitstop.evaluation.cv import run_group_cv
from f1pitstop.features.build import (
    E13_FULL_LEAKAGE_SAFE_FEATURES,
    build_engineered_frame,
    prepare_X_for_feature_set,
)
from f1pitstop.features.temporal import (
    PHASE14_CANDIDATE_FEATURE_NAMES,
    add_phase14_candidate_features,
)
from f1pitstop.models.baselines import SEED
from f1pitstop.models.ensemble import build_oof_feature_matrix, make_e25_logit_stack
from f1pitstop.models.manual_models import DIVERSITY_REGISTRY, make_e15_hgb_e13
from f1pitstop.tracking.mlflow_utils import log_run, setup_mlflow

TARGET = "PitNextLap"
VALIDATION_NAME = "V1_group_kfold_race_year_5fold"
FEATURE_SET_NAME = "E13_full_leakage_safe_features"
TABLES_DIR = Path("artifacts/tables")

# Best params de E20 (Fase 7, README "Manual models (Fase 7)"). Se compara
# contra la version YA TUNEADA, no contra el default sin tuning — ver
# docstring del modulo.
E20_BEST_PARAMS = {
    "learning_rate": 0.127,
    "max_iter": 152,
    "max_leaf_nodes": 38,
    "min_samples_leaf": 35,
    "l2_regularization": 0.84,
}


def make_e20_tuned():
    model = make_e15_hgb_e13()
    model.set_params(**E20_BEST_PARAMS)
    return model


def load_dev_e13() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    train, _test, _sample_sub, _reports = load_raw()
    holdout_ids = load_frozen_holdout_ids()
    dev = train[~train["id"].isin(holdout_ids)].reset_index(drop=True)
    engineered = build_engineered_frame(dev)
    # `build_engineered_frame()` calcula E10-E13 (Fase 6), no las 2
    # candidatas de Fase 14 (Tier 2) — se agregan aparte para no tocar el
    # registry E10-E13 ya cerrado. `X`/`y`/`groups` de 14a no se ven
    # afectados (prepare_X_for_feature_set solo selecciona columnas de
    # E13_FULL_LEAKAGE_SAFE_FEATURES); solo 14b usa las columnas extra.
    engineered = add_phase14_candidate_features(engineered)
    X = prepare_X_for_feature_set(engineered, FEATURE_SET_NAME)
    y = engineered[TARGET]
    groups = make_group_key(engineered)
    return X, y, groups, engineered


def compute_oof_predictions(make_model, X: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int = SEED) -> np.ndarray:
    """Predicciones out-of-fold sobre el MISMO split V1 (mismo seed) que
    usa `run_group_cv` — cada fila se predice con un modelo que nunca la
    vio en entrenamiento. Necesario para armar la matriz de features del
    stacker E25 sin leakage entre modelos base y meta-modelo."""
    oof = np.full(len(X), np.nan)
    for train_idx, val_idx in v1_group_stratified_kfold(y, groups, n_splits=5, seed=seed):
        model = make_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    assert not np.isnan(oof).any(), "Toda fila debe caer en el fold de validacion de exactamente un split"
    return oof


def step_14a_diversity(X, y, groups) -> pd.DataFrame:
    """Tier 1: compara E22/E23/E24 (defaults) contra E20 (tuneado, Fase 7)
    sobre CV V1. Ninguno de los 3 nuevos se tunea individualmente — regla
    explicita de esta fase (ver docstring del modulo)."""
    print("\n### 14a: Diversidad controlada (E22-E24) vs E20 tuneado ###")
    rows = []

    candidates = {"E20_hist_gradient_boosting": {"make_model": make_e20_tuned, "model_family": "hist_gradient_boosting"}}
    candidates.update(DIVERSITY_REGISTRY)

    for run_name, spec in candidates.items():
        result = run_group_cv(run_name, spec["make_model"], X, y, groups=groups, seed=SEED)
        metrics = result.to_metrics_dict()

        tags = {
            "project": "f1_pitstop",
            "stage": "tuning",
            "model_family": spec["model_family"],
            "feature_set": FEATURE_SET_NAME,
            "validation": VALIDATION_NAME,
            "seed": str(SEED),
        }
        params = {"n_splits": result.n_folds, "tuned": run_name.startswith("E20")}
        run_id = log_run(run_name, tags=tags, params=params, metrics=metrics, log_models=False)

        print(
            f"{run_name}: ROC-AUC {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f} | "
            f"fit {metrics['fit_seconds']:.2f}s | mlflow run_id={run_id}"
        )
        rows.append({"run_name": run_name, "model_family": spec["model_family"], **metrics, "mlflow_run_id": run_id})

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "phase14_diversity_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"Guardado en {out_path}")
    return df


def step_14a_ensemble(X, y, groups) -> dict:
    """E25: logit-stack sobre predicciones OOF de E20/E22/E23/E24. La CV
    del stacker reusa `run_group_cv` sobre la matriz OOF como si fuera un
    feature set mas — el mismo seed reproduce el mismo split V1 usado para
    generar las OOF, asi que no hay leakage entre el nivel base y el meta."""
    print("\n### 14a: E25 ensemble (logit-stack sobre OOF de E20/E22/E23/E24) ###")

    base_models = {"E20_hist_gradient_boosting": make_e20_tuned}
    base_models.update({name: spec["make_model"] for name, spec in DIVERSITY_REGISTRY.items()})

    oof_predictions = {}
    for run_name, make_model in base_models.items():
        print(f"  Generando OOF para {run_name}...")
        oof_predictions[run_name] = compute_oof_predictions(make_model, X, y, groups, seed=SEED)

    oof_matrix = build_oof_feature_matrix(oof_predictions)
    result = run_group_cv("E25_ensemble_logit_stack", make_e25_logit_stack, oof_matrix, y, groups=groups, seed=SEED)
    metrics = result.to_metrics_dict()

    tags = {
        "project": "f1_pitstop",
        "stage": "tuning",
        "model_family": "logistic_regression_stack",
        "feature_set": "oof_e20_e22_e23_e24",
        "validation": VALIDATION_NAME,
        "seed": str(SEED),
    }
    params = {"base_models": ",".join(base_models.keys())}
    run_id = log_run("E25_ensemble_logit_stack", tags=tags, params=params, metrics=metrics, log_models=False)

    print(
        f"E25_ensemble_logit_stack: ROC-AUC {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f} | "
        f"mlflow run_id={run_id}"
    )
    return {"run_name": "E25_ensemble_logit_stack", "model_family": "logistic_regression_stack", **metrics, "mlflow_run_id": run_id}


def step_14b_feature_ablation(engineered: pd.DataFrame, y: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Tier 2: agrega cada candidata UNA A LA VEZ sobre E13 (mismo patron
    que `scripts/phase6_feature_isolation.py`), mismo modelo de referencia
    (HGB tuneado E20) que el resto de la fase."""
    print("\n### 14b: ablation individual de candidatas Fase 14 sobre E13 ###")
    rows = []

    X_e13 = prepare_X_for_feature_set(engineered, FEATURE_SET_NAME)
    e13_result = run_group_cv("E13_reference", make_e20_tuned, X_e13, y, groups=groups, seed=SEED)
    e13_auc = e13_result.roc_auc_mean
    print(f"E13 (referencia, sin candidatas Fase 14): ROC-AUC {e13_auc:.4f}")

    for feature_name in PHASE14_CANDIDATE_FEATURE_NAMES:
        run_name = f"E13_plus_{feature_name}"
        cols = E13_FULL_LEAKAGE_SAFE_FEATURES + [feature_name]
        X = engineered[cols].copy()
        X["Compound"] = X["Compound"].astype("category")

        result = run_group_cv(run_name, make_e20_tuned, X, y, groups=groups, seed=SEED)
        metrics = result.to_metrics_dict()
        delta = result.roc_auc_mean - e13_auc

        tags = {
            "project": "f1_pitstop",
            "stage": "features",
            "model_family": "hist_gradient_boosting",
            "feature_set": f"e13_plus_{feature_name}",
            "validation": VALIDATION_NAME,
            "seed": str(SEED),
        }
        params = {"n_splits": result.n_folds, "features": ",".join(cols)}
        run_id = log_run(run_name, tags=tags, params=params, metrics=metrics, log_models=False)

        print(
            f"+{feature_name}: ROC-AUC {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f} "
            f"(delta vs E13: {delta:+.4f}) | mlflow run_id={run_id}"
        )
        rows.append(
            {
                "feature_name": feature_name,
                "roc_auc_mean": metrics["cv_roc_auc_mean"],
                "roc_auc_std": metrics["cv_roc_auc_std"],
                "delta_roc_auc_vs_e13": delta,
                "mlflow_run_id": run_id,
            }
        )

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "phase14_feature_isolation_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Guardado en {out_path}")
    return df


def main() -> None:
    setup_mlflow()
    X, y, groups, engineered = load_dev_e13()
    print(f"dev set (excluye holdout final 2025): {len(X)} filas, {X.shape[1]} columnas ({FEATURE_SET_NAME})")
    print("NOTA: este script NUNCA carga ni evalua sobre el holdout congelado (Year==2025) —")
    print("ver leakage-and-validation.md seccion 9.")

    diversity_df = step_14a_diversity(X, y, groups)
    ensemble_row = step_14a_ensemble(X, y, groups)
    feature_df = step_14b_feature_ablation(engineered, y, groups)

    print("\n=== Resumen Fase 14 ===")
    print("\n-- 14a: diversidad controlada + ensemble --")
    all_14a = pd.concat([diversity_df, pd.DataFrame([ensemble_row])], ignore_index=True)
    print(all_14a[["run_name", "cv_roc_auc_mean", "cv_roc_auc_std"]].to_string(index=False))

    e20_auc = diversity_df.loc[diversity_df["run_name"] == "E20_hist_gradient_boosting", "cv_roc_auc_mean"].iloc[0]
    e20_std = diversity_df.loc[diversity_df["run_name"] == "E20_hist_gradient_boosting", "cv_roc_auc_std"].iloc[0]
    best_row = all_14a.loc[all_14a["cv_roc_auc_mean"].idxmax()]
    margin = best_row["cv_roc_auc_mean"] - e20_auc
    print(f"\nE20 (incumbente, tuneado): {e20_auc:.4f}±{e20_std:.4f}")
    print(f"Mejor candidato Fase 14: {best_row['run_name']} ({best_row['cv_roc_auc_mean']:.4f}), delta={margin:+.4f}")
    if margin <= e20_std:
        print("DECISION: delta dentro de 1 std de E20 -> se mantiene E20 como candidato final.")
    else:
        print("DECISION: delta fuera del margen de ruido de E20 -> revisar si se justifica el cambio.")

    print("\n-- 14b: features candidatas --")
    print(feature_df[["feature_name", "roc_auc_mean", "delta_roc_auc_vs_e13"]].to_string(index=False))
    for _, row in feature_df.iterrows():
        verdict = "SE ADOPTA" if row["delta_roc_auc_vs_e13"] > 0 else "SE DESCARTA"
        print(f"{row['feature_name']}: {verdict} (delta={row['delta_roc_auc_vs_e13']:+.4f})")


if __name__ == "__main__":
    main()
