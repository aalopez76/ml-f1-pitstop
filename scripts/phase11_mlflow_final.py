"""Fase 11 — MLflow final.

MLflow es el registro central de experimentos. Registra E20_hist_gradient_boosting
como el modelo finalista con stage=final, tags obligatorios y métricas de CV.

Uso: `uv run python scripts/phase11_mlflow_final.py`
"""

from __future__ import annotations

import mlflow
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

from f1pitstop.data.ingest import load_raw
from f1pitstop.data.split import load_frozen_holdout_ids, make_group_key
from f1pitstop.features.build import (
    build_engineered_frame,
    prepare_X_for_feature_set,
)
from f1pitstop.models.baselines import SEED
from f1pitstop.models.manual_models import make_e15_hgb_e13
from f1pitstop.tracking.mlflow_utils import setup_mlflow

TARGET = "PitNextLap"


def main():
    """Registrar E20 como stage=final en MLflow."""

    setup_mlflow()

    print("Cargando datos y features...")
    df_train, _, _, _ = load_raw()
    holdout_ids = load_frozen_holdout_ids()
    df_full = build_engineered_frame(df_train)
    df_dev = df_full[~df_full.index.isin(holdout_ids)].reset_index(drop=True)

    print(f"Dev set: {len(df_dev)} filas")

    # Preparar datos
    X_full = prepare_X_for_feature_set(df_dev, "E13_full_leakage_safe_features")
    y_full = df_dev[TARGET].reset_index(drop=True)
    group_key = make_group_key(df_dev)

    # CV V1
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    # E20 con best params
    print("\nEntrenando E20 con CV V1 para métricas finales...")
    e20_model = make_e15_hgb_e13()
    e20_best_params = {
        "learning_rate": 0.127,
        "max_iter": 152,
        "max_leaf_nodes": 38,
        "min_samples_leaf": 35,
        "l2_regularization": 0.84,
    }
    e20_model.set_params(**e20_best_params)

    # Cross-validation
    cv_scores = cross_validate(
        e20_model,
        X_full,
        y_full,
        cv=splitter,
        groups=group_key,
        scoring=["roc_auc", "average_precision"],
    )

    roc_auc_mean = cv_scores["test_roc_auc"].mean()
    roc_auc_std = cv_scores["test_roc_auc"].std()
    pr_auc_mean = cv_scores["test_average_precision"].mean()

    print(f"E20 ROC-AUC: {roc_auc_mean:.4f}±{roc_auc_std:.4f}")
    print(f"E20 PR-AUC:  {pr_auc_mean:.4f}")

    # Registrar en MLflow como stage=final
    print("\nRegistrando en MLflow como stage=final...")
    with mlflow.start_run(run_name="E20_final_candidate"):
        # Tags obligatorios
        mlflow.set_tag("project", "f1_pitstop")
        mlflow.set_tag("stage", "final")
        mlflow.set_tag("model_family", "hist_gradient_boosting")
        mlflow.set_tag("feature_set", "E13_full_leakage_safe_features")
        mlflow.set_tag("validation", "V1_group_stratified_kfold")
        mlflow.set_tag("seed", str(SEED))

        # Métricas obligatorias (CV)
        mlflow.log_metrics({
            "cv_roc_auc_mean": roc_auc_mean,
            "cv_roc_auc_std": roc_auc_std,
            "cv_pr_auc_mean": pr_auc_mean,
        })

        # Parámetros
        mlflow.log_params(e20_best_params)

        # Registro de modelo (no loguear artifacts, solo referencias)
        mlflow.log_metric("n_features", X_full.shape[1])

        print("  [OK] MLflow run registrado como stage=final")

    print("\nFase 11 completada")
    print("Candidato final E20 registrado en MLflow")


if __name__ == "__main__":
    main()
