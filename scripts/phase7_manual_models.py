"""Fase 7 — Modelos manuales.

Procedimiento del spec (seccion 12):
1. comparar defaults razonables (E14/E15/E16) sobre el feature set ganador
   de Fase 6 (`E13_full_leakage_safe_features`, 10 columnas) con CV V1;
2. seleccionar las 2 familias con mejor `cv_roc_auc_mean`;
3. tuning limitado (`RandomizedSearchCV`, 20 configuraciones, CV V1) SOLO
   sobre esas 2 (E20/E21 en la matriz del spec).

Ademas resuelve el punto abierto dejado en Fase 6 (ver HANDOFF.md,
"Proxima accion concreta"): ¿ayuda quitar `Stint` crudo de E13 ahora que
Fase 1/2 demostro que no es monotono en 81.6% de los grupos, dado que ya
existe `recomputed_stint` como alternativa leakage-safe?

Corre SOLO sobre el conjunto de desarrollo (excluye el holdout final
`Year == 2025`). Uso: `uv run python scripts/phase7_manual_models.py`

**Runtime esperado en esta maquina (8 cores): ~90-100 min en total**,
dominado por el tuning de `extra_trees` (~65-90 min por si solo, ver
`SEARCH_N_JOBS` mas abajo). Resultados de referencia (ya corridos y
documentados en `README.md`, seccion "Manual models (Fase 7)"): E20 HGB
tuneado 0.8611 ROC-AUC, E21 ExtraTrees tuneado 0.8530.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from f1pitstop.data.ingest import load_raw
from f1pitstop.data.split import load_frozen_holdout_ids, make_group_key, v1_group_stratified_kfold
from f1pitstop.evaluation.cv import run_group_cv
from f1pitstop.features.build import (
    E13_FULL_LEAKAGE_SAFE_FEATURES,
    build_engineered_frame,
    prepare_X_for_feature_set,
)
from f1pitstop.models.baselines import SEED
from f1pitstop.models.manual_models import (
    MANUAL_DEFAULTS_REGISTRY,
    PARAM_DISTRIBUTIONS,
    make_e15_hgb_e13,
    make_tunable_model,
)
from f1pitstop.tracking.mlflow_utils import log_run, setup_mlflow

TARGET = "PitNextLap"
VALIDATION_NAME = "V1_group_kfold_race_year_5fold"
FEATURE_SET_NAME = "E13_full_leakage_safe_features"
N_TUNING_ITER = 20
TABLES_DIR = Path("artifacts/tables")


def load_dev_e13() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    train, _test, _sample_sub, _reports = load_raw()
    holdout_ids = load_frozen_holdout_ids()
    dev = train[~train["id"].isin(holdout_ids)].reset_index(drop=True)
    engineered = build_engineered_frame(dev)
    X = prepare_X_for_feature_set(engineered, FEATURE_SET_NAME)
    y = engineered[TARGET]
    groups = make_group_key(engineered)
    return X, y, groups


def step1_compare_defaults(X, y, groups) -> pd.DataFrame:
    print("\n### Paso 1: comparar defaults (E14/E15/E16) sobre E13 con CV V1 ###")
    rows = []
    for run_name, spec in MANUAL_DEFAULTS_REGISTRY.items():
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
        params = {"n_splits": result.n_folds, "features": ",".join(E13_FULL_LEAKAGE_SAFE_FEATURES)}
        run_id = log_run(run_name, tags=tags, params=params, metrics=metrics, log_models=False)

        print(
            f"{run_name}: ROC-AUC {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f} | "
            f"fit {metrics['fit_seconds']:.2f}s | mlflow run_id={run_id}"
        )
        rows.append(
            {"run_name": run_name, "model_family": spec["model_family"], **metrics, "mlflow_run_id": run_id}
        )

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "phase7_defaults_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"Guardado en {out_path}")
    return df


def step2_select_top2(defaults_df: pd.DataFrame) -> list[str]:
    ranked = defaults_df.sort_values("cv_roc_auc_mean", ascending=False)
    top2 = ranked["model_family"].head(2).tolist()
    print(f"\n### Paso 2: familias seleccionadas para tuning: {top2} ###")
    return top2


# n_jobs de RandomizedSearchCV por familia: -1 (todos los cores) para
# familias sin paralelismo interno propio; un valor acotado para
# `extra_trees`, cuyo estimador ya usa n_jobs=2 internamente (ver
# `make_tunable_model`) -- anidar dos `n_jobs=-1` sobre-suscribe los
# cores de la maquina (8 aqui) y el tuning deja de avanzar en un tiempo
# razonable (observado: >70 min sin completar antes de este fix).
SEARCH_N_JOBS = {"extra_trees": 4}


def step3_tune(model_family: str, X, y, groups, cv_folds, run_id_suffix: str) -> dict:
    run_name = f"{run_id_suffix}_{model_family}"
    print(f"\n### Paso 3: tuning {run_name} (RandomizedSearchCV, n_iter={N_TUNING_ITER}) ###")

    estimator = make_tunable_model(model_family)
    param_distributions = PARAM_DISTRIBUTIONS[model_family]
    search_n_jobs = SEARCH_N_JOBS.get(model_family, -1)

    mlflow.sklearn.autolog(log_models=False, disable=False, silent=True)
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_distributions,
        n_iter=N_TUNING_ITER,
        scoring="roc_auc",
        cv=cv_folds,
        random_state=SEED,
        n_jobs=search_n_jobs,
        refit=False,
    )
    search.fit(X, y)
    mlflow.sklearn.autolog(disable=True)

    best_idx = search.best_index_
    cv_results = search.cv_results_
    metrics = {
        "cv_roc_auc_mean": float(cv_results["mean_test_score"][best_idx]),
        "cv_roc_auc_std": float(cv_results["std_test_score"][best_idx]),
        "fit_seconds": float(cv_results["mean_fit_time"][best_idx]),
        "n_features": X.shape[1],
    }
    tags = {
        "project": "f1_pitstop",
        "stage": "tuning",
        "model_family": model_family,
        "feature_set": FEATURE_SET_NAME,
        "validation": VALIDATION_NAME,
        "seed": str(SEED),
    }
    params = {
        "n_iter": N_TUNING_ITER,
        "search_strategy": "RandomizedSearchCV",
        **{k: str(v) for k, v in search.best_params_.items()},
    }
    run_id = log_run(run_name, tags=tags, params=params, metrics=metrics, log_models=False)

    print(
        f"{run_name}: best ROC-AUC {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f} | "
        f"best_params={search.best_params_} | mlflow run_id={run_id}"
    )
    return {
        "run_name": run_name,
        "model_family": model_family,
        **metrics,
        "best_params": search.best_params_,
        "mlflow_run_id": run_id,
    }


def stint_ablation(X, y, groups) -> pd.DataFrame:
    """Resuelve el punto abierto de Fase 6: comparar E13 completo vs E13
    sin `Stint` crudo (dejando `recomputed_stint`), mismo modelo (HGB,
    ganador de Fase 4/5/6), misma CV V1."""
    print("\n### Ablation: Stint crudo vs recomputed_stint dentro de E13 (HGB) ###")
    rows = []

    result_full = run_group_cv("E13_with_raw_stint", make_e15_hgb_e13, X, y, groups=groups, seed=SEED)
    rows.append({"variant": "E13_with_raw_stint", "n_features": X.shape[1], **result_full.to_metrics_dict()})

    X_no_stint = X.drop(columns=["Stint"])
    result_no_stint = run_group_cv(
        "E13_without_raw_stint", make_e15_hgb_e13, X_no_stint, y, groups=groups, seed=SEED
    )
    rows.append(
        {"variant": "E13_without_raw_stint", "n_features": X_no_stint.shape[1], **result_no_stint.to_metrics_dict()}
    )

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "phase7_stint_ablation.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"Guardado en {out_path}")
    return df


def main() -> None:
    setup_mlflow()
    X, y, groups = load_dev_e13()
    print(f"dev set (excluye holdout final 2025): {len(X)} filas, {X.shape[1]} columnas ({FEATURE_SET_NAME})")

    defaults_df = step1_compare_defaults(X, y, groups)
    top2_families = step2_select_top2(defaults_df)

    cv_folds = v1_group_stratified_kfold(y, groups, n_splits=5, seed=SEED)
    tuning_rows = [
        step3_tune(family, X, y, groups, cv_folds, run_id_suffix="E20" if i == 0 else "E21")
        for i, family in enumerate(top2_families)
    ]
    tuning_df = pd.DataFrame(
        [{k: v for k, v in row.items() if k != "best_params"} for row in tuning_rows]
    )
    tuning_out = TABLES_DIR / "phase7_tuning_results.csv"
    tuning_df.to_csv(tuning_out, index=False)
    print(f"\nGuardado en {tuning_out}")
    for row in tuning_rows:
        print(f"{row['run_name']}: best_params={row['best_params']}")

    stint_ablation(X, y, groups)

    print("\n=== Resumen Fase 7 ===")
    print(defaults_df[["run_name", "cv_roc_auc_mean", "cv_roc_auc_std"]].to_string(index=False))
    print(tuning_df[["run_name", "cv_roc_auc_mean", "cv_roc_auc_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
