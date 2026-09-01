"""Fase 6 — Feature engineering F1: ablation E10 -> E11 -> E12 -> E13.

Cada familia entra sola contra el mismo baseline (E10 = feature set
leakage-safe de Fase 4/5, sin cambios): E11 = E10 + basic domain, E12 =
E10 + temporal, E13 = E10 + ambas. Mismo modelo (HGB, la familia mas
fuerte de Fase 4/5) y misma CV V1 en las 4 corridas — la unica variable es
el feature set, para poder atribuir cualquier cambio de ROC-AUC a la
familia de features y no a otra cosa.

Corre SOLO sobre el conjunto de desarrollo — el holdout final (`Year ==
2025`) nunca se toca aqui.

Uso: `uv run python scripts/phase6_feature_ablation.py`
Loguea a MLflow y escribe `artifacts/tables/feature_ablation_results.csv`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from f1pitstop.data.ingest import load_raw
from f1pitstop.data.split import load_frozen_holdout_ids, make_group_key
from f1pitstop.evaluation.cv import run_group_cv
from f1pitstop.features.build import (
    FEATURE_SET_REGISTRY,
    build_engineered_frame,
    prepare_X_for_feature_set,
)
from f1pitstop.models.baselines import SEED, make_e02_hgb
from f1pitstop.tracking.mlflow_utils import log_run, setup_mlflow

TARGET = "PitNextLap"
VALIDATION_NAME = "V1_group_kfold_race_year_5fold"


def main() -> None:
    train, _test, _sample_sub, _reports = load_raw()
    holdout_ids = load_frozen_holdout_ids()
    dev = train[~train["id"].isin(holdout_ids)].reset_index(drop=True)
    print(f"dev set (excluye holdout final 2025): {len(dev)} filas, {len(train)} filas totales")

    print("Calculando features engineered (una sola pasada para todas las familias)...")
    engineered = build_engineered_frame(dev)

    y = dev[TARGET]
    groups = make_group_key(dev)

    setup_mlflow()

    rows = []
    for run_name, feature_cols in FEATURE_SET_REGISTRY.items():
        print(f"\n=== {run_name} ({len(feature_cols)} features) ===")
        X = prepare_X_for_feature_set(engineered, run_name)
        result = run_group_cv(run_name, make_e02_hgb, X, y, groups=groups, seed=SEED)
        metrics = result.to_metrics_dict()

        tags = {
            "project": "f1_pitstop",
            "stage": "features",
            "model_family": "hist_gradient_boosting",
            "feature_set": run_name,
            "validation": VALIDATION_NAME,
            "seed": str(SEED),
        }
        params = {"n_splits": result.n_folds, "features": ",".join(feature_cols)}
        run_id = log_run(run_name, tags=tags, params=params, metrics=metrics, log_models=False)

        print(
            f"ROC-AUC {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f} | "
            f"PR-AUC {metrics['cv_pr_auc_mean']:.4f} | "
            f"n_features {metrics['n_features']} | "
            f"mlflow run_id={run_id}"
        )

        rows.append({"run_name": run_name, **metrics, "mlflow_run_id": run_id})

    result_df = pd.DataFrame(rows)
    baseline_auc = result_df.loc[result_df["run_name"] == "E10_raw_features", "cv_roc_auc_mean"].iloc[0]
    result_df["delta_roc_auc_vs_e10"] = result_df["cv_roc_auc_mean"] - baseline_auc

    out_path = Path("artifacts/tables/feature_ablation_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"\nGuardado en {out_path}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
