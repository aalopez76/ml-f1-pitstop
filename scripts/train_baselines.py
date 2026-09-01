"""Fase 4 — Baselines: E00_dummy, E01_logreg_basic, E02_hgb_basic.

Objetivo (spec, Fase 4): establecer el valor incremental real de un modelo
simple sobre el prior. Usa el feature set "leakage-safe" y la estrategia
de CV V1 decididos en Fase 3 (ver `README.md`, seccion "Validation
strategy"). Corre SOLO sobre el conjunto de desarrollo — el holdout final
(`Year == 2025`) nunca se toca aqui.

Uso: `uv run python scripts/train_baselines.py`
Loguea a MLflow (experimento `f1_pitstop`, backend sqlite en `mlruns/`) y
escribe `artifacts/tables/baseline_results.csv`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from f1pitstop.data.ingest import load_raw
from f1pitstop.data.split import load_frozen_holdout_ids, make_group_key
from f1pitstop.evaluation.cv import run_group_cv
from f1pitstop.models.baselines import BASELINE_REGISTRY, LEAKAGE_SAFE_FEATURES, SEED, prepare_X
from f1pitstop.tracking.mlflow_utils import log_run, setup_mlflow

TARGET = "PitNextLap"
VALIDATION_NAME = "V1_group_kfold_race_year_5fold"


def main() -> None:
    train, _test, _sample_sub, _reports = load_raw()
    holdout_ids = load_frozen_holdout_ids()
    dev = train[~train["id"].isin(holdout_ids)].reset_index(drop=True)
    print(f"dev set (excluye holdout final 2025): {len(dev)} filas, {len(train)} filas totales")

    X = prepare_X(dev)
    y = dev[TARGET]
    groups = make_group_key(dev)

    setup_mlflow()

    rows = []
    for run_name, spec in BASELINE_REGISTRY.items():
        print(f"\n=== {run_name} ===")
        result = run_group_cv(run_name, spec["make_model"], X, y, groups=groups, seed=SEED)
        metrics = result.to_metrics_dict()

        tags = {
            "project": "f1_pitstop",
            "stage": "baseline",
            "model_family": spec["model_family"],
            "feature_set": "leakage_safe_v1",
            "validation": VALIDATION_NAME,
            "seed": str(SEED),
        }
        params = {
            "n_splits": result.n_folds,
            "features": ",".join(LEAKAGE_SAFE_FEATURES),
        }
        run_id = log_run(run_name, tags=tags, params=params, metrics=metrics, log_models=False)

        print(
            f"ROC-AUC {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f} | "
            f"PR-AUC {metrics['cv_pr_auc_mean']:.4f} | "
            f"fit {metrics['fit_seconds']:.2f}s | "
            f"predict {metrics['predict_ms_per_1k_rows']:.2f} ms/1k rows | "
            f"mlflow run_id={run_id}"
        )

        row = {"run_name": run_name, **metrics, "mlflow_run_id": run_id}
        rows.append(row)

    result_df = pd.DataFrame(rows)
    out_path = Path("artifacts/tables/baseline_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"\nGuardado en {out_path}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
