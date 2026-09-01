"""Fase 6 — aislamiento por-feature de la familia "temporal".

`E12_temporal_features` combinaba 3 sub-features. Un primer corrido del
ablation completo (`scripts/phase6_feature_ablation.py`) dio un resultado
peor que el baseline (E10), señal de que al menos una de las 3
sub-features era perjudicial (regla del spec: "no mezclar variables y
atribuir la mejora/el costo a la familia completa" — hay que aislar).

Este script agrega, UNA A LA VEZ, cada sub-feature de la familia temporal
sobre E10, para poder atribuir el efecto a la sub-feature exacta en vez de
a la familia completa. Resultado usado para decidir
`UNSTABLE_TEMPORAL_FEATURE_NAMES` en `src/f1pitstop/features/temporal.py`
(regla no negociable 9 de CLAUDE.md: toda afirmacion de mejora/costo
requiere validacion reproducible, no una corrida suelta).

Uso: `uv run python scripts/phase6_feature_isolation.py`
Loguea a MLflow y escribe `artifacts/tables/feature_isolation_results.csv`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from f1pitstop.data.ingest import load_raw
from f1pitstop.data.split import load_frozen_holdout_ids, make_group_key
from f1pitstop.evaluation.cv import run_group_cv
from f1pitstop.features.build import E10_RAW_FEATURES, build_engineered_frame
from f1pitstop.models.baselines import SEED, make_e02_hgb
from f1pitstop.tracking.mlflow_utils import log_run, setup_mlflow

TARGET = "PitNextLap"
VALIDATION_NAME = "V1_group_kfold_race_year_5fold"

# Las 3 sub-features originalmente propuestas para la familia "temporal"
# (ver TEMPORAL_FEATURE_NAMES / UNSTABLE_TEMPORAL_FEATURE_NAMES en
# src/f1pitstop/features/temporal.py).
CANDIDATE_TEMPORAL_FEATURES = ["laptime_delta_prev", "laptime_roll_mean_3", "laps_since_last_pit"]


def main() -> None:
    train, _test, _sample_sub, _reports = load_raw()
    holdout_ids = load_frozen_holdout_ids()
    dev = train[~train["id"].isin(holdout_ids)].reset_index(drop=True)
    print(f"dev set (excluye holdout final 2025): {len(dev)} filas, {len(train)} filas totales")

    engineered = build_engineered_frame(dev)
    y = dev[TARGET]
    groups = make_group_key(dev)

    setup_mlflow()

    rows = []
    # fila de referencia: E10 solo, para computar el delta de cada sub-feature
    X_e10 = engineered[E10_RAW_FEATURES].copy()
    X_e10["Compound"] = X_e10["Compound"].astype("category")
    e10_result = run_group_cv("E10_reference", make_e02_hgb, X_e10, y, groups=groups, seed=SEED)
    e10_auc = e10_result.roc_auc_mean
    print(f"\n=== E10_raw_features (referencia) ===\nROC-AUC {e10_auc:.4f}")

    for feature_name in CANDIDATE_TEMPORAL_FEATURES:
        run_name = f"E12_isolation_{feature_name}"
        cols = E10_RAW_FEATURES + [feature_name]
        X = engineered[cols].copy()
        X["Compound"] = X["Compound"].astype("category")

        result = run_group_cv(run_name, make_e02_hgb, X, y, groups=groups, seed=SEED)
        metrics = result.to_metrics_dict()
        delta = result.roc_auc_mean - e10_auc

        tags = {
            "project": "f1_pitstop",
            "stage": "features",
            "model_family": "hist_gradient_boosting",
            "feature_set": f"e10_plus_{feature_name}",
            "validation": VALIDATION_NAME,
            "seed": str(SEED),
        }
        params = {"n_splits": result.n_folds, "features": ",".join(cols)}
        run_id = log_run(run_name, tags=tags, params=params, metrics=metrics, log_models=False)

        print(
            f"\n=== +{feature_name} ===\n"
            f"ROC-AUC {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f} "
            f"(delta vs E10: {delta:+.4f}) | mlflow run_id={run_id}"
        )
        rows.append(
            {
                "feature_name": feature_name,
                "roc_auc_mean": metrics["cv_roc_auc_mean"],
                "roc_auc_std": metrics["cv_roc_auc_std"],
                "delta_roc_auc_vs_e10": delta,
                "mlflow_run_id": run_id,
            }
        )

    result_df = pd.DataFrame(rows)
    out_path = Path("artifacts/tables/feature_isolation_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"\nGuardado en {out_path}")
    print(f"E10 baseline ROC-AUC: {e10_auc:.4f}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
