"""Fase 8 — AutoGluon challenger.

Benchmark AutoML honesto contra el modelo manual ganador de Fase 7
(`E20_hist_gradient_boosting`, 0.8611 ROC-AUC). Protocolo de evaluacion
(ver `src/f1pitstop/models/autogluon_runner.py` y
`.claude/rules/leakage-and-validation.md` §7): CV V1, 5 folds, mismo
protocolo que los modelos manuales. El holdout final (`Year==2025`) NO
se toca — se reserva para la evaluacion confirmatoria unica de la Fase
13, comun a este candidato y al manual si AutoGluon resulta finalista.

Entradas:
- A00 = `E10_raw_features` (Fase 4/5, leakage-safe, sin feature
  engineering de Fase 6).
- A01 = `E13_full_leakage_safe_features` (Fase 6, ganador del ablation).

Primera corrida (esta): `presets="medium_quality"`, `time_limit=120s`
por fold (10 fits totales: 2 inputs x 5 folds). Una segunda corrida con
`good_quality`/mas presupuesto solo se justifica si esta muestra una
mejora real sobre 0.8611.

Uso: `uv run python scripts/phase8_autogluon.py`

**Runtime esperado:** variable segun cuantos modelos base entrenables
alcance AutoGluon en 120s por fold sobre ~275k filas de train; estimar
15-30 min totales para los 10 fits (ver leccion de Fase 7 sobre
paralelismo anidado — AutoGluon gestiona su propio paralelismo interno,
no se anida con nada externo aqui).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from f1pitstop.data.ingest import load_raw
from f1pitstop.data.split import load_frozen_holdout_ids, make_group_key
from f1pitstop.features.build import (
    E10_RAW_FEATURES,
    E13_FULL_LEAKAGE_SAFE_FEATURES,
    build_engineered_frame,
)
from f1pitstop.models.autogluon_runner import (
    DEFAULT_PRESETS,
    DEFAULT_TIME_LIMIT_SECONDS,
    TARGET,
    run_autogluon_group_cv,
)
from f1pitstop.models.baselines import SEED
from f1pitstop.tracking.mlflow_utils import log_run, setup_mlflow

VALIDATION_NAME = "V1_group_kfold_race_year_5fold"
TABLES_DIR = Path("artifacts/tables")

# E20_hist_gradient_boosting tuneado (Fase 7, artifacts/tables/phase7_tuning_results.csv).
MANUAL_CHAMPION_ROC_AUC = 0.8610546687688133

AUTOGLUON_INPUT_REGISTRY = {
    "A00_autogluon_raw": {
        "features": E10_RAW_FEATURES,
        "feature_set_name": "E10_raw_features",
    },
    "A01_autogluon_engineered": {
        "features": E13_FULL_LEAKAGE_SAFE_FEATURES,
        "feature_set_name": "E13_full_leakage_safe_features",
    },
}


def load_dev() -> tuple[pd.DataFrame, pd.Series]:
    train, _test, _sample_sub, _reports = load_raw()
    holdout_ids = load_frozen_holdout_ids()
    dev = train[~train["id"].isin(holdout_ids)].reset_index(drop=True)
    engineered = build_engineered_frame(dev)
    groups = make_group_key(engineered)
    return engineered, groups


def main() -> None:
    setup_mlflow()
    df, groups = load_dev()
    print(f"dev set (excluye holdout final 2025): {len(df)} filas")

    rows = []
    for run_name, spec in AUTOGLUON_INPUT_REGISTRY.items():
        print(f"\n=== {run_name} (presets={DEFAULT_PRESETS}, time_limit={DEFAULT_TIME_LIMIT_SECONDS}s/fold) ===")
        result = run_autogluon_group_cv(
            run_name,
            df,
            feature_cols=spec["features"],
            target_col=TARGET,
            groups=groups,
            seed=SEED,
            presets=DEFAULT_PRESETS,
            time_limit=DEFAULT_TIME_LIMIT_SECONDS,
        )
        metrics = result.to_metrics_dict()

        tags = {
            "project": "f1_pitstop",
            "stage": "automl",
            "model_family": "autogluon",
            "feature_set": spec["feature_set_name"],
            "validation": VALIDATION_NAME,
            "seed": str(SEED),
        }
        params = {
            "n_splits": result.n_folds,
            "presets": DEFAULT_PRESETS,
            "time_limit_seconds": DEFAULT_TIME_LIMIT_SECONDS,
            "features": ",".join(spec["features"]),
        }
        run_id = log_run(run_name, tags=tags, params=params, metrics=metrics, log_models=False)

        print(
            f"{run_name}: ROC-AUC {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f} | "
            f"fit {metrics['fit_seconds']:.1f}s | mlflow run_id={run_id}"
        )
        rows.append({"run_name": run_name, **metrics, "mlflow_run_id": run_id})

    result_df = pd.DataFrame(rows)
    result_df["delta_vs_manual_champion"] = result_df["cv_roc_auc_mean"] - MANUAL_CHAMPION_ROC_AUC
    out_path = TABLES_DIR / "phase8_autogluon_results.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nGuardado en {out_path}")
    print(result_df.to_string(index=False))
    print(f"\nModelo manual ganador (Fase 7, E20_hist_gradient_boosting): {MANUAL_CHAMPION_ROC_AUC:.4f} ROC-AUC")


if __name__ == "__main__":
    main()
