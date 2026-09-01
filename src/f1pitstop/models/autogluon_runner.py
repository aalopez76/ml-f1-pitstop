"""AutoGluon challenger (Fase 8): benchmark AutoML honesto contra el
modelo manual ganador de Fase 7 (`E20_hist_gradient_boosting`, 0.8611
ROC-AUC).

**Protocolo de evaluacion (ambiguedad del holdout resuelta, ver
`.claude/rules/leakage-and-validation.md` §7):** se usa el MISMO CV V1
(`StratifiedGroupKFold` por `(Race, Year)`, 5 folds) que los modelos
manuales de Fase 4-7 — NO el holdout final congelado (`Year==2025`),
que se reserva intacto para la evaluacion confirmatoria unica de la
Fase 13 (regla no negociable 6 de CLAUDE.md). Cada fold reentrena un
`TabularPredictor` desde cero sobre el train del fold y predice sobre
el val del fold, exactamente igual que `evaluation.cv.run_group_cv()`
para los candidatos sklearn.

Entradas (spec, Fase 8): `A0` = raw cleaned data, `A1` = leakage-safe
engineered data. En este proyecto eso mapea directamente a los feature
sets ya establecidos: `A0 = E10_raw_features` (Fase 4/5, "crudo" en el
sentido de sin feature engineering de Fase 6, pero ya limpio de
leakage) y `A1 = E13_full_leakage_safe_features` (Fase 6, ganador del
ablation).

El predictor de cada fold se entrena en un directorio temporal
(`tempfile.TemporaryDirectory`) y se descarta al terminar — igual que
los modelos sklearn de `run_group_cv()`, que tampoco persisten pesos
por fold (`log_models=False` en runs preliminares, ver
`.claude/rules/experiment-tracking.md`). Evita acumular varios GB de
artefactos de AutoGluon en disco para 10 fits exploratorios (2 inputs x
5 folds) que no son el modelo final.

**Limitacion conocida del benchmark (hallazgo A_REVISAR del
`leakage-auditor`, no bloqueante, cierre de Fase 8):**
`predictor.fit(train_df, ...)` no recibe `tuning_data` explicito, asi
que AutoGluon hace su propio split/bagging interno DENTRO de
`train_df` (que ya es solo el train del fold externo V1). Ese split
interno NO es group-aware — podria mezclar filas de la misma
`(Race, Year)` entre el train y la validacion interna de AutoGluon.
Esto NO contamina la metrica reportada (`X_val`/`y_val` del fold
externo, con grupos disjuntos garantizados, solo se tocan en
`predict_proba`, nunca durante `fit()`), pero SI podria hacer que
AutoGluon optimice su seleccion de modelos/ensamble contra una senal
interna optimista — la misma clase de sesgo que V0 vs V1 cuantifico en
Fase 3, ocurriendo aqui dentro de la caja negra de AutoGluon en vez de
en la CV externa. No invalida la comparacion (ambos candidatos se
miden con el mismo V1 externo), pero es una asimetria real: el modelo
manual nunca tiene este problema porque no hace ningun split interno
propio.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from f1pitstop.data.split import make_group_key, v1_group_stratified_kfold

TARGET = "PitNextLap"
SEED = 42

# "Primera corrida" del spec (seccion 13): preset rapido, time_limit
# acotado. No se exige best_quality. Una segunda corrida con
# good_quality/mas tiempo solo se justifica si esta primera pasada
# muestra una mejora real sobre el modelo manual (0.8611 ROC-AUC).
DEFAULT_PRESETS = "medium_quality"
DEFAULT_TIME_LIMIT_SECONDS = 120


@dataclass
class AutoGluonCVResult:
    run_name: str
    roc_auc_scores: list[float]
    pr_auc_scores: list[float]
    fit_seconds: list[float]
    predict_ms_per_1k_rows: list[float]
    n_features: int
    n_folds: int
    presets: str
    time_limit: int

    @property
    def roc_auc_mean(self) -> float:
        return float(np.mean(self.roc_auc_scores))

    @property
    def roc_auc_std(self) -> float:
        return float(np.std(self.roc_auc_scores))

    @property
    def pr_auc_mean(self) -> float:
        return float(np.mean(self.pr_auc_scores))

    @property
    def pr_auc_std(self) -> float:
        return float(np.std(self.pr_auc_scores))

    @property
    def fit_seconds_mean(self) -> float:
        return float(np.mean(self.fit_seconds))

    @property
    def predict_ms_per_1k_rows_mean(self) -> float:
        return float(np.mean(self.predict_ms_per_1k_rows))

    def to_metrics_dict(self) -> dict:
        return {
            "cv_roc_auc_mean": self.roc_auc_mean,
            "cv_roc_auc_std": self.roc_auc_std,
            "cv_pr_auc_mean": self.pr_auc_mean,
            "cv_pr_auc_std": self.pr_auc_std,
            "fit_seconds": self.fit_seconds_mean,
            "predict_ms_per_1k_rows": self.predict_ms_per_1k_rows_mean,
            "n_features": self.n_features,
        }


def run_autogluon_group_cv(
    run_name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = TARGET,
    groups: pd.Series | None = None,
    n_splits: int = 5,
    seed: int = SEED,
    presets: str = DEFAULT_PRESETS,
    time_limit: int = DEFAULT_TIME_LIMIT_SECONDS,
) -> AutoGluonCVResult:
    """Corre V1 (`StratifiedGroupKFold` por `(Race, Year)`) con un
    `TabularPredictor` de AutoGluon reentrenado desde cero en cada fold,
    exactamente igual que `evaluation.cv.run_group_cv()` para los
    candidatos sklearn — misma particion externa, misma metrica."""
    from autogluon.tabular import TabularPredictor

    # Cinturon de seguridad barato (hallazgo A_REVISAR del leakage-auditor
    # en el cierre de Fase 8): si alguien agrega el target por error a un
    # feature set, nada mas aqui lo detectaria (pandas no rechaza columnas
    # duplicadas en `feature_cols + [target_col]` silenciosamente, pero
    # tampoco lo valida).
    assert target_col not in feature_cols, (
        f"'{target_col}' no debe estar en feature_cols (fuga directa del target)"
    )

    if groups is None:
        groups = make_group_key(df)
    y = df[target_col]

    roc_aucs, pr_aucs, fit_times, predict_ms = [], [], [], []
    for train_idx, val_idx in v1_group_stratified_kfold(y, groups, n_splits=n_splits, seed=seed):
        train_df = df.iloc[train_idx][feature_cols + [target_col]]
        val_df = df.iloc[val_idx]
        y_val = val_df[target_col]
        X_val = val_df[feature_cols]

        with tempfile.TemporaryDirectory(prefix=f"autogluon_{run_name}_") as tmp_dir:
            predictor = TabularPredictor(
                label=target_col,
                problem_type="binary",
                eval_metric="roc_auc",
                path=str(Path(tmp_dir) / "model"),
                verbosity=0,
            )
            t0 = time.perf_counter()
            predictor.fit(train_df, presets=presets, time_limit=time_limit)
            fit_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            proba = predictor.predict_proba(X_val)[1]
            predict_seconds = time.perf_counter() - t0
            predict_ms.append(predict_seconds * 1000 / (len(X_val) / 1000))

            roc_aucs.append(roc_auc_score(y_val, proba))
            pr_aucs.append(average_precision_score(y_val, proba))

    return AutoGluonCVResult(
        run_name=run_name,
        roc_auc_scores=roc_aucs,
        pr_auc_scores=pr_aucs,
        fit_seconds=fit_times,
        predict_ms_per_1k_rows=predict_ms,
        n_features=len(feature_cols),
        n_folds=n_splits,
        presets=presets,
        time_limit=time_limit,
    )
