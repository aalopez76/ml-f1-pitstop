"""Evaluacion por cross-validation (Fase 4+).

Usa exclusivamente la estrategia de CV oficial decidida en Fase 3: V1
(`StratifiedGroupKFold` por `(Race, Year)`, ver `README.md` seccion
"Validation strategy" y `src/f1pitstop/data/split.py`). V0 (aleatorio)
NUNCA se usa aqui — quedo demostrado en Fase 3 que infla el ROC-AUC de
forma optimista.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from f1pitstop.data.split import make_group_key, v1_group_stratified_kfold


@dataclass
class CVResult:
    run_name: str
    roc_auc_scores: list[float]
    pr_auc_scores: list[float]
    fit_seconds: list[float]
    predict_ms_per_1k_rows: list[float]
    n_features: int
    n_folds: int

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


def run_group_cv(
    run_name: str,
    make_model,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None = None,
    df_for_groups: pd.DataFrame | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> CVResult:
    """Corre V1 (`StratifiedGroupKFold` por `(Race, Year)`) y devuelve metricas
    por fold. `groups` se puede pasar precalculado, o derivarse de
    `df_for_groups` via `make_group_key`."""
    if groups is None:
        if df_for_groups is None:
            raise ValueError("Pasar `groups` o `df_for_groups` para derivar (Race, Year)")
        groups = make_group_key(df_for_groups)

    roc_aucs, pr_aucs, fit_times, predict_ms = [], [], [], []
    for train_idx, val_idx in v1_group_stratified_kfold(y, groups, n_splits=n_splits, seed=seed):
        model = make_model()

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        proba = model.predict_proba(X_val)[:, 1]
        predict_seconds = time.perf_counter() - t0
        predict_ms.append(predict_seconds * 1000 / (len(X_val) / 1000))

        roc_aucs.append(roc_auc_score(y_val, proba))
        pr_aucs.append(average_precision_score(y_val, proba))

    return CVResult(
        run_name=run_name,
        roc_auc_scores=roc_aucs,
        pr_auc_scores=pr_aucs,
        fit_seconds=fit_times,
        predict_ms_per_1k_rows=predict_ms,
        n_features=X.shape[1],
        n_folds=n_splits,
    )
