"""Fase 3 — cuantificar H1: ¿cuanto infla V0 (aleatorio) el ROC-AUC frente
a V1 (group-aware) y una demostracion de V2 (temporal), comparado con una
estrategia group-aware?

Se ejecuta ANTES de tocar el holdout final: usa solo el conjunto de
desarrollo (anios 2022-2024). El anio 2025, ya congelado como holdout final
en `artifacts/tables/final_holdout_ids.csv`, NUNCA se usa aqui (regla no
negociable 6 de CLAUDE.md).

Modelo deliberadamente simple (HistGradientBoostingClassifier sobre
features "crudas" no derivadas, ninguna de las columnas marcadas
SUSPECTED_LEAKAGE): el objetivo de este experimento es aislar el efecto de
la estrategia de split, no medir el mejor modelo posible (eso es Fase 4+).

Uso: `uv run python scripts/phase3_quantify_h1.py`
Escribe: `artifacts/tables/cv_strategy_comparison.csv`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from f1pitstop.data.ingest import load_raw
from f1pitstop.data.split import (
    load_frozen_holdout_ids,
    make_group_key,
    v0_stratified_kfold,
    v1_group_stratified_kfold,
    v2_temporal_split,
)

SEED = 42
N_SPLITS = 5

# Features "crudas", sin ninguna de las columnas SUSPECTED_LEAKAGE
# (LapTime_Delta, Cumulative_Degradation, RaceProgress, Position_Change) ni
# Race/Driver/Year (ver justificacion en el modulo, docstring superior).
RAW_FEATURES = ["LapNumber", "TyreLife", "Stint", "Position", "PitStop", "Compound"]
TARGET = "PitNextLap"


def _prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[RAW_FEATURES].copy()
    X["Compound"] = X["Compound"].astype("category")
    return X


def _fit_eval(X_train, y_train, X_val, y_val) -> float:
    model = HistGradientBoostingClassifier(
        random_state=SEED, categorical_features=["Compound"]
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, proba)


def main() -> None:
    train, _test, _sample_sub, _reports = load_raw()

    holdout_ids = load_frozen_holdout_ids()
    dev = train[~train["id"].isin(holdout_ids)].reset_index(drop=True)
    print(f"dev set (excluye holdout final 2025): {len(dev)} filas, {len(train)} filas totales")

    X = _prepare_X(dev)
    y = dev[TARGET]
    groups = make_group_key(dev)

    rows = []

    # V0: StratifiedKFold aleatorio
    v0_aucs = []
    for train_idx, val_idx in v0_stratified_kfold(y, n_splits=N_SPLITS, seed=SEED):
        auc = _fit_eval(X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx], y.iloc[val_idx])
        v0_aucs.append(auc)
    rows.append(
        {
            "strategy": "V0_random_kfold",
            "roc_auc_mean": np.mean(v0_aucs),
            "roc_auc_std": np.std(v0_aucs),
            "n_folds": len(v0_aucs),
            "note": "ignora grupos (Race, Year); baseline optimista de referencia",
        }
    )

    # V1: StratifiedGroupKFold por (Race, Year)
    v1_aucs = []
    for train_idx, val_idx in v1_group_stratified_kfold(y, groups, n_splits=N_SPLITS, seed=SEED):
        auc = _fit_eval(X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx], y.iloc[val_idx])
        v1_aucs.append(auc)
    rows.append(
        {
            "strategy": "V1_group_kfold",
            "roc_auc_mean": np.mean(v1_aucs),
            "roc_auc_std": np.std(v1_aucs),
            "n_folds": len(v1_aucs),
            "note": "carrera (Race, Year) nunca aparece en train y val a la vez",
        }
    )

    # V2 (demostracion, dentro de dev): train en 2022-2023, valida en 2024.
    # 2023 tiene una anomalia de tasa de pit casi nula (ver README de esta
    # fase); se deja igual en train porque asi luciria en produccion real
    # (el modelo no elige que anios ve).
    train_idx, val_idx = v2_temporal_split(dev, holdout_years=(2024,))
    auc_v2 = _fit_eval(
        X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx], y.iloc[val_idx]
    )
    rows.append(
        {
            "strategy": "V2_temporal_2022_2023_train_2024_val",
            "roc_auc_mean": auc_v2,
            "roc_auc_std": np.nan,
            "n_folds": 1,
            "note": "demostracion temporal dentro de dev; no usa el holdout final congelado",
        }
    )

    result = pd.DataFrame(rows)
    out_path = Path("artifacts/tables/cv_strategy_comparison.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(result.to_string(index=False))
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
