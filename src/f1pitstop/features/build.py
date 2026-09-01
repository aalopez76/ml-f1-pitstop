"""Orquestacion de feature sets para el ablation de Fase 6 (E10-E13).

Cada familia entra por ablation (base + una familia a la vez, regla del
spec: "no mezclar 40 nuevas variables y atribuir la mejora a 'feature
engineering'"). Reusa el feature set leakage-safe de Fase 4/5
(`LapNumber`, `TyreLife`, `Stint`, `Position`, `PitStop`, `Compound` —
`LapTime (s)` cruda excluida por inestabilidad, ver
`src/f1pitstop/models/baselines.py`).
"""

from __future__ import annotations

import pandas as pd

from f1pitstop.features.temporal import (
    BASIC_DOMAIN_FEATURE_NAMES,
    TEMPORAL_FEATURE_NAMES,
    add_basic_domain_features,
    add_temporal_features,
    add_winsorized_laptime,
)
from f1pitstop.models.baselines import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# E10: exactamente el feature set leakage-safe de Fase 4/5, sin cambios.
E10_RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# E11: E10 + familia "basic domain" (pit_stops_so_far, recomputed_stint).
E11_BASIC_DOMAIN_FEATURES = E10_RAW_FEATURES + BASIC_DOMAIN_FEATURE_NAMES

# E12: E10 + familia "temporal" (laptime_delta_prev, laptime_roll_mean_3,
# laps_since_last_pit) — requiere LapTime_s_winsorized, no la columna cruda.
E12_TEMPORAL_FEATURES = E10_RAW_FEATURES + TEMPORAL_FEATURE_NAMES

# E13: E10 + ambas familias.
E13_FULL_LEAKAGE_SAFE_FEATURES = E10_RAW_FEATURES + BASIC_DOMAIN_FEATURE_NAMES + TEMPORAL_FEATURE_NAMES

FEATURE_SET_REGISTRY = {
    "E10_raw_features": E10_RAW_FEATURES,
    "E11_basic_domain_features": E11_BASIC_DOMAIN_FEATURES,
    "E12_temporal_features": E12_TEMPORAL_FEATURES,
    "E13_full_leakage_safe_features": E13_FULL_LEAKAGE_SAFE_FEATURES,
}


def build_engineered_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula TODAS las familias de features de una vez sobre `df`
    (una sola pasada, mas eficiente que llamar cada builder por separado
    cuando se necesitan varios feature sets del registry sobre el mismo
    `df`). Preserva filas y orden original (misma longitud e indice)."""
    out = add_winsorized_laptime(df)
    out = add_basic_domain_features(out)
    out = add_temporal_features(out)
    return out


def prepare_X_for_feature_set(df_engineered: pd.DataFrame, feature_set_name: str) -> pd.DataFrame:
    """Selecciona las columnas del feature set `feature_set_name` (uno de
    `FEATURE_SET_REGISTRY`) de un frame ya pasado por `build_engineered_frame()`,
    y castea `Compound` a categorica (igual que `models.baselines.prepare_X`)."""
    if feature_set_name not in FEATURE_SET_REGISTRY:
        raise ValueError(
            f"feature_set_name desconocido: {feature_set_name!r}. "
            f"Opciones: {list(FEATURE_SET_REGISTRY)}"
        )
    columns = FEATURE_SET_REGISTRY[feature_set_name]
    X = df_engineered[columns].copy()
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X
