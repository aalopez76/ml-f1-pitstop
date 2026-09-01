"""Modelos baseline (Fase 4): E00 dummy, E01 logreg, E02 HistGradientBoosting.

Objetivo del spec (Fase 4): establecer el valor incremental real de un
modelo simple frente al prior. Preprocesamiento "sencillo" a proposito —
la ingenieria de features completa es Fase 6, esto es deliberadamente
minimo para medir una linea base honesta.

Todos los modelos usan el mismo feature set "leakage-safe" definido en
Fase 3 (ver `artifacts/reports/leakage_checklist_fase3.md`):
`LEAKAGE_SAFE_FEATURES`.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Feature set "leakage-safe" decidido en Fase 3 (leakage_checklist_fase3.md):
# PitStop, Position y LapTime (s) se aceptaron respecto al checklist de leakage;
# RaceProgress, Cumulative_Degradation, LapTime_Delta y Position_Change se
# excluyeron (respuesta incierta al checklist de 5 preguntas). Driver se deja
# fuera del baseline por alta cardinalidad y comportamiento inconsistente con un
# grid real de F1 (ver eda_report.md, hipotesis 7).
#
# `LapTime (s)` paso el checklist de leakage (no usa t+1, no usa el target) pero
# se excluye igual del set por defecto de Fase 4: un ablation con V1 (CV
# group-aware, ver README "Validation strategy") mostro que incluirla le cuesta
# ~0.075 ROC-AUC al HGB baseline (0.815 -> 0.740). La columna tiene outliers
# extremos (hasta 2507s vs media ~91s, ver artifacts/tables/baseline_results.csv
# y HANDOFF.md) que probablemente son vueltas con safety car/bandera roja —
# artefactos especificos de cada carrera que no generalizan a carreras nuevas en
# V1. No es un problema de leakage (no usa informacion de t+1 ni del target), es
# un problema de estabilidad/generalizacion — precisamente el tipo de cosa que
# V1 (group-aware) esta disenado para exponer y V0 (aleatorio) hubiese ocultado.
# Candidata a Fase 6 para winsorizar/clip/log-transform y re-evaluar.
UNSTABLE_FEATURES = ["LapTime (s)"]

NUMERIC_FEATURES = ["LapNumber", "TyreLife", "Stint", "Position", "PitStop"]
CATEGORICAL_FEATURES = ["Compound"]
LEAKAGE_SAFE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

SEED = 42


def make_e00_dummy() -> DummyClassifier:
    """E00: DummyClassifier(strategy='prior') — predice la tasa base del target."""
    return DummyClassifier(strategy="prior", random_state=SEED)


def make_e01_logreg() -> Pipeline:
    """E01: LogisticRegression con preprocesamiento sencillo
    (StandardScaler + OneHotEncoder), sin regularizacion ajustada (eso es Fase 7+)."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )


def make_e02_hgb() -> HistGradientBoostingClassifier:
    """E02: HistGradientBoostingClassifier, soporte nativo de categoricas
    (sin one-hot), sin tuning de hiperparametros (eso es Fase 7+)."""
    return HistGradientBoostingClassifier(
        random_state=SEED, categorical_features=CATEGORICAL_FEATURES
    )


BASELINE_REGISTRY = {
    "E00_dummy": {
        "make_model": make_e00_dummy,
        "model_family": "dummy",
        "needs_categorical_dtype": False,
    },
    "E01_logreg_basic": {
        "make_model": make_e01_logreg,
        "model_family": "logistic_regression",
        "needs_categorical_dtype": False,
    },
    "E02_hgb_basic": {
        "make_model": make_e02_hgb,
        "model_family": "hist_gradient_boosting",
        "needs_categorical_dtype": True,
    },
}


def prepare_X(df):
    """Selecciona el feature set leakage-safe y castea categoricas.
    `df` debe tener las columnas de `LEAKAGE_SAFE_FEATURES`."""
    X = df[LEAKAGE_SAFE_FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")
    return X
