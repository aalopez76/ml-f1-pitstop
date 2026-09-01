"""Pipelines skrub (Fase 5): ¿skrub simplifica el preprocessing sin
degradar la calidad?

`skrub.tabular_pipeline(estimator)` construye automaticamente un
`TableVectorizer` (mas imputacion/escalado si el estimador lo necesita)
configurado segun el tipo de estimador final — para
`HistGradientBoostingClassifier` usa soporte nativo de categoricas
(`ToCategorical`, sin one-hot); para `LogisticRegression` imputa e
imputa/escala con `SquashingScaler`. No se usa `TextEncoder` ni
embeddings: no hay campos textuales reales en este dataset (regla
explicita del spec, Fase 5, "No usar").

Mismo feature set y mismo seed que los baselines manuales de Fase 4
(`src/f1pitstop/models/baselines.py`) para que la comparacion sea
apples-to-apples.
"""

from __future__ import annotations

import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from f1pitstop.models.baselines import SEED, make_e01_logreg, make_e02_hgb


def make_e04_skrub_logreg() -> Pipeline:
    """E04: skrub.tabular_pipeline con LogisticRegression como estimador final."""
    return skrub.tabular_pipeline(LogisticRegression(max_iter=1000, random_state=SEED))


def make_e06_skrub_hgb() -> Pipeline:
    """E06: skrub.tabular_pipeline con HistGradientBoostingClassifier como
    estimador final (usa soporte nativo de categoricas, sin one-hot)."""
    return skrub.tabular_pipeline(HistGradientBoostingClassifier(random_state=SEED))


# Nombres de run segun el spec (Fase 5, seccion "Experimentos"). E03/E05
# reusan los mismos modelos que los baselines de Fase 4 (E01/E02) — son el
# mismo preprocesamiento manual, corridos bajo el nombre de esta
# comparacion para que quede junto a su contraparte skrub en MLflow.
SKRUB_COMPARISON_REGISTRY = {
    "E03_manual_preprocessing_logreg": {
        "make_model": make_e01_logreg,
        "model_family": "logistic_regression",
        "preprocessing": "manual",
    },
    "E04_skrub_tabular_pipeline_logreg": {
        "make_model": make_e04_skrub_logreg,
        "model_family": "logistic_regression",
        "preprocessing": "skrub",
    },
    "E05_manual_preprocessing_hgb": {
        "make_model": make_e02_hgb,
        "model_family": "hist_gradient_boosting",
        "preprocessing": "manual",
    },
    "E06_skrub_tabular_pipeline_hgb": {
        "make_model": make_e06_skrub_hgb,
        "model_family": "hist_gradient_boosting",
        "preprocessing": "skrub",
    },
}


def count_output_columns(model, X) -> int:
    """Numero de columnas que ve el estimador final tras el preprocessing,
    fiteado sobre `X` completo (fuera de CV, solo para medir complejidad de
    salida, no para evaluar). Si `model` no es un Pipeline (ej. el HGB
    manual, que recibe las columnas crudas sin transformar), retorna
    `X.shape[1]`."""
    if isinstance(model, Pipeline) and len(model.steps) > 1:
        preprocessing = model[:-1]
        preprocessing.fit(X)
        return preprocessing.transform(X).shape[1]
    return X.shape[1]
