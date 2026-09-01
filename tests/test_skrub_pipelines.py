"""Tests de src/f1pitstop/models/skrub_pipelines.py (Fase 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1pitstop.models.baselines import make_e02_hgb, prepare_X
from f1pitstop.models.skrub_pipelines import (
    SKRUB_COMPARISON_REGISTRY,
    count_output_columns,
    make_e04_skrub_logreg,
    make_e06_skrub_hgb,
)


def _toy_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "LapNumber": rng.integers(1, 60, n),
            "TyreLife": rng.integers(0, 30, n).astype(float),
            "Stint": rng.integers(1, 4, n),
            "Position": rng.integers(1, 20, n),
            "PitStop": rng.integers(0, 2, n),
            "Compound": rng.choice(["SOFT", "MEDIUM", "HARD"], n),
            "PitNextLap": rng.integers(0, 2, n),
        }
    )


def test_skrub_comparison_registry_has_four_expected_runs():
    assert set(SKRUB_COMPARISON_REGISTRY.keys()) == {
        "E03_manual_preprocessing_logreg",
        "E04_skrub_tabular_pipeline_logreg",
        "E05_manual_preprocessing_hgb",
        "E06_skrub_tabular_pipeline_hgb",
    }
    preprocessing_values = {spec["preprocessing"] for spec in SKRUB_COMPARISON_REGISTRY.values()}
    assert preprocessing_values == {"manual", "skrub"}


def test_e04_skrub_logreg_fits_and_predicts_proba():
    df = _toy_df()
    X = prepare_X(df)
    y = df["PitNextLap"]
    model = make_e04_skrub_logreg()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(df), 2)


def test_e06_skrub_hgb_fits_and_predicts_proba():
    df = _toy_df()
    X = prepare_X(df)
    y = df["PitNextLap"]
    model = make_e06_skrub_hgb()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(df), 2)


def test_count_output_columns_for_manual_hgb_equals_input_columns():
    """El HGB manual no tiene paso de preprocesamiento explicito (pasa las
    columnas crudas, usa soporte nativo de categoricas)."""
    df = _toy_df()
    X = prepare_X(df)
    n_cols = count_output_columns(make_e02_hgb(), X)
    assert n_cols == X.shape[1]


def test_count_output_columns_for_skrub_logreg_is_positive_and_reasonable():
    df = _toy_df()
    X = prepare_X(df)
    n_cols = count_output_columns(make_e04_skrub_logreg(), X)
    # al menos tantas columnas como el input (imputacion agrega indicadores,
    # one-hot expande Compound)
    assert n_cols >= X.shape[1]
