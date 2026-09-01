"""Tests de src/f1pitstop/models/baselines.py (Fase 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1pitstop.models.baselines import (
    BASELINE_REGISTRY,
    CATEGORICAL_FEATURES,
    LEAKAGE_SAFE_FEATURES,
    NUMERIC_FEATURES,
    make_e00_dummy,
    make_e01_logreg,
    make_e02_hgb,
    prepare_X,
)


def _toy_df(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "LapNumber": rng.integers(1, 60, n),
            "TyreLife": rng.integers(0, 30, n).astype(float),
            "Stint": rng.integers(1, 4, n),
            "Position": rng.integers(1, 20, n),
            "PitStop": rng.integers(0, 2, n),
            "LapTime (s)": rng.normal(90, 5, n),
            "Compound": rng.choice(["SOFT", "MEDIUM", "HARD"], n),
            "PitNextLap": rng.integers(0, 2, n),
        }
    )


def test_leakage_safe_features_matches_numeric_plus_categorical():
    assert LEAKAGE_SAFE_FEATURES == NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_prepare_X_selects_only_leakage_safe_columns():
    df = _toy_df()
    X = prepare_X(df)
    assert list(X.columns) == LEAKAGE_SAFE_FEATURES


def test_prepare_X_casts_categorical_columns():
    df = _toy_df()
    X = prepare_X(df)
    for c in CATEGORICAL_FEATURES:
        assert str(X[c].dtype) == "category"


def test_e00_dummy_fits_and_predicts_proba():
    df = _toy_df()
    X = prepare_X(df)
    y = df["PitNextLap"]
    model = make_e00_dummy()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(df), 2)


def test_e01_logreg_fits_and_predicts_proba():
    df = _toy_df()
    X = prepare_X(df)
    y = df["PitNextLap"]
    model = make_e01_logreg()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(df), 2)


def test_e02_hgb_fits_and_predicts_proba_with_native_categorical():
    df = _toy_df()
    X = prepare_X(df)
    y = df["PitNextLap"]
    model = make_e02_hgb()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(df), 2)


def test_baseline_registry_has_three_expected_runs():
    assert set(BASELINE_REGISTRY.keys()) == {"E00_dummy", "E01_logreg_basic", "E02_hgb_basic"}
    for spec in BASELINE_REGISTRY.values():
        assert "make_model" in spec
        assert "model_family" in spec
