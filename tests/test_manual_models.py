"""Tests de src/f1pitstop/models/manual_models.py (Fase 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1pitstop.features.build import E13_FULL_LEAKAGE_SAFE_FEATURES
from f1pitstop.models.baselines import CATEGORICAL_FEATURES
from f1pitstop.models.manual_models import (
    DIVERSITY_REGISTRY,
    MANUAL_DEFAULTS_REGISTRY,
    NUMERIC_FEATURES_E13,
    PARAM_DISTRIBUTIONS,
    make_e14_logreg_e13,
    make_e15_hgb_e13,
    make_e16_extratrees_e13,
    make_e22_xgboost_e13,
    make_e23_catboost_e13,
    make_e24_lightgbm_e13,
    make_tunable_model,
)


def _toy_X_e13(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "LapNumber": rng.integers(1, 60, n).astype(float),
            "TyreLife": rng.integers(0, 30, n).astype(float),
            "Stint": rng.integers(1, 4, n).astype(float),
            "Position": rng.integers(1, 20, n).astype(float),
            "PitStop": rng.integers(0, 2, n).astype(float),
            "pit_stops_so_far": rng.integers(0, 3, n).astype(float),
            "recomputed_stint": rng.integers(1, 4, n).astype(float),
            "laptime_delta_prev": rng.normal(0, 2, n),
            "laps_since_last_pit": rng.integers(0, 20, n).astype(float),
            "Compound": rng.choice(["SOFT", "MEDIUM", "HARD"], n),
        }
    )
    # NaN en la primera vuelta de cada grupo, como produce features/temporal.py.
    df.loc[0, "laptime_delta_prev"] = np.nan
    # Mismo casteo que prepare_X_for_feature_set() en produccion (features/build.py):
    # XGBoost/LightGBM (E22/E24) exigen dtype "category" real, no "object", para
    # soporte nativo de categoricas.
    df["Compound"] = df["Compound"].astype("category")
    return df[E13_FULL_LEAKAGE_SAFE_FEATURES]


def test_numeric_features_e13_excludes_categorical():
    assert CATEGORICAL_FEATURES[0] not in NUMERIC_FEATURES_E13
    assert set(NUMERIC_FEATURES_E13) | set(CATEGORICAL_FEATURES) == set(
        E13_FULL_LEAKAGE_SAFE_FEATURES
    )


def test_e14_logreg_fits_and_predicts_proba_with_nan():
    X = _toy_X_e13()
    y = pd.Series(np.random.default_rng(1).integers(0, 2, len(X)))
    model = make_e14_logreg_e13()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_e15_hgb_fits_and_predicts_proba_with_native_nan_and_categorical():
    X = _toy_X_e13()
    y = pd.Series(np.random.default_rng(1).integers(0, 2, len(X)))
    model = make_e15_hgb_e13()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_e16_extratrees_fits_and_predicts_proba_with_nan():
    X = _toy_X_e13()
    y = pd.Series(np.random.default_rng(1).integers(0, 2, len(X)))
    model = make_e16_extratrees_e13()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_manual_defaults_registry_has_three_expected_runs():
    assert set(MANUAL_DEFAULTS_REGISTRY.keys()) == {
        "E14_logreg_e13_features",
        "E15_hgb_e13_features",
        "E16_extratrees_e13_features",
    }
    for spec in MANUAL_DEFAULTS_REGISTRY.values():
        assert "make_model" in spec
        assert "model_family" in spec


def test_param_distributions_cover_all_model_families():
    assert set(PARAM_DISTRIBUTIONS.keys()) == {
        spec["model_family"] for spec in MANUAL_DEFAULTS_REGISTRY.values()
    }


def test_make_tunable_model_fits_for_each_family():
    X = _toy_X_e13()
    y = pd.Series(np.random.default_rng(1).integers(0, 2, len(X)))
    for family in PARAM_DISTRIBUTIONS:
        model = make_tunable_model(family)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)


def test_e22_xgboost_fits_and_predicts_proba_with_native_nan_and_categorical():
    X = _toy_X_e13()
    y = pd.Series(np.random.default_rng(1).integers(0, 2, len(X)))
    model = make_e22_xgboost_e13()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_e23_catboost_fits_and_predicts_proba_with_native_nan_and_categorical():
    X = _toy_X_e13()
    y = pd.Series(np.random.default_rng(1).integers(0, 2, len(X)))
    model = make_e23_catboost_e13()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_e24_lightgbm_fits_and_predicts_proba_with_native_nan_and_categorical():
    X = _toy_X_e13()
    y = pd.Series(np.random.default_rng(1).integers(0, 2, len(X)))
    model = make_e24_lightgbm_e13()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_diversity_registry_has_three_expected_runs():
    assert set(DIVERSITY_REGISTRY.keys()) == {
        "E22_xgboost_e13_features",
        "E23_catboost_e13_features",
        "E24_lightgbm_e13_features",
    }
    for spec in DIVERSITY_REGISTRY.values():
        assert "make_model" in spec
        assert "model_family" in spec
