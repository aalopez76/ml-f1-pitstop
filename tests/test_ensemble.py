"""Tests de src/f1pitstop/models/ensemble.py (Fase 14, E25)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pitstop.models.ensemble import build_oof_feature_matrix, make_e25_logit_stack


def test_build_oof_feature_matrix_one_column_per_model():
    oof = {
        "E20_hist_gradient_boosting": np.array([0.1, 0.8, 0.3]),
        "E22_xgboost_e13_features": np.array([0.2, 0.7, 0.4]),
    }
    X = build_oof_feature_matrix(oof)
    assert list(X.columns) == ["E20_hist_gradient_boosting", "E22_xgboost_e13_features"]
    assert len(X) == 3


def test_build_oof_feature_matrix_raises_on_empty_dict():
    with pytest.raises(ValueError):
        build_oof_feature_matrix({})


def test_build_oof_feature_matrix_raises_on_mismatched_lengths():
    oof = {
        "a": np.array([0.1, 0.2, 0.3]),
        "b": np.array([0.1, 0.2]),
    }
    with pytest.raises(ValueError):
        build_oof_feature_matrix(oof)


def test_make_e25_logit_stack_fits_and_predicts_proba():
    rng = np.random.default_rng(0)
    oof = {
        "E20_hist_gradient_boosting": rng.uniform(0, 1, 50),
        "E22_xgboost_e13_features": rng.uniform(0, 1, 50),
        "E23_catboost_e13_features": rng.uniform(0, 1, 50),
        "E24_lightgbm_e13_features": rng.uniform(0, 1, 50),
    }
    X = build_oof_feature_matrix(oof)
    y = pd.Series(rng.integers(0, 2, 50))

    model = make_e25_logit_stack()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (50, 2)
