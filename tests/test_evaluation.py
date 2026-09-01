"""Tests de src/f1pitstop/evaluation/cv.py (Fase 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pitstop.evaluation.cv import run_group_cv
from f1pitstop.models.baselines import make_e00_dummy, make_e01_logreg


def _toy_dev(n_groups: int = 20, rows_per_group: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    races = [f"Race_{i}" for i in range(n_groups // 4)]
    years = [2022, 2023, 2024, 2025]
    rows = []
    for race in races:
        for year in years:
            base_rate = rng.uniform(0.1, 0.4)
            for lap in range(rows_per_group):
                rows.append(
                    {
                        "Race": race,
                        "Year": year,
                        "LapNumber": lap,
                        "TyreLife": float(lap % 15),
                        "Stint": lap // 15 + 1,
                        "Position": rng.integers(1, 20),
                        "PitStop": int(rng.random() < 0.1),
                        "LapTime (s)": rng.normal(90, 5),
                        "Compound": rng.choice(["SOFT", "MEDIUM", "HARD"]),
                        "PitNextLap": int(rng.random() < base_rate),
                    }
                )
    return pd.DataFrame(rows)


def test_run_group_cv_returns_expected_number_of_folds():
    df = _toy_dev()
    from f1pitstop.models.baselines import prepare_X

    X = prepare_X(df)
    y = df["PitNextLap"]
    result = run_group_cv("test_dummy", make_e00_dummy, X, y, df_for_groups=df, n_splits=5)
    assert result.n_folds == 5
    assert len(result.roc_auc_scores) == 5
    assert len(result.pr_auc_scores) == 5
    assert len(result.fit_seconds) == 5
    assert len(result.predict_ms_per_1k_rows) == 5


def test_run_group_cv_requires_groups_or_df():
    df = _toy_dev()
    from f1pitstop.models.baselines import prepare_X

    X = prepare_X(df)
    y = df["PitNextLap"]
    with pytest.raises(ValueError):
        run_group_cv("test_dummy", make_e00_dummy, X, y)


def test_dummy_baseline_roc_auc_near_random():
    """DummyClassifier(strategy='prior') no discrimina: ROC-AUC ~0.5."""
    df = _toy_dev()
    from f1pitstop.models.baselines import prepare_X

    X = prepare_X(df)
    y = df["PitNextLap"]
    result = run_group_cv("test_dummy", make_e00_dummy, X, y, df_for_groups=df, n_splits=5)
    assert 0.35 <= result.roc_auc_mean <= 0.65


def test_logreg_beats_dummy_on_informative_toy_data():
    """En datos toy donde PitStop=1 predice determinísticamente PitNextLap
    (senal fuerte inyectada), logreg debe superar claramente al dummy."""
    rng = np.random.default_rng(1)
    races = [f"Race_{i}" for i in range(5)]
    years = [2022, 2023, 2024, 2025]
    rows = []
    for race in races:
        for year in years:
            for lap in range(30):
                pitstop = int(rng.random() < 0.2)
                rows.append(
                    {
                        "Race": race,
                        "Year": year,
                        "LapNumber": lap,
                        "TyreLife": float(lap % 15),
                        "Stint": lap // 15 + 1,
                        "Position": rng.integers(1, 20),
                        "PitStop": pitstop,
                        "LapTime (s)": rng.normal(90, 5),
                        "Compound": rng.choice(["SOFT", "MEDIUM", "HARD"]),
                        "PitNextLap": pitstop,  # senal casi perfecta
                    }
                )
    df = pd.DataFrame(rows)
    from f1pitstop.models.baselines import prepare_X

    X = prepare_X(df)
    y = df["PitNextLap"]

    dummy_result = run_group_cv("dummy", make_e00_dummy, X, y, df_for_groups=df, n_splits=5)
    logreg_result = run_group_cv("logreg", make_e01_logreg, X, y, df_for_groups=df, n_splits=5)

    assert logreg_result.roc_auc_mean > dummy_result.roc_auc_mean
    assert logreg_result.roc_auc_mean > 0.9


def test_cv_result_to_metrics_dict_has_required_keys():
    df = _toy_dev()
    from f1pitstop.models.baselines import prepare_X

    X = prepare_X(df)
    y = df["PitNextLap"]
    result = run_group_cv("test_dummy", make_e00_dummy, X, y, df_for_groups=df, n_splits=5)
    metrics = result.to_metrics_dict()
    expected_keys = {
        "cv_roc_auc_mean",
        "cv_roc_auc_std",
        "cv_pr_auc_mean",
        "cv_pr_auc_std",
        "fit_seconds",
        "predict_ms_per_1k_rows",
        "n_features",
    }
    assert expected_keys <= set(metrics.keys())
