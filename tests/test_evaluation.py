"""Tests de src/f1pitstop/evaluation/cv.py (Fase 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pitstop.evaluation.cv import CVResult, run_group_cv
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


# --- Persistencia por fold (auditoria 2026-09-10) ---
#
# `to_metrics_dict()` colapsa los folds a mean/std, lo que impide comparaciones
# pareadas entre modelos. `to_fold_rows()` conserva el detalle.


def _toy_cv_result(n_folds: int = 5) -> CVResult:
    return CVResult(
        run_name="E99_toy",
        roc_auc_scores=[0.80 + 0.01 * i for i in range(n_folds)],
        pr_auc_scores=[0.50 + 0.01 * i for i in range(n_folds)],
        fit_seconds=[1.0] * n_folds,
        predict_ms_per_1k_rows=[2.0] * n_folds,
        n_features=10,
        n_folds=n_folds,
    )


def test_to_fold_rows_emits_one_row_per_fold():
    result = _toy_cv_result(n_folds=5)
    rows = result.to_fold_rows()

    assert len(rows) == result.n_folds
    assert [r["fold_idx"] for r in rows] == [0, 1, 2, 3, 4]
    assert all(r["run_name"] == "E99_toy" for r in rows)


def test_to_fold_rows_is_consistent_with_to_metrics_dict():
    """Las medias reconstruidas desde las filas por fold deben coincidir con
    las que ya publica `to_metrics_dict()` — si divergieran, el CSV por fold
    y el CSV agregado contarian historias distintas."""
    result = _toy_cv_result()
    rows = result.to_fold_rows()
    metrics = result.to_metrics_dict()

    assert np.mean([r["roc_auc"] for r in rows]) == pytest.approx(metrics["cv_roc_auc_mean"])
    assert np.mean([r["pr_auc"] for r in rows]) == pytest.approx(metrics["cv_pr_auc_mean"])


def test_paired_delta_is_computable_from_fold_rows():
    """El delta pareado exige el detalle por fold: restar dos medias no
    permite conocer la dispersion de la diferencia."""
    a = _toy_cv_result()
    b = _toy_cv_result()
    b.run_name = "E98_toy"
    b.roc_auc_scores = [s + 0.002 for s in b.roc_auc_scores]

    deltas = [
        rb["roc_auc"] - ra["roc_auc"]
        for ra, rb in zip(a.to_fold_rows(), b.to_fold_rows())
    ]
    assert len(deltas) == a.n_folds
    assert np.mean(deltas) == pytest.approx(0.002)
