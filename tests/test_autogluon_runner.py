"""Tests de src/f1pitstop/models/autogluon_runner.py (Fase 8).

`time_limit` muy corto (patron ya usado en `scripts/smoke_test_stack.py`,
paso 10) para que el test sea rapido — no mide calidad del modelo, solo
que el wiring de CV V1 + AutoGluon (fit/predict_proba/columnas) funciona.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pitstop.models.autogluon_runner import run_autogluon_group_cv

pytestmark = pytest.mark.slow


def _toy_df(n_groups: int = 6, rows_per_group: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for g in range(n_groups):
        for lap in range(rows_per_group):
            rows.append(
                {
                    "Race": f"race{g}",
                    "Year": 2023,
                    "LapNumber": lap,
                    "TyreLife": rng.integers(0, 30),
                    "Stint": rng.integers(1, 4),
                    "Position": rng.integers(1, 20),
                    "PitStop": rng.integers(0, 2),
                    "Compound": rng.choice(["SOFT", "MEDIUM", "HARD"]),
                    "PitNextLap": rng.integers(0, 2),
                }
            )
    return pd.DataFrame(rows)


def test_run_autogluon_group_cv_returns_metrics_for_each_fold():
    df = _toy_df()
    feature_cols = ["LapNumber", "TyreLife", "Stint", "Position", "PitStop", "Compound"]
    result = run_autogluon_group_cv(
        "test_run",
        df,
        feature_cols=feature_cols,
        n_splits=3,
        time_limit=10,
    )
    assert result.n_folds == 3
    assert len(result.roc_auc_scores) == 3
    assert len(result.pr_auc_scores) == 3
    assert result.n_features == len(feature_cols)
    metrics = result.to_metrics_dict()
    assert 0.0 <= metrics["cv_roc_auc_mean"] <= 1.0


def test_run_autogluon_group_cv_rejects_target_in_feature_cols():
    """Cinturon de seguridad barato (hallazgo A_REVISAR del
    leakage-auditor, cierre de Fase 8): si el target se cuela en
    `feature_cols`, `run_autogluon_group_cv` debe fallar rapido, sin
    llegar a entrenar nada."""
    df = _toy_df()
    with pytest.raises(AssertionError, match="PitNextLap"):
        run_autogluon_group_cv(
            "test_run",
            df,
            feature_cols=["LapNumber", "PitNextLap"],
            n_splits=3,
            time_limit=10,
        )
