"""Tests de integracion sobre los artefactos de Fase 14 (auditoria 2026-09-10).

El CSV por fold lleva su propio estado metodologico para que la advertencia
sobre E25 viaje CON el dato: quien abra el artefacto dentro de seis meses no
depende de recordar un documento para saber que su evaluacion no es valida
como evidencia de promocion.

Se omiten (skip) si el artefacto no esta generado todavia, de modo que la
suite siga siendo ejecutable en un clon limpio sin correr Fase 14 entera.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FOLD_SCORES_PATH = Path("artifacts/tables/phase14_fold_level_scores.csv")
PAIRED_DELTAS_PATH = Path("artifacts/tables/phase14_paired_deltas.csv")

INCUMBENT = "E20_hist_gradient_boosting"
CONTAMINATED = "E25_ensemble_logit_stack"
VALID_CHALLENGERS = [
    "E22_xgboost_e13_features",
    "E23_catboost_e13_features",
    "E24_lightgbm_e13_features",
]
EXPECTED_FOLDS = 5


@pytest.fixture
def fold_scores() -> pd.DataFrame:
    if not FOLD_SCORES_PATH.exists():
        pytest.skip(f"{FOLD_SCORES_PATH} no generado; correr scripts/phase14_model_selection_framework.py")
    return pd.read_csv(FOLD_SCORES_PATH)


@pytest.fixture
def paired_deltas() -> pd.DataFrame:
    if not PAIRED_DELTAS_PATH.exists():
        pytest.skip(f"{PAIRED_DELTAS_PATH} no generado; correr scripts/phase14_model_selection_framework.py")
    return pd.read_csv(PAIRED_DELTAS_PATH)


def test_fold_scores_has_one_row_per_run_and_fold(fold_scores):
    n_runs = fold_scores["run_name"].nunique()
    assert n_runs == 5, f"esperados 5 runs (E20, E22, E23, E24, E25), hay {n_runs}"
    assert len(fold_scores) == n_runs * EXPECTED_FOLDS

    per_run = fold_scores.groupby("run_name")["fold_idx"].nunique()
    assert (per_run == EXPECTED_FOLDS).all(), f"algun run no tiene {EXPECTED_FOLDS} folds: {per_run.to_dict()}"


def test_fold_scores_marks_the_three_evaluation_states(fold_scores):
    status = dict(zip(fold_scores["run_name"], fold_scores["evaluation_status"]))
    valid = dict(zip(fold_scores["run_name"], fold_scores["promotion_evaluation_valid"].astype(str)))

    assert status[INCUMBENT] == "incumbent_reference"
    assert valid[INCUMBENT] == "not_applicable"

    for run_name in VALID_CHALLENGERS:
        assert status[run_name] == "valid_challenger"
        assert valid[run_name] == "true"

    assert status[CONTAMINATED] == "exploratory_contaminated"
    assert valid[CONTAMINATED] == "false", (
        "E25 evalua el stacker sobre los mismos folds que generaron sus OOF: "
        "su metrica no es evidencia valida de rendimiento incremental"
    )


def test_all_runs_share_the_same_folds(fold_scores):
    """El delta pareado solo es valido si los folds son identicos entre runs."""
    folds_per_run = fold_scores.groupby("run_name")["fold_idx"].apply(lambda s: sorted(s.tolist()))
    referencia = folds_per_run.loc[INCUMBENT]
    for run_name, folds in folds_per_run.items():
        assert folds == referencia, f"{run_name} no comparte los folds de {INCUMBENT}"


def test_paired_deltas_exclude_the_incumbent_and_flag_the_contaminated_run(paired_deltas):
    assert INCUMBENT not in set(paired_deltas["run_name"]), "el incumbente no se compara consigo mismo"
    assert len(paired_deltas) == 4

    # pandas infiere dtype bool en este CSV (solo true/false) pero deja texto en
    # el de folds (mezclado con "not_applicable"): se normaliza antes de comparar.
    fila_e25 = paired_deltas.loc[paired_deltas["run_name"] == CONTAMINATED].iloc[0]
    assert str(fila_e25["promotion_evaluation_valid"]).lower() == "false"
    assert fila_e25["evaluation_status"] == "exploratory_contaminated"

    for col in ["mean_paired_delta", "median_paired_delta", "min_paired_delta", "max_paired_delta"]:
        assert col in paired_deltas.columns
    assert (paired_deltas["n_folds"] == EXPECTED_FOLDS).all()
