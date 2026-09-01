"""Tests de src/f1pitstop/data/split.py (Fase 3).

Test obligatorio del spec (Fase 3): verificar que los grupos protegidos
(`Race`, `Year`) no se solapan entre train/validation cuando la estrategia
seleccionada asi lo exige (V1, V2), y que SI se solapan en V0 (para dejar
constancia de por que V0 no es la estrategia oficial).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pitstop.data.split import (
    assert_no_group_overlap,
    freeze_final_holdout,
    load_frozen_holdout_ids,
    make_group_key,
    v0_stratified_kfold,
    v1_group_stratified_kfold,
    v2_temporal_split,
)


def _toy_df(n_groups: int = 12, rows_per_group: int = 20, seed: int = 0) -> pd.DataFrame:
    """Simula el dataset real: varias `Race` que se repiten entre `Year`
    distintos (misma estructura que el hallazgo de Fase 2), balance de
    target moderado por grupo.
    """
    rng = np.random.default_rng(seed)
    races = [f"Race_{i}" for i in range(n_groups // 4)]
    years = [2022, 2023, 2024, 2025]
    rows = []
    rid = 0
    for race in races:
        for year in years:
            for _ in range(rows_per_group):
                rows.append(
                    {
                        "id": rid,
                        "Race": race,
                        "Year": year,
                        "PitNextLap": int(rng.random() < 0.2),
                    }
                )
                rid += 1
    return pd.DataFrame(rows)


def test_make_group_key_combines_race_and_year():
    df = _toy_df()
    keys = make_group_key(df)
    assert keys.nunique() == df.groupby(["Race", "Year"]).ngroups
    assert (keys == df["Race"].astype(str) + "|" + df["Year"].astype(str)).all()


def test_make_group_key_missing_column_raises():
    df = pd.DataFrame({"Race": ["A"]})
    with pytest.raises(ValueError):
        make_group_key(df)


def test_v0_random_kfold_can_split_same_group_across_folds():
    """V0 ignora grupos: con suficientes filas por grupo, es esperable que
    un mismo grupo (Race, Year) quede en ambos lados de al menos un fold.
    Esto documenta por que V0 NO es una estrategia group-safe."""
    df = _toy_df()
    folds = v0_stratified_kfold(df["PitNextLap"], n_splits=5, seed=42)
    groups = make_group_key(df)
    any_overlap = False
    for train_idx, val_idx in folds:
        overlap = set(groups.iloc[train_idx]) & set(groups.iloc[val_idx])
        if overlap:
            any_overlap = True
            break
    assert any_overlap, "V0 deberia solapar grupos entre train/val en al menos un fold"


def test_v1_group_kfold_never_overlaps_groups():
    df = _toy_df()
    groups = make_group_key(df)
    folds = v1_group_stratified_kfold(df["PitNextLap"], groups, n_splits=5, seed=42)
    assert len(folds) == 5
    for train_idx, val_idx in folds:
        assert_no_group_overlap(df, train_idx, val_idx)


def test_v1_group_kfold_covers_all_rows_across_folds():
    df = _toy_df()
    groups = make_group_key(df)
    folds = v1_group_stratified_kfold(df["PitNextLap"], groups, n_splits=5, seed=42)
    all_val_idx = np.concatenate([val_idx for _, val_idx in folds])
    assert sorted(all_val_idx) == list(range(len(df)))


def test_v2_temporal_split_isolates_holdout_years():
    df = _toy_df()
    train_idx, holdout_idx = v2_temporal_split(df, holdout_years=(2025,))
    assert (df.iloc[holdout_idx]["Year"] == 2025).all()
    assert (df.iloc[train_idx]["Year"] != 2025).all()
    assert len(train_idx) + len(holdout_idx) == len(df)


def test_v2_temporal_split_never_overlaps_groups():
    df = _toy_df()
    train_idx, holdout_idx = v2_temporal_split(df, holdout_years=(2025,))
    assert_no_group_overlap(df, train_idx, holdout_idx)


def test_v2_temporal_split_raises_if_year_absent():
    df = _toy_df()
    with pytest.raises(ValueError):
        v2_temporal_split(df, holdout_years=(1999,))


def test_v2_temporal_split_raises_if_all_years_in_holdout():
    df = _toy_df()
    with pytest.raises(ValueError):
        v2_temporal_split(df, holdout_years=(2022, 2023, 2024, 2025))


def test_assert_no_group_overlap_raises_on_overlap():
    df = _toy_df()
    idx_a = np.array([0, 1, 2])
    idx_b = np.array([0, 1, 2])  # mismos indices -> mismo grupo, debe fallar
    with pytest.raises(AssertionError):
        assert_no_group_overlap(df, idx_a, idx_b)


def test_freeze_final_holdout_writes_ids_and_no_group_overlap(tmp_path):
    df = _toy_df()
    ids_path = tmp_path / "final_holdout_ids.csv"
    report = freeze_final_holdout(df, holdout_years=(2025,), ids_path=ids_path)

    assert ids_path.exists()
    assert report.n_holdout > 0
    assert report.n_dev > 0
    assert report.n_dev + report.n_holdout == len(df)

    holdout_ids = load_frozen_holdout_ids(ids_path)
    expected_ids = set(df[df["Year"] == 2025]["id"])
    assert set(holdout_ids) == expected_ids

    # ninguna carrera (Race, Year) del holdout aparece en dev
    dev_ids = set(df["id"]) - expected_ids
    dev_groups = make_group_key(df[df["id"].isin(dev_ids)])
    holdout_groups = make_group_key(df[df["id"].isin(expected_ids)])
    assert not (set(dev_groups) & set(holdout_groups))


def test_load_frozen_holdout_ids_raises_if_missing(tmp_path):
    missing_path = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError):
        load_frozen_holdout_ids(missing_path)
