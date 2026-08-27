"""Tests de src/f1pitstop/data/schema.py (Fase 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1pitstop.data.schema import cardinality_summary, validate_schema


def _toy_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "Driver": ["A", "B", "A", "C"],
            "LapNumber": [1, 2, 3, 4],
            "Cumulative_Degradation": [0.1, 0.2, 0.3, 0.4],
            "PitNextLap": [0, 1, 0, 1],
        }
    )


def _toy_test() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [4, 5],
            "Driver": ["A", "D"],
            "LapNumber": [1, 2],
            "Cumulative_Degradation": [0.5, 0.6],
        }
    )


def test_valid_schema_has_no_errors():
    report = validate_schema(_toy_train(), _toy_test())
    assert not report.has_errors


def test_target_missing_in_train_is_error():
    train = _toy_train().drop(columns=["PitNextLap"])
    report = validate_schema(train, _toy_test())
    assert report.has_errors
    assert any(i.check == "target_presence" for i in report.issues)


def test_target_present_in_test_is_error():
    test = _toy_test().copy()
    test["PitNextLap"] = [0, 1]
    report = validate_schema(_toy_train(), test)
    assert report.has_errors
    assert any(i.check == "target_presence" for i in report.issues)


def test_duplicated_id_is_error():
    train = _toy_train().copy()
    train.loc[1, "id"] = 0  # duplica el id de la fila 0
    report = validate_schema(train, _toy_test())
    assert report.has_errors
    assert any(i.check == "id_uniqueness" for i in report.issues)


def test_id_overlap_between_train_and_test_is_error():
    test = _toy_test().copy()
    test.loc[0, "id"] = 0  # coincide con un id de train
    report = validate_schema(_toy_train(), test)
    assert report.has_errors
    assert any(i.check == "id_overlap" for i in report.issues)


def test_mismatched_columns_is_error():
    test = _toy_test().drop(columns=["Driver"])
    report = validate_schema(_toy_train(), test)
    assert report.has_errors
    assert any(i.check == "column_names" for i in report.issues)


def test_target_out_of_range_is_error():
    train = _toy_train().copy()
    train.loc[0, "PitNextLap"] = 2
    report = validate_schema(train, _toy_test())
    assert report.has_errors
    assert any(i.check == "target_range" for i in report.issues)


def test_infinite_values_are_error():
    train = _toy_train().copy()
    train.loc[0, "Cumulative_Degradation"] = np.inf
    report = validate_schema(train, _toy_test())
    assert report.has_errors
    assert any(i.check == "infinite_values" for i in report.issues)


def test_constant_column_is_warning_not_error():
    train = _toy_train().copy()
    train["constant_col"] = 1
    test = _toy_test().copy()
    test["constant_col"] = 1
    report = validate_schema(train, test)
    constant_issues = [i for i in report.issues if i.check == "constant_columns"]
    assert constant_issues
    assert all(i.severity == "warning" for i in constant_issues)


def test_suspected_leakage_name_hint_flags_known_columns():
    report = validate_schema(_toy_train(), _toy_test())
    flagged = {
        i.message.split("'")[1]
        for i in report.issues
        if i.check == "suspected_leakage_name"
    }
    assert "Cumulative_Degradation" in flagged


def test_cardinality_summary_shape_and_columns():
    df = _toy_train()
    summary = cardinality_summary(df)
    assert len(summary) == len(df.columns)
    assert set(summary.columns) == {"column", "dtype", "n_unique", "pct_missing", "pct_unique"}


def test_cardinality_summary_empty_dataframe_does_not_raise():
    empty = pd.DataFrame({"a": pd.Series(dtype="int64"), "b": pd.Series(dtype="float64")})
    summary = cardinality_summary(empty)
    assert (summary["pct_missing"] == 0.0).all()
    assert (summary["pct_unique"] == 0.0).all()
