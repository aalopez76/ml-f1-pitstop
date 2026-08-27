"""Tests de src/f1pitstop/data/ingest.py (Fase 1)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from f1pitstop.data.ingest import file_sha256, load_raw, split_target


def _write_toy_raw(tmp_path: Path) -> Path:
    train = pd.DataFrame({"id": [0, 1], "x": [1.0, 2.0], "PitNextLap": [0, 1]})
    test = pd.DataFrame({"id": [2, 3], "x": [3.0, 4.0]})
    sample_submission = pd.DataFrame({"id": [2, 3], "PitNextLap": [0.5, 0.5]})

    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    sample_submission.to_csv(tmp_path / "sample_submission.csv", index=False)
    return tmp_path


def test_load_raw_returns_expected_shapes_and_reports(tmp_path):
    data_dir = _write_toy_raw(tmp_path)
    train, test, sample_submission, reports = load_raw(data_dir)

    assert train.shape == (2, 3)
    assert test.shape == (2, 2)
    assert sample_submission.shape == (2, 2)
    assert {r.name for r in reports} == {"train", "test", "sample_submission"}
    for r in reports:
        assert len(r.sha256) == 64  # hex digest de sha256


def test_load_raw_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path)


def test_load_raw_target_missing_in_train_raises(tmp_path):
    train = pd.DataFrame({"id": [0], "x": [1.0]})  # sin PitNextLap
    test = pd.DataFrame({"id": [1], "x": [2.0]})
    sample_submission = pd.DataFrame({"id": [1], "PitNextLap": [0.5]})
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    sample_submission.to_csv(tmp_path / "sample_submission.csv", index=False)

    with pytest.raises(ValueError, match="PitNextLap"):
        load_raw(tmp_path)


def test_load_raw_target_leaked_into_test_raises(tmp_path):
    train = pd.DataFrame({"id": [0], "x": [1.0], "PitNextLap": [0]})
    test = pd.DataFrame({"id": [1], "x": [2.0], "PitNextLap": [1]})  # fuga
    sample_submission = pd.DataFrame({"id": [1], "PitNextLap": [0.5]})
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    sample_submission.to_csv(tmp_path / "sample_submission.csv", index=False)

    with pytest.raises(ValueError, match="test.csv"):
        load_raw(tmp_path)


def test_split_target_separates_y_and_drops_from_X():
    train = pd.DataFrame({"id": [0, 1], "x": [1.0, 2.0], "PitNextLap": [0, 1]})
    X, y = split_target(train)
    assert "PitNextLap" not in X.columns
    assert list(y) == [0, 1]


def test_split_target_missing_target_raises():
    train = pd.DataFrame({"id": [0], "x": [1.0]})
    with pytest.raises(ValueError):
        split_target(train)


def test_file_sha256_is_deterministic(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello world")
    assert file_sha256(p) == file_sha256(p)
    p2 = tmp_path / "g.txt"
    p2.write_text("different content")
    assert file_sha256(p) != file_sha256(p2)
