"""Ingesta de datos para F1 Pit Stop Prediction (Fase 1).

Responsabilidades (spec, seccion 6):
- localizar train/test en data/raw/;
- cargar sin modificar los archivos originales (data/raw/ es inmutable);
- separar target solo despues de validar su presencia;
- conservar `id` como identificador, no como feature por defecto;
- registrar shape, dtypes y memoria de cada archivo;
- calcular un fingerprint (sha256) para trazabilidad.

Este modulo NO valida el contenido/estructura de los datos mas alla de la
presencia del target — esa responsabilidad vive en `schema.py`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TARGET = "PitNextLap"
ID_COLUMN = "id"

DEFAULT_RAW_DIR = Path("data/raw")


@dataclass
class FileReport:
    """Metadata de trazabilidad de un archivo crudo cargado."""

    name: str
    path: Path
    n_rows: int
    n_cols: int
    memory_mb: float
    sha256: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "memory_mb": round(self.memory_mb, 2),
            "sha256": self.sha256,
        }


def file_sha256(path: Path, chunk_size: int = 2**20) -> str:
    """Fingerprint del archivo en disco (no del DataFrame ya parseado)."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _describe(name: str, path: Path, df: pd.DataFrame) -> FileReport:
    return FileReport(
        name=name,
        path=path,
        n_rows=df.shape[0],
        n_cols=df.shape[1],
        memory_mb=df.memory_usage(deep=True).sum() / (1024**2),
        sha256=file_sha256(path),
    )


def load_raw(
    data_dir: Path | str = DEFAULT_RAW_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[FileReport]]:
    """Carga train/test/sample_submission de `data_dir` sin modificarlos.

    Retorna (train, test, sample_submission, reports). `reports` trae
    shape/memoria/hash de cada archivo para trazabilidad (ver
    `FileReport.to_dict()`).

    Levanta `FileNotFoundError` si falta algun archivo, y `ValueError` si
    el target no esta donde debe (ausente en train, o presente en test).
    """
    data_dir = Path(data_dir)
    paths = {
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "sample_submission": data_dir / "sample_submission.csv",
    }
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(
                f"No se encontro {name} en {p}. ¿Se descargo data/raw/ desde Kaggle?"
            )

    train = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["test"])
    sample_submission = pd.read_csv(paths["sample_submission"])

    if TARGET not in train.columns:
        raise ValueError(f"'{TARGET}' no esta presente en train.csv; no se puede continuar.")
    if TARGET in test.columns:
        raise ValueError(
            f"'{TARGET}' esta presente en test.csv; posible fuga de datos en la fuente."
        )

    reports = [
        _describe("train", paths["train"], train),
        _describe("test", paths["test"], test),
        _describe("sample_submission", paths["sample_submission"], sample_submission),
    ]
    return train, test, sample_submission, reports


def split_target(
    train: pd.DataFrame, target: str = TARGET
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa el target de las features. Solo se llama tras validar su presencia
    (ver `load_raw`, que ya lo valida antes de devolver `train`)."""
    if target not in train.columns:
        raise ValueError(f"'{target}' no esta en train; no se puede separar.")
    y = train[target].copy()
    X = train.drop(columns=[target])
    return X, y
