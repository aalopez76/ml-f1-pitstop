"""Estrategias de validacion cruzada y holdout final (Fase 3).

Leer `.claude/rules/leakage-and-validation.md` completo antes de modificar
este archivo (regla no negociable 7 de CLAUDE.md).

Unidad de dependencia real: una fila es una vuelta de un piloto en una
carrera. `Driver` NO es un identificador de piloto confiable (ver
`eda_report.md`, hallazgo Fase 1/2: 887 valores unicos, no se comporta
como grid real de F1). La clave de agrupacion valida para "evento de
carrera" es `(Race, Year)` — `Race` sola se repite entre anios distintos.

Estrategias comparadas (spec, Fase 3):
- V0 = StratifiedKFold aleatorio reproducible (fila a fila, ignora grupos).
- V1 = group-aware: StratifiedGroupKFold por `(Race, Year)`, ninguna
  carrera aparece partida entre train y validation dentro de un fold.
- V2 = holdout temporal: los anios mas recientes quedan fuera de
  entrenamiento por completo (simula "carreras futuras nunca vistas").

V0 simula un escenario que NO existe en produccion (el modelo nunca vera
vueltas de una carrera que ya conoce en train y no en validation al mismo
tiempo) y por eso se espera que infle el ROC-AUC de forma optimista
respecto a V1/V2 — cuantificar esa inflacion es el objetivo central de
H1 (ver `eda_report.md`, hipotesis 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

GROUP_COLS = ("Race", "Year")

DEFAULT_HOLDOUT_YEARS = (2025,)

DEFAULT_HOLDOUT_IDS_PATH = Path("artifacts/tables/final_holdout_ids.csv")


def make_group_key(df: pd.DataFrame, group_cols: tuple[str, ...] = GROUP_COLS) -> pd.Series:
    """Clave de evento de carrera: `Race` sola no es unica entre anios distintos
    (mismas 26 carreras se repiten en los 4 anios del dataset, ver eda_report.md).
    """
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de agrupacion en el DataFrame: {missing}")
    return df[list(group_cols)].astype(str).agg("|".join, axis=1)


def v0_stratified_kfold(
    y: pd.Series, n_splits: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """V0: StratifiedKFold aleatorio, fila a fila, ignora grupos de carrera.

    Sirve como baseline optimista de referencia para H1, NO como estrategia
    oficial de CV del proyecto (ver docstring de modulo).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def v1_group_stratified_kfold(
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """V1: group-aware. Ninguna carrera `(Race, Year)` aparece en train y
    validation del mismo fold a la vez. `groups` debe venir de
    `make_group_key`.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, groups=groups))


def v2_temporal_split(
    df: pd.DataFrame,
    holdout_years: tuple[int, ...] = DEFAULT_HOLDOUT_YEARS,
    year_col: str = "Year",
) -> tuple[np.ndarray, np.ndarray]:
    """V2: holdout temporal. Filas de `holdout_years` quedan fuera de train
    por completo (simula carreras/temporadas futuras nunca vistas).

    Retorna (train_idx, holdout_idx) como posiciones enteras (iloc-based).
    """
    if year_col not in df.columns:
        raise ValueError(f"Falta columna temporal '{year_col}'")
    is_holdout = df[year_col].isin(holdout_years).to_numpy()
    holdout_idx = np.where(is_holdout)[0]
    train_idx = np.where(~is_holdout)[0]
    if len(holdout_idx) == 0:
        raise ValueError(f"Ningun anio de {holdout_years} presente en '{year_col}'")
    if len(train_idx) == 0:
        raise ValueError(f"Todos los anios caen en holdout_years={holdout_years}")
    return train_idx, holdout_idx


def assert_no_group_overlap(
    df: pd.DataFrame,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    group_cols: tuple[str, ...] = GROUP_COLS,
) -> None:
    """Levanta AssertionError si algun grupo `(Race, Year)` aparece en ambos
    conjuntos de indices. Usar en tests y antes de aceptar cualquier fold V1
    o el holdout V2 como validos."""
    groups = make_group_key(df, group_cols)
    overlap = set(groups.iloc[idx_a]) & set(groups.iloc[idx_b])
    if overlap:
        raise AssertionError(
            f"{len(overlap)} grupos {group_cols} se solapan entre los dos conjuntos: "
            f"{sorted(overlap)[:5]}..."
        )


@dataclass
class HoldoutReport:
    n_dev: int
    n_holdout: int
    holdout_years: tuple[int, ...]
    n_holdout_groups: int
    n_dev_groups: int
    ids_path: Path

    def to_dict(self) -> dict:
        d = vars(self).copy()
        d["ids_path"] = str(d["ids_path"])
        return d


def freeze_final_holdout(
    df: pd.DataFrame,
    holdout_years: tuple[int, ...] = DEFAULT_HOLDOUT_YEARS,
    id_column: str = "id",
    ids_path: Path | str = DEFAULT_HOLDOUT_IDS_PATH,
) -> HoldoutReport:
    """Congela el holdout final (V2, temporal) y persiste sus `id` en disco.

    Se llama UNA sola vez, en Fase 3. A partir de aqui el holdout NUNCA se
    usa para decisiones de modelado (regla no negociable 6 de CLAUDE.md);
    solo se vuelve a cargar en Fase 13 para la evaluacion confirmatoria.
    Escribe `ids_path` (no toca `data/raw/`, que es inmutable).
    """
    train_idx, holdout_idx = v2_temporal_split(df, holdout_years=holdout_years)
    assert_no_group_overlap(df, train_idx, holdout_idx)

    ids_path = Path(ids_path)
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_ids = df.iloc[holdout_idx][[id_column]].copy()
    holdout_ids.to_csv(ids_path, index=False)

    groups = make_group_key(df)
    return HoldoutReport(
        n_dev=len(train_idx),
        n_holdout=len(holdout_idx),
        holdout_years=holdout_years,
        n_holdout_groups=groups.iloc[holdout_idx].nunique(),
        n_dev_groups=groups.iloc[train_idx].nunique(),
        ids_path=ids_path,
    )


def load_frozen_holdout_ids(ids_path: Path | str = DEFAULT_HOLDOUT_IDS_PATH) -> pd.Series:
    """Recarga los `id` del holdout congelado (solo debe usarse en Fase 13)."""
    ids_path = Path(ids_path)
    if not ids_path.exists():
        raise FileNotFoundError(
            f"No existe {ids_path}; correr freeze_final_holdout() primero (Fase 3)."
        )
    return pd.read_csv(ids_path)["id"]
