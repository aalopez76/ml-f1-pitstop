"""Validaciones de esquema para train/test (Fase 1).

Checklist minimo (spec, seccion 6):
- target presente solo en train;
- test no contiene target;
- id no duplicado;
- dtypes consistentes train/test;
- mismos nombres de features train/test;
- proporciones de missing;
- valores infinitos;
- duplicados exactos;
- columnas constantes;
- cardinalidad;
- rango del target {0,1};
- columnas que parecen identificadores;
- columnas que podrian codificar informacion futura (heuristica, requiere
  revision humana — ver SUSPECTED_LEAKAGE en notebooks/01_data_audit).

Este modulo NO reemplaza el audit de leakage completo de la Fase 3: aqui
solo se detectan senales estructurales (nombres, cardinalidad) que ameritan
revision manual con el checklist de 5 preguntas de
`.claude/rules/leakage-and-validation.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TARGET = "PitNextLap"
ID_COLUMN = "id"

# Nombres que sugieren agregados/estado acumulado dentro de una carrera y que
# por lo tanto ameritan revision explicita antes de usarse como feature (no
# se excluyen automaticamente: se marcan para revision en Fase 3).
SUSPECTED_LEAKAGE_NAME_HINTS = ("cumulative", "progress", "delta", "change")


@dataclass
class SchemaIssue:
    check: str
    severity: str  # "error" | "warning" | "info"
    message: str


@dataclass
class SchemaReport:
    issues: list[SchemaIssue] = field(default_factory=list)

    def add(self, check: str, severity: str, message: str) -> None:
        self.issues.append(SchemaIssue(check, severity, message))

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    def to_frame(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(columns=["check", "severity", "message"])
        return pd.DataFrame([vars(i) for i in self.issues])


def validate_schema(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str = TARGET,
    id_column: str = ID_COLUMN,
) -> SchemaReport:
    """Corre el checklist de validaciones minimas y devuelve un SchemaReport.

    No levanta excepciones: acumula issues con severidad. El caller decide
    que hacer con `report.has_errors` (p.ej. tests/test_schema.py debe
    fallar si hay errores en las columnas conocidas del dataset).
    """
    report = SchemaReport()

    # target presente solo en train
    if target not in train.columns:
        report.add("target_presence", "error", f"'{target}' no esta en train")
    if target in test.columns:
        report.add("target_presence", "error", f"'{target}' esta presente en test (fuga)")

    # id no duplicado
    if id_column in train.columns:
        n_dup = int(train[id_column].duplicated().sum())
        if n_dup:
            report.add("id_uniqueness", "error", f"{n_dup} ids duplicados en train")
    if id_column in test.columns:
        n_dup = int(test[id_column].duplicated().sum())
        if n_dup:
            report.add("id_uniqueness", "error", f"{n_dup} ids duplicados en test")

    # overlap de ids train/test
    if id_column in train.columns and id_column in test.columns:
        overlap = set(train[id_column]) & set(test[id_column])
        if overlap:
            report.add(
                "id_overlap", "error", f"{len(overlap)} ids se repiten entre train y test"
            )

    # mismos nombres de features (excluyendo target) + dtypes consistentes
    feature_cols_train = [c for c in train.columns if c != target]
    if set(feature_cols_train) != set(test.columns):
        only_train = set(feature_cols_train) - set(test.columns)
        only_test = set(test.columns) - set(feature_cols_train)
        report.add(
            "column_names",
            "error",
            f"columnas distintas train/test. Solo train: {only_train}. Solo test: {only_test}",
        )
    else:
        for c in feature_cols_train:
            if train[c].dtype != test[c].dtype:
                report.add(
                    "dtype_consistency",
                    "warning",
                    f"'{c}' dtype difiere: train={train[c].dtype}, test={test[c].dtype}",
                )

    # proporciones de missing
    for name, df in (("train", train), ("test", test)):
        miss = df.isna().mean()
        for c, pct in miss.items():
            if pct > 0:
                report.add("missing_values", "info", f"{name}.{c}: {pct:.2%} missing")

    # valores infinitos
    for name, df in (("train", train), ("test", test)):
        num_cols = df.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            n_inf = int(np.isinf(df[c]).sum())
            if n_inf:
                report.add("infinite_values", "error", f"{name}.{c}: {n_inf} valores infinitos")

    # duplicados exactos
    for name, df in (("train", train), ("test", test)):
        n_dup = int(df.duplicated().sum())
        if n_dup:
            report.add("exact_duplicates", "warning", f"{name}: {n_dup} filas duplicadas")

    # columnas constantes
    for name, df in (("train", train), ("test", test)):
        for c in df.columns:
            if df[c].nunique(dropna=False) <= 1:
                report.add("constant_columns", "warning", f"{name}.{c} es constante")

    # rango del target {0, 1}
    if target in train.columns:
        vals = set(train[target].dropna().unique())
        if not vals <= {0, 1}:  # 0.0/1.0 son iguales a 0/1 en un set de Python
            report.add("target_range", "error", f"target tiene valores fuera de {{0,1}}: {vals}")

    # columnas que parecen identificadores (cardinalidad casi igual al numero de filas)
    for c in train.columns:
        if c in (id_column, target) or len(train) == 0:
            continue
        nunique = train[c].nunique()
        if nunique / len(train) > 0.9:
            report.add(
                "id_like_columns",
                "info",
                f"'{c}' tiene cardinalidad muy alta ({nunique}/{len(train)}), "
                "revisar si es identificador",
            )

    # columnas cuyo nombre sugiere agregado/estado acumulado (revision manual, Fase 3)
    for c in train.columns:
        if c in (id_column, target):
            continue
        lowered = c.lower()
        if any(hint in lowered for hint in SUSPECTED_LEAKAGE_NAME_HINTS):
            report.add(
                "suspected_leakage_name",
                "info",
                f"'{c}' sugiere agregado/estado acumulado por su nombre; "
                "revisar con checklist de leakage-and-validation.md antes de usarla",
            )

    return report


def cardinality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla dtype/n_unique/%missing/%unique por columna (para schema_summary.csv)."""
    rows = []
    n = len(df)
    for c in df.columns:
        nunique = df[c].nunique()
        rows.append(
            {
                "column": c,
                "dtype": str(df[c].dtype),
                "n_unique": nunique,
                "pct_missing": round(df[c].isna().mean() * 100, 4) if n else 0.0,
                "pct_unique": round(nunique / n * 100, 4) if n else 0.0,
            }
        )
    return pd.DataFrame(rows)
