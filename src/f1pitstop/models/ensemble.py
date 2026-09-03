"""Ensemble simple (Fase 14, E25): logit-stacking sobre predicciones OOF.

Mismo patron que describio Chris Deotte (2do lugar, competencia Kaggle
real que inspira este dataset, ver
`artifacts/reports/model_selection_framework.md`): combinar predicciones
out-of-fold de varios modelos base con una `LogisticRegression` simple,
en vez de un promedio fijo o un stacker complejo.

Deliberadamente NO se usa Optuna aqui: con 4 modelos base (E20/E22/E23/
E24) el espacio de pesos es pequeno y `LogisticRegression` sobre los
logits ya encuentra una combinacion razonable sin busqueda adicional —
agregar un optimizador de hiperparametros para esto seria exactamente el
tipo de complejidad no justificada que este proyecto evita (ver regla no
negociable 3 de CLAUDE.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from f1pitstop.models.baselines import SEED


def make_e25_logit_stack() -> LogisticRegression:
    """E25: LogisticRegression sin regularizacion fuerte (`C=1.0`, default)
    sobre las probabilidades OOF de los modelos base como unicas features.
    Se entrena UNA vez por fold externo sobre las columnas de
    `build_oof_feature_matrix()` correspondientes a ese fold."""
    return LogisticRegression(max_iter=1000, random_state=SEED)


def build_oof_feature_matrix(oof_predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """Arma la matriz de features del stacker: una columna por modelo base,
    valor = probabilidad predicha OOF para esa fila.

    `oof_predictions` es un dict `{run_name: array_de_probabilidades}`,
    todas del mismo largo y en el mismo orden de filas (ver
    `scripts/phase14_model_selection_framework.py`, que genera predicciones
    OOF con la misma particion V1 para los 4 modelos base antes de armar
    esta matriz).
    """
    if not oof_predictions:
        raise ValueError("oof_predictions no puede estar vacio")
    lengths = {len(v) for v in oof_predictions.values()}
    if len(lengths) != 1:
        raise ValueError(f"Todas las predicciones OOF deben tener el mismo largo: {lengths}")
    return pd.DataFrame(oof_predictions)
