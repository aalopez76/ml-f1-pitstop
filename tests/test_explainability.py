"""Tests de explicabilidad SHAP (extension de Fase 9).

Verifica una propiedad matematica real de SHAP -- no un "corre sin error":
la suma de los valores SHAP de una fila mas el base_value debe reconstruir
exactamente la probabilidad predicha por el modelo (aditividad). Esto
tambien sirve de regresion de compatibilidad: si una futura actualizacion
de `shap` o `scikit-learn` rompe el manejo de la categorica nativa de HGB
(`categorical_features=[...]`), este test lo detecta aqui, no a mitad de
generar figuras sobre 346k filas reales.

Reusa `build_onehot_wrapper` de `f1pitstop.evaluation.explainability` (el
wrapper que hace viable SHAP sobre una categorica nativa que
`shap.TreeExplainer` no soporta en la version instalada, ver docstring de
ese modulo) en vez de reimplementar la logica de reconstruccion one-hot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.ensemble import HistGradientBoostingClassifier

from f1pitstop.evaluation.explainability import build_onehot_wrapper


def _toy_e13_df(n: int = 80, seed: int = 42) -> pd.DataFrame:
    """DataFrame toy con las 10 columnas del feature set E13, incluyendo
    `Compound` como categorica -- mismo esquema que `prepare_X_for_feature_set`
    produce sobre datos reales (ver `src/f1pitstop/features/build.py`)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "LapNumber": rng.integers(1, 60, n).astype(float),
        "TyreLife": rng.integers(0, 30, n).astype(float),
        "Stint": rng.integers(1, 4, n).astype(float),
        "Position": rng.integers(1, 20, n).astype(float),
        "PitStop": rng.integers(0, 2, n).astype(float),
        "Compound": rng.choice(["HARD", "MEDIUM", "SOFT"], n),
        "pit_stops_so_far": rng.integers(0, 3, n).astype(float),
        "recomputed_stint": rng.integers(1, 4, n).astype(float),
        "laptime_delta_prev": rng.normal(0, 1, n),
        "laps_since_last_pit": rng.integers(0, 20, n).astype(float),
    })
    df["Compound"] = df["Compound"].astype("category")
    return df


def test_onehot_wrapper_reproduces_original_predictions():
    """El wrapper debe reconstruir EXACTAMENTE las mismas predicciones que
    el modelo original -- si no, SHAP explicaria un modelo distinto del
    que realmente se usa, y la explicacion no seria confiable."""
    X = _toy_e13_df()
    y = np.random.default_rng(0).integers(0, 2, len(X))
    model = HistGradientBoostingClassifier(random_state=42, categorical_features=["Compound"])
    model.fit(X, y)

    predict_fn, to_onehot, _ = build_onehot_wrapper(model, X)

    proba_direct = model.predict_proba(X)[:, 1]
    proba_via_wrapper = predict_fn(to_onehot(X).values)

    assert np.allclose(proba_direct, proba_via_wrapper)


def test_shap_values_satisfy_additivity():
    """Propiedad de aditividad de SHAP: sum(shap_values_row) + base_value
    debe reconstruir predict_proba(row) dentro de una tolerancia numerica.
    Esta es la verificacion real de que el explainer esta bien calibrado
    contra el modelo, no una comprobacion de que el codigo "no truena"."""
    X = _toy_e13_df(n=60)
    y = np.random.default_rng(0).integers(0, 2, len(X))
    model = HistGradientBoostingClassifier(random_state=42, categorical_features=["Compound"])
    model.fit(X, y)

    predict_fn, to_onehot, feature_names = build_onehot_wrapper(model, X)

    background = to_onehot(X.sample(15, random_state=42)).values
    sample = to_onehot(X.sample(10, random_state=1)).values

    explainer = shap.Explainer(predict_fn, background, feature_names=feature_names)
    shap_values = explainer(sample)

    proba = predict_fn(sample)
    reconstructed = shap_values.values.sum(axis=1) + shap_values.base_values

    assert reconstructed == pytest.approx(proba, abs=1e-3)


def test_treeexplainer_does_not_support_native_categorical():
    """Documenta y fija en un test la limitacion real encontrada durante el
    desarrollo (no una suposicion): `shap.TreeExplainer` sobre
    `HistGradientBoostingClassifier` con `categorical_features=[...]` y
    dtype pandas `"category"` falla al castear la columna a float.

    Si una futura version de `shap` soluciona esto, este test empieza a
    fallar (porque el `pytest.raises` ya no se dispara) -- eso es una senal
    util para simplificar `build_onehot_wrapper` y volver a `TreeExplainer`
    directo, mas rapido que el wrapper generico actual."""
    X = _toy_e13_df(n=30)
    y = np.random.default_rng(0).integers(0, 2, len(X))
    model = HistGradientBoostingClassifier(random_state=42, categorical_features=["Compound"])
    model.fit(X, y)

    with pytest.raises((ValueError, TypeError)):
        explainer = shap.TreeExplainer(model)
        explainer(X.sample(5, random_state=42))
