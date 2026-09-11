"""Explicabilidad con SHAP (extension de Fase 9).

`HistGradientBoostingClassifier` con `categorical_features=[...]` (soporte
nativo de categoricas, ver `src/f1pitstop/models/manual_models.py`) no es
compatible con `shap.TreeExplainer` en la version instalada
(`shap==0.50.0`): internamente intenta castear todo el array de entrada a
float y falla con cualquier columna de dtype `"category"`. Verificado con
un smoke test dedicado antes de construir esto -- no es una suposicion.

`build_onehot_wrapper` resuelve esto envolviendo `model.predict_proba` en
una funcion que opera sobre un espacio completamente numerico (one-hot de
la columna categorica) y reconstruye la fila original antes de llamar al
modelo real. Esto permite usar `shap.Explainer` generico
(`PermutationExplainer`), model-agnostic, sin sustituir el modelo por uno
distinto solo para poder explicarlo.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def build_onehot_wrapper(
    model, X: pd.DataFrame, categorical_column: str = "Compound"
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[pd.DataFrame], pd.DataFrame], list[str]]:
    """Envuelve `model.predict_proba` para operar sobre un espacio one-hot.

    Devuelve `(predict_fn, to_onehot, feature_names_onehot)`:
    - `predict_fn(X_onehot_array) -> np.ndarray` de probabilidades P(y=1),
      apto para pasar directo a `shap.Explainer`.
    - `to_onehot(df) -> pd.DataFrame` convierte un DataFrame con la columna
      categorica original al espacio one-hot que `predict_fn` espera.
    - `feature_names_onehot`: nombres de columnas en el orden que produce
      `to_onehot`, para pasar a `shap.Explainer(..., feature_names=...)`.

    Invariante verificado por quien llama (ver
    `tests/test_explainability.py::test_onehot_wrapper_reproduces_original_predictions`):
    `predict_fn(to_onehot(X).values)` debe ser identico a
    `model.predict_proba(X)[:, 1]` para cualquier `X` con el mismo esquema
    que se uso al entrenar `model`.
    """
    categories = X[categorical_column].cat.categories.tolist()
    numeric_cols = [c for c in X.columns if c != categorical_column]
    onehot_cols = [f"{categorical_column}_{c}" for c in categories]

    def to_onehot(df_orig: pd.DataFrame) -> pd.DataFrame:
        onehot = pd.get_dummies(df_orig[categorical_column], prefix=categorical_column)
        for c in onehot_cols:
            if c not in onehot.columns:
                onehot[c] = 0
        onehot = onehot[onehot_cols].astype(float)
        return pd.concat(
            [df_orig[numeric_cols].reset_index(drop=True), onehot.reset_index(drop=True)],
            axis=1,
        )

    def from_onehot(X_onehot: np.ndarray) -> pd.DataFrame:
        X_onehot = np.asarray(X_onehot)
        df_num = pd.DataFrame(X_onehot[:, : len(numeric_cols)], columns=numeric_cols)
        onehot_part = X_onehot[:, len(numeric_cols) :]
        idx = onehot_part.argmax(axis=1)
        df_num[categorical_column] = pd.Categorical(
            [categories[i] for i in idx], categories=categories
        )
        return df_num[numeric_cols + [categorical_column]]

    def predict_fn(X_onehot: np.ndarray) -> np.ndarray:
        return model.predict_proba(from_onehot(X_onehot))[:, 1]

    return predict_fn, to_onehot, numeric_cols + onehot_cols


def stratified_sample(
    X: pd.DataFrame, y: pd.Series, n: int, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Muestra estratificada por clase, seed fijo, reproducible.

    Usado para acotar el costo de SHAP con `PermutationExplainer`
    (model-agnostic, ~400ms/fila medido sobre el modelo real E20) a un
    tamano manejable, en vez de correrlo sobre el dev set completo.
    """
    y_arr = y.to_numpy()
    frac = n / len(X)
    sampled_idx = []
    for cls in np.unique(y_arr):
        cls_idx = X.index[y_arr == cls]
        n_cls = max(1, round(len(cls_idx) * frac))
        sampled_idx.append(
            pd.Series(cls_idx).sample(n_cls, random_state=seed).to_numpy()
        )
    idx = np.concatenate(sampled_idx)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)  # evitar filas agrupadas por clase en el resultado
    return X.loc[idx], y.loc[idx]
