"""Modelos manuales (Fase 7): comparacion de defaults + tuning limitado.

Todos los candidatos usan el feature set ganador de Fase 6
(`E13_full_leakage_safe_features`, ver `src/f1pitstop/features/build.py`):
9 columnas numericas + `Compound` categorica.

Candidatos (maximo 4-6 del spec, aqui 3 para no convertir esto en un zoo
de modelos):
- `LogisticRegression` (contraste lineal, con preprocessing manual).
- `HistGradientBoostingClassifier` (ganador en Fase 4/5/6, soporte nativo
  de categoricas y de NaN).
- `ExtraTreesClassifier` (contraste de tree ensemble no boosting, mas
  rapido de entrenar que RandomForest a igual n_estimators).

`laptime_delta_prev` tiene NaN en la primera vuelta VISIBLE de cada grupo
(no hay `t-1`, ver `features/temporal.py`) — HGB lo soporta nativamente;
LogisticRegression y ExtraTrees no soportan NaN, se imputa con la mediana
(fiteada solo sobre train de cada fold via Pipeline, sin leakage).
"""

from __future__ import annotations

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from f1pitstop.features.build import E13_FULL_LEAKAGE_SAFE_FEATURES
from f1pitstop.models.baselines import CATEGORICAL_FEATURES, SEED

NUMERIC_FEATURES_E13 = [c for c in E13_FULL_LEAKAGE_SAFE_FEATURES if c not in CATEGORICAL_FEATURES]


def make_e14_logreg_e13() -> Pipeline:
    """E14: LogisticRegression sobre E13 (imputacion mediana + escalado +
    one-hot), defaults de sklearn salvo `max_iter`."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                NUMERIC_FEATURES_E13,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )


def make_e15_hgb_e13() -> HistGradientBoostingClassifier:
    """E15: HistGradientBoostingClassifier sobre E13, defaults de sklearn.
    Soporte nativo de categoricas y de NaN (no requiere imputar
    `laptime_delta_prev`)."""
    return HistGradientBoostingClassifier(
        random_state=SEED, categorical_features=CATEGORICAL_FEATURES
    )


def make_e16_extratrees_e13(n_jobs: int = -1) -> Pipeline:
    """E16: ExtraTreesClassifier sobre E13 (imputacion mediana + one-hot,
    no soporta NaN ni categoricas nativas), defaults de sklearn salvo
    `n_estimators=200` (el default de 100 es deliberadamente bajo).

    `n_jobs` es parametrizable porque el paralelismo interno del modelo
    (n_jobs=-1) y el paralelismo externo de `RandomizedSearchCV`
    (tambien n_jobs=-1) se sobre-suscriben mutuamente cuando se anidan
    (ver `make_tunable_model`, que fuerza `n_jobs=1` aqui durante el
    tuning)."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES_E13),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                ExtraTreesClassifier(n_estimators=200, random_state=SEED, n_jobs=n_jobs),
            ),
        ]
    )


def make_e22_xgboost_e13() -> XGBClassifier:
    """E22 (Fase 14): XGBClassifier sobre E13, defaults salvo lo minimo
    para soportar categoricas/NaN nativos (`enable_categorical=True`
    requiere `tree_method="hist"`). Diversidad controlada frente a E20
    (HGB): mismo feature set y CV, algoritmo boosting distinto — ver
    `artifacts/reports/model_selection_framework.md`."""
    return XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
    )


def make_e23_catboost_e13() -> CatBoostClassifier:
    """E23 (Fase 14): CatBoostClassifier sobre E13, defaults salvo
    `cat_features` (soporte nativo de `Compound`) y `verbose=False` (evitar
    ruido de log por fold durante CV). Soporta NaN nativamente."""
    return CatBoostClassifier(
        cat_features=CATEGORICAL_FEATURES,
        random_state=SEED,
        verbose=False,
    )


def make_e24_lightgbm_e13() -> LGBMClassifier:
    """E24 (Fase 14): LGBMClassifier sobre E13, defaults. Soporte nativo
    de categoricas (dtype `category`, ya casteado por
    `prepare_X_for_feature_set`) y de NaN. `verbose=-1` evita el log de
    warnings por fold durante CV."""
    return LGBMClassifier(random_state=SEED, verbose=-1)


# Comparacion de defaults (paso 1 del procedimiento de Fase 7): mismo
# feature set E13, mismo CV V1, sin tuning todavia.
MANUAL_DEFAULTS_REGISTRY = {
    "E14_logreg_e13_features": {
        "make_model": make_e14_logreg_e13,
        "model_family": "logistic_regression",
    },
    "E15_hgb_e13_features": {
        "make_model": make_e15_hgb_e13,
        "model_family": "hist_gradient_boosting",
    },
    "E16_extratrees_e13_features": {
        "make_model": make_e16_extratrees_e13,
        "model_family": "extra_trees",
    },
}


# Diversidad controlada (Fase 14, paso 14a): candidatos boosting adicionales
# sobre EXACTAMENTE el mismo feature set E13 y CV V1 que E20 — la pregunta
# no es "encontrar el mejor algoritmo posible" (eso es el juego de
# 186/218 modelos de los ganadores de Kaggle, ver
# artifacts/reports/model_selection_framework.md), es medir si alguno de
# estos 3 supera a E20 fuera del margen de ruido de CV lo suficiente para
# justificar el costo de mantenerlo. Ninguno se tunea individualmente
# (regla explicita de esta fase: solo defaults, igual que
# MANUAL_DEFAULTS_REGISTRY antes del tuning de Fase 7).
DIVERSITY_REGISTRY = {
    "E22_xgboost_e13_features": {
        "make_model": make_e22_xgboost_e13,
        "model_family": "xgboost",
    },
    "E23_catboost_e13_features": {
        "make_model": make_e23_catboost_e13,
        "model_family": "catboost",
    },
    "E24_lightgbm_e13_features": {
        "make_model": make_e24_lightgbm_e13,
        "model_family": "lightgbm",
    },
}


# Espacios de busqueda para RandomizedSearchCV (paso 3, solo sobre las 2
# familias seleccionadas en el paso 2). Prefijo "model__" porque los 3
# candidatos son (o se envuelven en) un Pipeline con paso final "model".
PARAM_DISTRIBUTIONS = {
    "logistic_regression": {
        "model__C": loguniform(1e-3, 1e2),
        "model__class_weight": [None, "balanced"],
    },
    "hist_gradient_boosting": {
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__max_iter": randint(100, 400),
        "model__max_leaf_nodes": randint(15, 255),
        "model__min_samples_leaf": randint(10, 100),
        "model__l2_regularization": uniform(0.0, 1.0),
    },
    "extra_trees": {
        "model__n_estimators": randint(100, 400),
        "model__max_depth": randint(5, 30),
        "model__min_samples_leaf": randint(1, 50),
        "model__max_features": ["sqrt", "log2", None],
    },
}


def make_tunable_model(model_family: str):
    """Instancia sin fitear para `model_family`, lista para envolver en
    `RandomizedSearchCV` con `PARAM_DISTRIBUTIONS[model_family]`.

    `HistGradientBoostingClassifier` no es un `Pipeline` (no necesita
    preprocessing), pero sus hiperparametros en `PARAM_DISTRIBUTIONS`
    usan igualmente el prefijo `model__` por consistencia con los otros
    dos candidatos — se envuelve en un `Pipeline` de un solo paso
    ("model") para que el prefijo funcione sin bifurcar el codigo del
    script de tuning.
    """
    make_fns = {
        "logistic_regression": make_e14_logreg_e13,
        "hist_gradient_boosting": lambda: Pipeline([("model", make_e15_hgb_e13())]),
        # n_jobs=2 explicito (no -1): evita sobre-suscripcion de cores al
        # anidar dentro de RandomizedSearchCV -- el script de tuning usa
        # n_jobs=4 para la busqueda externa (2*4=8, el numero de cores de
        # esta maquina) en vez de -1*-1 (ver docstring de
        # make_e16_extratrees_e13 y scripts/phase7_manual_models.py).
        "extra_trees": lambda: make_e16_extratrees_e13(n_jobs=2),
    }
    if model_family not in make_fns:
        raise ValueError(f"model_family desconocido: {model_family!r}")
    return make_fns[model_family]()
