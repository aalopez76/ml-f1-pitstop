"""Fase 12 — Serializacion con skops.

Serializa E20_hist_gradient_boosting con skops.dump() para persistencia
reproducible. Verifica deserializacion y que las predicciones sean identicas
al modelo original.

Uso: `uv run python scripts/phase12_skops_serialization.py`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from skops.io import dump, load

from f1pitstop.data.ingest import load_raw
from f1pitstop.data.split import load_frozen_holdout_ids, make_group_key
from f1pitstop.features.build import (
    build_engineered_frame,
    prepare_X_for_feature_set,
)
from f1pitstop.models.baselines import SEED
from f1pitstop.models.manual_models import make_e15_hgb_e13

TARGET = "PitNextLap"
MODELS_DIR = Path("models/sklearn")


def main():
    """Serializar E20 con skops."""

    print("Cargando datos y features...")
    df_train, _, _, _ = load_raw()
    holdout_ids = load_frozen_holdout_ids()
    df_full = build_engineered_frame(df_train)
    df_dev = df_full[~df_full.index.isin(holdout_ids)].reset_index(drop=True)

    print(f"Dev set: {len(df_dev)} filas")

    # Preparar datos
    X_full = prepare_X_for_feature_set(df_dev, "E13_full_leakage_safe_features")
    y_full = df_dev[TARGET].reset_index(drop=True)
    group_key = make_group_key(df_dev)

    # CV V1
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Entrenar E20 en todos los datos (para serializar el mejor modelo)
    print("\nEntrenando E20 en dev set completo...")
    e20_model = make_e15_hgb_e13()
    e20_best_params = {
        "learning_rate": 0.127,
        "max_iter": 152,
        "max_leaf_nodes": 38,
        "min_samples_leaf": 35,
        "l2_regularization": 0.84,
    }
    e20_model.set_params(**e20_best_params)
    e20_model.fit(X_full, y_full)

    # Generar predicciones en fold 1 (para verificacion posterior)
    _, val_idx = next(splitter.split(X_full, y_full, groups=group_key))
    X_val = X_full.iloc[val_idx]
    y_pred_original = e20_model.predict_proba(X_val)[:, 1]

    # Serializar con skops
    print("\nSerializando E20 con skops.dump()...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "e20_final.skops"

    try:
        dump(e20_model, model_path)
        print(f"  [OK] Modelo serializado a {model_path}")
    except Exception as e:
        print(f"  [FAIL] Error al serializar: {e}")
        raise

    # Verificar deserializacion
    print("\nDeserializando y verificando reproducibilidad...")
    trusted_types = ["functools.partial", "sklearn.utils.validation.check_array"]
    try:
        e20_loaded = load(model_path, trusted=trusted_types)
        print("  [OK] Modelo deserializado exitosamente")
    except Exception as e:
        print(f"  [FAIL] Error al deserializar: {e}")
        raise

    # Comparar predicciones
    print("\nVerificando que las predicciones son identicas...")
    y_pred_loaded = e20_loaded.predict_proba(X_val)[:, 1]

    if np.allclose(y_pred_original, y_pred_loaded, rtol=1e-10):
        print(f"  [OK] Predicciones identicas (maxima diferencia: {np.max(np.abs(y_pred_original - y_pred_loaded)):.2e})")
    else:
        print(f"  [FAIL] Predicciones divergentes (diferencia: {np.max(np.abs(y_pred_original - y_pred_loaded)):.4f})")
        raise ValueError("Reproducibilidad de predicciones fallida")

    # Resumen
    print("\n" + "="*70)
    print("Fase 12 completada")
    print("="*70)
    print(f"Modelo E20 serializado: {model_path}")
    print(f"Tamano del archivo: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("Reproducibilidad: [OK] Predicciones identicas")

    print("\nProximo paso: Fase 13 (Holdout final y submission a Kaggle)")


if __name__ == "__main__":
    main()
