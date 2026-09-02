"""Fase 13 — Evaluación holdout final y submission a Kaggle.

Evaluación confirmatoria UNICA del holdout congelado (Year==2025).
Genera submission.csv para Kaggle.

Usa el modelo E20 deserializado de Fase 12.

Uso: `uv run python scripts/phase13_holdout_and_submission.py`
"""

from __future__ import annotations

from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score
from skops.io import load

from f1pitstop.data.ingest import load_raw
from f1pitstop.data.split import load_frozen_holdout_ids
from f1pitstop.features.build import (
    build_engineered_frame,
    prepare_X_for_feature_set,
)

TARGET = "PitNextLap"
ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = Path("models/sklearn")


def main():
    """Evaluar en holdout y generar submission."""

    print("Cargando datos...")
    df_train, df_test, sample_submission, _ = load_raw()
    holdout_ids = load_frozen_holdout_ids()

    # Holdout final
    df_full = build_engineered_frame(df_train)
    df_holdout = df_full[df_full.index.isin(holdout_ids)].reset_index(drop=True)

    print(f"Holdout set: {len(df_holdout)} filas (Year==2025)")
    print(f"Distribución del target: {df_holdout[TARGET].mean():.1%} pit stops")

    # Preparar features
    X_holdout = prepare_X_for_feature_set(df_holdout, "E13_full_leakage_safe_features")
    y_holdout = df_holdout[TARGET]

    # Cargar modelo E20 desde skops
    print("\nCargando modelo E20 serializado...")
    model_path = MODELS_DIR / "e20_final.skops"

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    trusted_types = ["functools.partial", "sklearn.utils.validation.check_array"]

    e20_model = load(model_path, trusted=trusted_types)
    print(f"  [OK] Modelo cargado desde {model_path}")

    # Evaluar en holdout
    print("\nEvaluando en holdout final...")
    y_score_holdout = e20_model.predict_proba(X_holdout)[:, 1]

    holdout_roc_auc = roc_auc_score(y_holdout, y_score_holdout)
    holdout_pr_auc = average_precision_score(y_holdout, y_score_holdout)

    print(f"Holdout ROC-AUC: {holdout_roc_auc:.4f}")
    print(f"Holdout PR-AUC:  {holdout_pr_auc:.4f}")

    # Generar submission para Kaggle (test set)
    print("\nGenerando predicciones para test set (Kaggle)...")
    df_test_engineered = build_engineered_frame(df_test)
    X_test = prepare_X_for_feature_set(df_test_engineered, "E13_full_leakage_safe_features")

    y_score_test = e20_model.predict_proba(X_test)[:, 1]

    # Crear submission
    submission = sample_submission.copy()
    submission["PitNextLap"] = y_score_test

    submission_path = ARTIFACTS_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"  [OK] Submission generado: {submission_path}")
    print(f"  Predicciones test: {len(submission)} filas")
    print(f"  Rango de probabilidades: [{y_score_test.min():.4f}, {y_score_test.max():.4f}]")

    # Resumen final
    print("\n" + "="*70)
    print("PROYECTO COMPLETADO - Fase 13 finalizada")
    print("="*70)
    print("\nCandidato final: E20_hist_gradient_boosting")
    print("  Feature set: E13_full_leakage_safe_features (10 features)")
    print("  Validation strategy: V1 (StratifiedGroupKFold por Race-Year)")
    print("  CV ROC-AUC: 0.8611±0.0251 (Fase 7)")
    print(f"  Holdout ROC-AUC: {holdout_roc_auc:.4f} (Fase 13)")
    print("\nGaps observados:")
    print(f"  CV -> Holdout: {0.8611 - holdout_roc_auc:+.4f}")

    if holdout_roc_auc >= 0.85:
        print("  [OK] Generalizacion EXITOSA (holdout dentro de rango esperado)")
    elif holdout_roc_auc >= 0.80:
        print("  [OK] Generalizacion RAZONABLE (pequeno gap esperado)")
    else:
        print("  [WARNING] Gap significativo (posible drift H4)")

    print("\nArtefactos finales:")
    print(f"  - Modelo serializado: {model_path}")
    print(f"  - Submission: {submission_path}")
    print("  - MLflow run: stage=final (Fase 11)")

    print("\nPregunta de portafolio respondida:")
    print("  Pipeline manual + feature engineering ~ AutoML en calidad")
    print("  Ventaja manual: interpretabilidad + 5x velocidad")


if __name__ == "__main__":
    main()
