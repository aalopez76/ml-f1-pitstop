"""Fase 10 — Análisis de errores.

Segmentación de errores del candidato ganador E20_hist_gradient_boosting
para identificar dónde el modelo falla, si hay oportunidad de ensemble,
y diversidad entre modelos (manual vs AutoGluon).

Variables de segmentación:
- Race: evento de carrera (26 valores únicos)
- Year: año (4 valores)
- Compound: tipo de neumático (5 valores)
- Stint: rango de stint (grupos)
- Position: rango de posición (inicio, medio, fin de carrera)
- RaceProgress: fase de carrera (temprana, media, tardía)

Métricas por segmento:
- n_samples, n_errors, error_rate
- AUC, PR-AUC
- Predicciones y confianza (y_score)

Uso: `uv run python scripts/phase10_error_analysis.py`

**Runtime esperado:** ~5-10 min (reentrenamiento CV V1 + análisis).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from f1pitstop.data.ingest import load_raw
from f1pitstop.data.split import load_frozen_holdout_ids, make_group_key
from f1pitstop.features.build import (
    build_engineered_frame,
    prepare_X_for_feature_set,
)
from f1pitstop.models.baselines import SEED
from f1pitstop.models.manual_models import make_e15_hgb_e13

TARGET = "PitNextLap"
ARTIFACTS_DIR = Path("artifacts")


def create_error_analysis_table(
    df: pd.DataFrame,
    y_true: pd.Series,
    y_score: pd.Series,
) -> pd.DataFrame:
    """Crear tabla de análisis de errores con segmentación."""

    # Predicción con umbral 0.5
    y_pred = (y_score >= 0.5).astype(int)
    errors = (y_true != y_pred).astype(int)

    # DataFrame base con predicciones y errores
    analysis_df = pd.DataFrame({
        "race": df["Race"],
        "year": df["Year"],
        "compound": df["Compound"],
        "stint": df["Stint"],
        "position": df["Position"],
        "race_progress": df["RaceProgress"],
        "y_true": y_true.values,
        "y_score": y_score,
        "y_pred": y_pred,
        "is_error": errors,
        "confidence": pd.Series(y_score).clip(0, 1),
    })

    return analysis_df


def segment_analysis(df: pd.DataFrame) -> dict:
    """Analizar errores por segmentos."""

    segments = {}

    # Por Race
    print("\n" + "="*70)
    print("Análisis por Race (evento de carrera)")
    print("="*70)
    race_segments = []
    for race_name in sorted(df["race"].unique()):
        mask = df["race"] == race_name
        subset = df[mask]

        try:
            auc = roc_auc_score(subset["y_true"], subset["y_score"])
        except ValueError:
            auc = None

        try:
            pr_auc = average_precision_score(subset["y_true"], subset["y_score"])
        except ValueError:
            pr_auc = None

        race_segments.append({
            "segment": race_name,
            "n_samples": len(subset),
            "n_errors": subset["is_error"].sum(),
            "error_rate": subset["is_error"].mean(),
            "auc": auc,
            "pr_auc": pr_auc,
            "pos_rate": subset["y_true"].mean(),
        })

    race_df = pd.DataFrame(race_segments).sort_values("auc", na_position="last")
    print(race_df[["segment", "n_samples", "error_rate", "auc"]].to_string(index=False))
    segments["by_race"] = race_df

    # Por Year
    print("\n" + "="*70)
    print("Análisis por Year")
    print("="*70)
    year_segments = []
    for year in sorted(df["year"].unique()):
        mask = df["year"] == year
        subset = df[mask]

        try:
            auc = roc_auc_score(subset["y_true"], subset["y_score"])
        except ValueError:
            auc = None

        year_segments.append({
            "year": year,
            "n_samples": len(subset),
            "n_errors": subset["is_error"].sum(),
            "error_rate": subset["is_error"].mean(),
            "auc": auc,
            "pos_rate": subset["y_true"].mean(),
        })

    year_df = pd.DataFrame(year_segments).sort_values("year")
    print(year_df.to_string(index=False))
    segments["by_year"] = year_df

    # Por Compound
    print("\n" + "="*70)
    print("Análisis por Compound (neumático)")
    print("="*70)
    compound_segments = []
    for compound in sorted(df["compound"].dropna().unique()):
        mask = df["compound"] == compound
        subset = df[mask]

        try:
            auc = roc_auc_score(subset["y_true"], subset["y_score"])
        except ValueError:
            auc = None

        compound_segments.append({
            "compound": compound,
            "n_samples": len(subset),
            "error_rate": subset["is_error"].mean(),
            "auc": auc,
            "pos_rate": subset["y_true"].mean(),
        })

    compound_df = pd.DataFrame(compound_segments).sort_values("auc", na_position="last")
    print(compound_df.to_string(index=False))
    segments["by_compound"] = compound_df

    # Por Stint (rangos)
    print("\n" + "="*70)
    print("Análisis por rango de Stint")
    print("="*70)
    df_stint = df.copy()
    df_stint["stint_range"] = pd.cut(df_stint["stint"], bins=5)

    stint_segments = []
    for stint_range in sorted(df_stint["stint_range"].cat.categories):
        mask = df_stint["stint_range"] == stint_range
        subset = df_stint[mask]

        if len(subset) == 0:
            continue

        try:
            auc = roc_auc_score(subset["y_true"], subset["y_score"])
        except ValueError:
            auc = None

        stint_segments.append({
            "stint_range": str(stint_range),
            "n_samples": len(subset),
            "error_rate": subset["is_error"].mean(),
            "auc": auc,
            "pos_rate": subset["y_true"].mean(),
        })

    stint_df = pd.DataFrame(stint_segments)
    print(stint_df.to_string(index=False))
    segments["by_stint"] = stint_df

    # Por Position (rangos)
    print("\n" + "="*70)
    print("Análisis por rango de Position")
    print("="*70)
    df_pos = df.copy()
    df_pos["position_range"] = pd.cut(df_pos["position"], bins=5)

    position_segments = []
    for pos_range in sorted(df_pos["position_range"].cat.categories):
        mask = df_pos["position_range"] == pos_range
        subset = df_pos[mask]

        if len(subset) == 0:
            continue

        try:
            auc = roc_auc_score(subset["y_true"], subset["y_score"])
        except ValueError:
            auc = None

        position_segments.append({
            "position_range": str(pos_range),
            "n_samples": len(subset),
            "error_rate": subset["is_error"].mean(),
            "auc": auc,
            "pos_rate": subset["y_true"].mean(),
        })

    position_df = pd.DataFrame(position_segments)
    print(position_df.to_string(index=False))
    segments["by_position"] = position_df

    # Por RaceProgress (fases de carrera)
    print("\n" + "="*70)
    print("Análisis por RaceProgress (fase de carrera)")
    print("="*70)
    df_rp = df.copy()
    df_rp["race_phase"] = pd.cut(
        df_rp["race_progress"],
        bins=[0, 0.33, 0.67, 1.0],
        labels=["early", "mid", "late"],
    )

    race_phase_segments = []
    for phase in ["early", "mid", "late"]:
        mask = df_rp["race_phase"] == phase
        subset = df_rp[mask]

        if len(subset) == 0:
            continue

        try:
            auc = roc_auc_score(subset["y_true"], subset["y_score"])
        except ValueError:
            auc = None

        race_phase_segments.append({
            "race_phase": phase,
            "n_samples": len(subset),
            "error_rate": subset["is_error"].mean(),
            "auc": auc,
            "pos_rate": subset["y_true"].mean(),
        })

    race_phase_df = pd.DataFrame(race_phase_segments)
    print(race_phase_df.to_string(index=False))
    segments["by_race_phase"] = race_phase_df

    return segments


def main():
    """Ejecutar análisis de errores Fase 10."""

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

    # Generar predicciones OOF
    print("\nGenerando predicciones OOF para E20...")
    e20_model = make_e15_hgb_e13()
    e20_best_params = {
        "learning_rate": 0.127,
        "max_iter": 152,
        "max_leaf_nodes": 38,
        "min_samples_leaf": 35,
        "l2_regularization": 0.84,
    }
    e20_model.set_params(**e20_best_params)

    y_true_oof = pd.Series(index=range(len(y_full)), dtype=float)
    y_score_oof = pd.Series(index=range(len(y_full)), dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(
        splitter.split(X_full, y_full, groups=group_key)
    ):
        print(f"  Fold {fold_idx + 1}/5...")
        X_train, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_train, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

        e20_model.fit(X_train, y_train)
        y_score_oof.iloc[val_idx] = e20_model.predict_proba(X_val)[:, 1]
        y_true_oof.iloc[val_idx] = y_val.values

    # Crear tabla de análisis
    print("\nCreando tabla de análisis de errores...")
    analysis_df = create_error_analysis_table(df_dev, y_true_oof, y_score_oof)

    # Segmentar análisis
    print("\nSegmentando errores...")
    segments = segment_analysis(analysis_df)

    # Guardar artefactos
    artifacts_dir = ARTIFACTS_DIR / "error_analysis"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    analysis_df.to_csv(artifacts_dir / "e20_oof_predictions.csv", index=False)

    for segment_name, segment_df in segments.items():
        segment_df.to_csv(artifacts_dir / f"e20_analysis_{segment_name}.csv", index=False)

    # Resumen
    print("\n" + "="*70)
    print("Fase 10 completada")
    print("="*70)
    print(f"Artefactos guardados en {artifacts_dir}")
    print(f"Total OOF predictions: {len(analysis_df)}")
    print(f"Total errors: {analysis_df['is_error'].sum()} ({analysis_df['is_error'].mean():.1%})")

    # Calidad general
    overall_auc = roc_auc_score(y_true_oof, y_score_oof)
    overall_pr_auc = average_precision_score(y_true_oof, y_score_oof)
    print(f"Overall AUC: {overall_auc:.4f}")
    print(f"Overall PR-AUC: {overall_pr_auc:.4f}")

    print("\nPróximo paso: Fase 11 (MLflow final) o Fase 12 (skops).")


if __name__ == "__main__":
    main()
