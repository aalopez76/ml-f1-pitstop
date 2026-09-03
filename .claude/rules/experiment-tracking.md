# Convenciones de experimentos y tracking — F1 Pit Stop

Leer antes de crear un nuevo run de MLflow, nombrar un experimento nuevo, o
decidir si algo se registra en skore, en MLflow, o en ambos.

## Division de responsabilidades

- **skore** = evaluacion, diagnostico y reportes (ROC/PR curves, calibration,
  permutation importance, ComparisonReport).
- **MLflow** = historial transversal de runs, parametros, metricas y
  artefactos a lo largo de todo el proyecto.

No duplicar todo automaticamente en los dos. Si algo ya vive bien en skore
(un reporte visual puntual), no hace falta tambien subirlo entero a MLflow
run por run; basta con referenciar el artefacto exportado.

## Nombres de experimento en MLflow

Experimento de MLflow: `f1_pitstop`.

Nombres de run estables, alineados con la matriz de experimentos del spec
(seccion 20):

```
E00_dummy               E13_full_leakage_safe_features
E01_logreg_basic        E20_tuned_manual_candidate_a
E02_hgb_basic           E21_tuned_manual_candidate_b
E03_manual_preprocessing A00_autogluon_raw
E04_skrub_preprocessing  A01_autogluon_engineered
E10_raw_features         F00_final_sklearn
E11_basic_domain_features F01_final_autogluon
E12_temporal_features
E22_xgboost_e13_features   E24_lightgbm_e13_features
E23_catboost_e13_features  E25_ensemble_logit_stack
```

`E22`-`E25` (Fase 14, "Model Selection Framework"): candidatos de
diversidad controlada sobre el mismo feature set E13 y CV V1 — ver
`artifacts/reports/model_selection_framework.md`. Se comparan y deciden
enteramente sobre CV en `dev`, nunca sobre el holdout congelado (ver
`.claude/rules/leakage-and-validation.md` §9).

## Tags obligatorios por run

```
project=f1_pitstop
stage=baseline|features|tuning|automl|final
model_family=...
feature_set=...
validation=...
seed=...
```

## Metricas obligatorias por run

```
cv_roc_auc_mean
cv_roc_auc_std
holdout_roc_auc       # solo en runs finalistas (Fase 13)
holdout_pr_auc        # solo en runs finalistas
fit_seconds
predict_ms_per_1k_rows  # si se mide consistentemente
```

## Registro de modelos

No usar `log_models=True` en todos los runs preliminares — el disco crece
sin control. Usar `log_models=False` para runs exploratorios y registrar el
modelo completo solo para el ganador y los finalistas (Fase 11 en
adelante).

## Artefactos pesados

`mlruns/` NUNCA se versiona en Git (ya deberia estar en `.gitignore`). Para
compartir el mejor run en el portafolio basta con las metricas y graficos
ya exportados a `artifacts/`.
