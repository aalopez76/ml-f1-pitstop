# HANDOFF — F1 Pit Stop Prediction

> Convencion propia del portafolio (no estandar de Claude Code): este
> archivo es la fuente de verdad del estado exacto del trabajo entre
> sesiones. Se lee al empezar una sesion y se actualiza al terminarla.
> Optimizar para que quien lea esto (humano o Claude, sin memoria de la
> sesion anterior) pueda continuar sin tener que releer todo el spec.

## Estado actual

- **Fase activa:** Fase 1 (Ingesta y auditoria) — no iniciada todavia.
- **Ultimo criterio de salida cumplido:** Fase 0 (Smoke test del stack) —
  los 10 pasos pasaron en el mismo entorno, sin necesidad de separar
  AutoGluon en un entorno aparte.
- **Entorno:** creado con `uv` (Python 3.11.9). `uv.lock` generado y
  commiteado. Repo git inicializado (`uv init` lo creo automaticamente).

## Ultima sesion

- **Fecha:** 2026-08-26
- **Que se hizo:**
  - Se inicializo el proyecto con `uv init` (layout `src/f1pitstop/`,
    `scripts/`, `tests/`).
  - Se agregaron las dependencias del stack: pandas, scikit-learn, skrub,
    skore, skops, mlflow, pytest, ruff, y `autogluon.tabular`.
  - Se escribio y ejecuto `scripts/smoke_test_stack.py` cubriendo los 10
    pasos de la Fase 0 (spec, seccion 5).
  - Smoke test paso completo (exit code 0) en un UNICO entorno — no hizo
    falta separar AutoGluon como preveia la nota de dependencias del spec.
  - Se genero `uv.lock` y se preparo el primer commit de dependencias.
- **Que se aprendio / decidio:**
  1. **scikit-learn 1.9.0 + AutoGluon 1.6.1 conviven sin conflicto** en este
     entorno (Python 3.11.9, Windows). No se necesito el plan B de dos
     entornos separados que anticipaba el spec.
  2. **MLflow 3.15.2 deprecó el backend de filesystem** (`./mlruns` via
     `set_tracking_uri(path)`) — lanza `MlflowException` pidiendo migrar a
     un backend de base de datos. Decision: usar backend **SQLite local**
     (`sqlite:///.../mlflow.db`) en vez del opt-out deprecado
     (`MLFLOW_ALLOW_FILE_STORE=true`). Esto aplica tambien para el MLflow
     "de verdad" desde la Fase 4 en adelante — usar URI `sqlite:///...`,
     no una ruta de carpeta plana.
  3. **skops marca `numpy.dtype` como tipo no confiable por defecto** en
     `get_untrusted_types()`, aunque es un tipo estandar presente en
     practicamente cualquier pipeline sklearn serializado. Se establecio
     una allowlist explicita y revisada manualmente (`KNOWN_SAFE_TYPES =
     {"numpy.dtype"}`) en el smoke test; el patron a seguir en Fase 12
     (persistencia final) es el mismo: inspeccionar, no aceptar a ciegas,
     documentar por que cada tipo se considera seguro.
  4. **AutoGluon corrio solo con sus modelos base** en el smoke test: los
     extras opcionales `torch`, `lightgbm`, `catboost`, `xgboost` no estan
     instalados, asi que esos model families fallan al importar y AutoGluon
     los salta automaticamente (no rompe el fit). Esto es aceptable para el
     smoke test (solo probaba que el stack funciona), pero **hay que
     decidir explicitamente en la Fase 8** si se instalan esos extras
     (`autogluon.tabular[lightgbm,catboost,xgboost]` y/o `torch`) para que
     el challenger de AutoML compita con su set completo de modelos, o si
     se documenta la limitacion como parte de la comparacion manual vs
     AutoML.

## Proxima accion concreta

Ejecutar Fase 1 del spec (seccion 6): descargar `train.csv`, `test.csv`,
`sample_submission.csv` de Kaggle (Playground Series S6E5) a `data/raw/`
(inmutable, protegido por hook), y escribir:
- `src/f1pitstop/data/ingest.py`
- `src/f1pitstop/data/schema.py`

Criterio de salida de la Fase 1: poder explicar cada columna y producir las
listas `FEATURE_CANDIDATES`, `ID_COLUMNS`, `TARGET`, `SUSPECTED_LEAKAGE`,
`EXCLUDED_COLUMNS`.

## Bloqueadores / dudas abiertas

- Decidir en Fase 8 si se instalan los extras opcionales de AutoGluon
  (torch/lightgbm/catboost/xgboost) — ver punto 4 de "que se aprendio"
  arriba.
- Queda un directorio local `mlruns_smoke_test/` (SQLite del smoke test)
  sin borrar — ya esta en `.gitignore`, no se commitea, pero no se elimino
  del disco por no ejecutar un `rm -rf` sin confirmacion explicita.

## Historial de fases (actualizar segun se cierre cada una)

| Fase | Estado | Fecha cierre | Notas |
|---|---|---|---|
| 0 — Smoke test | **cerrada** | 2026-08-26 | Un unico entorno; fix de MLflow (sqlite) y skops (allowlist) documentados arriba |
| 1 — Ingesta y auditoria | pendiente | | |
| 2 — EDA dirigido | pendiente | | |
| 3 — Validacion y leakage | pendiente | | |
| 4 — Baselines | pendiente | | |
| 5 — skrub | pendiente | | |
| 6 — Feature engineering | pendiente | | |
| 7 — Modelos manuales | pendiente | | |
| 8 — AutoGluon challenger | pendiente | | decidir extras opcionales (ver bloqueadores) |
| 9 — skore | pendiente | | |
| 10 — Error analysis | pendiente | | |
| 11 — MLflow final | pendiente | | |
| 12 — skops | pendiente | | reusar patron de allowlist del smoke test |
| 13 — Holdout final y Kaggle | pendiente | | |
