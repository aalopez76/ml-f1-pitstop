# F1 Pit Stop Prediction — Guia de Trabajo

## Que es esto

Proyecto de portafolio ML: predecir `PitNextLap` (Kaggle Playground Series
S6E5, metrica ROC-AUC). Pregunta de portafolio: *"How much does a carefully
designed, leakage-aware ML pipeline gain or lose against AutoML, and what is
the cost in complexity, compute and interpretability?"*

Fuente de verdad completa: `01_F1_Pit_Stop_ML_Project_Spec.txt` (24 fases,
criterios de salida, matriz de experimentos, definition of done). Prioridad
frente a los otros proyectos del portafolio: `00_Scope_And_Priority.md` —
este es TIER 1 / FLAGSHIP, se ejecuta completo, sin recortes.

## Protocolo de sesion

1. Al empezar: leer `HANDOFF.md` para saber en que fase quedo el proyecto,
   que se hizo en la ultima sesion y cual es la siguiente accion concreta.
2. No avanzar a la siguiente fase si el criterio de salida de la fase actual
   (spec) no esta cumplido. Si no esta claro si se cumple, decirlo
   explicitamente en vez de asumir que si.
3. Al cerrar una sesion de trabajo: actualizar `HANDOFF.md` con el estado
   real (incluyendo bloqueos y dudas abiertas), no un resumen optimista.

## Reglas no negociables

(resumen operativo; el detalle y el porque de cada una esta en el spec,
seccion "PRINCIPIO DE EJECUCION" y "Reglas obligatorias")

1. No empezar por AutoGluon, tuning ni ensembles.
2. No activar MLflow antes de tener una ejecucion local minima funcionando
   (a partir de Fase 4).
3. Ninguna herramienta del stack entra sin una pregunta experimental que la
   justifique.
4. Nada de MLOps completo en este proyecto: sin Kubernetes/Airflow/serving/
   feature store/CI-CD de modelos.
5. Notebooks son para exploracion/comunicacion. Toda logica reusable vive en
   `src/f1pitstop/`, nunca solo en un notebook.
6. Se conserva un holdout final que NUNCA se usa para decisiones de
   modelado, solo para la evaluacion confirmatoria de la Fase 13.
7. Antes de escribir o modificar codigo de split o de feature engineering,
   leer `.claude/rules/leakage-and-validation.md` completo.
8. El leaderboard de Kaggle es evidencia externa, no el mecanismo de
   seleccion del modelo final.
9. Toda afirmacion de mejora requiere validacion reproducible (CV con seed
   fijo comparando folds equivalentes), nunca una unica corrida suelta.

## Entorno

- Python 3.11. `uv` para instalar. Lockfile (`uv.lock`) se genera DESPUES de
  que el smoke test (Fase 0) pase completo, nunca antes.
- AutoGluon puede exigir una version de scikit-learn mas antigua que la que
  necesitan skrub/skore. Si el smoke test no logra una instalacion unica,
  documentarlo y separar en dos entornos (ver spec, "NOTA SOBRE
  DEPENDENCIAS CRITICAS"). No forzar una instalacion unica a costa de
  tiempo de proyecto.
- Codigo, nombres de variables y README tecnico en ingles. Notas de trabajo
  y commits pueden ir en espanol.

## Estructura del repo

Arbol completo en el spec, seccion 4. Resumen de donde vive cada cosa:

```
src/f1pitstop/{data,features,models,evaluation,tracking,persistence}/  # logica reusable
scripts/            # entrypoints ejecutables (smoke test, train, evaluate, submission)
notebooks/          # 01_data_audit, 02_eda, 03_leakage_and_validation, 04_results_review
configs/            # yaml de data/validation/features/models/experiments
artifacts/          # reports, figures, tables, model_cards
models/             # artefactos entrenados (sklearn/, autogluon/)
tests/              # test_schema, test_split, test_features, test_pipeline, test_submission
```

## Cuando leer cada rules file

- `.claude/rules/leakage-and-validation.md`: SIEMPRE antes de tocar
  `data/split.py`, cualquier archivo bajo `features/`, o de disenar/ajustar
  la estrategia de CV. Relevante en Fases 3, 6, 8 y 13.
- `.claude/rules/experiment-tracking.md`: antes de crear un nuevo run de
  MLflow, nombrar un experimento nuevo, o decidir que se registra en skore
  vs MLflow. Relevante desde Fase 4 en adelante.

## Piezas de Claude Code disponibles en este proyecto

- **Subagente** `.claude/agents/leakage-auditor.md`: invocarlo (Agent tool)
  antes de dar por cerrada la Fase 6 (feature engineering) o la Fase 8
  (AutoGluon), para una revision de leakage con ojos frescos sobre
  `src/f1pitstop/features/` y `src/f1pitstop/data/split.py`. Es de solo
  lectura: reporta hallazgos, no edita codigo.
- **Skill** `/new-experiment`: usarla al arrancar cada nuevo experimento de
  la matriz (Exx/Axx/Fxx, seccion 20 del spec) para mantener consistencia
  en nombres, tags de MLflow y estructura del script.

## Calidad

- `uv run pytest` debe pasar antes de considerar cerrada cualquier fase.
- `uv run ruff check .` sin errores antes de cualquier commit.
- `data/raw/` y `data/external/` son inmutables. Un hook de Claude Code
  (`.claude/settings.json` + `.claude/hooks/protect_raw_data.py`) bloquea
  escrituras/ediciones ahi automaticamente. Si de verdad hace falta tocar
  esos archivos, hazlo manualmente fuera de Claude Code y documenta el
  motivo en el README.
