# HANDOFF — F1 Pit Stop Prediction

> Convencion propia del portafolio (no estandar de Claude Code): este
> archivo es la fuente de verdad del estado exacto del trabajo entre
> sesiones. Se lee al empezar una sesion y se actualiza al terminarla.
> Optimizar para que quien lea esto (humano o Claude, sin memoria de la
> sesion anterior) pueda continuar sin tener que releer todo el spec.

## Estado actual

- **Fase activa:** Fase 2 (EDA dirigido) — no iniciada todavia.
- **Ultimo criterio de salida cumplido:** Fase 1 (Ingesta y auditoria) —
  ver `artifacts/reports/data_audit.md` para el detalle completo. Las
  cuatro listas requeridas (`FEATURE_CANDIDATES`, `ID_COLUMNS`, `TARGET`,
  `SUSPECTED_LEAKAGE`, `EXCLUDED_COLUMNS`) estan en la seccion 6 de ese
  reporte.
- **Entorno:** creado con `uv` (Python 3.11.9). `uv.lock` generado y
  commiteado.

## Datos (Fase 1 cerrada)

`train.csv` (439,140 x 16), `test.csv` (188,165 x 15),
`sample_submission.csv` (188,165 x 2) descargados via `kaggle` CLI a
`data/raw/` (inmutable, protegido por hook). Fingerprints sha256 en
`artifacts/reports/data_audit.md` seccion 1.

**Discrepancia con el spec:** el spec (linea 70) dice "el dataset publicado
contiene 33 columnas"; el CSV real tiene 16. No se corrige el spec, se
documenta la discrepancia y se trabaja con la estructura real (detalle en
el audit report, seccion 2).

**Contexto de Kaggle no cubierto por el spec:** el dataset esta inspirado
en un dataset real de estrategia F1; se removio intencionalmente
`Normalized_TyreLife` porque hacia la prediccion trivial. El link al
dataset original esta roto en Kaggle (relevante para H4, ver bloqueadores).

Codigo reusable: `src/f1pitstop/data/ingest.py` (carga + fingerprint +
validacion de presencia del target) y `src/f1pitstop/data/schema.py`
(checklist estructural + `cardinality_summary()`). Cubiertos por
`tests/test_ingest.py` y `tests/test_schema.py` (19 tests, todos pasan).
`validate_schema()` sobre los datos reales: `has_errors = False`, solo 4
issues `info` por heuristica de nombre.

## HALLAZGO CRITICO para Fase 3 — leer antes de disenar el split

`Stint` **no es monotono** dentro de `(Driver, Race, Year)` ordenado por
`LapNumber` en el 80.4% de una muestra de 2,000 grupos (fisicamente
imposible en una carrera real). Ademas `LapNumber` no es consecutivo
dentro de un mismo grupo (hay huecos). Ver `data_audit.md` seccion 5.4
para el detalle completo.

**Esto significa que agrupar por `(Driver, Race, Year)` y asumir una
trayectoria continua de vueltas NO es valido sin mas investigacion.**
Afecta directamente:
- la Fase 3 (que clave de agrupacion usar para el split V1/V2);
- la Fase 6 (cualquier feature rolling/lag que asuma continuidad de
  vueltas, regla de oro de `.claude/rules/leakage-and-validation.md`
  seccion 5, se apoya en un supuesto que hay que verificar primero).

Otros hallazgos del audit (no criticos, pero a resolver en Fase 2/3):
- `Driver` no se comporta como un grid real de F1 (887 valores unicos,
  414 distintos en una sola carrera) — no asumir que es un identificador
  de piloto consistente.
- `Race` sola no identifica un evento unico (se repite entre `Year`
  distintos); la clave de agrupacion candidata es `(Race, Year)`.
- `PitStop` vs `PitNextLap`: diferencia moderada (19.1% vs 24.8% de tasa
  de pit en la siguiente vuelta), no concluyente por si sola — confirmar
  semantica temporal exacta antes de usar `PitStop` como feature.

## Ultima sesion

- **Fecha:** 2026-08-26
- **Que se hizo:**
  - Fase 0: entorno `uv`, smoke test de los 10 pasos del stack (OK, un
    unico entorno), `uv.lock`, primer commit.
  - Fase 1: descarga de datos de Kaggle (requirio aceptar reglas de la
    competencia para que la API dejara el 403), `ingest.py`, `schema.py`,
    19 tests, `artifacts/reports/data_audit.md`,
    `artifacts/tables/schema_summary.csv`.
  - Auditoria manual mas alla del checklist automatico: balance del
    target (80/20), estructura de `Race`/`Driver`/`Year`, y el hallazgo
    critico de `Stint` no monotono (ver seccion arriba).
- **Que se aprendio / decidio (Fase 0, referencia rapida):**
  1. scikit-learn 1.9.0 + AutoGluon 1.6.1 conviven sin conflicto — no hizo
     falta separar entornos.
  2. MLflow 3.15.2 deprecó el backend de filesystem; usar
     `sqlite:///.../mlflow.db` (aplica tambien desde Fase 4).
  3. skops marca `numpy.dtype` como no confiable por defecto; se acepta
     via allowlist explicita revisada manualmente (mismo patron a seguir
     en Fase 12).
  4. AutoGluon corrio solo con modelos base en el smoke test (sin
     torch/lightgbm/catboost/xgboost instalados) — decidir en Fase 8 si
     se instalan esos extras.

## Proxima accion concreta

Ejecutar Fase 2 del spec (EDA dirigido, notebook `02_eda.ipynb`) con foco
en responder, en este orden de prioridad:
1. **Investigar el hallazgo de `Stint` no monotono** (ver seccion arriba)
   — es el bloqueador conceptual mas importante antes de poder disenar
   bien la Fase 3.
2. Confirmar si `(Race, Year)` es la clave de agrupacion correcta.
3. Confirmar semantica temporal de `PitStop`, `LapTime_Delta`,
   `Cumulative_Degradation`, `RaceProgress`, `Position_Change` (columnas
   en `SUSPECTED_LEAKAGE`, ver `data_audit.md` seccion 6).

Nota: no se genero todavia `notebooks/02_eda.ipynb` ni el previo
`notebooks/01_data_audit.ipynb` mencionado en la arquitectura del spec —
el criterio de salida de la Fase 1 (explicar columnas + 4 listas) se
cubrio via el reporte `.md` + tests, que es reproducible. Si se quiere el
notebook de Fase 1 como artefacto de comunicacion, sigue pendiente pero no
bloquea avanzar a Fase 2.

## Bloqueadores / dudas abiertas

- **Hallazgo de `Stint` no monotono** (ver seccion dedicada arriba) — la
  duda abierta mas importante del proyecto en este momento.
- Decidir en Fase 8 si se instalan los extras opcionales de AutoGluon
  (torch/lightgbm/catboost/xgboost).
- Directorio local `mlruns_smoke_test/` (SQLite del smoke test) sin
  borrar — esta en `.gitignore`, no se commitea, pero no se elimino del
  disco por no ejecutar un `rm -rf` sin confirmacion explicita.
- El link al dataset F1 original (para `data/external/`, relevante para
  H4) esta roto en Kaggle. No bloquea el trabajo actual; si se retoma H4
  hay que buscar el dataset por otra via (ej. FastF1) y documentar la
  fuente exacta.
- `notebooks/01_data_audit.ipynb` y `02_eda.ipynb` de la arquitectura del
  spec no se crearon todavia (ver "Proxima accion concreta").

## Historial de fases (actualizar segun se cierre cada una)

| Fase | Estado | Fecha cierre | Notas |
|---|---|---|---|
| 0 — Smoke test | **cerrada** | 2026-08-26 | Un unico entorno; fix de MLflow (sqlite) y skops (allowlist) documentados arriba |
| 1 — Ingesta y auditoria | **cerrada** | 2026-08-26 | Ver `artifacts/reports/data_audit.md`; hallazgo critico de `Stint` no monotono queda abierto para Fase 3 |
| 2 — EDA dirigido | pendiente | | prioridad: investigar hallazgo de Stint |
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
