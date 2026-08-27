# HANDOFF — F1 Pit Stop Prediction

> Convencion propia del portafolio (no estandar de Claude Code): este
> archivo es la fuente de verdad del estado exacto del trabajo entre
> sesiones. Se lee al empezar una sesion y se actualiza al terminarla.
> Optimizar para que quien lea esto (humano o Claude, sin memoria de la
> sesion anterior) pueda continuar sin tener que releer todo el spec.

## Estado actual

- **Fase activa:** Fase 3 (Validacion y leakage) — no iniciada todavia.
  **Leer `.claude/rules/leakage-and-validation.md` completo antes de tocar
  `split.py`.**
- **Ultimo criterio de salida cumplido:** Fase 2 (EDA dirigido) — ver
  `artifacts/reports/eda_report.md`. Lista priorizada de 10 hipotesis en
  la seccion final de ese reporte.
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

## HALLAZGO CRITICO para Fase 3 — leer antes de disenar el split (confirmado en Fase 2)

`Stint` **no es monotono** dentro de `(Driver, Race, Year)` ordenado por
`LapNumber` en ~80% de una muestra de grupos (fisicamente imposible en
una carrera real). Ademas `LapNumber` no es consecutivo dentro de un
mismo grupo (hay huecos). Ver `data_audit.md` seccion 5.4.

**Investigacion adicional en Fase 2 (ver `eda_report.md`, pregunta 8)
confirmo y explico el mecanismo:**
- `Position_Change` NO coincide con la diferencia de `Position` entre
  filas visibles consecutivas → las columnas derivadas
  (`Position_Change`, `RaceProgress`, `Cumulative_Degradation`,
  `LapTime_Delta`) se calcularon sobre una secuencia completa oculta, y
  el CSV publico es un submuestreo de esa secuencia.
- **Prueba matematica de que la no-monotonicidad de `Stint` NO es un
  artefacto de submuestreo:** quitar filas de una secuencia no-decreciente
  nunca puede producir una caida visible. Se observaron caidas de hasta
  -5 en 81.6% de una muestra de 3,000 grupos → es una inconsistencia real
  de la generacion sintetica, no solo un efecto de que faltan filas.

**Conclusion:** el dataset es una generacion sintetica que aproxima
distribuciones/relaciones marginales (Kaggle: *"Feature distributions are
close to, but not exactly the same, as the original"*) sin garantizar
restricciones de consistencia secuencial estricta dentro de cada grupo.

**Esto significa que agrupar por `(Driver, Race, Year)` y asumir una
trayectoria continua de vueltas NO es valido sin tratamiento cuidadoso.**
Afecta directamente:
- la Fase 3 (que clave de agrupacion usar para el split V1/V2 — usar
  `(Race, Year)`, no `Driver`, ver mas abajo);
- la Fase 6 (cualquier feature rolling/lag debe usar `LapNumber` como
  distancia real entre vueltas, no solo el orden de fila, y `Stint` no
  debe tratarse como contador confiable sin verificacion adicional).

Otros hallazgos (Fase 1 + Fase 2, no criticos pero a resolver en Fase 3):
- `Driver` no se comporta como un grid real de F1 (887 valores unicos,
  414 distintos en una sola carrera) — no asumir que es un identificador
  de piloto consistente.
- `Race` sola no identifica un evento unico (se repite entre `Year`
  distintos, y train/test comparten las mismas 26 carreras y 4 anios); la
  clave de agrupacion candidata es `(Race, Year)`.
- `PitStop` vs `PitNextLap`: diferencia moderada (19.1% vs 24.8% de tasa
  de pit en la siguiente vuelta), no concluyente por si sola — confirmar
  semantica temporal exacta antes de usar `PitStop` como feature.
- Train y test de Kaggle comparten exactamente las mismas carreras/anios
  (split row-level, no agrupado) — no imitar ese split para la validacion
  interna; el objetivo de H1 es evaluar el escenario mas realista
  (carrera nueva no vista), independiente de como particiono Kaggle.

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
  - Fase 2: EDA dirigido por las 8 preguntas obligatorias del spec.
    Investigacion a fondo del hallazgo de `Stint` (confirmado que NO es
    artefacto de submuestreo, con prueba matematica). 7 figuras en
    `artifacts/figures/`. `artifacts/reports/eda_report.md` con las 8
    respuestas y 10 hipotesis priorizadas para Fase 3+.
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

Ejecutar Fase 3 del spec (Validacion y leakage). **Leer
`.claude/rules/leakage-and-validation.md` completo primero** (regla no
negociable 7 de CLAUDE.md). En orden de prioridad, guiado por las
hipotesis 1–3 de `eda_report.md`:
1. Disenar `src/f1pitstop/data/split.py` comparando V0 (StratifiedKFold
   aleatorio), V1 (group-aware por `(Race, Year)` — no por `Driver` solo,
   ver hallazgos arriba) y V2 (holdout temporal por `Year`).
2. Cuantificar cuanto se infla el ROC-AUC con V0 vs V1/V2 (test directo
   de H1 del spec).
3. Congelar el holdout final (nunca se usa para decisiones de modelado,
   solo Fase 13).
4. Revisar `PitStop` y las columnas de `SUSPECTED_LEAKAGE`
   (`LapTime_Delta`, `Cumulative_Degradation`, `RaceProgress`,
   `Position_Change`) con el checklist de 5 preguntas antes de aceptarlas.
5. Invocar el subagente `leakage-auditor` antes de dar por cerrada esta
   fase.

Nota: `notebooks/01_data_audit.ipynb` y `02_eda.ipynb` de la arquitectura
del spec no se crearon todavia — los criterios de salida de Fase 1 y 2 se
cubrieron via reportes `.md` + tests + figuras, que son reproducibles.
Pendiente si se quieren como artefacto de comunicacion, no bloquea Fase 3.

## Bloqueadores / dudas abiertas

- Decidir en Fase 3 la clave de agrupacion exacta para el split
  (`(Race, Year)` es la candidata, ver hallazgos arriba).
- Decidir en Fase 8 si se instalan los extras opcionales de AutoGluon
  (torch/lightgbm/catboost/xgboost).
- Directorio local `mlruns_smoke_test/` (SQLite del smoke test) sin
  borrar — esta en `.gitignore`, no se commitea, pero no se elimino del
  disco por no ejecutar un `rm -rf` sin confirmacion explicita.
- El link al dataset F1 original (para `data/external/`, relevante para
  H4) esta roto en Kaggle. No bloquea el trabajo actual. **Verificado en
  esta sesion:** `fastf1` (pip/uv, `uv run --with fastf1 ...`) es una
  libreria real, funcional en este entorno (probado con
  `get_event_schedule` y `session.laps` de Bahrain 2023), que trae datos
  via API sin descargas manuales. Sus columnas (`Driver`, `LapNumber`,
  `Stint`, `Compound`, `TyreLife`, `Position`, `PitInTime`/`PitOutTime`)
  coinciden casi exactamente con el esquema sintetico de Kaggle — fuente
  muy plausible (no 100% confirmada, el link roto impide verificarlo con
  certeza) para el dataset original que menciona H4. Dato interesante:
  en FastF1 real, `LapNumber` es consecutivo y `Stint` es monotono,
  reforzando la conclusion de la Fase 2 (el CSV de Kaggle es sintetico y
  no preserva esa consistencia). No se instalo como dependencia permanente
  (regla 3 de CLAUDE.md: sin pregunta experimental que lo justifique
  todavia — H4 sigue siendo de baja prioridad). Si se retoma H4, este es
  el camino a seguir.
- `notebooks/01_data_audit.ipynb` y `02_eda.ipynb` de la arquitectura del
  spec no se crearon todavia (ver "Proxima accion concreta").

## Historial de fases (actualizar segun se cierre cada una)

| Fase | Estado | Fecha cierre | Notas |
|---|---|---|---|
| 0 — Smoke test | **cerrada** | 2026-08-26 | Un unico entorno; fix de MLflow (sqlite) y skops (allowlist) documentados arriba |
| 1 — Ingesta y auditoria | **cerrada** | 2026-08-26 | Ver `artifacts/reports/data_audit.md`; hallazgo critico de `Stint` no monotono queda abierto para Fase 3 |
| 2 — EDA dirigido | **cerrada** | 2026-08-26 | Ver `artifacts/reports/eda_report.md`; confirmado que Stint no es artefacto de submuestreo; 10 hipotesis para Fase 3+ |
| 3 — Validacion y leakage | pendiente | | prioridad: comparar V0/V1/V2, cuantificar H1 |
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
