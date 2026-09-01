# HANDOFF — F1 Pit Stop Prediction

> Convencion propia del portafolio (no estandar de Claude Code): este
> archivo es la fuente de verdad del estado exacto del trabajo entre
> sesiones. Se lee al empezar una sesion y se actualiza al terminarla.
> Optimizar para que quien lea esto (humano o Claude, sin memoria de la
> sesion anterior) pueda continuar sin tener que releer todo el spec.

## Estado actual

- **Fase activa:** Fase 8 (AutoGluon challenger) — no iniciada todavia.
- **Ultimo criterio de salida cumplido:** Fase 7 (Modelos manuales) — ver
  `README.md` seccion "Manual models (Fase 7)",
  `scripts/phase7_manual_models.py`,
  `artifacts/tables/phase7_defaults_comparison.csv`,
  `artifacts/tables/phase7_tuning_results.csv`,
  `artifacts/tables/phase7_stint_ablation.csv`. Ganador:
  **E20_hist_gradient_boosting (tuneado), ROC-AUC 0.8611±0.0251** (vs
  0.8599 default, ganancia marginal de tuning; ExtraTrees tuneado
  0.8530 queda por debajo). Candidato manual que se lleva a Fase 8 para
  comparar contra AutoGluon. Ablation de Stint (punto abierto de Fase 6)
  resuelto: quitar `Stint` crudo cuesta -0.030 ROC-AUC, se mantiene en
  el feature set (E13, 10 columnas, sin cambios respecto a Fase 6).
- **Commit al dia:** el trabajo de Fase 3 a Fase 7 se commiteo el
  2026-08-31 (`7153148`, 37 archivos) tras confirmacion explicita del
  usuario. Ya no es un bloqueador para Fase 8.
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

- **Fecha:** 2026-08-31 (Fase 7, sesion continua)
- **Que se hizo (Fase 7 — Modelos manuales, cerrada):**
  - `src/f1pitstop/models/manual_models.py`: 3 candidatos sobre el
    feature set `E13_full_leakage_safe_features` (10 columnas, sin
    cambios respecto a Fase 6): `make_e14_logreg_e13` (imputacion
    mediana + escalado + one-hot), `make_e15_hgb_e13` (soporte nativo de
    NaN/categoricas), `make_e16_extratrees_e13` (imputacion + one-hot).
    `laptime_delta_prev` tiene NaN en la primera vuelta VISIBLE de cada
    grupo (esperado, ver `features/temporal.py`) — HGB lo soporta
    nativo, logreg/extratrees imputan con mediana dentro del Pipeline
    (sin leakage, fiteado solo sobre train de cada fold). `registry`
    `MANUAL_DEFAULTS_REGISTRY`, espacios de busqueda
    `PARAM_DISTRIBUTIONS` y `make_tunable_model()` para el tuning.
  - `scripts/phase7_manual_models.py`: reproducible, corre los 3 pasos
    del procedimiento del spec (comparar defaults -> seleccionar top-2
    -> `RandomizedSearchCV` n_iter=20 sobre esas 2) mas el ablation de
    Stint, todo con CV V1 y logueado a MLflow (`stage=tuning`).
  - **Resultados paso 1 (defaults, sin tuning):** E14 logreg 0.777±0.040,
    **E15 HGB 0.860±0.027** (coincide exactamente con E13 de Fase 6 —
    mismo modelo/features, check de reproducibilidad pasado), E16
    ExtraTrees 0.829±0.021. Top-2 seleccionadas: `hist_gradient_boosting`,
    `extra_trees` (logreg descartada, muy por debajo).
  - **Resultados paso 3 (tuning, RandomizedSearchCV n_iter=20, CV V1):**
    **E20_hist_gradient_boosting 0.8611±0.0251** (+0.0012 vs default,
    ganancia marginal — el default ya estaba cerca de su techo en este
    feature set), E21_extra_trees 0.8530±0.0230 (+0.0239 vs default,
    recupera terreno pero no alcanza a HGB). **Ganador: E20** (best
    params: `learning_rate≈0.127`, `max_iter=152`, `max_leaf_nodes=38`,
    `min_samples_leaf=35`, `l2_regularization≈0.84`).
  - **Bug de rendimiento encontrado y corregido en esta sesion:**
    `RandomizedSearchCV(n_jobs=-1)` anidado con
    `ExtraTreesClassifier(n_jobs=-1)` sobre-suscribe los 8 cores de la
    maquina y el tuning quedo mas de 70 minutos sin completar los
    100 fits (20 configs x 5 folds) — confirmado con CPU activa (no
    colgado) pero muy ineficiente por thrashing. Se mato el proceso y se
    corrigio: `ExtraTreesClassifier(n_jobs=2)` (parametro `n_jobs` en
    `make_e16_extratrees_e13`, usado por `make_tunable_model`) +
    `RandomizedSearchCV(n_jobs=4)` para esa familia especificamente
    (`SEARCH_N_JOBS` en `phase7_manual_models.py`, 2×4=8). Con el fix,
    el tuning de ExtraTrees completo en ~65-90 min (compute-heavy mismo,
    ya no thrashing). El script commiteado YA tiene el fix — correrlo de
    cero desde `main()` reproduce el mismo resultado sin el problema
    (documentado el runtime esperado en el docstring del script).
  - **Ablation de Stint (resuelve el punto abierto de Fase 6):** HGB
    sobre E13 con vs sin `Stint` crudo, misma CV V1. Con `Stint`:
    0.860±0.027 (10 features). Sin `Stint`: 0.830±0.025 (9 features).
    **Quitar `Stint` crudo cuesta -0.030 ROC-AUC** — pese a ser no
    monotono en 81.6% de los grupos (Fase 1/2), sigue aportando senal
    real que `recomputed_stint` no reemplaza del todo. **Decision: se
    mantiene en el feature set por defecto** (sin cambios de codigo,
    solo documentacion — cierra la duda dejada en Fase 6).
  - Resultados completos en `artifacts/tables/phase7_defaults_comparison.csv`,
    `artifacts/tables/phase7_tuning_results.csv`,
    `artifacts/tables/phase7_stint_ablation.csv`. Detalle narrativo en
    `README.md`, seccion "Manual models (Fase 7)".
  - `tests/test_manual_models.py` (7 tests) nuevos, con datos toy
    (incluye NaN centinela en `laptime_delta_prev` para probar
    imputacion). Suite completa: 74 tests, todos pasan. `ruff check .`
    limpio.
  - No se invoco el subagente `leakage-auditor` en esta fase: no se
    toco `data/split.py` ni `features/` (regla de CLAUDE.md aplica a
    Fases 3/6/8/13 explicitamente, no a Fase 7 — esta fase solo agrega
    modelos sobre un feature set ya auditado en Fase 6).
- **Que se hizo (Fase 3 a Fase 6, sesion 2026-08-27/28):**
- **Que se hizo (Fase 6 — Feature engineering F1, cerrada):**
  - `src/f1pitstop/features/temporal.py`: unidad de agrupacion
    `(Driver, Race, Year)` (DISTINTA de `(Race, Year)` usada en
    `split.py` para CV — documentado explicitamente el porque). Familias:
    `add_winsorized_laptime` (cap FIJO de 150s, no percentil derivado —
    evita violar la pregunta 5 del checklist de leakage),
    `add_basic_domain_features` (`pit_stops_so_far`, `recomputed_stint`),
    `add_temporal_features` (`laptime_delta_prev`, `laptime_roll_mean_3`,
    `laps_since_last_pit`, con `shift(1)` SIEMPRE antes de `rolling`).
  - `src/f1pitstop/features/build.py`: registry `FEATURE_SET_REGISTRY`
    con E10 (raw, = Fase 4/5) / E11 (+basic domain) / E12 (+temporal) /
    E13 (ambas), `build_engineered_frame()` (una sola pasada) y
    `prepare_X_for_feature_set()`.
  - `tests/test_features.py`: test adversarial OBLIGATORIO del spec
    (DataFrame toy de 5 vueltas, vuelta 3 no ve `lap_time` de vueltas 4/5,
    con centinelas 999.0 para hacerlo inequivoco) + tests de dos autos
    intercalados en la misma carrera (verifica que `groupby` no mezcla
    historiales) + tests de casos borde. 19 tests nuevos.
  - `scripts/phase6_feature_ablation.py`: ablation E10->E13, mismo modelo
    (HGB) y misma CV (V1) en las 4 corridas.
  - **Resultado del primer corrido (con las 3 sub-features temporales
    juntas):** E10=0.815, E11=0.858 (+0.043), E12=0.792 (**-0.023**, peor
    que el baseline), E13=0.823 (+0.009, mucho menos que E11 sola).
  - **Diagnostico:** un ablation por-feature aislo la causa de que E12
    empeorara: `laptime_roll_mean_3` (rolling mean sobre
    `LapTime_s_winsorized`, EL ejemplo trabajado del spec para esta fase)
    le cuesta SOLA ~0.057 ROC-AUC (0.815 -> 0.757), heredando la misma
    inestabilidad de `LapTime (s)` cruda de Fase 4 (el winsorizing a 150s
    no alcanza a arreglarlo). Las otras 2 (`laptime_delta_prev` +0.005,
    `laps_since_last_pit` +0.024) SI ayudan. Se excluyo
    `laptime_roll_mean_3` del feature set por defecto
    (`UNSTABLE_TEMPORAL_FEATURE_NAMES` en `temporal.py`) — se sigue
    calculando y testeando (es el caso de referencia del test
    adversarial), solo no entra en E12/E13.
  - **Re-corrido con las familias corregidas — resultado final:**
    E10=0.815, E11=0.858 (+0.043), E12=0.844 (+0.029), **E13=0.860
    (+0.045, el mejor)**. E13 ahora si supera a E11 sola, confirmando que
    las familias se combinan bien una vez removida la feature ruidosa.
  - **Subagente `leakage-auditor` invocado antes de cerrar la fase**
    (regla de CLAUDE.md). Hallazgo **BLOQUEANTE** real: los numeros del
    ablation por-feature citados en README/comentarios NO eran
    reproducibles desde ningun script/artefacto commiteado (se habian
    corrido ad-hoc por consola) — violacion de la regla no negociable 9
    de CLAUDE.md ("toda afirmacion de mejora requiere validacion
    reproducible, nunca una corrida suelta"). **Corregido en la misma
    sesion:** se creo `scripts/phase6_feature_isolation.py`
    (reproducible, loguea a MLflow) que reproduce EXACTAMENTE los mismos
    numeros (+0.0050, -0.0573, +0.0236), guardados en
    `artifacts/tables/feature_isolation_results.csv`.
  - Tambien se cerraron 3 hallazgos A_REVISAR del mismo auditor: (1) se
    verifico empiricamente que ningun `Driver` corresponde a 2 autos
    fisicos distintos en la misma carrera (0 `LapNumber` duplicados,
    grupos maximo 51 filas, sobre train+test completos) — documentado en
    `temporal.py`; (2) se agregaron tests de dos autos intercalados; (3)
    se corrigio el texto que llamaba a `recomputed_stint` "reemplazo" de
    `Stint` crudo (en realidad se agrega, no reemplaza — `Stint` crudo
    sigue en el feature set, la comparacion "con vs sin Stint crudo"
    queda pendiente para Fase 7).
  - Suite completa: 67 tests, todos pasan. `ruff check .` limpio.
- **Que se hizo (Fase 5 — skrub, cerrada):**
  - `src/f1pitstop/models/skrub_pipelines.py`: `make_e04_skrub_logreg()` y
    `make_e06_skrub_hgb()` via `skrub.tabular_pipeline(estimator)`.
    `SKRUB_COMPARISON_REGISTRY` con los 4 runs del spec (E03/E05 reusan
    los mismos modelos manuales de Fase 4 — `make_e01_logreg`/
    `make_e02_hgb` — bajo el nombre de esta comparacion). Helper
    `count_output_columns()` para medir complejidad de salida del
    preprocessing.
  - `scripts/phase5_skrub_comparison.py`: reproducible, misma CV V1 y
    mismo feature set de Fase 4 (sin `LapTime (s)`), loguea a MLflow con
    `stage=features` y tag extra `preprocessing=manual|skrub`.
  - Resultados (V1, dev set = 346,246 filas):

    | run | preprocessing | ROC-AUC (mean±std) | PR-AUC | cols salida | fit (s) |
    |---|---|---|---|---|---|
    | E03_manual_preprocessing_logreg | manual | 0.732±0.041 | 0.351 | 10 | 1.44 |
    | E04_skrub_tabular_pipeline_logreg | skrub | 0.736±0.040 | 0.353 | 10 | 1.87 |
    | E05_manual_preprocessing_hgb | manual | 0.815±0.023 | 0.424 | 6 | 7.16 |
    | E06_skrub_tabular_pipeline_hgb | skrub | 0.815±0.023 | 0.424 | 6 | 4.24 |

  - **Decision:** calidad practicamente identica (HGB: match exacto;
    logreg: diferencia de 0.004, dentro de ruido) — este dataset no
    estresa las fortalezas de skrub (ya es limpio, una sola columna
    categorica de baja cardinalidad). La ganancia ergonomica es real pero
    modesta: 1 linea (`skrub.tabular_pipeline(estimator)`) reemplaza un
    `ColumnTransformer` manual, y para HGB detecta automaticamente que
    debe usar categoricas nativas en vez de que lo codifiquemos a mano
    (`categorical_features=[...]`). Se adopta skrub donde reduce codigo
    sin costo de calidad (candidato fuerte para Fase 6/7 cuando entre
    `Driver`, 887 valores — el caso de uso real de `StringEncoder`), sin
    forzarlo en todo el pipeline (regla del spec: "si skrub no aporta
    mejora... usarlo solo donde sea util").
  - `tests/test_skrub_pipelines.py` (5 tests) nuevos. Suite completa: 48
    tests, todos pasan. `ruff check .` limpio.
  - No hay "Criterio de salida" explicito para Fase 5 en el spec (a
    diferencia de Fases 3/4/6+) — el criterio de facto es la "Decision"
    documentada arriba, ya cumplida.
- **Que se hizo (Fase 4 — Baselines, cerrada):**
  - `src/f1pitstop/models/baselines.py`: E00 (`DummyClassifier(strategy=
    "prior")`), E01 (`LogisticRegression` + `ColumnTransformer` simple),
    E02 (`HistGradientBoostingClassifier` con soporte nativo de
    categoricas). `BASELINE_REGISTRY` centraliza los 3 runs.
  - `src/f1pitstop/evaluation/cv.py`: `run_group_cv()` reusable, corre
    SIEMPRE V1 (nunca V0), devuelve ROC-AUC/PR-AUC por fold, fit_seconds,
    predict_ms_per_1k_rows, n_features.
  - `src/f1pitstop/tracking/mlflow_utils.py`: `setup_mlflow()` (sqlite en
    `mlruns/`, experimento `f1_pitstop`) y `log_run()` que fuerza los tags
    obligatorios de `.claude/rules/experiment-tracking.md` (levanta
    `ValueError` si falta alguno).
  - `scripts/train_baselines.py`: entrypoint reproducible, corre los 3
    baselines sobre `dev` (excluye holdout 2025), loguea a MLflow con
    `log_models=False`, guarda `artifacts/tables/baseline_results.csv`.
  - **Hallazgo importante (revision de la decision de Fase 3):** el
    primer corrido de baselines dio E02 ROC-AUC=0.740, mucho mas bajo que
    el 0.815 de la cuantificacion de H1 en Fase 3 con casi el mismo
    feature set. Ablation aislo la causa: `LapTime (s)` (incluida en Fase
    3 como "leakage-safe") le cuesta ~0.075 ROC-AUC al HGB baseline.
    Motivo: outliers extremos (hasta 2507s vs media ~91s, probablemente
    vueltas con safety car/bandera roja) que son artefactos especificos
    de cada carrera y no generalizan bajo V1 (group-aware). **No es un
    problema de leakage temporal** (no usa `t+1` ni el target) — es un
    problema de estabilidad/generalizacion, precisamente el tipo de cosa
    que V1 esta disenado para exponer. Se retiro `LapTime (s)` del
    feature set por defecto (`UNSTABLE_FEATURES` en `baselines.py`),
    documentado en `README.md` y como addendum en
    `artifacts/reports/leakage_checklist_fase3.md`. Se re-corrio el
    script con el feature set corregido: resultados finales limpios
    (E00=0.500, E01=0.732, E02=0.815 — el 0.815 coincide exactamente con
    Fase 3, confirmando reproducibilidad).
  - `tests/test_evaluation.py` (5 tests) y `tests/test_models.py` (7
    tests) nuevos, con datos toy (no tocan `data/raw/`). Suite completa:
    43 tests, todos pasan. `ruff check .` limpio.
  - Resultados finales (V1, dev set = 346,246 filas):

    | run | ROC-AUC (mean±std) | PR-AUC | fit (s) | predict (ms/1k) |
    |---|---|---|---|---|
    | E00_dummy | 0.500±0.000 | 0.176 | 0.02 | 0.03 |
    | E01_logreg_basic | 0.732±0.041 | 0.351 | 0.92 | 0.39 |
    | E02_hgb_basic | 0.815±0.023 | 0.424 | 2.70 | 3.59 |

- **Que se hizo (Fase 3 — Validacion y leakage, cerrada):**
  - `src/f1pitstop/data/split.py`: V0 (`StratifiedKFold` aleatorio), V1
    (`StratifiedGroupKFold` por `(Race, Year)`), V2 (`v2_temporal_split`
    por anio), `assert_no_group_overlap`, `freeze_final_holdout` /
    `load_frozen_holdout_ids`. `tests/test_split.py` (12 tests, incluye el
    test obligatorio del spec de no-overlap de grupos).
  - Holdout final congelado: `Year == 2025` (92,894 filas, 26 grupos
    `(Race, Year)`) vs dev 346,246 filas / 78 grupos. Persistido en
    `artifacts/tables/final_holdout_ids.csv`. NUNCA se evaluo ningun
    modelo sobre el (verificado por el subagente `leakage-auditor`).
  - Cuantificacion de H1 (`scripts/phase3_quantify_h1.py`, sobre dev
    unicamente): V0 ROC-AUC 0.844±0.001, V1 ROC-AUC 0.815±0.023 → **gap de
    ~0.03 ROC-AUC atribuible a validacion aleatoria ingenua**. V2 demo
    (train 2022-2023 → val 2024) = 0.839 (un solo fold). Resultado en
    `artifacts/tables/cv_strategy_comparison.csv`.
  - **Decision de CV oficial: V1** (justificacion completa en `README.md`,
    seccion "Validation strategy").
  - **Hallazgo nuevo de Fase 3** (no reportado en Fase 2): el anio 2023
    tiene tasa de pit (`PitNextLap` y `PitStop`) anormalmente baja
    (~1%) en TODAS las carreras de ese anio (vs ~19-30% en 2022/2024/2025)
    — drift real del generador sintetico, no solo en `Pre-Season
    Testing`. Documentado en `notebooks/03_leakage_and_validation.ipynb`
    seccion 2 y en `README.md`.
  - Checklist de 5 preguntas aplicado a `PitStop`, `Position`,
    `LapTime (s)`, `RaceProgress`, `Cumulative_Degradation`,
    `LapTime_Delta`, `Position_Change` — detalle completo en
    `artifacts/reports/leakage_checklist_fase3.md`. **Feature set
    "leakage-safe" para Fase 4:** `LapNumber`, `TyreLife`, `Stint`,
    `Position`, `PitStop`, `Compound`, `LapTime (s)`.
  - `notebooks/03_leakage_and_validation.ipynb`: notebook ejecutado (con
    outputs reales, no solo celdas vacias) que cumple el criterio de
    salida de la fase (demuestra la diferencia V0 vs V1/V2 y documenta la
    decision). Se ejecuto con un script auxiliar propio en vez de
    `jupyter nbconvert --execute` porque el kernel de Jupyter no arranca
    en este entorno Windows (bug conocido de zmq/asyncio
    ProactorEventLoop) — no se instalo jupyter/nbclient como dependencia
    permanente del proyecto (regla 3 de CLAUDE.md), solo via
    `uv run --with` puntual.
  - Subagente `leakage-auditor` invocado antes de cerrar la fase: **sin
    hallazgos bloqueantes** (sin overlap de grupos, sin uso indebido del
    holdout). Detecto 2 huecos de documentacion (rigor de `RaceProgress`,
    falta de entrada explicita para `Position`/`LapTime (s)`) que se
    cerraron en la misma sesion — esto **cambio la decision original**:
    `RaceProgress` se penso primero como segura ("incluir con nota") y se
    re-clasifico a "excluir del set leakage-safe" tras verificar
    empiricamente que su denominador implicito (vueltas totales) no es
    monotono en 24.2% de los grupos (misma inconsistencia que `Stint`).
  - `uv run pytest` (31 tests) y `uv run ruff check .` pasan limpios al
    cierre de la fase.
- **Que se hizo (sesiones anteriores):**
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

1. **Commitear el trabajo de Fase 3 a Fase 7** (ver "PENDIENTE ANTES DE
   SEGUIR" arriba) — pedir confirmacion explicita al usuario primero.
2. Ejecutar Fase 8 del spec (AutoGluon challenger). Ver spec seccion 13:
   - Archivo: `autogluon_runner.py`. Objetivo: benchmark AutoML honesto,
     el holdout final NUNCA se pasa como `tuning_data`.
   - Primera corrida: `presets="medium_quality"`, `time_limit` acotado.
     Segunda corrida (`good_quality` o equivalente) solo si la primera
     justifica el costo. No exigir `best_quality`.
   - Entradas: A0 = raw cleaned data, A1 = leakage-safe engineered data
     (feature set E13 de Fase 6/7).
   - **Ambiguedad a resolver ANTES de escribir codigo de Fase 8** (no
     asumir, seguir la regla 2 del protocolo de sesion de CLAUDE.md):
     `leakage-and-validation.md` §7 dice "AutoGluon se evalua con el
     MISMO holdout externo que el modelo manual", pero la regla no
     negociable 6 de CLAUDE.md dice que el holdout final "solo se usa
     para la evaluacion confirmatoria de la Fase 13". Aclarar si el
     "holdout externo" de §7 se refiere al holdout congelado
     (`Year==2025`) — y en ese caso la comparacion real ocurre recien en
     Fase 13, no en Fase 8 — o a una particion de validacion V1 dentro
     de dev. El criterio de salida de Fase 8 en el spec ("comparacion
     manual vs AutoML con la misma particion externa") es compatible con
     ambas lecturas.
   - Guardar leaderboard, score_val, score en holdout (si aplica segun
     lo anterior), pred_time, fit_time, stack_level, espacio en disco
     del predictor.
   - MLflow: registrar manualmente (sin autolog magico) preset,
     time_limit, eval_metric, filas/columnas, leaderboard.csv, holdout
     metrics, runtime, path/version del predictor.
   - Decidir primero si se instalan los extras opcionales de AutoGluon
     (torch/lightgbm/catboost/xgboost) — ver "Bloqueadores".
   - Criterio de salida: comparacion manual vs AutoML con la misma
     particion externa y la misma metrica.
   - Usar la skill `/new-experiment` para A00/A01.

Nota: `notebooks/01_data_audit.ipynb` y `02_eda.ipynb` de la arquitectura
del spec no se crearon todavia — los criterios de salida de Fase 1 y 2 se
cubrieron via reportes `.md` + tests + figuras, que son reproducibles.
`03_leakage_and_validation.ipynb` (Fase 3) si se creo, porque el spec lo
exige explicitamente como criterio de salida de esa fase.

## Bloqueadores / dudas abiertas

- Decidir en Fase 8 si se instalan los extras opcionales de AutoGluon
  (torch/lightgbm/catboost/xgboost).
- `jupyter`/`nbclient`/`ipykernel` no estan en el entorno permanente del
  proyecto (no se justifico como dependencia via regla 3 de CLAUDE.md); si
  se necesita ejecutar mas notebooks con `nbconvert --execute`, recordar
  que el kernel de Jupyter no arranca en este entorno Windows (zmq/asyncio
  ProactorEventLoop) — usar el patron de ejecucion celda-a-celda con
  `exec()`/`ast` en un script auxiliar (ver como se hizo para el notebook
  03, script ya borrado tras usarlo) en vez de pelear con el kernel.
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
| 3 — Validacion y leakage | **cerrada** | 2026-08-27 | V1 (StratifiedGroupKFold por (Race,Year)) elegida como CV oficial; holdout Year==2025 congelado; ver `notebooks/03_leakage_and_validation.ipynb` y `leakage_checklist_fase3.md`.|
| 4 — Baselines | **cerrada** | 2026-08-27 | E00=0.500, E01=0.732, E02=0.815 ROC-AUC (V1); MLflow activado; `LapTime (s)` retirada del feature set por defecto (hallazgo de estabilidad, no leakage).|
| 5 — skrub | **cerrada** | 2026-08-28 | calidad practicamente identica (HGB match exacto, logreg +0.004); skrub adoptado donde reduce codigo, no forzado.|
| 6 — Feature engineering | **cerrada** | 2026-08-28 | E13 gana con 0.860 ROC-AUC (+0.045 vs E10); `laptime_roll_mean_3` invalidada por ablation por-feature reproducible.|
| 7 — Modelos manuales | **cerrada** | 2026-08-31 | E20_hist_gradient_boosting (tuneado) gana con 0.8611 ROC-AUC; ExtraTrees tuneado 0.8530; Stint crudo confirmado como necesario (-0.030 sin el).|
| 8 — AutoGluon challenger | pendiente | | decidir extras opcionales (ver bloqueadores); resolver ambiguedad sobre uso del holdout (ver "Proxima accion concreta") |
| 9 — skore | pendiente | | |
| 10 — Error analysis | pendiente | | |
| 11 — MLflow final | pendiente | | |
| 12 — skops | pendiente | | reusar patron de allowlist del smoke test |
| 13 — Holdout final y Kaggle | pendiente | | |
