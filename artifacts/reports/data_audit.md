# Data Audit — F1 Pit Stop Prediction (Fase 1)

Fecha: 2026-08-26
Fuente: Kaggle Playground Series S6E5, competencia `playground-series-s6e5`
(descargado via `kaggle competitions download`).

## 1. Archivos y fingerprint

| archivo | filas | columnas | memoria (MB) | sha256 |
|---|---|---|---|---|
| train.csv | 439,140 | 16 | 127.25 | `f004e79d...cce52ccc4128` |
| test.csv | 188,165 | 15 | 53.10 | `95b449a8...e0140c2ea7` |
| sample_submission.csv | 188,165 | 2 | 2.87 | `1f0cc0c4...b83382d009018` |

Reproducible via `f1pitstop.data.ingest.load_raw()`.

## 2. Discrepancia con el spec del proyecto

El spec (`01_F1_Pit_Stop_ML_Project_Spec.txt`, linea 70) describe el
dataset como de **33 columnas**. El CSV real descargado tiene **16
columnas** en train (15 en test, sin el target). No se "corrige" el spec:
se documenta la discrepancia y se trabaja con la estructura real. Hipotesis
mas probable: el spec describe una version generica/desactualizada de la
competencia.

**Contexto adicional confirmado en la pagina de Kaggle** (no estaba en el
spec original): el dataset esta inspirado en un dataset real de estrategia
de F1, y se **removio intencionalmente la columna `Normalized_TyreLife`**
porque hacia la prediccion trivial. El link al dataset original esta roto
en la pagina de Kaggle a la fecha de este audit.

## 3. Columnas (train)

| columna | dtype | n_unique | rol candidato |
|---|---|---|---|
| `id` | int64 | 439,140 (100%) | identificador |
| `Driver` | object | 887 | feature candidata (categorica, alta cardinalidad) |
| `Compound` | object | 5 | feature candidata (categorica, baja cardinalidad) |
| `Race` | object | 26 | feature candidata / posible clave de agrupacion para split |
| `Year` | int64 | 4 | feature candidata / posible eje temporal |
| `PitStop` | int64 (binario 0/1) | 2 | feature candidata — **revisar semantica** (ver seccion 5) |
| `LapNumber` | int64 | 78 | feature candidata |
| `Stint` | int64 | 8 | feature candidata — **ver hallazgo de no-monotonicidad** (seccion 5) |
| `TyreLife` | float64 | 78 | feature candidata |
| `Position` | int64 | 20 | feature candidata |
| `LapTime (s)` | float64 | 37,719 | feature candidata |
| `LapTime_Delta` | float64 | 57,532 | **sospechosa por nombre** — revisar en Fase 3 |
| `Cumulative_Degradation` | float64 | 142,701 | **sospechosa por nombre** — revisar en Fase 3 |
| `RaceProgress` | float64 | 1,898 | **sospechosa por nombre** — revisar en Fase 3 |
| `Position_Change` | float64 | 37 | **sospechosa por nombre** — revisar en Fase 3 |
| `PitNextLap` | float64 (binario 0/1) | 2 | **TARGET** |

Tabla completa y reproducible en `artifacts/tables/schema_summary.csv`
(via `f1pitstop.data.schema.cardinality_summary()`).

## 4. Resultado de `validate_schema()`

`has_errors = False`. Sin errores estructurales:

- target presente solo en train, ausente en test — OK
- sin `id` duplicados en train ni en test — OK
- sin overlap de `id` entre train y test — OK
- mismas columnas (excepto target) y mismos dtypes en train/test — OK
- sin missing values en ninguna columna — OK
- sin valores infinitos — OK
- sin filas duplicadas exactas — OK
- sin columnas constantes — OK
- target dentro de `{0, 1}` — OK

4 issues de severidad `info` (heuristica por nombre, no error):
`LapTime_Delta`, `Cumulative_Degradation`, `RaceProgress`,
`Position_Change` sugieren agregados/estado acumulado y se marcan para
revision con el checklist de 5 preguntas de
`.claude/rules/leakage-and-validation.md` antes de la Fase 6.

## 5. Hallazgos de auditoria manual (mas alla de `validate_schema`)

Estos hallazgos surgen de exploracion dirigida sobre `train.csv` y son
**criticos para el diseno de validacion de la Fase 3** — no se resuelven
aqui, solo se documentan.

### 5.1 Balance del target

`PitNextLap`: 351,759 negativos (80.1%) vs 87,381 positivos (19.9%).
Desbalanceado pero no extremo; razonable usar ROC-AUC (metrica de Kaggle)
directamente, sin necesitar resampling agresivo a priori.

### 5.2 `Race` como posible clave de agrupacion (V1 del spec)

26 carreras distintas (`Race`), con conteos entre 3,185 (French GP) y
24,462 (Dutch GP) filas. Es la columna candidata mas clara para un split
group-aware (evitar que la misma carrera aparezca en train y validation).
Sin embargo, **`Race` sola no identifica un evento unico**: el mismo
nombre de carrera se repite en distintos `Year` (2022–2025), por lo que la
clave de agrupacion real deberia ser `(Race, Year)`, no `Race` sola. A
confirmar/decidir en Fase 3.

### 5.3 `Driver` no se comporta como el grid real de F1

Un grid de F1 real tiene ~20 pilotos por temporada. En este dataset, una
sola carrera (`Canadian Grand Prix`, 2022) tiene **414 valores distintos**
de `Driver`, y en total hay 887 valores unicos de `Driver` en todo el
dataset. Los valores mezclan formatos: codigos de 3 letras estilo F1 real
(`ALB`, `ALO`, `BOT`, `ZON`) y codigos sinteticos (`D001`, `D002`, ...,
`D109`). **Conclusion: `Driver` NO es un identificador de piloto real
consistente y no debe asumirse como tal.** Es una feature categorica de
alta cardinalidad, pero no sirve para razonar sobre "el mismo piloto en
distintas carreras" sin verificacion adicional.

### 5.4 Hallazgo critico: `Stint` no es monotono dentro de `(Driver, Race, Year)`

Se tomo una muestra de 2,000 grupos `(Driver, Race, Year)` y se ordeno cada
grupo por `LapNumber`. **1,608 de 2,000 grupos (80.4%) tienen `Stint` no
monotono** (baja despues de haber subido), lo cual es fisicamente
imposible en una carrera real (el numero de stint solo puede aumentar o
mantenerse). Ademas, dentro de un mismo `(Driver, Race, Year)` los valores
de `LapNumber` no son consecutivos (ej.: `2, 4, 5, 12, 15, 16, 18, 21, 26,
34, 37, 39, 50` para un caso revisado manualmente) — no hay una fila por
cada vuelta.

**Implicacion directa para la Fase 3 y la Fase 6:** agrupar por
`(Driver, Race, Year)` y asumir que las filas ordenadas por `LapNumber`
representan la trayectoria continua de un mismo auto **no es valido tal
cual**. Cualquier feature de tipo rolling/lag (regla de oro de
`.claude/rules/leakage-and-validation.md`, seccion 5) que asuma esa
continuidad debe primero verificar/resolver este problema — posiblemente
el dataset esta muestreando snapshots de vueltas no consecutivas en vez de
secuencias completas, o `(Driver, Race, Year)` no es la clave de
agrupacion fisica correcta. **Este es el principal punto abierto para
Fase 3, mas importante que las 4 columnas marcadas por nombre.**

No hay duplicados de `(Driver, Race, Year, LapNumber)` (maximo 1 fila por
combinacion), asi que al menos cada snapshot de vuelta es unico dentro de
su grupo.

### 5.5 `PitStop` vs `PitNextLap` (posible relacion, no concluyente)

Tabla cruzada:

| PitStop (vuelta actual) | PitNextLap=0 | PitNextLap=1 | %PitNextLap=1 |
|---|---|---|---|
| 0 | 306,798 | 72,567 | 19.1% |
| 1 | 44,961 | 14,814 | 24.8% |

Diferencia moderada (19.1% vs 24.8%), no un leak evidente de
determinismo, pero la semantica exacta de `PitStop` (¿fue a boxes ESTA
vuelta?) debe confirmarse en Fase 2 antes de usarla como feature — si
`PitStop` se computa con informacion posterior a `t` para la fila `t`,
seria fuga.

## 6. Listas requeridas (criterio de salida de la Fase 1)

```
TARGET = "PitNextLap"

ID_COLUMNS = ["id"]

FEATURE_CANDIDATES = [
    "Driver", "Compound", "Race", "Year", "PitStop", "LapNumber",
    "Stint", "TyreLife", "Position", "LapTime (s)",
]

SUSPECTED_LEAKAGE = [
    "LapTime_Delta",           # nombre sugiere delta entre vueltas; confirmar direccion temporal
    "Cumulative_Degradation",  # nombre sugiere acumulado; confirmar que no usa vueltas futuras
    "RaceProgress",            # nombre sugiere fraccion de carrera completada; confirmar como se computa
    "Position_Change",         # nombre sugiere delta de posicion; confirmar direccion temporal
    "PitStop",                 # semantica temporal a confirmar (ver 5.5); candidata a EXCLUDED si es ambigua
]

EXCLUDED_COLUMNS = [
    "id",  # identificador, no feature
]
```

`SUSPECTED_LEAKAGE` no implica exclusion automatica: cada columna debe
pasar el checklist de 5 preguntas de
`.claude/rules/leakage-and-validation.md` en la Fase 3 antes de aceptarse
o descartarse definitivamente. `PitStop` se incluye ahi (y no directamente
en `EXCLUDED_COLUMNS`) porque la evidencia de la seccion 5.5 no es
concluyente por si sola.

## 7. Abierto para Fase 2/3 (no resuelto en este audit)

1. Confirmar si `(Race, Year)` es la clave de agrupacion correcta para el
   split group-aware (V1), en vez de `Race` sola.
2. Investigar el hallazgo de `Stint` no-monotono (seccion 5.4) — determina
   si se puede construir cualquier feature rolling/lag tal como esta
   planteado en el spec, o si hace falta repensar la unidad de secuencia.
3. Confirmar la semantica exacta de `PitStop`, `LapTime_Delta`,
   `Cumulative_Degradation`, `RaceProgress` y `Position_Change` (¿en que
   momento se calculan respecto a la vuelta `t`?).
4. El link al dataset F1 original (relevante para H4) esta roto en Kaggle;
   evaluar via otra fuente (ej. FastF1) si se retoma esa hipotesis.
