# Checklist de leakage — Fase 3

Aplicacion del checklist de 5 preguntas de
`.claude/rules/leakage-and-validation.md` seccion 4 a `PitStop` y a las
columnas marcadas `SUSPECTED_LEAKAGE` en `src/f1pitstop/data/schema.py`
(`LapTime_Delta`, `Cumulative_Degradation`, `RaceProgress`,
`Position_Change`). Contexto previo relevante: `eda_report.md` seccion 8
(hallazgo de que estas columnas derivadas se calcularon sobre una
secuencia oculta y el CSV publico es un submuestreo de esa secuencia).

Las 5 preguntas del checklist:

1. ¿Se conoce en el instante `t`?
2. ¿Usa informacion de `t+1` o del futuro respecto a `t`?
3. ¿Usa el target de forma directa o indirecta?
4. ¿Usa agregados calculados con datos de validation/test?
5. ¿Usa estadisticas globales que deberian computarse dentro de cada fold?

Regla: si la respuesta a 2, 3 o 4 es "si" (o no se puede responder con
certeza), la feature se descarta o se corrige — no se documenta como
"limitacion" y se deja pasar.

## `PitStop`

| pregunta | respuesta |
|---|---|
| 1. ¿conocida en t? | Si — describe si la vuelta ACTUAL (t) fue de pit stop |
| 2. ¿usa t+1? | No hay evidencia; es conceptualmente un indicador del propio registro de la fila t |
| 3. ¿usa el target? | No, es una columna independiente del target `PitNextLap` |
| 4. ¿usa agregados de val/test? | No aplica, es un valor por fila |
| 5. ¿usa estadisticas globales? | No aplica |

**Evidencia empirica:** AUC univariada = 0.521 (eda_report.md, pregunta 4)
— asociacion leve con el target, no una fuga evidente (una fuga directa
mostraria AUC cercano a 1.0). Diferencia moderada de tasa de pit-siguiente
entre `PitStop=0` (19.1%) vs `PitStop=1` (24.8%), coherente con una senal
util pero no determinista.

**Decision: incluir.** Es un feature legitimo del instante `t`.

## `RaceProgress`

`RaceProgress` = `LapNumber` / vueltas totales de la carrera (implicito
por el nombre y por su correlacion r=0.965 con `LapNumber`, ver
eda_report.md seccion 5).

| pregunta | respuesta |
|---|---|
| 1. ¿conocida en t? | Inicialmente se penso que si (el numero total de vueltas de una carrera de F1 es conocido de antemano) — **revisado abajo con evidencia empirica que lo pone en duda** |
| 2. ¿usa t+1? | No debiera, si el denominador fuera realmente fijo por carrera — **ver evidencia empirica** |
| 3. ¿usa el target? | No |
| 4. ¿usa agregados de val/test? | El denominador (vueltas totales) deberia ser un atributo fijo de la carrera, conocido antes de que empiece — misma duda que en 1/2 |
| 5. ¿usa estadisticas globales fuera de fold? | No — es especifico por carrera, no una estadistica global del dataset completo |

**Evidencia empirica (revision post-auditoria del subagente `leakage-auditor`):**
el subagente reconstruyo el denominador implicito (`LapNumber / RaceProgress`)
por grupo `(Driver, Race, Year)` y encontro varianza enorme dentro de un
mismo grupo (ej. un grupo con denominador implicito entre 56 y 532 vueltas
— fisicamente imposible para una sola carrera). Verificacion adicional en
esta fase: `RaceProgress` es monotono creciente al ordenar por `LapNumber`
dentro de cada grupo `(Driver, Race, Year)` solo en el **75.8%** de los
grupos (equivalente al test de monotonicidad de `Stint` en `eda_report.md`,
que dio 18.4%). Un cociente con denominador verdaderamente fijo por carrera
NUNCA deberia des-ordenarse al ordenar por su propio numerador
(`LapNumber`) — este 24.2% de grupos no monotonos contradice la hipotesis
de "denominador fijo conocido de antemano" y es el mismo tipo de
inconsistencia de secuencia oculta/subsampleada que Fase 2 ya encontro en
`Stint` y `Position_Change`.

**Decision (revisada): excluir del set "leakage-safe" por defecto**, igual
que `Cumulative_Degradation`/`LapTime_Delta`/`Position_Change` (regla 4:
respuesta incierta a la pregunta 1/2 → se descarta). AUC univariada
individual es la mas alta del grupo sospechoso (0.664), lo cual hace mas
importante, no menos, tratarla con el mismo rigor. Queda disponible para
el mismo experimento ablation de Fase 6 que las demas columnas derivadas
de la secuencia oculta.

## `Cumulative_Degradation`, `LapTime_Delta`, `Position_Change`

Se tratan juntas porque comparten el mismo problema estructural.

| pregunta | respuesta |
|---|---|
| 1. ¿conocida en t? | **No se puede responder con certeza.** Fase 2 ya demostro que `Position_Change` NO coincide con la diferencia de `Position` entre filas visibles consecutivas del mismo grupo — se calculo sobre una secuencia oculta completa, no sobre las filas que efectivamente vemos. |
| 2. ¿usa t+1? | **No se puede descartar.** El mecanismo exacto de generacion es opaco (no hay codigo fuente del generador sintetico disponible); no hay forma de certificar que el "delta"/"cambio"/"acumulado" reportado en la fila visible t no incorpora informacion de una vuelta oculta posterior a t. |
| 3. ¿usa el target? | No de forma directa (son columnas separadas de `PitNextLap`), pero no se puede descartar una dependencia indirecta dado el punto anterior |
| 4. ¿usa agregados de val/test? | No aplica directamente (son valores por fila, no agregados del dataset), pero comparten el mismo problema de opacidad del punto 1 |
| 5. ¿usa estadisticas globales? | No aplica |

**Evidencia empirica adicional (esta fase):** se verifico si
`Cumulative_Degradation` y `TyreLife` "resetean" (bajan respecto a la fila
anterior visible) en la MISMA fila donde `PitStop=1` — si el reseteo
coincidiera siempre con `PitStop=1`, seria evidencia a favor de que la
columna respeta el orden visible. Resultado sobre 506 pit stops
muestreados: el reseteo coincide con la fila de `PitStop=1` solo el 46.2%
de las veces para `Cumulative_Degradation` y el 38.5% para `TyreLife`. Esto
es consistente con el hallazgo de Fase 2 (subsampleo de una secuencia
oculta) — **no confirma leakage de futuro, pero tampoco permite
descartarlo**, y confirma que estas columnas no se comportan como una
secuencia confiable fila-a-fila en los datos que efectivamente tenemos.

**Decision: excluir del set "leakage-safe" por defecto** (regla 4:
respuesta incierta a la pregunta 2 → se descarta). No se eliminan del
dataset ni se prohibe su uso permanentemente: quedan disponibles para un
experimento ablation dedicado en Fase 6 (comparar modelo con vs sin estas
columnas, ver hipotesis 1 de `eda_report.md`), donde se puede volver a
evaluar con mas tiempo si el generador sintetico se documenta mejor o si
aparece evidencia adicional.

## `Position`

Posicion en pista de la vuelta actual (t).

| pregunta | respuesta |
|---|---|
| 1. ¿conocida en t? | Si — es un valor de estado de carrera en tiempo real, no un derivado de agregados; el hallazgo de Fase 2 fue que `Position_Change` (la diferencia) no coincide con la diferencia real entre filas visibles, no que `Position` en si sea invalida |
| 2. ¿usa t+1? | No hay evidencia ni mecanismo plausible — es la posicion instantanea de la vuelta t |
| 3. ¿usa el target? | No |
| 4. ¿usa agregados de val/test? | No aplica, valor por fila |
| 5. ¿usa estadisticas globales? | No aplica |

**Evidencia empirica:** AUC univariada = 0.516 (confirmado en esta fase,
coincide con `eda_report.md`) — asociacion muy leve.

**Decision: incluir.**

## `LapTime (s)`

Tiempo de vuelta de la vuelta actual (t). No habia recibido tratamiento
explicito en la primera version de este documento (senalado por el
subagente `leakage-auditor`).

| pregunta | respuesta |
|---|---|
| 1. ¿conocida en t? | Si, bajo la interpretacion adoptada en este proyecto para todas las columnas de estado (`TyreLife`, `Position`, `Stint`): la fila `t` representa el estado **al completar** la vuelta `t`, y `PitNextLap` predice si la vuelta `t+1` sera de pit. `LapTime (s)` de la vuelta `t` esta disponible en el instante de prediccion (justo despues de cruzar meta en `t`, antes de que empiece `t+1`) |
| 2. ¿usa t+1? | No — es el tiempo de la vuelta ya completada, no de la siguiente |
| 3. ¿usa el target? | No |
| 4. ¿usa agregados de val/test? | No aplica, valor por fila |
| 5. ¿usa estadisticas globales? | No aplica |

**Evidencia empirica:** AUC univariada = 0.540 (confirmado en esta fase,
coincide con `eda_report.md`) — asociacion leve, coherente con que vueltas
mas lentas (degradacion) preceden a un pit.

**Decision: incluir.** Nota para Fase 4: esta interpretacion ("fila t =
estado al completar la vuelta t") es la que sostiene tambien la inclusion
de `TyreLife`, `Position` y `Stint` como conocidas en el instante de
prediccion; si en Fase 6 aparece evidencia de que el corte real es distinto
(p.ej. que `LapTime (s)` en realidad ya refleja parcialmente la vuelta
`t+1`), esta decision debe revisarse para las cuatro columnas a la vez, no
solo para `LapTime (s)`.

## Resumen

| columna | decision |
|---|---|
| `PitStop` | incluir |
| `Position` | incluir |
| `LapTime (s)` | paso el checklist de leakage, pero **se excluyo del set por defecto en Fase 4** por inestabilidad/generalizacion (ver addendum abajo) — no es una decision de leakage |
| `RaceProgress` | **excluir** del set leakage-safe por defecto (revisado tras auditoria — denominador implicito no es tan fijo como se penso, 24.2% de grupos no monotonos) |
| `Cumulative_Degradation` | excluir del set leakage-safe por defecto |
| `LapTime_Delta` | excluir del set leakage-safe por defecto |
| `Position_Change` | excluir del set leakage-safe por defecto |

Feature set "leakage-safe" resultante para Fase 4 (baselines, ya
actualizado con el addendum de abajo): `LapNumber`, `TyreLife`, `Stint`,
`Position`, `PitStop`, `Compound`. `Driver` queda pendiente de decision de
encoding (Fase 6/7, ver hipotesis 7 de `eda_report.md`), no se incluye en
el baseline por defecto dada su alta cardinalidad y comportamiento
inconsistente con un grid real de F1.

## Addendum (Fase 4): `LapTime (s)` se retira del set por defecto (no por leakage)

Durante Fase 4 (baselines), un ablation con V1 mostro que `LapTime (s)`
(que si paso este checklist — ver arriba, sin evidencia de leakage) le
cuesta ~0.075 ROC-AUC al HGB baseline (0.815 → 0.740) cuando se incluye.
La columna tiene outliers extremos (hasta 2507s vs media ~91s, ver
`artifacts/tables/baseline_results.csv`), probablemente vueltas con
safety car/bandera roja — artefactos especificos de cada carrera que no
generalizan a carreras nuevas bajo V1. Se retira del feature set por
defecto por **estabilidad/generalizacion, no por leakage temporal** (no
usa `t+1` ni el target). Detalle en `src/f1pitstop/models/baselines.py`
(`UNSTABLE_FEATURES`) y `README.md` seccion "Baselines (Fase 4)".

## Revision post-auditoria (subagente `leakage-auditor`, misma sesion de Fase 3)

El subagente de solo lectura revisó `split.py`, `test_split.py`,
`scripts/phase3_quantify_h1.py`, este documento, el notebook y el README.
Sin hallazgos bloqueantes (sin overlap de grupos, sin uso indebido del
holdout). Los dos puntos "a revisar" que senalo (`RaceProgress` sin
suficiente rigor, `Position`/`LapTime (s)` sin entrada explicita) se
cerraron en esta misma revision con las secciones de arriba.
