# Model Selection Framework — Incumbent Challenge Evaluation

**Revisado 2026-09-10** tras una auditoria que encontro una inferencia
estadistica invalida, una evaluacion contaminada y una regla de parada
construida despues de ver los resultados. Las tres cosas se documentan aqui
en vez de reescribirse.

---

## 1. La pregunta

No es *que algoritmo obtiene mas ROC-AUC*. Es:

> **¿Existe evidencia suficiente para reemplazar al modelo en uso?**

El proyecto llega a Fase 14 con un incumbente: `E20_hist_gradient_boosting`,
tuneado en Fase 7, CV ROC-AUC 0.8611 +/- 0.0251 bajo V1 group-aware. La
pregunta de esta fase es si algun candidato justifica desplazarlo, y con que
evidencia.

Ese encuadre cambia como se lee todo lo que sigue. E20 estaba tuneado; los
candidatos E22-E24 se corren con parametros por defecto. Eso **no** es un
benchmark desequilibrado por descuido: es un *challenger screening* barato,
que es la forma realista de decidir si merece la pena invertir mas computo en
una linea nueva.

```
Incumbente E20
      |
      v
Screening barato de challengers (defaults, mismo feature set, misma CV)
      |
      +-- evidencia clara de mejora --> tuning acotado --> evaluacion completa
      |
      +-- sin evidencia material -----> STOP
```

---

## 2. Como se compara: deltas pareados, no diferencias de medias

La version anterior de este documento afirmaba que la ganancia del ensemble
era "~60x mas chica que el ruido de CV", comparando un delta de +0.0004 contra
la desviacion tipica de 0.0246 entre folds.

**Esa comparacion es invalida.** La dispersion de los scores de un modelo
entre folds no es la incertidumbre de la *diferencia* entre dos modelos.

Como todos los runs comparten exactamente los mismos folds (mismo `groups`,
mismo `seed`), la comparacion correcta es el delta pareado por fold:

```
d_i = ROC-AUC(challenger, fold_i) - ROC-AUC(E20, fold_i),   i = 1..5
```

**Precision importante:** la *media* de los deltas pareados es
aritmeticamente identica a la diferencia de medias, porque
`mean(a_i - b_i) = mean(a_i) - mean(b_i)`. El analisis pareado no cambia ese
numero y no se presenta como si lo hiciera. Lo que aporta es lo que la
diferencia de medias no puede dar: **la dispersion de la diferencia** — en
cuantos folds el challenger va por delante, cual es el peor fold, y si la
direccion es consistente o el promedio esconde signos opuestos. Ese es
justamente el dato que faltaba cuando se contrastaba un delta contra la std
entre folds.

Con n=5 no se construye un p-value; se reportan estadisticos descriptivos y se
razona la decision a partir de ellos. Los scores por fold se persisten ahora
en `artifacts/tables/phase14_fold_level_scores.csv` y los deltas en
`artifacts/tables/phase14_paired_deltas.csv` — antes se descartaban al guardar
solo media y desviacion.

---

## 3. Tier 1 — Challenger screening

Mismo feature set (E13, 10 columnas), misma CV (V1, 5 folds, seed 42), sin
tuning individual de los challengers.

| run | ROC-AUC (mean +/- std) | delta pareado medio | mediana | rango [min, max] | folds favorables |
|---|---|---|---|---|---|
| **E20** (incumbente, tuneado) | **0.8611 +/- 0.0250** | — | — | — | — |
| E22 XGBoost (default) | 0.8590 +/- 0.0229 | -0.0021 | -0.0010 | [-0.0062, +0.0013] | 1/5 |
| E23 CatBoost (default) | 0.8606 +/- 0.0216 | -0.0005 | -0.0003 | [-0.0043, +0.0054] | 2/5 |
| E24 LightGBM (default) | 0.8593 +/- 0.0272 | -0.0018 | -0.0015 | [-0.0060, +0.0010] | 1/5 |
| E25 logit-stack | 0.8615 +/- 0.0246 | +0.0004 | +0.0009 | [-0.0007, +0.0010] | 4/5 [^e25] |

[^e25]: Evaluacion contaminada — ver la subseccion siguiente. Su delta no es
una estimacion valida de rendimiento incremental.

**Lectura.** Los tres challengers validos quedan por debajo del incumbente, y
la dispersion por fold lo confirma: ninguno gana en mas de 2 de 5 folds.
CatBoost es el mas cercano en media (-0.0005) pero tambien el mas inestable
(rango de casi 0.010, con signos opuestos entre folds) — su media pequena
promedia un fold claramente peor con otro claramente mejor, que es exactamente
lo que la diferencia de medias no dejaba ver.

**Y un detalle que conviene no pasar por alto:** el unico candidato con
direccion consistente es E25 — 4 de 5 folds favorables y el rango mas estrecho
de todos. Bajo una lectura ingenua eso pareceria la senal mas solida de la
tabla. Es, precisamente, el candidato cuya evaluacion esta contaminada. Una
fuga entre capas de stacking produce justamente esa firma: mejoras pequenas y
consistentes. La magnitud sigue siendo despreciable (+0.0004), asi que la
decision no cambia, pero el caso ilustra por que la consistencia entre folds
no sustituye a una evaluacion correctamente anidada.

Reproducibilidad: esta re-ejecucion reproduce las medias de la corrida original
de 2026-09-02 de forma exacta.

> **Caveat obligatorio.** E20 fue seleccionado y tuneado usando esta misma
> estructura de CV. Estas comparaciones pareadas evaluan challengers baratos
> contra un incumbente ya optimizado bajo el protocolo de validacion de
> referencia; **no** son un benchmark imparcial entre familias de algoritmos.

### E25: por que su numero no cuenta

El ensemble E25 (logit-stack sobre las OOF de E20/E22/E23/E24) aparecia como
el mejor candidato. Su evaluacion esta contaminada.

Las predicciones OOF se generan sobre los folds F1..F5. El stacker se evalua
despues con `run_group_cv` sobre **esos mismos** folds. Cuando F1 es el fold
de validacion del stacker, este se entrena con filas de F2..F5 cuyas
meta-features fueron producidas por modelos base entrenados en F1+F3+F4+F5 —
modelos que **si vieron F1**. Hay fuga entre capas.

El codigo afirmaba lo contrario: un docstring explicaba que reusar el mismo
seed garantizaba la ausencia de leakage. El razonamiento estaba invertido —
compartir los folds es lo que produce la contaminacion, no lo que la evita.

> E25 remains exploratory and is not eligible for model promotion because its
> evaluation is contaminated across stacking levels. The observed +0.0004
> therefore cannot be interpreted as a valid estimate of incremental
> generalization performance. Given the small observed delta under a
> methodologically invalid evaluation, together with the additional complexity
> required for a proper nested evaluation, no further compute was allocated.

No se afirma la direccion del sesgo. Este tipo de contaminacion suele producir
estimaciones optimistas, pero eso no puede garantizarse y **no hace falta**
para la decision: basta con que no exista evidencia valida para reemplazar al
incumbente.

Una evaluacion correcta requeriria stacking anidado (CV interna dentro de cada
fold externo para generar las meta-features, y modelos base reentrenados solo
sobre el train externo). No se construyo: el objetivo de esta fase es decidir
si merece la pena invertir mas, y un delta de esa magnitud bajo una evaluacion
invalida no justifica el gasto.

La marca viaja con el dato, no solo con este documento:
`phase14_fold_level_scores.csv` incluye las columnas `evaluation_status` y
`promotion_evaluation_valid`, donde E25 queda como
`exploratory_contaminated` / `false`. `promotion_evaluation_valid = true`
significa *"su evaluacion es metodologicamente valida para considerar una
promocion"*, no *"cumplio los criterios para ser promovido"*.

---

## 4. Tier 2 — Exploracion acotada de features

Cada candidata paso el checklist de 5 preguntas de
`.claude/rules/leakage-and-validation.md` y el test adversarial de
`tests/test_features.py` **antes** de entrar al ablation.

Resultados (`artifacts/tables/phase14_feature_isolation_results.csv`, mismo
modelo E20 tuneado, CV V1, seed 42):

| feature | ROC-AUC | delta vs E13 | veredicto |
|---|---|---|---|
| E13 (referencia) | 0.8611 | — | — |
| `laptime_roll_mean_5` | 0.8221 +/- 0.0187 | -0.0390 | rechazada |
| `pit_stops_rate_last3` | 0.8613 +/- 0.0249 | +0.0002 | rechazada |

**`laptime_roll_mean_5`** degrada de forma clara, igual que
`laptime_roll_mean_3` en Fase 6 (-0.057 aislada sobre E10).

La conclusion que se puede sostener es acotada: *las dos variantes de ventana
evaluadas (3 y 5 vueltas) degradaron la generalizacion group-aware de forma
sustancial, lo que no da evidencia para seguir invirtiendo en esa familia de
features.* No se afirma que ninguna ventana funcione — no se probaron 2, 4, 7,
10, EWMA, expanding mean ni variantes normalizadas.

**`pit_stops_rate_last3`** se rechaza pese al delta positivo. El argumento no
es que una columna extra sea cara de mantener; rara vez lo es. Es que +0.0002
no demuestra una mejora practica reproducible, y por tanto no existe razon
positiva para ampliar el feature set. La complejidad debe ganarse su sitio.

---

## 5. Tier 3 — No perseguido

| Opcion | Por que no |
|---|---|
| Ensemble grande (100+ modelos) | No existe una hipotesis experimental que justifique expandir de 4 a 100+ modelos bajo el presupuesto de computo y mantenimiento de este proyecto. **No se estima la ganancia**: la version anterior derivaba un rango de que dos ensembles publicos distasen 0.00004 entre si, lo cual no informa de cuanto aporto cada ensemble sobre su propio mejor modelo individual |
| Tuning intensivo de los challengers | El proposito de Tier 1 es screening barato. Un challenger que no muestra ventaja material en su configuracion por defecto no pasa a la etapa costosa. **No se afirma** saber como responderian XGBoost, CatBoost o LightGBM al tuning: la justificacion es de presupuesto experimental, no de conocimiento previo |
| Mitigacion ad-hoc del anio 2023 | Es una limitacion documentada de los datos, no un bug del pipeline. Ver mas abajo |

### El caso 2023, planteado como failure mode

`Year == 2023` presenta una tasa de pit cercana al 1% frente a 19-30% en el
resto de temporadas, en **todas** las carreras de ese anio. Fase 10 midio AUC
0.6668 en ese segmento contra 0.8620 global.

Ajustar el pipeline para "arreglarlo" seria modificar el modelo tras ver el
resultado. Pero documentarlo y parar tampoco agota la pregunta. En un sistema
en operacion, un cambio de distribucion de esa magnitud exigiria deteccion de
drift, recalibracion, un disparador de reentrenamiento y monitorizacion por
segmento. Nada de eso esta implementado aqui — es explicitamente fuera de
alcance — pero es el failure mode relevante y conviene nombrarlo como tal.

---

## 6. Tier 4 — No reabrir el holdout

Aparecio una tentacion razonable: comprobar si la importancia de features es
estable tambien sobre el holdout congelado.

Se decidio no hacerlo. La formulacion precisa:

> Computing permutation importance on the holdout would consume additional
> information from the reserved evaluation set. Although this would not
> constitute training leakage by itself, it could introduce post-selection
> feedback and would violate the pre-registered one-touch holdout policy.

La distincion importa: no es leakage de entrenamiento. El modelo esta
congelado y el resultado no se usaria para modificarlo. El problema es que el
holdout dejaria de ser una evaluacion independiente en el momento en que
cualquier informacion extraida de el alimente una decision posterior.

Esta es la parte mas solida de la fase, y por una razon concreta: la regla se
escribio en Fase 3, mucho antes de que existiera esta tentacion. Eso es lo que
distingue una politica de proceso de una racionalizacion.

---

## 7. La regla que faltaba

Fase 14 decidio si cada delta era "suficientemente grande" **despues** de
observarlo. Eso es construir el criterio para que encaje con el resultado.

No se corrige retroactivamente inventando un umbral que nunca existio:

> No minimum practical effect threshold had been pre-specified before these
> experiments. Therefore, the decision reported in Phase 14 is retrospective
> and combines observed predictive gain with engineering complexity. A formal
> challenger acceptance policy is introduced prospectively for subsequent
> experiments. **Policy effective from Phase 15 onward.**

### Challenger Acceptance Policy (prospectiva, Fase 15+)

| Gate | Criterio |
|---|---|
| **G0** Leakage | Pasa el checklist de 5 preguntas y, si toca features, el test adversarial |
| **G1** Predictive | La mejora pareada media alcanza el MPE especifico del experimento, definido antes de ejecutarlo |
| **G2** Consistency | Los deltas pareados no muestran inestabilidad material entre folds ni en los grupos criticos predefinidos. Se reportan media, mediana, min, max y numero de folds favorables, y se razona. Con 5 folds, 3/5 no es un PASS automatico |
| **G3** Secondary | La tolerancia de PR-AUC se define antes del experimento, igual que el MPE |
| **G4** Latency | Dentro del presupuesto, con protocolo de medicion documentado: mismo hardware, mismo batch, warm-up, numero de repeticiones, mediana/p95 |
| **G5** Complexity | Inventario cualitativo explicito — numero de modelos, dependencias nuevas, transformaciones de features, tamano de artefacto, tiempo de entrenamiento, cambios en el serving path. Sin inventar un "complexity score" |
| **G6** Reproducibility | Entorno pinneado + seeds deterministas + tests en verde + una segunda ejecucion que reproduce las metricas dentro de una tolerancia fijada **antes** de esa segunda ejecucion. Tests en verde no implica reproducible |

**No todos los gates aplican a todo challenger.** Si la latencia no es una
restriccion real de un experimento concreto, se declara `applicable: false`
con su motivo, en lugar de inventar un presupuesto para rellenar el fichero.
El objetivo es disciplinar decisiones, no producir formularios.

**Donde se pre-registra:** `configs/experiments/<experiment_id>.yaml`
(plantilla en `configs/experiments/_template.yaml`), commiteado **antes** de
ejecutar. El campo central es `mpe_rationale`: un MPE sin justificar deja
abierta exactamente la pregunta que la politica pretende cerrar. El commit del
YAML precede al de resultados, de modo que la frontera
`criterio -> experimento -> resultado` queda verificable en el historial de
Git, y no al reves.

---

## 8. Competition Protocol as an External Reference

Los write-ups publicos de la competicion de la que procede este dataset son
una referencia externa util, siempre que se citen como lo que son.

**Citable, por estar publicado por sus propios autores:** los write-ups
describen ensembles del orden de 200-250 corrientes de predicciones OOF
construidas sobre multiples feature sets, y algunos describen features
agregadas cross-year o dependientes de temporadas futuras. Los scores
publicos del leaderboard estan en torno a 0.955.

**No citable:** atribuir a nadie que "no auditara" o "no supiera cuando
parar". Un procedimiento ausente de un write-up no es un procedimiento
ausente. La distincion entre *not reported in the available write-up* y
*not performed* se mantiene siempre.

**Tampoco citable:** cuanto de un score se explica por una tecnica concreta.
Eso requeriria el ablation de esa solucion, que no esta publicado.

El contraste util es entre **protocolos y funciones objetivo**, no entre
personas. Una solucion optimizada para un leaderboard opera bajo restricciones
distintas: la interpretabilidad no puntua, el coste de mantenimiento no
puntua, y las features dependientes de informacion futura no estan penalizadas
porque el protocolo de evaluacion no las detecta. Este proyecto adopto
restricciones distintas — auditabilidad temporal, CPU, un solo artefacto — y
por tanto llega a una solucion distinta. Ninguna de las dos es la correcta en
abstracto.

**Sobre los numeros:** el score publico del leaderboard, la CV group-aware y
el holdout interno proceden de protocolos de evaluacion distintos y **nunca se
restan ni se comparan causalmente** entre si. Ver
`artifacts/reports/kaggle_late_submission.md` para el registro de la late
submission de este proyecto y su interpretacion.

---

## 9. Que queda de Fase 14

**Candidato final sin cambios: `E20_hist_gradient_boosting`.**

Ningun challenger aporto evidencia valida y suficiente para desplazarlo. Dos
features candidatas se rechazaron, una por degradacion clara y otra por no
demostrar mejora practica.

El resultado con mas valor de esta fase no es el modelo — es el mismo que
antes — sino lo que la auditoria encontro en el propio proceso: una
comparacion estadistica invalida, una evaluacion de ensemble contaminada que
el codigo documentaba como limpia, y la ausencia de un umbral de mejora fijado
de antemano. Las tres estan corregidas, y la tercera dio lugar a una politica
que entra en vigor de forma prospectiva.

---

## Referencias

- Scores por fold: `artifacts/tables/phase14_fold_level_scores.csv`
- Deltas pareados: `artifacts/tables/phase14_paired_deltas.csv`
- Ablation de features: `artifacts/tables/phase14_feature_isolation_results.csv`
- Script reproducible: `scripts/phase14_model_selection_framework.py`
- Plantilla de pre-registro: `configs/experiments/_template.yaml`
- Decision de clase de solucion: `artifacts/reports/modeling_strategy_decision_record.md`
- Reglas de validacion: `.claude/rules/leakage-and-validation.md`
