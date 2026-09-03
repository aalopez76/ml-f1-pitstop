# Model Selection Framework — Fase 14

## La pregunta que responde este documento

No es *"¿cual es el mejor score posible?"* — es **¿como decides que un
modelo es lo suficientemente bueno para dejar de optimizarlo, y como
documentas esa decision para que alguien mas pueda auditarla?**

Esta pregunta surgio al comparar este proyecto contra los writeups
publicados del 1er y 2do lugar de la competencia Kaggle real que inspira
este dataset (Playground Series S6E5). Ninguno de los dos documenta
explicitamente *cuando* paran de agregar modelos, ni el costo/beneficio de
seguir — solo persiguen el score maximo del leaderboard.

## Kaggle Top 2 vs este proyecto

| Aspecto | 1er lugar (Optimistix) | 2do lugar (Chris Deotte) | Este proyecto |
|---|---|---|---|
| Score final | 0.95506 (publico/privado) | 0.95502 (privado) | 0.8727 (holdout, Fase 13) |
| Margen de la victoria | +0.00001 sobre 2do | perdio por +0.00004 | N/A — objetivo distinto |
| # de modelos | 186 OOF | 218 modelos | 5 manuales + AutoGluon (Fase 7-8), +3 en esta fase |
| Codigo generado | Manual (implicito) | 230k lineas (LLM agent autonomo) | ~2k lineas, manual |
| Auditoria de leakage | No documentada | Nested folds (mencionado) | Checklist 5-preguntas + subagente `leakage-auditor` |
| Estrategia de CV | No documentada (probable V0) | No documentada | V0 vs V1 vs V2 comparadas y justificadas (Fase 3) |
| Reproducibilidad | Manual/ad-hoc | Delegada a un LLM agent | Scripts + seed fijo + 92+ tests |
| Gap CV->holdout | Desconocido | Desconocido | -0.0116 (Fase 13: holdout MEJOR que CV) |
| Documentacion de decisiones de parada | No | No | Este documento |

Insight clave de los comentarios de esos writeups (ver `HANDOFF.md` para
el detalle completo): un competidor de 5to lugar (publico #1) le dijo al
ganador *"you did things the right way"* — y aun asi no gano. Ganar
Kaggle y construir ML auditable/producible son objetivos distintos; este
proyecto persigue el segundo.

## Tier 1 — Diversidad controlada (SE IMPLEMENTA)

Candidatos nuevos sobre el MISMO feature set (E13) y la MISMA CV (V1) que
ya uso el candidato final de Fase 7 (E20, HistGradientBoosting tuneado).
Ninguno se tunea individualmente — la pregunta no es "¿cual es el mejor
XGBoost posible?", es "¿un candidato por defecto justifica su costo de
mantenimiento frente al que ya esta optimizado y en produccion?".

Resultados reales (`artifacts/tables/phase14_diversity_comparison.csv`,
CV V1, seed=42, dev=346,246 filas):

| run | ROC-AUC (mean±std) | PR-AUC | fit (s/fold) | predict (ms/1k) | delta vs E20 |
|---|---|---|---|---|---|
| **E20_hist_gradient_boosting** (incumbente, tuneado) | **0.8611±0.0250** | 0.5531 | 4.39 | 5.92 | — |
| E22_xgboost_e13_features (default) | 0.8590±0.0229 | 0.5484 | 2.24 | 1.62 | -0.0021 |
| E23_catboost_e13_features (default) | 0.8606±0.0216 | 0.5526 | **153.58** | 0.47 | -0.0005 |
| E24_lightgbm_e13_features (default) | 0.8593±0.0272 | 0.5523 | 1.46 | 4.08 | -0.0018 |
| E25_ensemble_logit_stack (E20+E22+E23+E24) | 0.8615±0.0246 | — | — | — | **+0.0004** |

**Decision tomada:** se mantiene **E20** como candidato final. Ninguno de
los 3 candidatos por defecto lo supera — los 3 quedan por debajo, con
diferencias (-0.0005 a -0.0021) muy por debajo de 1 std (0.025). El
ensemble E25 SI queda por encima (+0.0004), pero esa ganancia es ~60x mas
chica que el ruido de CV (std 0.0246) — estadisticamente indistinguible
de cero. Adoptar el ensemble significaria mantener 4 modelos en produccion
(incluyendo CatBoost, **35-100x mas lento** de entrenar que los otros 3:
153.58s/fold vs 1.46-4.39s/fold) para una ganancia que no se puede
distinguir del ruido de muestreo. El costo (4x la complejidad de
despliegue, un modelo con fit ~35x mas lento) no se justifica por una
ganancia no medible con confianza. Esto es exactamente el mismo patron
que Top 1/2 de Kaggle no documentaron: mas modelos, ganancia marginal, sin
evaluar si el costo se justifica.

**Advertencia metodologica (hallazgo del subagente `leakage-auditor`,
A_REVISAR no bloqueante):** E20 fue tuneado en Fase 7
(`RandomizedSearchCV`) sobre la MISMA particion V1/seed=42 que esta fase
reutiliza para comparar contra E22/E23/E24 sin tuning. Esto no es leakage
(ninguna fila de validacion se usa para entrenar en ningun caso), pero es
una ventaja estructural real a favor del incumbente: E20 ya "conoce" esa
particion especifica desde su seleccion de hiperparametros, mientras que
E22-E24 nunca la vieron durante ningun ajuste. Interpretacion correcta de
la tabla de arriba: *"¿un candidato por defecto supera a un incumbente ya
optimizado sobre la particion de referencia?"*, no *"¿cual es el mejor
algoritmo posible en igualdad de condiciones?"* — esa segunda pregunta
requeriria tunear los 4 por separado, que es precisamente lo que esta
fase decide NO hacer (ver Tier 3).

## Tier 2 — Exploracion acotada de features (SE IMPLEMENTA, 2 candidatas)

Cada candidata paso el checklist de 5 preguntas de
`.claude/rules/leakage-and-validation.md` y el test adversarial
obligatorio (`tests/test_features.py`) ANTES de entrar al ablation — no
se prueba nada que no pase ese filtro primero.

Resultados reales (`artifacts/tables/phase14_feature_isolation_results.csv`,
mismo modelo de referencia E20 tuneado, CV V1, seed=42):

| feature | ROC-AUC (mean±std) | delta vs E13 | veredicto |
|---|---|---|---|
| E13 (referencia, sin candidatas) | 0.8611 | — | — |
| `laptime_roll_mean_5` | 0.8221±0.0187 | **-0.0390** | rechazada (fuerte) |
| `pit_stops_rate_last3` | 0.8613±0.0249 | +0.0002 | rechazada (ruido) |

**Decision tomada:**

- `laptime_roll_mean_5` **se descarta**: hereda la misma inestabilidad
  que `laptime_roll_mean_3` (Fase 6, excluida por -0.057 aislada sobre
  E10) — ampliar la ventana de 3 a 5 vueltas no soluciona el problema de
  fondo, lo empeora (-0.039 sobre un feature set ya optimizado, E13). Un
  rolling de `LapTime_s_winsorized` de cualquier ventana sigue sin
  generalizar bien entre carreras bajo V1.
- `pit_stops_rate_last3` **se descarta tambien**, pese a tener delta
  positivo (+0.0002): esa ganancia es ~100x mas chica que el std de CV
  (0.0249) — indistinguible de ruido de muestreo. Adoptarla agregaria una
  columna mas a mantener, documentar y auditar por leakage (ya lo esta,
  pero cada feature nueva es superficie adicional) sin evidencia de que
  aporte senal real. **Un delta positivo no es automaticamente motivo
  para adoptar una feature** — el mismo criterio que aplica a Tier 1
  (¿el delta supera el margen de ruido?) se aplica aqui.

## Tier 3 — Rechazado explícitamente (NO SE IMPLEMENTA)

Sin codigo nuevo. Documentado aqui como decision activa, con el mismo
razonamiento costo/beneficio que ya senta el precedente en Fase 6
(`laptime_roll_mean_3` descartada por inestable) y Fase 8 (AutoGluon
`good_quality` no se corrio):

| Opcion considerada | Costo estimado | Ganancia estimada | Por que NO |
|---|---|---|---|
| Ensemble gigante (100+ modelos) | Semanas de computo + mantenimiento imposible de auditar | +0.0001–0.001 (ver Top 1/2: 218 modelos no le ganaron a 186) | El propio caso Optimistix vs Deotte demuestra que mas modelos no garantiza mejor resultado — el retorno decrece a casi cero |
| Tuning obsesivo (Optuna n_iter>50 por candidato) | Horas adicionales por candidato, 4+ candidatos en esta fase | Rendimientos decrecientes (Fase 7: tuning de E20 gano solo +0.0012 sobre default) | El tuning de Fase 7 ya establecio que la ganancia marginal es minima una vez que el feature set es bueno |
| Drift mitigation ad-hoc para 2023 (Fase 3/10: tasa de pit ~1% vs ~20-30% otros anios) | Reajustar pipeline completo para "arreglar" un anio anomalo | Score mas alto en ese segmento especifico, riesgo de sobreajustar a un artefacto sintetico | Es una limitacion documentada de los datos (H4 del spec), no un bug — "arreglarla" seria ajustar el modelo a ver el resultado, la regla 8 de `leakage-and-validation.md` lo prohibe explicitamente |

## Tier 4 — La decision de NO re-tocar el holdout

Se identifico una tentacion razonable: verificar si la importancia de
features (permutation importance, ya calculada en Fase 9 sobre CV/OOF) es
estable tambien en el holdout — dato que Fase 13 nunca calculo (solo
midio ROC-AUC/PR-AUC ahi).

**Se decidio NO hacerlo.** `leakage-and-validation.md` seccion 3 dice que
el holdout "se evalua una unica vez en la Fase 13" — y seccion 9 (agregada
en esta misma fase, ver mas abajo) aclara que eso aplica tambien a
analisis que parecen "solo diagnostico": correr `permutation_importance()`
sobre el holdout es una nueva inferencia del modelo sobre esas filas,
exactamente el tipo de uso que la regla busca prevenir.

Esto es, deliberadamente, el ejemplo mas fuerte de este documento: la
regla se escribio ANTES de tener este resultado especifico en la mano (en
Fase 3, mucho antes de que existiera la tentacion concreta de Fase 14) —
que es precisamente el punto de tener reglas de proceso escritas de
antemano en vez de decidir caso por caso cuando ya se sabe que se
"ganaria" con la excepcion.

## Conclusion: el framework es reusable

Los 4 tiers de esta fase no son especificos de F1 ni de este dataset. El
patron generaliza:

1. **Medir con evidencia real**, no estimar (Tier 1/2 se implementaron con
   resultados medidos, no solo proyectados).
2. **Documentar lo que se rechaza y por que**, con el mismo rigor que lo
   que se acepta (Tier 3).
3. **Reconocer cuando una regla de proceso pre-existente ya responde la
   pregunta**, en vez de reabrir la discusion cada vez que aparece un caso
   nuevo (Tier 4).
4. **El resultado real decide, no la expectativa** — si Tier 1 hubiera
   superado a E20 claramente, este documento diria eso; el hecho de que
   la conclusion final sea la que sea no estaba decidido de antemano.
