# EDA Report — F1 Pit Stop Prediction (Fase 2)

Fecha: 2026-08-26. Fuente: `data/raw/train.csv` (439,140 filas),
`data/raw/test.csv` (188,165 filas). Figuras en `artifacts/figures/`
(`00_missingness.png` a `06_train_vs_test_drift.png`).

No se concluye causalidad de nada de lo siguiente — solo asociacion
observacional, tal como exige el spec (Fase 2, "No concluir causalidad
desde EDA").

## Preguntas obligatorias

### 1. ¿Que tan desbalanceado esta `PitNextLap`?

80.1% negativos / 19.9% positivos. Desbalance moderado, no extremo.
ROC-AUC (metrica de Kaggle) es razonable sin resampling agresivo.
Figura: `01_target_balance.png`.

### 2. ¿Como cambia la tasa de pit por fase de carrera?

Forma de U invertida clara sobre `RaceProgress` (fraccion de carrera
completada, bineada en deciles):

| fase de carrera | tasa de pit |
|---|---|
| 0–11% | 6.4% |
| 11–21% | 11.2% |
| 21–31% | 18.3% |
| 31–41% | 28.4% |
| 41–51% | 35.3% |
| **51–61%** | **38.7%** |
| **61–70%** | **38.6%** |
| 70–80% | 28.7% |
| 80–90% | 14.9% |
| 90–100% | 5.1% |

La probabilidad de pit se concentra en la mitad de la carrera, cae fuerte
al inicio (recien salieron con neumaticos nuevos) y al final (ya no vale
la pena pitear). Coherente con estrategia real de F1. Figura:
`02_pit_rate_by_race_progress.png`.

### 3. ¿Existen diferencias por piloto/carrera/compuesto?

- **Compound**: variacion enorme — HARD 32.8% vs SOFT 19.3% vs
  INTERMEDIATE 15.2% vs MEDIUM 10.1% vs WET 2.5%. Tiene sentido: los
  autos suelen empezar en HARD/MEDIUM y pitear hacia compuestos mas
  blandos, y en mojado casi no se pitea a WET de nuevo. Figura:
  `03_pit_rate_by_compound.png`.
- **Race**: variacion sustancial — de 38.9% (Chinese GP) a 9.1% (Mexico
  City GP). Distintas pistas tienen distintas ventanas de pit / numero de
  paradas tipico.
- **Driver**: variacion mas modesta (std de la tasa de pit entre pilotos
  ≈ 0.099), probablemente confundida con equipo/estrategia mas que con
  habilidad individual del piloto — no se investiga mas a fondo aqui.

### 4. ¿Que variables muestran fuerte separacion univariada?

ROC-AUC de cada variable numerica usada sola como score (>0.5 = alguna
separacion; direccion no importa):

| variable | AUC univariada |
|---|---|
| **LapNumber** | **0.702** |
| **TyreLife** | **0.699** |
| **Stint** | **0.684** |
| RaceProgress | 0.664 |
| Cumulative_Degradation | 0.611 |
| Year | 0.595 |
| LapTime_Delta | 0.566 |
| LapTime (s) | 0.540 |
| Position_Change | 0.528 |
| PitStop | 0.521 |
| Position | 0.516 |

`LapNumber`, `TyreLife` y `Stint` son, individualmente, ya buenos
predictores (AUC ~0.68–0.70) — coherente con la intuicion de dominio
(neumaticos mas viejos → mas probable pitear). Figura:
`04_univariate_auc.png`.

### 5. ¿Existen grupos de filas muy correlacionadas?

(Interpretado como columnas muy correlacionadas entre si, que es lo que
permite construir la matriz de correlacion pedida como grafico minimo.)

Pares con `|r| > 0.7`:

| par | r |
|---|---|
| `LapNumber` vs `RaceProgress` | 0.965 |
| `LapNumber` vs `Stint` | 0.724 |
| `Stint` vs `RaceProgress` | 0.710 |

`LapNumber` y `RaceProgress` son casi redundantes (r=0.965) — coherente
con que `RaceProgress` = `LapNumber` / vueltas totales de la carrera.
Implicacion para Fase 6/7: evitar meter las tres juntas en modelos
lineales sin regularizacion; en modelos de arboles la redundancia es
menos problematica pero igual conviene documentarla. Figura:
`05_correlation_matrix.png`.

### 6. ¿Hay distribucion distinta train vs test?

**No.** Kolmogorov-Smirnov por variable numerica: todos los `ks_stat` son
minusculos (<0.004) y ningun `p_value` es significativo (todos > 0.12,
la mayoria > 0.5). Proporciones de `Compound` y `Race` tambien
practicamente identicas train/test (diferencias < 0.3 puntos
porcentuales). Figura: `06_train_vs_test_drift.png`.

**Hallazgo relevante para Fase 3:** train y test contienen exactamente
las mismas 26 carreras y los mismos 4 anios (`Year` 2022–2025), sin
ninguna carrera exclusiva de train o de test. Esto sugiere que el split
train/test de Kaggle es **row-level, no agrupado por carrera** — el
mismo evento aparece de ambos lados. Esto no significa que nuestra
validacion interna deba imitar ese mismo split: el objetivo de H1 del
spec es evaluar si una validacion aleatoria ingenua es optimista respecto
a un escenario mas realista (carrera nueva nunca vista), independientemente
de como Kaggle particiono su propio test set.

### 7. ¿La cardinalidad de categoricas puede generar problemas?

Ya cubierto en `data_audit.md` (Fase 1): `Driver` tiene 887 valores
unicos (alta cardinalidad para one-hot naive; considerar target/frequency
encoding o dejar que skrub/AutoGluon lo manejen). `Compound` (5) y `Race`
(26) tienen cardinalidad manejable. No hay diferencias de vocabulario:
mismos valores de `Race` en train y test (confirmado arriba).

### 8. ¿Hay columnas sinteticas con artefactos sospechosos?

**Si, y es el hallazgo mas importante de esta fase.** Investigacion
dirigida sobre el hallazgo de Fase 1 (`Stint` no monotono):

- **`Position_Change` no coincide con la diferencia de `Position` entre
  filas visibles consecutivas** del mismo `(Driver, Race, Year)`. Ejemplo
  verificado: en la vuelta 5 de un grupo, `Position_Change=-1.0` pero la
  diferencia real entre las filas visibles es `-4.0`. Esto, junto con que
  `LapNumber` no es consecutivo dentro de un grupo, sugiere fuertemente
  que **las columnas derivadas (`Position_Change`, `RaceProgress`,
  `Cumulative_Degradation`, `LapTime_Delta`) se calcularon sobre una
  secuencia completa oculta, y el CSV publico es un submuestreo de esa
  secuencia** (no vemos todas las vueltas).
- **Prueba matematica de que `Stint` no es solo un artefacto de
  submuestreo:** en una muestra de 3,000 grupos `(Driver, Race, Year)`,
  2,449 (81.6%) tienen al menos una caida de `Stint` al ordenar por
  `LapNumber` (caida maxima observada: -5). Quitar filas de una secuencia
  no-decreciente NUNCA puede producir una caida en la sub-secuencia
  visible — por lo tanto esto **no es explicable por submuestreo**, es
  una inconsistencia real en como se genero el dato sintetico.

**Conclusion combinada:** el dataset no es una simulacion fisica estricta
con garantias duras (ej. "Stint solo puede subir"); es una generacion
sintetica que aproxima distribuciones y relaciones marginales (consistente
con lo que dice la pagina de Kaggle: *"Feature distributions are close
to, but not exactly the same, as the original"*) sin necesariamente
preservar restricciones de consistencia secuencial estricta dentro de
cada grupo. Ver seccion de hipotesis mas abajo para las implicaciones en
Fase 3/6.

## Lista priorizada de hipotesis (criterio de salida de la Fase 2)

Ordenadas por impacto esperado en el diseno del proyecto, de mayor a
menor:

1. **[Diseno de validacion, prioridad maxima]** Las features derivadas
   (`RaceProgress`, `Cumulative_Degradation`, `LapTime_Delta`,
   `Position_Change`) fueron calculadas sobre una secuencia completa no
   visible y luego submuestreadas; construir features de rolling/lag
   propias asumiendo continuidad entre filas visibles consecutivas del
   mismo `(Driver, Race, Year)` produce una nocion de "vuelta anterior"
   que en realidad puede estar a muchas vueltas reales de distancia. →
   Experimento Fase 3: verificar el efecto de tratar el `LapNumber` como
   distancia real (usar el propio `LapNumber` como espaciador, no solo el
   orden de fila) al construir cualquier rolling/lag.
2. **[Diseno de validacion]** `Stint` no es confiable como contador
   estrictamente creciente (81.6% de grupos muestreados violan
   monotonicidad); no debe usarse como ancla de "inicio/fin de stint" sin
   verificacion adicional. → Experimento: comparar un modelo que usa
   `Stint` tal cual vs uno que la excluye o la recalcula desde `TyreLife`.
3. **[Split, H1]** Aunque el propio split train/test de Kaggle es
   row-level (mismas 26 carreras en ambos lados), la pregunta de
   portafolio exige evaluar el escenario mas realista: ¿el modelo veria
   la misma carrera en train e inferencia en produccion? → Experimento
   central de Fase 3: comparar CV aleatorio (V0) vs group-aware por
   `(Race, Year)` (V1) vs holdout temporal por `Year` (V2), y cuantificar
   cuanto se infla el ROC-AUC con V0 vs V1/V2 (test de H1).
4. **[Feature engineering]** `TyreLife`, `LapNumber` y `Stint` son ya
   fuertes predictores univariados (AUC 0.68–0.70) y estan correlacionados
   entre si (`LapNumber` vs `RaceProgress` r=0.965). Combinarlos con
   `Compound` (fuerte variacion de tasa de pit, 2.5%–32.8%) podria dar una
   interaccion util (ej. `TyreLife` normalizado por compuesto tipico). →
   Experimento Fase 6: probar una feature de interaccion
   `TyreLife x Compound` contra el baseline sin interaccion.
5. **[Feature engineering]** La forma de U invertida de la tasa de pit
   sobre `RaceProgress` (pico en 51–70%) sugiere que una version
   discretizada/binned de `RaceProgress`, o una feature no-lineal
   (ej. distancia al centro de carrera), podria ayudar a modelos lineales
   que no capturan la no-linealidad naturalmente. → Experimento: comparar
   LogisticRegression con `RaceProgress` cruda vs binned.
6. **[Validacion de features]** No se pudo confirmar la semantica exacta
   de `PitStop` en Fase 1 (¿la vuelta actual fue de pit stop?). La
   diferencia de tasa de pit-siguiente-vuelta entre `PitStop=0` (19.1%) y
   `PitStop=1` (24.8%) es moderada, no una fuga evidente, pero merece
   confirmacion explicita del momento en que se registra (¿antes o
   despues del stop de esa vuelta?) antes de aceptarla como feature
   valida. → Aplicar el checklist de 5 preguntas de
   `leakage-and-validation.md` en Fase 3.
7. **[Encoding]** `Driver` (887 valores unicos, alta cardinalidad, y no
   se comporta como el grid real de F1 segun el hallazgo de Fase 1) es
   candidata a target/frequency encoding en vez de one-hot naive, o a
   dejarse fuera del modelo manual base y solo probarse en un experimento
   dedicado. → Experimento Fase 6/7: modelo con vs sin `Driver`
   (encoded), medir impacto en CV.
8. **[Modelado]** `WET` (0.3% de las filas) e `INTERMEDIATE` (4.0%) son
   categorias raras de `Compound` con tasas de pit muy distintas al resto
   (2.5% y 15.2%); en folds pequenos podrian quedar mal representadas. →
   Verificar en Fase 3 que la estrategia de split no deje folds sin
   ejemplos de estas categorias.
9. **[H4, dataset original]** El link al dataset F1 original esta roto en
   Kaggle. Si se decide perseguir H4 (drift sintetico vs real), habria
   que reconstruir una fuente equivalente (ej. FastF1) antes de poder
   comparar distribuciones. → Bloqueador documentado, no accionable
   todavia.
10. **[Menor prioridad]** La variacion de tasa de pit por `Race` (9%–39%)
    podria deberse a diferencias reales de estrategia por circuito, pero
    tambien podria estar parcialmente confundida con el numero de vueltas
    total de cada carrera (`RaceProgress` normaliza por eso, pero el
    conteo absoluto de oportunidades de pit por carrera no se investigo
    aqui). → Explorar en un EDA de seguimiento si hace falta, no bloquea
    Fase 3.
