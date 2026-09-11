# Explainability Report — E20 (HistGradientBoostingClassifier)

Consolida los dos métodos de explicabilidad del proyecto — antes dispersos entre un script de Fase 9 (permutation importance) y este documento nuevo (SHAP) — en una sola narrativa. Ambos se calculan sobre el mismo modelo (E20, fiteado sobre `dev` completo, feature set E13) para que sean comparables.

Reproducible: `uv run python scripts/phase9_skore_evaluation.py` (permutation) + `uv run python scripts/phase9_shap_explainability.py` (SHAP).

## 1. Qué mide cada método, y por qué se reportan los dos

| | Permutation importance | SHAP |
|---|---|---|
| **Mide** | Cuánto degrada la métrica (ROC-AUC) al barajar una columna, manteniendo las demás intactas | Cuánto contribuye cada feature a la predicción de *una fila concreta*, de forma que la suma de contribuciones reconstruye exactamente `predict_proba` |
| **Nivel** | Solo global (una degradación promedio sobre todo el dataset) | Global (agregando `|SHAP|` sobre muchas filas) y **local** (una fila individual) |
| **Sensible a correlación entre features** | Sí — si dos features están correlacionadas, perturbar una no cambia mucho la predicción porque la otra la compensa, subestimando ambas | También, aunque de forma distinta (la atribución se reparte entre las correlacionadas) |
| **Costo computacional aquí** | Bajo — una pasada sobre el dev set completo (346,246 filas) | Alto — ver §4. El `TreeExplainer` nativo de `shap` no es compatible con la categórica nativa de HGB en la versión instalada; el método viable (`PermutationExplainer` genérico) corre a ~400 ms/fila, lo que fuerza a trabajar sobre una muestra |

Ningún método por sí solo responde ambas preguntas que un stakeholder real hace: *"¿qué importa en general?"* (global) y *"¿por qué esta predicción concreta?"* (local). Por eso se reportan los dos, no uno solo con una nota diciendo que el otro "también existe".

## 2. Importancia global — ¿coinciden ambos métodos?

| feature | permutation importance | SHAP `mean(|valor|)` |
|---|---|---|
| `Stint` | **0.0719** | **0.1034** |
| `TyreLife` | 0.0640 | 0.0822 |
| `pit_stops_so_far` | 0.0484 | 0.0819 |
| `LapNumber` | 0.0293 | 0.0340 |
| `Compound` (agregado) | 0.0182 | 0.0175 *(suma de las 5 categorías one-hot, ver abajo)* |
| `recomputed_stint` | 0.0060 | 0.0141 |
| `Position` | 0.0044 | 0.0068 |
| `laptime_delta_prev` | 0.0038 | 0.0125 |
| `laps_since_last_pit` | 0.0024 | 0.0071 |
| `PitStop` | 0.0019 | 0.0041 |

**Sí coinciden, en lo que importa.** El top 3 es idéntico en ambos métodos (`Stint`, `TyreLife`, `pit_stops_so_far`, con `TyreLife` y `pit_stops_so_far` prácticamente empatados en los dos), y `LapNumber` en cuarto lugar también en ambos. Dos métodos con mecanismos completamente distintos —uno mide degradación de la métrica por perturbación, el otro atribución aditiva por predicción— llegando al mismo ranking superior es la señal de robustez que un solo método no puede dar por sí mismo.

**Coincidencia adicional, no buscada a propósito:** SHAP descompone `Compound` en sus 5 categorías (`Compound_HARD` 0.0055, `Compound_MEDIUM` 0.0056, `Compound_SOFT` 0.0033, `Compound_INTERMEDIATE` 0.0030, `Compound_WET` 0.0001 — ver `artifacts/skore/e20_hgb_final/shap_importance.csv` para la tabla completa sin agregar). Sumadas, dan 0.0175 — casi idéntico a los 0.0182 que permutation importance mide tratando `Compound` como una sola columna. Esta cercanía no estaba garantizada por construcción (son magnitudes calculadas de formas distintas); que coincidan es una segunda validación cruzada entre métodos.

**Dónde difieren, y por qué es esperable:** el orden relativo de las features de menor impacto (`recomputed_stint`, `laptime_delta_prev`, `Position`, `laps_since_last_pit`, `PitStop`) cambia entre métodos. Esto es consistente con que varias de estas features están correlacionadas entre sí (`recomputed_stint` deriva de `Stint`; `laps_since_last_pit` y `pit_stops_so_far` comparten información) — ambos métodos son sensibles a correlación, pero de formas distintas, así que un reordenamiento en la cola de la tabla no es una contradicción, es la firma esperada de esa correlación.

## 3. Explicabilidad local — dos ejemplos, no arbitrarios

**Ejemplo 1 — predicción correcta de alta confianza.** `y_true=1` (pit stop real), `predict_proba=0.9023`. Elegido por ser el tipo de caso donde el modelo funciona como se espera: una confirmación de que las features de mayor peso global (`Stint`, `TyreLife`, `pit_stops_so_far`) también dominan localmente cuando el modelo acierta con confianza. Ver `artifacts/figures/e20_shap_local_example_correct.png`.

**Ejemplo 2 — un error real, anclado al segmento ya documentado en Fase 10.** `y_true=1` (sí hubo pit stop), `predict_proba=0.1054` — un falso negativo marcado, tomado específicamente del segmento `Year == 2023`, donde Fase 10 (`artifacts/reports/` — error analysis) ya había medido AUC 0.6668 contra 0.8620 global, por la anomalía de tasa de pit (~1% frente a 19-30% en el resto de temporadas). Esto no es un ejemplo elegido al azar: cierra el círculo entre *dónde* falla el modelo (ya sabido desde Fase 10) y *por qué* estaba confundido en un caso concreto de ese segmento (ahora visible con SHAP). Ver `artifacts/figures/e20_shap_local_example_2023_drift.png`.

## 4. Limitación técnica encontrada (no genérica — verificada en este proyecto)

`shap.TreeExplainer` (versión instalada: `shap==0.50.0`) **no es compatible** con el manejo nativo de categóricas de `HistGradientBoostingClassifier` (`categorical_features=["Compound"]`, dtype pandas `"category"`). Internamente intenta castear todo el array de entrada a `float` y falla con la columna `Compound`. Esto no es una suposición: se verificó con un smoke test dedicado antes de construir el resto del pipeline de explicabilidad, y queda fijado como test de regresión (`tests/test_explainability.py::test_treeexplainer_does_not_support_native_categorical`) — si una futura versión de `shap` soluciona esto, ese test empieza a fallar, señal para simplificar el código y volver al `TreeExplainer` nativo (más rápido).

**Solución aplicada:** `build_onehot_wrapper` (`src/f1pitstop/evaluation/explainability.py`) envuelve `model.predict_proba` en una función que opera sobre un espacio completamente numérico (one-hot de `Compound`) y reconstruye la fila original antes de llamar al modelo real — no un modelo sustituto, el mismo E20. Se verifica explícitamente (en el script y en un test dedicado) que `predict_fn(to_onehot(X)) == model.predict_proba(X)` de forma exacta antes de confiar en cualquier valor SHAP derivado.

**Costo de esta solución:** usa `shap.Explainer` genérico (`PermutationExplainer`, model-agnostic), no `TreeExplainer` — medido en ~400 ms/fila sobre el modelo real. Por eso el cálculo global se hizo sobre una **muestra estratificada de 500 filas** (por `PitNextLap`, seed fijo, reproducible), no sobre las 346,246 filas del dev set completo: a ese ritmo, el dev completo tomaría más de 38 horas. 500 filas es un compromiso deliberado costo/cobertura para un beeswarm plot legible, documentado aquí para que nadie lo interprete como una limitación de recursos ocultada.

**El holdout no se toca.** Todo el cálculo de SHAP (global y los dos ejemplos locales) opera sobre `dev`, nunca sobre el holdout congelado (`Year == 2025`) — consistente con la política de un único uso (`.claude/rules/leakage-and-validation.md` §9): calcular SHAP sobre el holdout consumiría información del conjunto reservado, exactamente el mismo argumento por el que Fase 14 decidió no correr `permutation_importance()` ahí.

## 5. Lo que esto no cubre

- **Explicación local en tiempo real servida en producción.** Esto es diseño de infraestructura, fuera del alcance de este proyecto (ver `modeling_strategy_decision_record.md`, sección de próximos pasos hacia producción).
- **SHAP sobre AutoGluon.** Su explicación viene de la estructura interna del ensemble, no de este mecanismo — ya documentado como límite en el ADR.
- **Interacciones de segundo orden (SHAP interaction values).** Computacionalmente más caras que la atribución simple; no se calcularon dado el costo ya identificado en §4.

## Referencias

- `scripts/phase9_skore_evaluation.py` — permutation importance
- `scripts/phase9_shap_explainability.py` — SHAP global y local
- `src/f1pitstop/evaluation/explainability.py` — wrapper one-hot reusable
- `tests/test_explainability.py` — verificación de aditividad SHAP y de la limitación de `TreeExplainer`
- `artifacts/skore/e20_hgb_final/permutation_importance.csv`, `.../shap_importance.csv` — datos completos sin agregar
- `artifacts/figures/e20_shap_summary.png`, `e20_shap_local_example_correct.png`, `e20_shap_local_example_2023_drift.png`
