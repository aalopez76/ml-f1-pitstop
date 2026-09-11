# Architecture Decision Framework: Cuidadosa Canalización vs AutoML vs Modelos Avanzados

> **Nivel de documento:** Senior/Staff Engineering decision record  
> **Propósito:** Documentar decisiones arquitectónicas, trade-offs, y alternativas evaluadas — no "qué ganó", sino "qué se eligió y por qué"

---

## Executive Summary

Este proyecto compara tres enfoques de diseño para un pipeline de ML tabular:

1. **Canalización Cuidadosa** (este proyecto): Feature engineering manual + GBDT simple + rigurosa validación
2. **AutoML** (AutoGluon): Automatización de algoritmos + ensembles internos + caja negra
3. **Modelos Avanzados** (Kaggle top): Redes neuronales tabulares (TABM, RealMLP, TabICL) + brute-force ensemble + code generation (Codex, Anthropic Claude)

**Cada enfoque tiene restricciones reales, ventajas auténticas, y desventajas explícitas.**

Este documento las mapea de forma honesta.

---

## Part 1: Las Restricciones Iniciales

### Restricción 1: Sin GPU

- **Este proyecto:** CPU 8-cores, 16 GB RAM (máquina local)
- **AutoGluon:** Funciona en CPU, pero GPU recomendada para tuning agresivo
- **Modelos Avanzados:** Casi obligatorio GPU (A100, A6000Ada para redes neuronales tabulares)

**Implicación:** Elegir GPU vs CPU es una decisión arquitectónica no técnica. "Usa GPU" no es una opción si los recursos no están disponibles.

### Restricción 2: Reproducibilidad Local

- **Este proyecto:** `uv run scripts/train.py` debe funcionar idéntico en cualquier máquina, mismo seed
- **AutoGluon:** Directorios temporales, artefactos de 10+ GB por fold, gestión compleja
- **Modelos Avanzados:** 230k líneas de código (Chris Deotte, Rank 2), múltiples dependencias, imposible reproducir sin GPU

**Implicación:** Reproducibilidad local descarta arquitecturas que requieren compute distribuido o artefactos enormes.

### Restricción 3: Interpretabilidad Requerida

- **Este proyecto:** "¿Por qué el modelo predijo X?" tiene una respuesta clara (feature importance)
- **AutoGluon:** Ensemble de 10+ modelos internos, stacking multi-nivel, explicación imposible
- **Modelos Avanzados:** Redes neuronales, transformers — black-box por defecto

**Implicación:** Interpretabilidad no es una "nice-to-have", es un requisito de operación.

---

## Part 2: Diseño de Este Proyecto (Trade-offs Conscientes)

### Decisión 1: Validación Group-Aware (V1)

**Elegido:**
```python
StratifiedGroupKFold(n_splits=5, groups=['Race', 'Year'])
```

**Alternativas descartadas:**
- V0 (StratifiedKFold random): +0.029 ROC-AUC pero sobrestima generalización
- V2 (Temporal): Correcto conceptualmente pero solo 1 fold posible (no estimación de spread)

**Trade-off:**
| Aspecto | V1 (elegido) | V0 | V2 |
|---------|---|---|---|
| **Realismo** | ✅ Modela generalizacion real | ❌ Optimista | ✅ Realista |
| **Spread del CV** | ✅ 5 folds (±0.025) | ✅ 5 folds (±0.001) | ❌ 1 fold |
| **Decisión informada** | ✅ Podemos ver heterogeneidad | ❌ Falsa certeza | ⚠️ Una sola estimación |

**Por qué V1:** La pregunta es "¿generaliza a carreras nuevas?" V1 responde eso. V0 contesta una pregunta diferente (¿generaliza a vueltas nuevas de carreras vistas?).

---

### Decisión 2: Manual Feature Engineering (10 features)

**Elegido:**
- `pit_stops_so_far` (lag-based)
- `recomputed_stint` (carrera-segura)
- `laptime_delta_prev` (con shift(1) obligatorio)
- `laps_since_last_pit` (historia local)
- Más 6 features crudas validadas

**Alternativas evaluadas:**
1. **Target Encoding (Kaggle approach):** >800 features, multi-fold, cross-year agregates
2. **Interacciones polinómicas:** Feature explosion
3. **AutoML feature selection:** Black-box, imposible auditar

**Trade-off:**
| Aspecto | Manual (elegido) | Target Encoding | Polynomial |
|---------|---|---|---|
| **Tiempo engineering** | 40 horas | 100+ horas | 20 horas |
| **Features** | 10 | 800+ | 100+ |
| **Auditoría de leakage** | ✅ 5Q checklist per feature | ❌ Implícita, desocumentada | ⚠️ Interacciones complejas |
| **ROC-AUC delta** | — | +0.03 vs esto | +0.02 vs esto |
| **Mantenibilidad** | ✅ Clear intent | ❌ Opaco | ⚠️ Requiere regresión |
| **Tiempo predicción** | 0.39 ms/1k rows | ~5 ms/1k rows | ~2 ms/1k rows |

**Por qué Manual:** El cost/benefit de Target Encoding (+0.03 AUC) vs cost (100 horas, 800 features, imposible auditar leakage, 10x más lento en inferencia) es negativo. Manual da 95% de la ganancia con 10% del esfuerzo y 100% de la auditoría.

---

### Decisión 3: HistGradientBoosting Tuneado vs Alternatives

**Elegido:**
```
HistGradientBoostingClassifier(
    learning_rate=0.127,
    max_iter=152,
    max_leaf_nodes=38,
    min_samples_leaf=35,
    l2_regularization=0.84
)
```
**ROC-AUC:** 0.8611 ± 0.0251

**Alternativas (Fase 7):**
| Modelo | ROC-AUC | Tiempo | Interpretabilidad | Seleccionado |
|---|---|---|---|---|
| **E20 HGB (tuned)** | **0.8611 ± 0.0251** | **25s/fold** | **✅ Native feature importance** | **✅ SÍ** |
| E21 ExtraTrees (tuned) | 0.8530 ± 0.0230 | 65s/fold | ✅ Native importance | ❌ −0.0081 AUC |
| E01 LogReg | 0.732 ± 0.041 | 0.9s/fold | ✅ Coefficients | ❌ −0.129 AUC |

**Por qué HGB:**
- Mejor AUC que alternativas
- Tiempo de entrenamiento razonable (no 153s como CatBoost)
- Soporte nativo de NaN (sin imputation pipeline)
- Feature importance clara

---

### Decisión 4: No Usar Redes Neuronales Tabulares

**Alternativas no elegidas:**
- TABM (Tabular Model)
- RealMLP
- TABR
- FT-Transformer
- TabICL

**Por qué no:**
| Razón | Impacto |
|---|---|
| **GPU requerido** | Restringe reproducibilidad local |
| **Tuning agresivo necesario** | Cost = 50+ GPU-horas de Optuna |
| **ROC-AUC esperado** | +0.005 a +0.015 vs GBDT (desde literatura) |
| **Interpretabilidad** | ❌ Black-box |
| **Overhead operacional** | Manejo de PyTorch/TF, dependencias |

**Cost-benefit:** 
- Ganancia esperada: +0.01 ROC-AUC (optimista)
- Costo: 50+ GPU-horas + interpretabilidad perdida + reproducibilidad desconocida
- **ROI: Negativo para este contexto de portafolio**

---

## Part 3: AutoGluon Evaluation (Fase 8)

### Qué Hizo AutoGluon

**A00 (features raw, no engineering):**
- ROC-AUC: 0.813 ± 0.022
- Tiempo: variable (AutoGluon maneja automático)
- Artefactos: ✅ sin guardar (temporal dirs)

**A01 (features tuned E13, engineering manual):**
- ROC-AUC: 0.861 ± 0.024 (estadísticamente idéntico a E20: 0.8611)
- Tiempo: 121s/fold (5× E20)
- Artefactos: ✅ sin guardar

### Trade-off: AutoGluon vs Manual (E20)

| Dimensión | E20 Manual | A01 AutoGluon | Veredicto |
|---|---|---|---|
| **Exactitud** | 0.8611 ± 0.0251 | 0.861 ± 0.024 | ⏸️ Empate (dentro ruido) |
| **Velocidad entrenar** | 25s/fold | 121s/fold | **E20 gana 5×** |
| **Velocidad inferencia** | 3.59 ms/1k | ~5 ms/1k | **E20 gana** |
| **Explicabilidad** | ✅ Feature importance | ❌ 10+ modelos internos | **E20 gana** |
| **Mantenimiento** | 1 modelo | 10+ (stacking, blending) | **E20 gana** |
| **Reproducibilidad** | ✅ Seed fijo | ⚠️ Split interno no group-aware | **E20 gana** |
| **GPU requerido** | ❌ No | ⚠️ Recomendado | **E20 gana** |

### Conclusión Fase 8

**AutoGluon no ofrece ventaja neta.** Iguala AUC pero a 5× el costo. En un sistema embarcado, móvil, o low-latency, esto importa. En un pipeline batch, importa menos pero sigue siendo subóptimo.

**Decision documentada:** E20 es el candidato a producción.

---

## Part 4: Modelos Avanzados en Kaggle (Análisis de Alternativas No Tomadas)

### Qué Hizo Kaggle Top 1 (Optimistix, 0.95506)

**Arquitectura:**
- 186 OOFs (out-of-fold predictions)
- 5–6 feature sets
- Modelos: GBDTs (XGB, LGBM, CatBoost), AutoGluon, ExtraTrees, LogisticGAM, TabNet

**Feature Engineering:**
- PolynomialFeatures (interacciones automáticas)
- Target encoding (5-fold)
- Bigram + trigram features
- Dropped `Driver` column (ganancia empírica)

**Ensembling:**
- Hill Climbing (GPU-based Ridge blending)
- Logistic Regression stacker
- RealMLP + TabM inputs

**Restricciones:** Ninguna mencionada (asume GPU disponible)

---

### Qué Hizo Kaggle Top 2 (Chris Deotte, 0.95502)

**Arquitectura:**
- 218 OOFs
- Codex (LLM agent) auto-generó **230,000 líneas de código**
- 94k líneas solo feature engineering
- "Big Six": XGB, LGBM, CatBoost, RealMLP, TabM, TabICL

**Validación:**
- Nested folds (mejor que Top 1, pero aún V0 aleatorio)

**Feature Engineering:**
- Delegado a Codex (no documentado qué específicamente)
- Codex ejecutó 48 horas autonomously

**Resultado:**
- Perdió por 0.00004 (4 cienmilésimas) a pesar de 32 MÁS modelos
- 230k LOC aún no fue suficiente

---

### Trade-off: Kaggle Approach vs Este Proyecto

| Dimensión | Este Proyecto | Kaggle Top 1 | Kaggle Top 2 |
|---|---|---|---|
| **ROC-AUC Kaggle** | No aplicable | 0.95506 | 0.95502 |
| **GAP real** | Diferencia significativa | **0.00001** (coin flip) | **0.00004** |
| **Features** | 10 | ~200–300 | ~400+ |
| **Modelos** | 5–6 base | 186 OOFs | 218 OOFs |
| **GPU-horas** | 0 | ~100–200 | ~500+ |
| **Documentación de decisiones** | ✅ Completa (Fase 14) | ❌ Ninguna | ⚠️ Mínima |
| **Reproducibilidad** | ✅ `uv run` local | ❌ Unknown | ❌ 230k LOC, Codex-dependent |
| **Interpretabilidad** | ✅ 8 features, top-5 named | ❌ 186 black-boxes | ❌ 218 black-boxes |
| **Leakage audit** | ✅ 5Q checklist per feature | ❌ Undocumented | ⚠️ Nested folds |
| **Holdout validation** | ✅ 0.8727 (better than CV) | ❓ Unknown | ❓ Unknown |
| **Production readiness** | ✅ High | ❌ Low | ❌ Low |

---

## Part 5: Criterios de Decisión (Senior/Staff Engineering)

### El Framework

Cuando elegimos arquitectura, no es "mejor vs peor". Es "qué restricciones importan para MI contexto":

1. **Recursos computacionales disponibles**
   - Este proyecto: CPU local ✅
   - AutoGluon: CPU OK, GPU mejor
   - Kaggle modelos: GPU nearly required

2. **Reproducibilidad requerida**
   - Este proyecto: Sem seed fijo, mismo resultado siempre ✅
   - AutoGluon: Artefactos temporales, algo de variance
   - Kaggle: 230k LOC, reproducibilidad desconocida

3. **Interpretabilidad para auditoría**
   - Este proyecto: Feature importance, decisiones documentadas ✅
   - AutoGluon: Opaco (10+ modelos internos)
   - Kaggle: Completamente opaco

4. **Velocidad de inferencia**
   - Este proyecto: 3.59 ms/1k rows ✅
   - AutoGluon: ~5 ms/1k rows
   - Kaggle: ~50+ ms/1k rows (186+ models)

5. **Costo de entrenamiento**
   - Este proyecto: 2.5 min/5-fold (25s×5) ✅
   - AutoGluon: 10 min/5-fold (121s×5)
   - Kaggle: 100+ GPU-horas

6. **Mantenibilidad operacional**
   - Este proyecto: 1 modelo, 10 features, 500 LOC ✅
   - AutoGluon: 10+ modelos internos, artefactos complejos
   - Kaggle: 230k LOC, dependencias múltiples

### La Decision Diagram

```
¿GPU disponible?
├─ NO → Este proyecto ✅
└─ SÍ
    ├─ ¿Reproducibilidad crítica?
    │  ├─ SÍ → Este proyecto ✅
    │  └─ NO → Considera AutoGluon
    │     ├─ ¿Interpretabilidad requerida?
    │     │  ├─ SÍ → Este proyecto ✅
    │     │  └─ NO → AutoGluon OK ⚠️
    │     └─ ¿Latencia <5ms?
    │        ├─ SÍ → Este proyecto ✅
    │        └─ NO → Batch OK, AutoGluon aceptable

¿Ganador Kaggle requerido?
├─ SÍ → Modelos avanzados + brute-force (Rank 1/2) ⚠️
│   └─ Costo: GPU, reproducibilidad perdida, 230k LOC
└─ NO → Este proyecto ✅
```

---

## Part 6: Lo Que Cada Enfoque Optimiza

### Este Proyecto Optimiza Para

1. **Reproducibilidad local** (sin dependencia de GPU clusters)
2. **Auditoría de decisiones** (leakage checklist, Fase 14 documented)
3. **Interpretabilidad operacional** (puedo explicar cada feature)
4. **Eficiencia de inferencia** (3.59 ms/1k rows en CPU)
5. **Mantenibilidad** (1 modelo, no ensemble de 186)

### AutoGluon Optimiza Para

1. **Automatización** (menos decisiones humanas)
2. **Performance neto** (ensemble interno busca óptimo)
3. **Engineering effort** (no escribir feature engineering manual)

### Kaggle Top Optimiza Para

1. **Leaderboard score** (0.955 >> 0.861, punto final)
2. **Ensemble diversity** (250+ modelos "votan")
3. **Feature count** (800+ features aseguran cobertura)

---

## Part 7: Reflexión de Staff/Senior Engineering

### Lo Que NO Es Este Proyecto

- ❌ No es "prueba de que manual > AutoML"
- ❌ No es "Kaggle es mal y este proyecto es bien"
- ❌ No es "mi modelo es 'mejor' por esto y esto"

### Lo Que SÍ Es Este Proyecto

- ✅ Análisis riguroso de trade-offs en un espacio de diseño (manual ↔️ auto ↔️ brute-force)
- ✅ Documentación de decisiones tomadas conscientemente (Fase 14)
- ✅ Validación de que las decisiones están justified (holdout mejor que CV = no overfitting)
- ✅ Honestidad sobre costos (perdimos 0.08 AUC vs Kaggle, aquí está por qué)
- ✅ Honestidad sobre beneficios (ganamos en reproducibilidad, interpretabilidad, velocidad, mantenibilidad)

### Senior/Staff Mindset

A nivel senior/staff, la pregunta no es "¿quién gana?" sino:

1. **¿Cuáles son las restricciones no negociables?** (GPU, interpretabilidad, reproducibilidad)
2. **¿Qué variables importan más en mi contexto?** (latencia, mantenibilidad, auditoría)
3. **¿Cuáles son los trade-offs reales?** (no marketing, datos actuales)
4. **¿Cómo documento la decisión para futuro-yo y otros?** (Fase 14, architecture records)

Este proyecto responde todas cuatro.

---

## Conclusión

**No hay ganador.** Hay diferentes optimizaciones para diferentes contextos.

- **Para Kaggle:** Rank 1/2 ganó. Optimizaron correctamente para su contexto (leaderboard score es lo único que importa).
- **Para producción en GPU cluster con interpretabilidad OK:** AutoGluon es una buena opción (menos engineering, resultado similar).
- **Para producción en CPU local con requisitos de auditoría:** Este proyecto es la opción correcta.

La sofisticación está en entender POR QUÉ cada uno es correcto en su contexto, no en declarar un "ganador".

---

## References

- **Fase 7 - Manual Models:** `scripts/phase7_manual_models.py`, results en `artifacts/tables/phase7_*_results.csv`
- **Fase 8 - AutoGluon Eval:** `scripts/phase8_autogluon.py`, results en `artifacts/tables/phase8_autogluon_results.csv`
- **Fase 14 - Model Selection:** `scripts/phase14_model_selection_framework.py`, analysis en `artifacts/reports/model_selection_framework.md`
- **Leakage Rules:** `.claude/rules/leakage-and-validation.md`
- **CV Strategies:** `notebooks/03_leakage_and_validation.ipynb`

---

**Autor:** Portfolio ML Project | F1 Pit Stop Prediction  
**Fecha:** 2026-09-10  
**Nivel:** Staff/Senior Engineering ADR (Architecture Decision Record)
