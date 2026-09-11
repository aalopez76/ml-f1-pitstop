# GitHub Reading Map

El repositorio no necesita una narrativa nueva — ya es la fuente primaria de rigor del proyecto. Este documento es el mapa de lectura: en qué orden un revisor técnico debería recorrer los documentos existentes para que la historia completa tenga sentido sin depender del blog o de Medium.

## El arco, en orden

```
README.md
  El arco completo, Fase 0 a Fase 14, con cada tabla de resultados
  reproducible desde su script correspondiente.
    │
    ├──▶ artifacts/reports/modeling_strategy_decision_record.md
    │      "¿Qué clase de solución, bajo qué restricciones, y qué se pierde?"
    │      V0 vs V1 vs V2 de validación · manual vs AutoML · alternativas
    │      de mayor capacidad consideradas y no perseguidas, con la razón.
    │
    ├──▶ artifacts/reports/model_selection_framework.md
    │      "¿Hay evidencia para reemplazar al modelo en uso?"
    │      Incumbent Challenge Evaluation · deltas pareados por fold ·
    │      Challenger Acceptance Policy (G0-G6) prospectiva desde Fase 15.
    │
    └──▶ artifacts/reports/explainability_report.md
           "¿Por qué el modelo predice lo que predice?"
           Permutation importance + SHAP, validados de forma cruzada ·
           dos ejemplos locales, uno anclado a un fallo real conocido.
```

## Por qué este orden y no otro

Un lector que abre el repo sin haber leído nada más necesita, en secuencia:

1. **Qué se construyó y con qué resultados** (README) — antes de evaluar cualquier decisión, hace falta ver el mapa completo del trabajo.
2. **Por qué se construyó así** (decision record) — las restricciones (CPU, reproducibilidad, auditabilidad, mantenimiento por una persona) son la premisa de cada elección; sin ellas, cualquier decisión parece arbitraria.
3. **Cómo se decide si algo debería reemplazar lo ya construido** (model selection framework) — un proyecto que se congela en su primera decisión no demuestra proceso; este documento muestra qué pasaría si apareciera un candidato mejor, y por qué ninguno de los evaluados lo fue.
4. **Cómo se explica lo que finalmente se usa** (explainability report) — el cierre natural: un modelo sin explicación auditable no está listo para que nadie confíe en él, sin importar cuán bueno sea su ROC-AUC.

## Lo que el repositorio deliberadamente no incluye

Documentos de diseño extensos sobre despliegue, serving o monitoreo en producción. Esa pieza vive como una sección breve al final del decision record ("Next Steps Toward Production") — el proyecto responde una pregunta de rigor de modelado, no de operación en producción; esa segunda pregunta es el objeto de otro proyecto del portafolio.

## Trazabilidad

Cada tabla de resultados en `README.md` es reproducible desde el script que la generó (`scripts/phase<N>_*.py`), y cada decisión de leakage está sujeta al checklist de 5 preguntas en `.claude/rules/leakage-and-validation.md`. Nada de lo narrado aquí ni en los tres documentos enlazados es una afirmación sin evidencia versionada en el repositorio.
