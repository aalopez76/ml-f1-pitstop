# Proyecto_1 — F1 Pit Stop Prediction — Alcance y Prioridad

Fecha de decision: 25 de agosto de 2026.
Documento base: `01_F1_Pit_Stop_ML_Project_Spec.txt` (mismo directorio).

## Prioridad asignada: TIER 1 — FLAGSHIP

Este es el proyecto que se ejecuta **primero** y **completo**, siguiendo el spec
original sin recortes. Es la pieza de portafolio que debe demostrar el nivel
maximo de rigor de los tres.

## Por que este es el flagship

- La habilidad central que demuestra — deteccion y prevencion de leakage
  temporal/de grupo (group-aware CV, `shift(1)` antes de rolling, holdout
  congelado antes del tuning) — es la mas transferible y la mas valorada en
  una entrevista tecnica real. Es un error comun; mostrar que se sabe
  prevenir con tests es una senal fuerte de seniority.
- El dominio (F1) es visualmente atractivo y facil de explicar a un no
  tecnico en la primera linea del README, lo que ayuda en la etapa de
  screening de reclutadores.
- La pregunta de portafolio (manual ML vs AutoML, coste vs beneficio) es
  clara y se presta a una figura de resumen fuerte
  (`06_manual_vs_automl_tradeoff.png`).

## Alcance: SIN RECORTES

Ejecutar el spec completo tal como esta escrito, incluyendo:

- Fase 0 (smoke test) a Fase 13 (holdout final y Kaggle), todas las fases.
- skrub, skore, skops, MLflow y AutoGluon Tabular, todos incluidos.
- Matriz de experimentos completa (E00–E21, A00–A01, F00–F01).
- Las 6 figuras minimas especificadas en la seccion 21.
- Definition of Done completo (seccion 22).

Unica libertad operativa: si el smoke test de Fase 0 revela una
incompatibilidad real entre AutoGluon y el resto del stack, seguir el
protocolo ya descrito en el spec (documentar y separar entornos) en vez de
forzar una solucion unica. Esto no es un recorte de alcance, es parte del
spec original.

## Orden sugerido de ejecucion

Ejecutar este proyecto **antes** que Proyecto_3 y Proyecto_2. Sirve como
banco de pruebas del stack completo (skrub/skore/skops/MLflow/AutoGluon);
las lecciones de compatibilidad y friccion que se aprendan aqui se
reutilizan al planificar los recortes de los otros dos proyectos.

## Nota de consistencia entre los tres proyectos

Los tres proyectos comparten stack y arquitectura de repo. Si durante la
ejecucion de este proyecto se descubre que alguna herramienta (skore en
particular) requiere ajustes no previstos en el spec, documentar la
solucion aqui mismo o en el README del proyecto para poder replicarla (o
evitarla a proposito) en Proyecto_2 y Proyecto_3.
