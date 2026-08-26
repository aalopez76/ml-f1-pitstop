---
name: new-experiment
description: Scaffolds a new experiment (Exx/Axx/Fxx) for the F1 Pit Stop project with consistent naming, MLflow tags, and script structure. Use whenever starting a new entry from the experiment matrix in the project spec (section 20).
---

# Nuevo experimento — F1 Pit Stop

Args esperados: un ID de experimento y un slug corto, por ejemplo
`/new-experiment E04 skrub_tabular_pipeline_logreg`.

## Pasos

1. Confirma que el ID sigue la convencion del spec (seccion 20): `E` para
   modelos manuales/preprocessing/features, `A` para AutoGluon, `F` para
   los finalistas. Si el ID no aparece en la matriz del spec, pregunta al
   usuario si es un experimento nuevo no planeado y por que se justifica
   (regla del proyecto: ninguna herramienta o experimento entra sin una
   pregunta experimental clara).

2. Antes de escribir codigo nuevo, lee `.claude/rules/experiment-tracking.md`
   para los tags y metricas obligatorias, y `.claude/rules/leakage-and-validation.md`
   si el experimento toca features o splits.

3. Crea (o localiza si ya existe) el script correspondiente en `scripts/`
   siguiendo el patron ya usado por experimentos anteriores del mismo tipo
   (`train_baselines.py`, `train_manual.py`, `train_autogluon.py`).

4. El run de MLflow debe:
   - usar el experimento `f1_pitstop`;
   - nombrarse exactamente `<ID>_<slug>` (ej. `E04_skrub_tabular_pipeline_logreg`);
   - registrar los tags obligatorios: `project`, `stage`, `model_family`,
     `feature_set`, `validation`, `seed`;
   - registrar las metricas obligatorias: `cv_roc_auc_mean`, `cv_roc_auc_std`,
     `fit_seconds`, y `holdout_roc_auc`/`holdout_pr_auc` solo si es un
     experimento finalista (F0x).

5. Actualiza la tabla de historial de fases en `HANDOFF.md` si este
   experimento cierra el criterio de salida de una fase.

6. Corre `uv run pytest` antes de dar el experimento por terminado.

No dupliques codigo entre experimentos similares: si dos experimentos
comparten preprocessing, extrae la parte comun a `src/f1pitstop/` en vez de
copiar y pegar entre scripts.
