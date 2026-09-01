"""Helpers de MLflow (Fase 4+).

Convenciones obligatorias en `.claude/rules/experiment-tracking.md`:
experimento `f1_pitstop`, nombres de run estables (`E00_dummy`, ...),
tags obligatorios, metricas obligatorias, `log_models=False` para runs
preliminares.

Backend sqlite (no filesystem, deprecado en MLflow 3.15 — ver hallazgo de
Fase 0 en `HANDOFF.md`). `mlruns/` esta en `.gitignore`, nunca se
versiona.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn

EXPERIMENT_NAME = "f1_pitstop"
DEFAULT_TRACKING_DIR = Path("mlruns")

REQUIRED_TAGS = ("project", "stage", "model_family", "feature_set", "validation", "seed")


def setup_mlflow(tracking_dir: Path | str = DEFAULT_TRACKING_DIR) -> None:
    """Configura tracking URI (sqlite) y el experimento `f1_pitstop`.
    Llamar una vez al inicio de cada script de entrenamiento."""
    tracking_dir = Path(tracking_dir)
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(tracking_dir / 'mlflow.db').resolve().as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_run(
    run_name: str,
    tags: dict,
    params: dict,
    metrics: dict,
    log_models: bool = False,
    model=None,
) -> str:
    """Loguea un run con los tags/metricas obligatorios de
    `.claude/rules/experiment-tracking.md`. Levanta `ValueError` si falta
    algun tag obligatorio (evita runs sin trazabilidad minima).

    Retorna el `run_id`.
    """
    missing = [t for t in REQUIRED_TAGS if t not in tags]
    if missing:
        raise ValueError(f"Faltan tags obligatorios: {missing}")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if log_models and model is not None:
            mlflow.sklearn.log_model(model, name="model")
        return run.info.run_id
