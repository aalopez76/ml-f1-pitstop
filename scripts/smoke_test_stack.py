"""Fase 0 — Smoke test del stack completo del proyecto F1 Pit Stop.

Objetivo (spec, seccion 5): probar que el entorno real puede importar y
ejecutar una operacion minima con cada libreria del stack ANTES de escribir
la arquitectura completa. Usa exclusivamente datos toy, nunca el dataset
real (que todavia no se descarga hasta la Fase 1).

Criterio de salida: los 10 pasos siguientes funcionan en el mismo entorno.
Solo entonces se genera uv.lock y se hace el primer commit de dependencias.
"""

from __future__ import annotations

import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd


def step(n: int, title: str) -> None:
    print(f"\n[{n}] {title}")
    print("-" * 60)


def main() -> None:
    # 1. Version de Python y de los paquetes del stack.
    step(1, "Version de Python y de los paquetes del stack")
    print("python:", sys.version.replace("\n", " "))
    packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "skrub",
        "skore",
        "skops",
        "mlflow",
        "autogluon.tabular",
    ]
    for pkg in packages:
        try:
            print(f"{pkg}: {version(pkg)}")
        except Exception as exc:  # noqa: BLE001 - smoke test, queremos ver el fallo
            print(f"{pkg}: NO DISPONIBLE ({exc})")

    # 2. DataFrame diminuto y mixto (numerico + categorico + booleano).
    step(2, "DataFrame diminuto y mixto")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "num_feature": rng.normal(size=20),
            "cat_feature": rng.choice(["a", "b", "c"], size=20),
            "bool_feature": rng.choice([True, False], size=20),
        }
    )
    y = pd.Series(rng.choice([0, 1], size=20), name="target")
    print(df.head())
    print("dtypes:\n", df.dtypes)

    # 3-4. skrub.tabular_pipeline(LogisticRegression()) + fit/predict_proba.
    step(3, "skrub.tabular_pipeline(LogisticRegression()) + fit/predict_proba")
    import skrub
    from sklearn.linear_model import LogisticRegression

    pipeline = skrub.tabular_pipeline(LogisticRegression())
    pipeline.fit(df, y)
    proba = pipeline.predict_proba(df)
    print("pipeline:", pipeline)
    print("predict_proba shape:", proba.shape)
    assert proba.shape == (20, 2), "predict_proba deberia tener shape (20, 2)"

    # 5. skore.evaluate(...)
    step(5, "skore.evaluate(...)")
    import skore

    report = skore.evaluate(pipeline, df, y, splitter=0.5)
    print("report:", report)
    metrics = report.metrics.summarize()
    print("metrics summary:\n", metrics.frame())

    # 6. Run local de MLflow con 1 parametro + 1 metrica.
    step(6, "Run local de MLflow (1 parametro + 1 metrica)")
    import mlflow

    # NOTA (hallazgo del smoke test, MLflow 3.15.2): el backend de filesystem
    # ("./mlruns") esta en modo mantenimiento y MLflow lo rechaza salvo que se
    # active MLFLOW_ALLOW_FILE_STORE=true. En vez de usar ese opt-out
    # deprecado, seguimos la recomendacion oficial y usamos un backend
    # SQLite local para el tracking.
    mlflow_dir = Path("mlruns_smoke_test")
    mlflow_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(mlflow_dir / 'mlflow.db').resolve().as_posix()}")
    mlflow.set_experiment("smoke_test")
    with mlflow.start_run(run_name="smoke_test_run") as run:
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("roc_auc_toy", 0.5)
        print("mlflow run_id:", run.info.run_id)
    print(f"tracking uri: {mlflow.get_tracking_uri()} (directorio temporal, no se versiona)")

    # 7. Serializar un Pipeline sklearn pequeno con skops.
    step(7, "Serializar Pipeline sklearn con skops")
    import skops.io as sio
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    small_pipeline = Pipeline(
        [
            ("impute", SimpleImputer()),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression()),
        ]
    )
    X_num = df[["num_feature"]]
    small_pipeline.fit(X_num, y)

    tmp_dir = Path(tempfile.mkdtemp(prefix="skops_smoke_"))
    skops_path = tmp_dir / "small_pipeline.skops"
    sio.dump(small_pipeline, skops_path)
    print("guardado en:", skops_path)

    # 8. Inspeccionar get_untrusted_types().
    step(8, "get_untrusted_types()")
    untrusted = sio.get_untrusted_types(file=skops_path)
    print("tipos no confiables:", untrusted)

    # 9. Cargar solo si los tipos son conocidos/aceptados.
    step(9, "Cargar solo si los tipos son conocidos/aceptados")
    # Allowlist revisada manualmente: tipos numpy/sklearn estandar que skops
    # marca como "no confiables" por defecto pero que son de uso normal y
    # documentado en cualquier pipeline sklearn serializado.
    KNOWN_SAFE_TYPES = {"numpy.dtype"}
    unknown = [t for t in untrusted if t not in KNOWN_SAFE_TYPES]
    if unknown:
        raise RuntimeError(
            f"skops reporta tipos no confiables NO reconocidos: {unknown}. "
            "No se carga el objeto sin revision manual."
        )
    if untrusted:
        print(f"tipos aceptados explicitamente (allowlist conocida): {untrusted}")
    loaded_pipeline = sio.load(skops_path, trusted=untrusted)
    print("cargado OK:", loaded_pipeline)
    assert np.allclose(
        loaded_pipeline.predict_proba(X_num), small_pipeline.predict_proba(X_num)
    ), "el pipeline cargado deberia predecir igual que el original"

    # 10. TabularPredictor de AutoGluon con time_limit muy corto sobre datos toy.
    step(10, "AutoGluon TabularPredictor (time_limit corto, datos toy)")
    from autogluon.tabular import TabularPredictor

    toy_train = df.copy()
    toy_train["target"] = y.values
    predictor = TabularPredictor(
        label="target",
        problem_type="binary",
        eval_metric="roc_auc",
        path=str(tmp_dir / "autogluon_smoke"),
        verbosity=0,
    ).fit(toy_train, time_limit=20, presets="medium_quality")
    ag_proba = predictor.predict_proba(df)
    print("AutoGluon predict_proba shape:", ag_proba.shape)

    print("\n" + "=" * 60)
    print("SMOKE TEST OK: los 10 pasos del stack funcionaron en el mismo entorno.")
    print("=" * 60)


if __name__ == "__main__":
    main()
