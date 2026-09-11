"""Fase 9 (extension) — Explicabilidad con SHAP.

Complementa la permutation importance de `phase9_skore_evaluation.py` con
SHAP: importancia global por atribucion aditiva + dos ejemplos locales.

No modifica `phase9_skore_evaluation.py` (evita re-ejecutar permutation
importance, que ya toma varios minutos, solo para iterar en SHAP).

**Limitacion de compatibilidad verificada (no generica):** `shap.TreeExplainer`
(shap==0.50.0) no soporta el manejo nativo de categoricas de
`HistGradientBoostingClassifier` (`categorical_features=[...]`, dtype pandas
`"category"`) — internamente intenta castear todo el array de entrada a
float y falla con `Compound`. El fallback fue verificado con un smoke test
propio: envolver `predict_proba` en una funcion que recibe la matriz
one-hot de `Compound` y la reconstruye a `category` antes de llamar al
modelo real (no un modelo sustituto), dejando que SHAP perturbe un espacio
completamente numerico. Esto usa `shap.Explainer` generico
(`PermutationExplainer`), no `TreeExplainer` — mas lento (~400 ms/fila
medido sobre datos reales), lo que fija el tamano de la muestra global en
500 filas en vez de miles: a este ritmo, 5000 filas tomarian ~33 min por
una ejecucion de un script que no necesita repetirse.

Uso: `uv run python scripts/phase9_shap_explainability.py`

**Runtime esperado:** ~5-7 min (fit de E20 + SHAP global sobre 500 filas +
2 ejemplos locales).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap

from f1pitstop.data.ingest import load_raw
from f1pitstop.data.split import load_frozen_holdout_ids
from f1pitstop.evaluation.explainability import build_onehot_wrapper, stratified_sample
from f1pitstop.features.build import build_engineered_frame, prepare_X_for_feature_set
from f1pitstop.models.baselines import SEED
from f1pitstop.models.manual_models import make_e15_hgb_e13
from f1pitstop.tracking.mlflow_utils import setup_mlflow

TARGET = "PitNextLap"
ARTIFACTS_DIR = Path("artifacts")
GLOBAL_SAMPLE_SIZE = 500

E20_BEST_PARAMS = {
    "learning_rate": 0.127,
    "max_iter": 152,
    "max_leaf_nodes": 38,
    "min_samples_leaf": 35,
    "l2_regularization": 0.84,
}


def main() -> None:
    setup_mlflow()

    print("Cargando datos y features...")
    df_train, _, _, _ = load_raw()
    holdout_ids = load_frozen_holdout_ids()
    df_full = build_engineered_frame(df_train)
    df_dev = df_full[~df_full.index.isin(holdout_ids)]
    X_dev = prepare_X_for_feature_set(df_dev, "E13_full_leakage_safe_features")
    y_dev = df_dev[TARGET]
    print(f"Dev set: {len(X_dev)} filas ({df_dev['Year'].nunique()} anios)")

    print("\nRefiteando E20 sobre dev completo (mismo procedimiento que Fase 9)...")
    model = make_e15_hgb_e13()
    model.set_params(**E20_BEST_PARAMS)
    model.fit(X_dev, y_dev)

    predict_fn, to_onehot, feature_names_onehot = build_onehot_wrapper(model, X_dev)

    # Verificacion de wrapper: predict_fn(to_onehot(X)) debe ser identico a
    # model.predict_proba(X) para no explicar un modelo distinto del real.
    check_sample = X_dev.sample(100, random_state=SEED)
    proba_direct = model.predict_proba(check_sample)[:, 1]
    proba_via_wrapper = predict_fn(to_onehot(check_sample).values)
    assert np.allclose(proba_direct, proba_via_wrapper), (
        "El wrapper one-hot no reconstruye las mismas predicciones que el "
        "modelo original - no se puede confiar en la explicacion resultante."
    )
    print("Wrapper one-hot verificado: predicciones identicas al modelo original.")

    artifacts_dir = ARTIFACTS_DIR / "skore" / "e20_hgb_final"
    figures_dir = ARTIFACTS_DIR / "figures"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # === SHAP global ===
    print(f"\nCalculando SHAP global sobre muestra de {GLOBAL_SAMPLE_SIZE} filas...")
    print("(muestra, no dev completo: PermutationExplainer generico es el unico")
    print(" camino viable con la categorica nativa de HGB en esta version de shap;")
    print(f" ~400ms/fila medido -> {GLOBAL_SAMPLE_SIZE} filas es un compromiso deliberado")
    print(" costo/cobertura, no una limitacion de recursos.)")

    background = to_onehot(X_dev.sample(50, random_state=SEED)).values
    global_sample_df, _ = stratified_sample(X_dev, y_dev, GLOBAL_SAMPLE_SIZE, seed=SEED)
    global_sample_onehot = to_onehot(global_sample_df).values

    explainer = shap.Explainer(predict_fn, background, feature_names=feature_names_onehot)
    shap_values_global = explainer(global_sample_onehot)

    global_importance = pd.DataFrame({
        "feature": feature_names_onehot,
        "mean_abs_shap": np.abs(shap_values_global.values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    global_importance.to_csv(artifacts_dir / "shap_importance.csv", index=False)
    print(f"\nTop 5 features por |SHAP| medio:\n{global_importance.head()}")

    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values_global.values,
        global_sample_onehot,
        feature_names=feature_names_onehot,
        show=False,
    )
    plt.tight_layout()
    fig.savefig(figures_dir / "e20_shap_summary.png", dpi=150)
    plt.close(fig)
    print(f"Guardado: {figures_dir / 'e20_shap_summary.png'}")

    # === SHAP local: dos ejemplos con criterio explicito ===
    print("\nCalculando SHAP local para 2 ejemplos...")

    # Ejemplo 1: prediccion correcta de alta confianza (verdadero positivo).
    proba_dev_sample = model.predict_proba(X_dev)[:, 1]
    correct_high_conf_mask = (y_dev.to_numpy() == 1) & (proba_dev_sample > 0.9)
    idx_correct = X_dev.index[correct_high_conf_mask][0]
    example_correct = X_dev.loc[[idx_correct]]

    # Ejemplo 2: error del modelo en el segmento Year==2023 (drift documentado
    # en Fase 10 - error_analysis, AUC 0.6668 en ese segmento vs 0.8620 global).
    # No arbitrario: ancla la explicabilidad local a un fallo real ya conocido.
    year_2023_mask = df_dev["Year"] == 2023
    proba_2023 = proba_dev_sample[year_2023_mask.to_numpy()]
    y_2023 = y_dev[year_2023_mask].to_numpy()
    X_2023 = X_dev[year_2023_mask]
    false_negative_mask = (y_2023 == 1) & (proba_2023 < 0.3)
    if false_negative_mask.sum() > 0:
        idx_error = X_2023.index[false_negative_mask][0]
    else:
        # si no hay falso negativo tan marcado, usar el error de mayor magnitud
        error_magnitude = np.abs(y_2023 - proba_2023)
        idx_error = X_2023.index[np.argmax(error_magnitude)]
    example_error = X_dev.loc[[idx_error]]

    for name, example_row, description in [
        ("correct", example_correct, "prediccion correcta de alta confianza"),
        ("2023_drift", example_error, "error del modelo en el segmento Year==2023"),
    ]:
        row_onehot = to_onehot(example_row).values
        shap_local = explainer(row_onehot)
        proba_row = predict_fn(row_onehot)[0]
        y_true_row = y_dev.loc[example_row.index[0]]

        print(f"\n[{name}] {description}")
        print(f"  y_true={y_true_row}, predict_proba={proba_row:.4f}")

        fig = plt.figure(figsize=(9, 5))
        shap.plots.waterfall(shap_local[0], show=False, max_display=10)
        plt.title(f"SHAP local ({description})\ny_true={y_true_row}, proba={proba_row:.3f}")
        plt.tight_layout()
        fig.savefig(figures_dir / f"e20_shap_local_example_{name}.png", dpi=150)
        plt.close(fig)
        print(f"  Guardado: {figures_dir / f'e20_shap_local_example_{name}.png'}")

    # === MLflow ===
    with mlflow.start_run(run_name="E20_shap_explainability"):
        mlflow.set_tag("project", "f1_pitstop")
        mlflow.set_tag("stage", "final")
        mlflow.set_tag("model_family", "hist_gradient_boosting")
        mlflow.set_tag("feature_set", "E13_full_leakage_safe_features")
        mlflow.set_tag("seed", str(SEED))
        mlflow.log_param("shap_global_sample_size", GLOBAL_SAMPLE_SIZE)
        mlflow.log_param("shap_explainer", "PermutationExplainer (generic, one-hot wrapper)")
        mlflow.log_artifact(str(artifacts_dir / "shap_importance.csv"))
        for fname in [
            "e20_shap_summary.png",
            "e20_shap_local_example_correct.png",
            "e20_shap_local_example_2023_drift.png",
        ]:
            mlflow.log_artifact(str(figures_dir / fname))

    print("\n" + "=" * 70)
    print("Fase 9 (SHAP) completada")
    print("=" * 70)
    print(f"Artefactos en {artifacts_dir} y {figures_dir}")


if __name__ == "__main__":
    main()
