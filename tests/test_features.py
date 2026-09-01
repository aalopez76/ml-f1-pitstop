"""Tests de src/f1pitstop/features/ (Fase 6).

Incluye el test adversarial obligatorio del spec (Fase 6, y
`.claude/rules/leakage-and-validation.md` seccion 6): un DataFrame toy de
5 vueltas donde se verifica explicitamente que el valor de rolling
calculado para la vuelta 3 NO incluye `lap_time` de la vuelta 4 ni de la 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pitstop.features.build import (
    E10_RAW_FEATURES,
    E11_BASIC_DOMAIN_FEATURES,
    E12_TEMPORAL_FEATURES,
    E13_FULL_LEAKAGE_SAFE_FEATURES,
    build_engineered_frame,
    prepare_X_for_feature_set,
)
from f1pitstop.features.temporal import (
    LAPTIME_WINSORIZE_CAP_SECONDS,
    add_basic_domain_features,
    add_temporal_features,
    add_winsorized_laptime,
)


def _five_lap_toy() -> pd.DataFrame:
    """5 vueltas de UN solo auto en UNA carrera, `LapTime (s)` elegido a
    proposito para que el test adversarial sea inequivoco: si la rolling
    de la vuelta 3 incluyera la vuelta 4 o 5 (valores muy altos, 999),
    el resultado seria muy distinto al esperado."""
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "Driver": ["D1"] * 5,
            "Race": ["Test GP"] * 5,
            "Year": [2024] * 5,
            "LapNumber": [1, 2, 3, 4, 5],
            "LapTime (s)": [90.0, 92.0, 88.0, 999.0, 999.0],
            "TyreLife": [1.0, 2.0, 3.0, 1.0, 2.0],
            "Stint": [1, 1, 1, 2, 2],
            "Position": [5, 5, 4, 4, 3],
            "PitStop": [0, 0, 0, 1, 0],
            "Compound": ["SOFT", "SOFT", "SOFT", "MEDIUM", "MEDIUM"],
        }
    )


def test_adversarial_rolling_does_not_use_future_laps():
    """El test obligatorio del spec: la rolling mean de la vuelta 3 (indice
    2, LapNumber=3) debe usar SOLO LapTime de las vueltas 1 y 2 (90, 92),
    nunca de la vuelta 4 (999) ni la 5 (999)."""
    df = _five_lap_toy()
    out = add_winsorized_laptime(df)
    out = add_temporal_features(out)

    lap3_row = out[out["LapNumber"] == 3].iloc[0]
    expected_mean = np.mean([90.0, 92.0])  # solo vueltas 1 y 2 (shift(1) antes de rolling)
    assert lap3_row["laptime_roll_mean_3"] == pytest.approx(expected_mean)

    # Ademas: el valor de la vuelta 3 no debe acercarse en absoluto a 999
    # (si el bug de mirar al futuro existiera, el promedio se dispararia)
    assert lap3_row["laptime_roll_mean_3"] < 200


def test_adversarial_rolling_lap4_uses_up_to_lap3_only():
    """La vuelta 4 (indice 3) debe usar vueltas 1,2,3 (90,92,88), nunca su
    propio LapTime (999, que ademas fue winsorizado a 150 antes)."""
    df = _five_lap_toy()
    out = add_winsorized_laptime(df)
    out = add_temporal_features(out)

    lap4_row = out[out["LapNumber"] == 4].iloc[0]
    expected_mean = np.mean([90.0, 92.0, 88.0])
    assert lap4_row["laptime_roll_mean_3"] == pytest.approx(expected_mean)


def test_laptime_delta_prev_first_lap_is_nan():
    df = _five_lap_toy()
    out = add_winsorized_laptime(df)
    out = add_temporal_features(out)
    lap1_row = out[out["LapNumber"] == 1].iloc[0]
    assert pd.isna(lap1_row["laptime_delta_prev"])


def test_laptime_delta_prev_matches_manual_diff():
    df = _five_lap_toy()
    out = add_winsorized_laptime(df)
    out = add_temporal_features(out)
    lap2_row = out[out["LapNumber"] == 2].iloc[0]
    # 92 - 90 = 2
    assert lap2_row["laptime_delta_prev"] == pytest.approx(2.0)


def test_winsorize_caps_extreme_laptimes():
    df = _five_lap_toy()
    out = add_winsorized_laptime(df)
    assert (out["LapTime_s_winsorized"] <= LAPTIME_WINSORIZE_CAP_SECONDS).all()
    # las vueltas normales (90, 92, 88) no se tocan
    assert out.loc[out["LapNumber"] == 1, "LapTime_s_winsorized"].iloc[0] == 90.0
    # las vueltas extremas (999) quedan exactamente en el cap
    assert out.loc[out["LapNumber"] == 4, "LapTime_s_winsorized"].iloc[0] == (
        LAPTIME_WINSORIZE_CAP_SECONDS
    )


def test_recomputed_stint_increments_on_tyre_reset():
    df = _five_lap_toy()
    out = add_basic_domain_features(df)
    # TyreLife: [1,2,3,1,2] -> reset entre vuelta 3 (TyreLife=3) y vuelta 4 (TyreLife=1)
    stints = out.sort_values("LapNumber")["recomputed_stint"].tolist()
    assert stints == [1, 1, 1, 2, 2]


def test_pit_stops_so_far_is_cumulative_including_current_row():
    df = _five_lap_toy()
    out = add_basic_domain_features(df)
    out = out.sort_values("LapNumber")
    # PitStop: [0,0,0,1,0] -> acumulado incluyendo la fila actual: [0,0,0,1,1]
    assert out["pit_stops_so_far"].tolist() == [0, 0, 0, 1, 1]


def test_laps_since_last_pit_resets_after_pit_and_does_not_count_current_stop():
    df = _five_lap_toy()
    out = add_winsorized_laptime(df)
    out = add_temporal_features(out)
    out = out.sort_values("LapNumber")
    # vuelta 4 tiene PitStop=1, pero "laps_since_last_pit" en la vuelta 4
    # mide vueltas desde el pit ANTERIOR (no hubo ninguno) -> sigue contando
    # normal (3, cuarta vuelta = indice 3 desde el inicio del grupo);
    # la vuelta 5 (justo despues del pit de la vuelta 4) debe resetear a 0.
    lap5_value = out[out["LapNumber"] == 5]["laps_since_last_pit"].iloc[0]
    assert lap5_value == 0


def test_add_basic_domain_features_preserves_row_count_and_index():
    df = _five_lap_toy()
    out = add_basic_domain_features(df)
    assert len(out) == len(df)
    assert list(out.index) == list(df.index)


def test_add_temporal_features_preserves_row_count_and_index():
    df = _five_lap_toy()
    out = add_winsorized_laptime(df)
    out = add_temporal_features(out)
    assert len(out) == len(df)
    assert list(out.index) == list(df.index)


def test_add_temporal_features_raises_without_winsorized_column():
    df = _five_lap_toy()
    with pytest.raises(ValueError):
        add_temporal_features(df)


def test_no_infinite_values_in_engineered_columns():
    df = _five_lap_toy()
    engineered = build_engineered_frame(df)
    new_cols = [
        "LapTime_s_winsorized",
        "pit_stops_so_far",
        "recomputed_stint",
        "laptime_delta_prev",
        "laptime_roll_mean_3",
        "laps_since_last_pit",
    ]
    for c in new_cols:
        assert not np.isinf(engineered[c].to_numpy(dtype=float, na_value=0.0)).any()


def test_build_engineered_frame_preserves_original_row_order():
    df = _five_lap_toy()
    engineered = build_engineered_frame(df)
    assert engineered["id"].tolist() == df["id"].tolist()
    assert engineered["LapNumber"].tolist() == df["LapNumber"].tolist()


def test_prepare_X_for_feature_set_e10_matches_raw_feature_list():
    df = _five_lap_toy()
    engineered = build_engineered_frame(df)
    X = prepare_X_for_feature_set(engineered, "E10_raw_features")
    assert list(X.columns) == E10_RAW_FEATURES


def test_prepare_X_for_feature_set_e13_has_all_families():
    df = _five_lap_toy()
    engineered = build_engineered_frame(df)
    X = prepare_X_for_feature_set(engineered, "E13_full_leakage_safe_features")
    assert list(X.columns) == E13_FULL_LEAKAGE_SAFE_FEATURES
    assert "recomputed_stint" in X.columns
    assert "laps_since_last_pit" in X.columns
    # laptime_roll_mean_3 se calcula (test adversarial la cubre aparte) pero
    # se excluye del feature set por defecto tras el ablation de Fase 6
    # (ver UNSTABLE_TEMPORAL_FEATURE_NAMES en features/temporal.py)
    assert "laptime_roll_mean_3" not in X.columns


def test_prepare_X_for_feature_set_unknown_name_raises():
    df = _five_lap_toy()
    engineered = build_engineered_frame(df)
    with pytest.raises(ValueError):
        prepare_X_for_feature_set(engineered, "not_a_real_feature_set")


def test_e11_and_e12_are_disjoint_additions_over_e10():
    """Ablation limpia: E11 solo agrega la familia basic-domain, E12 solo
    la familia temporal — ninguna repite columnas de la otra familia."""
    e11_extra = set(E11_BASIC_DOMAIN_FEATURES) - set(E10_RAW_FEATURES)
    e12_extra = set(E12_TEMPORAL_FEATURES) - set(E10_RAW_FEATURES)
    assert e11_extra.isdisjoint(e12_extra)


def _two_driver_interleaved_toy() -> pd.DataFrame:
    """Dos autos (D1, D2) en la MISMA carrera, con filas intercaladas en el
    DataFrame de entrada (no vienen ya agrupadas/ordenadas por Driver).
    Verifica que `groupby` no mezcla el historial de un auto con el del
    otro solo porque sus vueltas quedan intercaladas por `LapNumber` o por
    orden de fila. D1 tiene LapTime muy bajo (50s) y D2 muy alto (500s,
    winsorizado a 150s) — si el agrupamiento fallara y tratara todo como
    una sola secuencia, la rolling de D1 quedaria contaminada por los
    valores de D2."""
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "Driver": ["D1", "D2", "D1", "D2", "D1", "D2"],
            "Race": ["Test GP"] * 6,
            "Year": [2024] * 6,
            "LapNumber": [1, 1, 2, 2, 3, 3],
            "LapTime (s)": [50.0, 500.0, 51.0, 500.0, 52.0, 500.0],
            "TyreLife": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            "Stint": [1, 1, 1, 1, 1, 1],
            "Position": [1, 2, 1, 2, 1, 2],
            "PitStop": [0, 0, 0, 0, 0, 0],
            "Compound": ["SOFT"] * 6,
        }
    )


def test_groupby_does_not_mix_two_drivers_interleaved_in_the_same_race():
    df = _two_driver_interleaved_toy()
    out = add_winsorized_laptime(df)
    out = add_temporal_features(out)

    d1_lap3 = out[(out["Driver"] == "D1") & (out["LapNumber"] == 3)].iloc[0]
    d2_lap3 = out[(out["Driver"] == "D2") & (out["LapNumber"] == 3)].iloc[0]

    # D1: rolling de la vuelta 3 debe promediar SOLO sus propias vueltas 1-2
    # (50, 51), nunca los valores de D2 (500, winsorizados a 150)
    assert d1_lap3["laptime_roll_mean_3"] == pytest.approx(np.mean([50.0, 51.0]))
    assert d1_lap3["laptime_roll_mean_3"] < 100

    # D2: idem, solo sus propias vueltas 1-2 (500 winsorizado a 150 ambas)
    assert d2_lap3["laptime_roll_mean_3"] == pytest.approx(LAPTIME_WINSORIZE_CAP_SECONDS)


def test_groupby_does_not_mix_two_drivers_recomputed_stint():
    df = _two_driver_interleaved_toy()
    out = add_basic_domain_features(df)
    # ambos autos tienen TyreLife estrictamente creciente (sin reset) ->
    # recomputed_stint debe quedar en 1 para las 3 vueltas de CADA auto,
    # nunca incrementarse por "ver" el TyreLife del otro auto intercalado
    for driver in ["D1", "D2"]:
        stints = out[out["Driver"] == driver].sort_values("LapNumber")["recomputed_stint"]
        assert (stints == 1).all()
