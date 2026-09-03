"""Feature engineering temporal (Fase 6).

Leer `.claude/rules/leakage-and-validation.md` completo antes de tocar
este archivo (regla no negociable 7 de CLAUDE.md), en particular la
seccion 5 ("regla de oro para rolling/lags": la fila `t` NUNCA puede usar
informacion posterior a `t`; si la variable a predecir es `t+1`, aplicar
`shift(1)` ANTES de cualquier `rolling`) y la seccion 6 (test adversarial
obligatorio).

**Unidad de agrupacion de este modulo:** `(Driver, Race, Year)` — la
secuencia de vueltas de UN auto especifico dentro de UNA carrera. Esto es
DISTINTO de la unidad de agrupacion de `src/f1pitstop/data/split.py`
(`(Race, Year)`, usada para que el CV no vea partes de la misma carrera en
train y validation). Aqui el proposito es reconstruir, para cada fila, el
historial propio de ese auto hasta el instante `t` — el hallazgo de Fase
1/2 de que `Driver` no es un identificador real de piloto NO invalida su
uso como clave de agrupacion de una secuencia de vueltas dentro de una
carrera (sigue siendo la mejor aproximacion disponible a "este auto en
esta carrera").

**Advertencia heredada de Fase 2:** `LapNumber` no es consecutivo dentro
de un grupo (hay huecos, el CSV es un submuestreo de una secuencia
oculta) y `Stint` no es monotono. Las features de este modulo tratan la
fila anterior VISIBLE (ordenada por `LapNumber`) como "la vuelta
anterior", que es una aproximacion, NO la vuelta `t-1` fisica exacta —
sigue siendo estrictamente pasada respecto a `t` (nunca futura), que es
la unica garantia que exige la regla de oro.

**Verificado en Fase 6 (revision del subagente `leakage-auditor`):** ¿un
mismo valor de `Driver` podria corresponder a DOS autos fisicos distintos
dentro de la misma `(Race, Year)`, corrompiendo el orden temporal
reconstruido aqui? Chequeado sobre el dataset completo (train+test): 0
filas con `(Driver, Race, Year, LapNumber)` duplicado, y ningun grupo
`(Driver, Race, Year)` supera 51 filas (muy por debajo de cualquier tope
fisico de vueltas de una carrera real). Ambas senales serian las
esperables si dos autos compartieran un valor de `Driver` (mismo
`LapNumber` repetido, o un grupo con mas filas que vueltas posibles) y
ninguna aparece — evidencia razonable de que `Driver` SI aisla
correctamente un auto dentro de una carrera, aunque (por el hallazgo de
Fase 1) no sea un identificador de piloto real consistente ENTRE carreras.
"""

from __future__ import annotations

import pandas as pd

FEATURE_GROUP_COLS = ("Driver", "Race", "Year")
ORDER_COL = "LapNumber"

# Cap fijo de winsorizing para `LapTime (s)` (Fase 4: la columna cruda tiene
# outliers extremos -- hasta 2507s vs media ~91s/std ~19.8s -- que le
# costaron ~0.075 ROC-AUC al HGB baseline por ser artefactos especificos de
# cada carrera, probablemente vueltas de safety car/bandera roja, que no
# generalizan bajo V1). Se usa una CONSTANTE FIJA (no un percentil derivado
# del dataset) para no violar la pregunta 5 del checklist de leakage
# ("estadisticas globales que deberian computarse dentro de cada fold") —
# el cap no depende de que filas caen en cada fold. mean + 3*std ~ 91+3*19.8
# ~ 150s da margen generoso para vueltas genuinamente lentas (lluvia,
# trafico) mientras excluye las vueltas con parada/incidente incluidos.
LAPTIME_WINSORIZE_CAP_SECONDS = 150.0


def _sorted_by_group(df: pd.DataFrame, group_cols=FEATURE_GROUP_COLS, order_col=ORDER_COL):
    """Ordena por grupo + `LapNumber` con sort estable; retorna el frame
    ordenado (el índice original se preserva para poder reindexar al final)."""
    return df.sort_values(list(group_cols) + [order_col], kind="mergesort")


def add_winsorized_laptime(df: pd.DataFrame) -> pd.DataFrame:
    """`LapTime (s)` recortada a `LAPTIME_WINSORIZE_CAP_SECONDS`. Row-wise,
    no requiere orden ni agrupacion — no hay riesgo de leakage temporal."""
    out = df.copy()
    out["LapTime_s_winsorized"] = out["LapTime (s)"].clip(upper=LAPTIME_WINSORIZE_CAP_SECONDS)
    return out


def add_basic_domain_features(
    df: pd.DataFrame, group_cols=FEATURE_GROUP_COLS, order_col=ORDER_COL
) -> pd.DataFrame:
    """Familia 'basic domain' (E11): usan solo informacion en o antes de `t`,
    sin rolling/shift explicito mas alla de una comparacion con la fila
    anterior VISIBLE.

    - `pit_stops_so_far`: cuenta acumulada de `PitStop` hasta E INCLUYENDO
      `t` (ya establecido como leakage-safe en Fase 3/4 checklist: `PitStop`
      describe el estado de la vuelta actual).
    - `recomputed_stint`: contador de stints reconstruido a partir de
      caidas de `TyreLife` respecto a la fila anterior VISIBLE — version
      leakage-safe ADICIONAL a `Stint` crudo (que Fase 1/2 demostro NO
      monotono en 81.6% de los grupos; peor que `RaceProgress`, excluida
      por un problema similar en Fase 3). Solo compara `t` contra `t-1`
      (pasado), nunca mira adelante. **No reemplaza `Stint` crudo en el
      feature set por defecto** (`Stint` sigue en `E10_RAW_FEATURES`,
      heredado de Fase 4/5, ya evaluado y sin hallazgos bloqueantes por el
      subagente `leakage-auditor` en esas fases) — el ablation de Fase 6
      prueba si AGREGAR `recomputed_stint` ayuda, no si sustituirlo por
      `Stint` ayuda mas; esa comparacion queda pendiente para Fase 7
      (tuning/seleccion de features).
    """
    out = _sorted_by_group(df, group_cols, order_col)
    grp = out.groupby(list(group_cols), sort=False)

    out["pit_stops_so_far"] = grp["PitStop"].cumsum()

    tyre_prev = grp["TyreLife"].shift(1)
    tyre_reset = (out["TyreLife"] < tyre_prev).fillna(False).astype(int)
    out["_tyre_reset"] = tyre_reset
    out["recomputed_stint"] = out.groupby(list(group_cols))["_tyre_reset"].cumsum() + 1
    out = out.drop(columns=["_tyre_reset"])

    return out.reindex(df.index)


def add_temporal_features(
    df: pd.DataFrame, group_cols=FEATURE_GROUP_COLS, order_col=ORDER_COL
) -> pd.DataFrame:
    """Familia 'temporal' (E12): rolling/lag causales, `shift(1)` SIEMPRE
    antes de cualquier `rolling` (regla de oro, seccion 5 de
    leakage-and-validation.md). Requiere `LapTime_s_winsorized` — llamar
    `add_winsorized_laptime()` primero si no esta presente.

    - `laptime_delta_prev`: `LapTime_s_winsorized[t] - LapTime_s_winsorized[t-1]`
      (fila anterior VISIBLE). NaN en la primera fila de cada grupo (no
      hay `t-1`) — es un NaN esperado, no un bug.
    - `laptime_roll_mean_3`: media movil de las hasta 3 vueltas anteriores
      VISIBLES, con `shift(1)` aplicado ANTES del `rolling` — este es
      exactamente el ejemplo trabajado del spec (Fase 6): para la fila
      `t`, usa `LapTime_s_winsorized` de `t-1`, `t-2`, `t-3`, nunca de `t`
      en adelante. Test adversarial de 5 vueltas en `tests/test_features.py`.
    - `laps_since_last_pit`: vueltas transcurridas desde el ULTIMO pit
      stop ANTERIOR a `t` (no cuenta el propio `PitStop` de `t`). 0 en la
      primera vuelta despues de un pit (o en la primera vuelta del grupo
      si nunca hubo pit).
    """
    if "LapTime_s_winsorized" not in df.columns:
        raise ValueError(
            "Falta 'LapTime_s_winsorized'; llamar add_winsorized_laptime() primero."
        )

    out = _sorted_by_group(df, group_cols, order_col)
    grp_cols = list(group_cols)

    laptime_prev = out.groupby(grp_cols)["LapTime_s_winsorized"].shift(1)
    out["laptime_delta_prev"] = out["LapTime_s_winsorized"] - laptime_prev
    out["laptime_roll_mean_3"] = out.groupby(grp_cols)["LapTime_s_winsorized"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )

    prior_pit_cumsum = out.groupby(grp_cols)["PitStop"].transform(
        lambda s: s.shift(1).fillna(0).cumsum()
    )
    out["_prior_pit_cumsum"] = prior_pit_cumsum
    out["laps_since_last_pit"] = out.groupby(grp_cols + ["_prior_pit_cumsum"]).cumcount()
    out = out.drop(columns=["_prior_pit_cumsum"])

    return out.reindex(df.index)


BASIC_DOMAIN_FEATURE_NAMES = ["pit_stops_so_far", "recomputed_stint"]

# `laptime_roll_mean_3` (rolling mean causal, EL ejemplo trabajado del spec
# para esta fase) se sigue calculando y testeando (ver tests/test_features.py,
# test adversarial obligatorio) porque es el caso de referencia de la "regla
# de oro" de rolling — pero un ablation por-feature reproducible
# (`scripts/phase6_feature_isolation.py`, resultado en
# `artifacts/tables/feature_isolation_results.csv`) mostro que agregarla
# SOLA le cuesta ~0.057 ROC-AUC al HGB (0.815 -> 0.757), incluso ya
# calculada sobre `LapTime_s_winsorized`. Hereda la misma inestabilidad
# de `LapTime (s)` documentada en Fase 4 (`models/baselines.py`,
# `UNSTABLE_FEATURES`) — el winsorizing a 150s no alcanza a arreglarlo. Se
# excluye del feature set "temporal" validado, siguiendo la misma logica
# aplicada a `LapTime (s)` en Fase 4: no es leakage (no usa `t+1` ni el
# target), es una feature demasiado ruidosa/inestable entre carreras bajo
# V1 para generalizar.
UNSTABLE_TEMPORAL_FEATURE_NAMES = ["laptime_roll_mean_3"]

# Features temporales validadas por ablation individual (todas mejoran el
# HGB baseline por si solas: laptime_delta_prev +0.005, laps_since_last_pit
# +0.024 ROC-AUC sobre E10).
TEMPORAL_FEATURE_NAMES = ["laptime_delta_prev", "laps_since_last_pit"]


def add_phase14_candidate_features(
    df: pd.DataFrame, group_cols=FEATURE_GROUP_COLS, order_col=ORDER_COL
) -> pd.DataFrame:
    """Familia "candidatas Fase 14" (Tier 2 del framework de seleccion de
    modelos, ver `artifacts/reports/model_selection_framework.md`): 2
    features exploratorias, cada una sujeta al mismo ablation individual
    reproducible que ya se aplico a la familia "temporal" en Fase 6 — no
    se asume que ayudan solo por pasar el checklist de leakage.
    Requiere `LapTime_s_winsorized` y `pit_stops_so_far` ya calculados
    (llamar `add_winsorized_laptime()` y `add_basic_domain_features()`
    primero).

    - `laptime_roll_mean_5`: igual que `laptime_roll_mean_3` (Fase 6,
      excluida por inestable, ver `UNSTABLE_TEMPORAL_FEATURE_NAMES`) pero
      con ventana de 5 vueltas en vez de 3 — hipotesis: una ventana mas
      ancha promedia mas ruido de vuelta a vuelta y podria ser menos
      inestable entre carreras bajo V1. `shift(1)` SIEMPRE antes del
      `rolling` (regla de oro, seccion 5 de leakage-and-validation.md):
      para la fila `t` usa `LapTime_s_winsorized` de `t-1` .. `t-5`, nunca
      de `t` en adelante.
    - `pit_stops_rate_last3`: tasa de pit stops en las hasta 3 vueltas
      VISIBLES anteriores a `t` (`shift(1)` antes de `rolling(3).mean()`
      sobre `PitStop`, que ya es 0/1 leakage-safe segun el checklist de
      Fase 3 — su media movil hereda esa misma garantia). NaN en la
      primera vuelta VISIBLE de cada grupo (no hay `t-1`), igual criterio
      que `laptime_delta_prev` — es un NaN esperado, no un bug; los
      candidatos que no soportan NaN nativo (LogisticRegression,
      ExtraTrees) lo imputan con mediana dentro del Pipeline, igual que ya
      hacen con `laptime_delta_prev`.
    """
    if "LapTime_s_winsorized" not in df.columns:
        raise ValueError(
            "Falta 'LapTime_s_winsorized'; llamar add_winsorized_laptime() primero."
        )
    if "pit_stops_so_far" not in df.columns:
        raise ValueError(
            "Falta 'pit_stops_so_far'; llamar add_basic_domain_features() primero."
        )

    out = _sorted_by_group(df, group_cols, order_col)
    grp_cols = list(group_cols)

    out["laptime_roll_mean_5"] = out.groupby(grp_cols)["LapTime_s_winsorized"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    out["pit_stops_rate_last3"] = out.groupby(grp_cols)["PitStop"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )

    return out.reindex(df.index)


PHASE14_CANDIDATE_FEATURE_NAMES = ["laptime_roll_mean_5", "pit_stops_rate_last3"]
