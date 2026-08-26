# Reglas de validacion y leakage — F1 Pit Stop

Leer completo antes de escribir o modificar: `src/f1pitstop/data/split.py`,
cualquier archivo bajo `src/f1pitstop/features/`, o antes de disenar/ajustar
la estrategia de cross-validation. Este archivo es un checklist operativo,
no un resumen — el detalle narrativo completo esta en el spec, Fases 3, 6,
8 y 13.

## 1. Unidades de dependencia

No asumir nombres de columnas de agrupacion. Antes de elegir la estrategia
de split, inspeccionar si existen columnas equivalentes a `race`, `driver`,
`season`, `event`, `lap`, u otras que impliquen filas relacionadas.

## 2. Estrategias a comparar (minimo)

- V0 = StratifiedKFold aleatorio reproducible (seed fijo).
- V1 = group-aware si existe un identificador de evento/carrera adecuado.
- V2 = holdout temporal/por eventos posteriores si la estructura temporal
  lo permite.

La estrategia mas "dificil" no es automaticamente la correcta: debe
corresponder al uso que se afirma simular (¿el modelo en produccion veria
datos de la misma carrera en train y en inferencia, o no?).

## 3. El holdout final

Se congela ANTES del tuning y no se usa para tomar ninguna decision de
modelado (ni de feature engineering, ni de seleccion de modelo, ni de
threshold). Se evalua una unica vez en la Fase 13.

## 4. Checklist por cada feature engineered

Para cada feature nueva, responder explicitamente estas cinco preguntas
antes de darla por valida:

1. ¿Se conoce en el instante `t` (el momento de la prediccion)?
2. ¿Usa informacion de `t+1` o del futuro respecto a `t`?
3. ¿Usa el target de forma directa o indirecta?
4. ¿Usa agregados calculados con datos de validation/test?
5. ¿Usa estadisticas globales que deberian computarse dentro de cada fold,
   no sobre el dataset completo?

Si la respuesta a 2, 3 o 4 es "si" (o no se puede responder con certeza),
la feature se descarta o se corrige — no se documenta como "limitacion" y
se deja pasar.

## 5. Regla de oro para rolling / lags

La fila `t` NUNCA puede usar informacion posterior a `t`. Si la variable a
predecir es `t+1` (como `PitNextLap`), aplicar `shift(1)` ANTES de calcular
cualquier `rolling`. Ejemplo concreto (ya en el spec, Fase 6): para una
rolling mean de `lap_time` a 3 vueltas prediciendo la vuelta `t+1`, la
feature en la fila `t` debe usar `lap_time` de `t-1`, `t-2`, `t-3` — nunca
de `t`, `t+1` o posteriores.

## 6. Test adversarial obligatorio

`tests/test_features.py` debe incluir un DataFrame toy de 5 vueltas donde
se verifique explicitamente que el valor de rolling calculado para la
vuelta 3 NO incluye `lap_time` de la vuelta 4 ni de la 5. Si este test no
existe todavia para una familia de features nueva, la familia no esta
terminada.

## 7. AutoGluon y el holdout

El holdout final NUNCA se pasa como `tuning_data` a AutoGluon. AutoGluon se
evalua con el MISMO holdout externo que el modelo manual, para que la
comparacion sea justa.

## 8. Kaggle score vs holdout interno

Si el score publico de Kaggle es mucho mas bajo que el holdout interno, la
hipotesis por defecto es drift entre el dataset sintetico y el original
(H4 del spec), no un bug de leakage recien descubierto que haya que
"arreglar" reajustando todo el pipeline despues de ver el leaderboard. No
se reentrenar ni se modifica el modelo para mejorar el score de Kaggle
despues de haberlo visto — eso es fuga de informacion del propio proceso
de evaluacion.
