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

El holdout final NUNCA se pasa como `tuning_data` a AutoGluon.

**Ambiguedad resuelta (2026-08-31, confirmada con el usuario antes de
empezar Fase 8):** "el MISMO holdout externo que el modelo manual" NO
significa el holdout final congelado (`Year==2025`) — ese es el mismo
`holdout` de la seccion 3 de este archivo, que la regla no negociable 6
de CLAUDE.md dice que NUNCA se usa para decisiones de modelado, solo
para la evaluacion confirmatoria de la Fase 13 (consistente con
`.claude/rules/experiment-tracking.md`, que restringe
`holdout_roc_auc`/`holdout_pr_auc` a runs finalistas). Decidir en Fase 8
si AutoGluon supera al modelo manual usando el holdout final SERIA una
decision de modelado y violaria esa regla.

Significa, en cambio, el mismo protocolo de evaluacion externa que ya
uso el modelo manual en las Fases 4-7: **CV V1 sobre `dev`** (nunca V0).
AutoGluon en Fase 8 se compara contra el modelo manual con el mismo CV
V1, sin tocar el holdout congelado. El holdout final se evalua UNA
unica vez en la Fase 13, sobre AMBOS finalistas (`F00_final_sklearn` y
`F01_final_autogluon`) por igual — ahi si es el mismo holdout externo
para los dos.

## 8. Kaggle score vs holdout interno

Si el score publico de Kaggle es mucho mas bajo que el holdout interno, la
hipotesis por defecto es drift entre el dataset sintetico y el original
(H4 del spec), no un bug de leakage recien descubierto que haya que
"arreglar" reajustando todo el pipeline despues de ver el leaderboard. No
se reentrenar ni se modifica el modelo para mejorar el score de Kaggle
despues de haberlo visto — eso es fuga de informacion del propio proceso
de evaluacion.

## 9. Candidatos que surgen DESPUES de la Fase 13

La Fase 13 es la evaluacion confirmatoria del holdout — "una unica vez"
(seccion 3) significa una unica vez en la vida del proyecto, no una vez
por candidato ni una vez por fase. Si una fase posterior (p.ej. Fase 14+)
descubre o construye un candidato nuevo que supera a los finalistas
originales en CV, **eso NO reabre el holdout**: ni para seleccionar ese
candidato, ni para verificar cualquier propiedad suya (incluida la
estabilidad de feature importance via permutation importance, que
requiere volver a correr el modelo sobre las filas del holdout).

**Ambiguedad resuelta (2026-09-02, confirmada con el usuario antes de
empezar Fase 14):** esto se aplica incluso cuando el analisis parece "solo
diagnostico" y no "decision de modelado" en sentido estricto — correr
`permutation_importance()` sobre el holdout es una nueva inferencia del
modelo sobre esas filas, y la seccion 3 no distingue entre usar el
holdout para *decidir* y usar el holdout para *medir algo mas*. Ambos
casos violan "se evalua una unica vez".

Consecuencia practica: cualquier fase posterior a la 13 que compare
candidatos nuevos (manuales, AutoML, ensembles) se decide y se cierra
enteramente sobre CV V1 en `dev`, igual que las Fases 4-7. Si el candidato
ganador de esa fase reemplazara al finalista de Fase 13 en un
hipotetico despliegue real, la evaluacion confirmatoria en datos nunca
vistos tendria que venir de un holdout NUEVO (otro corte temporal, otra
temporada), no de reabrir `Year==2025` — eso es trabajo fuera de alcance
de este proyecto de portafolio, no algo que se resuelva "reutilizando" el
holdout existente.
