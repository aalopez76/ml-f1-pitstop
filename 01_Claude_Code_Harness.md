# Proyecto_1 — Harness de Claude Code: que se preparo y por que

Fecha: 25 de agosto de 2026.
Este documento existe porque el usuario pidio, antes de empezar a
desarrollar cualquiera de los 4 proyectos, dejar preparada la
configuracion de Claude Code (CLAUDE.md, reglas, skills, subagentes,
hooks, MCP, permisos, y la convencion propia de HANDOFF.md) para cada uno
por separado. Este archivo documenta las decisiones tomadas para
Proyecto_1 y por que.

Principio aplicado: el mismo que rige el proyecto de ML en si — ninguna
pieza del harness se anade porque "existe la opcion"; cada una debe
resolver un problema concreto de este proyecto. Se aplico literalmente la
regla del propio spec ("Cada herramienta se incorpora solo si resuelve un
problema verificable") al propio tooling de Claude Code.

## Piezas incluidas y por que

### CLAUDE.md
Justificacion: es la unica pieza que se carga automaticamente en cada
sesion. Sin esto, cada sesion nueva tendria que re-derivar (o peor,
olvidar) las reglas no negociables del spec (no empezar por AutoGluon, no
tocar el holdout, disciplina de leakage), que son criticas y se repiten a
lo largo de 14 fases. Se mantuvo deliberadamente corto (~100 lineas): solo
lo que aplica SIEMPRE. El detalle especifico de cada fase se deja en el
spec original, no se duplica.

### HANDOFF.md (convencion propia, no estandar de Claude Code)
Justificacion: este es un proyecto de 14 fases que se ejecutara a lo largo
de muchas sesiones independientes, sin calendario fijo. Sin un registro
explicito del estado exacto (fase actual, ultima accion, bloqueos), cada
sesion nueva perderia tiempo reconstruyendo el contexto desde el spec y el
historial de git, con riesgo real de repetir trabajo o —peor— de asumir
que un criterio de salida esta cumplido cuando no lo esta. Se actualiza al
cierre de cada sesion.

### .claude/rules/leakage-and-validation.md
Justificacion: el leakage es, segun el propio spec, "la fase central" del
proyecto (Fase 3) y reaparece en las Fases 6, 8 y 13. Es exactamente el
tipo de contenido que conviene NO tener siempre cargado en CLAUDE.md (no
hace falta en fases de EDA o de submission), pero que debe leerse por
completo, sin excepciones, antes de tocar split o feature engineering. Por
eso vive en un archivo aparte que CLAUDE.md senala explicitamente cuando
leer, en vez de inyectarlo siempre.

### .claude/rules/experiment-tracking.md
Justificacion: mismo razonamiento que el anterior mfaplicado a las
convenciones de MLflow/skore (nombres de experimento, tags obligatorios,
division skore/MLflow). Solo relevante desde la Fase 4 en adelante; no
tiene sentido cargarlo en Fase 0-3.

### .claude/agents/leakage-auditor.md (subagente)
Justificacion: el spec pide explicitamente una comparacion "manual vs
AutoML" honesta y un manejo de leakage a prueba de auditoria externa. Un
subagente de solo lectura (sin Edit/Write) que revisa con "ojos frescos"
el codigo de split/features antes de cerrar una fase es una aplicacion
directa y barata de esa exigencia: no reemplaza el juicio del desarrollador,
pero anade una segunda pasada sistematica sin coste de contexto en la
conversacion principal. Se restringio a herramientas de lectura (Read,
Grep, Glob, Bash) a proposito: su trabajo es encontrar y reportar, no
arreglar — así no hay riesgo de que "corrija" silenciosamente algo sin que
quede registrado como decision explicita.

### .claude/skills/new-experiment/SKILL.md
Justificacion: el proyecto define una matriz de ~15 experimentos con una
convencion de nombres estricta (Exx/Axx/Fxx) y tags de MLflow obligatorios.
Sin una skill que lo recuerde, es facil que la nomenclatura se desvie entre
sesiones distintas (mas aun sin calendario fijo, donde puede pasar tiempo
entre una sesion y la siguiente). Es un caso legitimo de "tarea repetida
con estructura fija", que es exactamente lo que una skill esta pensada
para resolver.

### .claude/settings.json — permisos
Justificacion: reduce interrupciones para los comandos no destructivos que
se van a ejecutar decenas de veces (`uv run`, `pytest`, `ruff`, `git
status/diff/log`, `mlflow ui`, scripts en `scripts/`). Se dejo
deliberadamente FUERA del allowlist cualquier variante de `git commit` o
`git push`: seguen exigiendo confirmacion explicita, en linea con la
politica general de no commitear sin que el usuario lo pida. Se anadieron
denies explicitos (`rm -rf`, `git push --force`, `git reset --hard`) como
defensa en profundidad, redundante con la politica general pero util como
documentacion visible del criterio de seguridad para quien audite el repo.

### .claude/hooks/protect_raw_data.py — hook PreToolUse
Justificacion: el Definition of Done del spec exige explicitamente "raw
data no modificado". Confiar en que Claude (o un humano) recuerde esa
regla en todas las sesiones futuras es mas fragil que hacerla cumplir de
forma automatica. El hook intercepta Write/Edit/NotebookEdit y bloquea
cualquier intento de tocar `data/raw/` o `data/external/`, en cualquiera de
las dos convenciones de path (POSIX o Windows). Probado con 4 casos antes
de integrarlo (bloqueo POSIX, bloqueo Windows, ruta segura permitida, y un
falso positivo evitado con `data/raw_backup/`). Alcance reconocido: NO
cubre escrituras hechas via `Bash` (ej. un `cp` o una redireccion de shell
hacia `data/raw/`) — cubrir eso con una regex sobre comandos de shell es
fragil y propenso a falsos positivos/negativos, asi que se dejo fuera a
proposito; esa via queda cubierta por disciplina y por `git status` antes
de cualquier commit.

## Piezas consideradas y descartadas (con motivo)

- **MCP servers**: no hay ningun servicio externo que este proyecto
  necesite consultar en vivo desde Claude Code (Kaggle es una descarga
  puntual via CLI, MLflow UI es local). Anadir un MCP server sin un
  problema concreto violaria el mismo principio de "no herramienta sin
  justificacion" que rige el proyecto de ML.
- **Agent teams**: el proyecto son 14 fases con dependencias secuenciales
  fuertes (no se puede hacer Fase 6 sin que Fase 3 este cerrada). Un equipo
  de subagentes trabajando en paralelo no encaja con esa estructura; anade
  coordinacion sin resolver un cuello de botella real. Se prefirio un unico
  subagente puntual (leakage-auditor) invocado bajo demanda.
- **Plugins**: este es un proyecto de portafolio individual y autocontenido,
  no una pieza de tooling que se vaya a reutilizar entre repos o equipos.
  Empaquetarlo como plugin anadiria una capa de distribucion sin
  consumidor real.
- **Hooks adicionales** (ej. autoformateo con ruff tras cada Edit, o correr
  pytest automaticamente tras cada cambio en `src/`): se considero, pero se
  descarto por ahora porque anadiria latencia a cada edicion sin una
  necesidad demostrada todavia (a diferencia de la proteccion de datos
  crudos, que es una regla explicita del Definition of Done). Si en la
  practica el ritmo de trabajo lo justifica, se puede anadir despues — el
  criterio es el mismo que rige el resto del proyecto: no anticipar
  necesidades hipoteticas.
- **auto-memory (memoria global entre sesiones de Claude)**: las decisiones,
  hallazgos y estado de este proyecto pertenecen a este repo (CLAUDE.md,
  HANDOFF.md, ADRs/README), no a la memoria global del asistente. La
  memoria global se reserva para preferencias del usuario que trasciendan
  este proyecto especifico (ej. como le gusta que se le presenten
  comparaciones de alternativas), no para hechos de este repo.

## Estructura final relevante a Claude Code

```
Proyecto_1/
|-- 01_F1_Pit_Stop_ML_Project_Spec.txt   # spec original (fuente de verdad de fases)
|-- 00_Scope_And_Priority.md             # prioridad frente a los otros 3 proyectos
|-- 01_Claude_Code_Harness.md            # este documento
|-- CLAUDE.md                            # cargado automaticamente cada sesion
|-- HANDOFF.md                           # estado exacto entre sesiones (convencion propia)
`-- .claude/
    |-- settings.json                    # permisos + registro de hooks
    |-- hooks/
    |   `-- protect_raw_data.py          # bloquea escritura en data/raw, data/external
    |-- rules/
    |   |-- leakage-and-validation.md    # leer antes de tocar split/features
    |   `-- experiment-tracking.md       # leer antes de crear un run de MLflow
    |-- agents/
    |   `-- leakage-auditor.md           # subagente de solo lectura, bajo demanda
    `-- skills/
        `-- new-experiment/
            `-- SKILL.md                 # /new-experiment <ID> <slug>
```

Cuando se inicialice el repo real (`git init` + estructura `src/f1pitstop/`
del spec, seccion 4), estos archivos ya quedan en su lugar correcto en la
raiz del repo — no requieren moverse.
