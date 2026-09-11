# Addendum para el Medium post — "How I Built an F1 Pit Stop Predictor"

> Instrucciones de uso: pegar esta seccion completa al FINAL del post
> existente (https://medium.com/@aalpzp/how-i-built-an-f1-pit-stop-predictor-andtl-dr-68c3fe2b1cb6),
> antes del cierre/bio del autor si tiene uno. Esta escrito en ingles para
> mantener consistencia con el resto del post.
>
> **Regla editorial (revisada 2026-09-10):** no se nombra a participantes
> concretos de la competicion ni se les atribuyen conductas metodologicas a
> partir de lo que sus write-ups no dicen. Se habla de "publicly reported
> scores" y "published write-ups report...". Se distingue siempre
> *not reported* de *not performed*. Y nunca se resta el score publico del
> leaderboard contra la CV o el holdout interno: son protocolos distintos.

---

## Postscript: I Audited My Own Project, and It Failed

A few weeks after publishing this, I did something uncomfortable: I went back
and audited my own write-up against my own code.

It did not survive intact.

### Three claims I had to retract

**1. "My holdout beat my CV, so there is no overfitting."**

My holdout scored 0.8727 against a cross-validation mean of 0.8611 +/- 0.0251.
I read that as proof the model generalizes.

It is not proof of anything of the sort. A holdout result landing slightly
above the CV mean simply means it fell inside the variability I had already
measured. The honest sentence is: *the holdout result falls within the
variability observed during group-aware CV, providing no evidence of a
material generalization gap.* That is a weaker claim, and it is the one the
data supports.

**2. "The ensemble's gain was sixty times smaller than the noise."**

I had compared a +0.0004 improvement against the 0.0246 standard deviation of
scores across folds, and concluded the gain was negligible.

Wrong comparison. The spread of a model's scores across folds is not the
uncertainty of the *difference* between two models. Because every model ran on
the same folds, I could have computed the paired difference fold by fold —
which is the comparison that actually answers the question. I had the data. I
was throwing it away at save time, persisting only means and standard
deviations.

**3. The one that actually stings: my ensemble evaluation was contaminated.**

My stacking ensemble generated out-of-fold predictions across five folds, then
evaluated the meta-model using *the same five folds*. When fold 1 served as
validation for the stacker, the stacker had been trained on rows whose
meta-features came from base models that had seen fold 1.

That is leakage between stacking levels. The evaluation was never nested.

The worst part: my code contained a docstring explaining that reusing the same
seed meant there was *no* leakage. The reasoning was exactly backwards. Using
the same folds is what creates the contamination, not what prevents it.

### What I did about it

I did not build a nested stacking pipeline to rescue the number. The ensemble
had shown a +0.0004 delta; investing more compute to measure it properly would
have been optimizing something I had already decided not to ship.

Instead I reclassified it. The ensemble is now marked in the results file
itself — not just in a document someone has to remember to read — as
`exploratory_contaminated`, with a column stating its evaluation is not valid
grounds for promotion. The observed +0.0004 cannot be interpreted as an
estimate of incremental performance at all.

I also found the gap in my own process that let this through. I had been
deciding whether each improvement was "big enough" *after* seeing it. That is
how you end up building the rule to fit the result. There was no pre-specified
threshold, and I am not going to pretend retroactively that there was. The
project now says so explicitly, and a formal acceptance policy — thresholds
declared and committed to git *before* each experiment runs — applies going
forward, not backwards.

### On the competition scores

I had also written a comparison between my holdout score and publicly reported
leaderboard scores, complete with a tidy decomposition of the gap.

I deleted all of it, for a simple reason: I had never submitted. There was no
score for my model under the competition protocol, so there was no comparison
to make. The decomposition I had written summed neatly to the gap because I had
constructed it to sum neatly. Nothing measured it.

I have since made a late submission, and I report the number as what it is: an
external reference point from a different evaluation protocol. Three quantities
now live in the project — group-aware CV, a reserved internal holdout evaluated
once, and that late-submission score — and none of them is ever subtracted from
another.

As for what the top solutions did: published write-ups describe ensembles on
the order of 200 to 250 out-of-fold prediction streams, and some describe
cross-year or future-aware aggregate features that would not satisfy the
temporal information constraints I adopted here. That contrast is genuinely
interesting — it is a contrast between objective functions and validation
protocols. It is not a contrast between people, and my earlier draft treated it
as one. Absence of a procedure from a write-up is not evidence the procedure
was absent.

### What I would keep

The part of this project I am most confident about is not the score. It is a
rule I wrote in Phase 3 that cost me something in Phase 14.

I wanted to check whether my model's feature importances stayed stable on the
frozen holdout. It felt harmless — read-only, purely diagnostic. But my own
rule said the holdout gets touched exactly once, and running permutation
importance on it would consume information from the reserved evaluation set.
Not training leakage by itself, but it would introduce post-selection feedback
and break a policy I had committed to before the temptation existed.

So I did not do it, and wrote down why.

That is the thing worth showing. Not a model that scores well, but a process
that caught its own errors and a rule that held when it was inconvenient.

---

*Full audit trail, corrected documents and code are in the
[GitHub repo](https://github.com/aalopez76/ml-f1-pitstop): see
`artifacts/reports/modeling_strategy_decision_record.md` and
`artifacts/reports/model_selection_framework.md`.*
