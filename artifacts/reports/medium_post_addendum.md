# Addendum para el Medium post — "How I Built an F1 Pit Stop Predictor"

> Instrucciones de uso: pegar esta seccion completa al FINAL del post
> existente (https://medium.com/@aalpzp/how-i-built-an-f1-pit-stop-predictor-andtl-dr-68c3fe2b1cb6),
> antes del cierre/bio del autor si tiene uno. Esta escrito en ingles para
> mantener consistencia con el resto del post. No requiere cambiar nada
> del contenido anterior.

---

## Postscript: What the Actual Kaggle Winners Taught Me (And Why They'd Fail in Production)

A few weeks after publishing this, I did something I probably should have done before writing a single line of code: I read the actual solution writeups from Ranks 1, 2, 11, and 17 of the real Kaggle competition this dataset is modeled after ([Playground Series S6E5](https://www.kaggle.com/competitions/playground-series-s6e5/)).

Here's what I found that shocked me.

### The Leaderboard Reality Check

**Rank 1 vs Rank 2:**
- Rank 1: 186 OOFs, beat Rank 2 by **+0.00001** (one hundred-thousandth)
- Rank 2: 218 OOFs, Codex-generated 230,000 lines of code over 48 hours, STILL lost by 0.00004

This margin is literally a coin flip. The person with *more* compute, *more* models, and AI-generated code still lost. Yet Rank 1 provides no explanation for *when or why* they stopped optimizing.

**Rank 11 vs Rank 17:**
- Rank 11 (Shiv Satyam): 250 OOFs, 5-6 feature sets, scored 0.952
- Rank 17 (Ravi & Arun): 130+ OOFs, >800 features, scored 0.953
- Gap: 0.001 (one thousandth)

At this scale, more models = more noise, not more signal. Yet neither competitor documents a stopping rule.

### The Leakage Problem Nobody Talks About

I dug into the feature engineering details from Ranks 11 and 17.

Rank 17's writeup explicitly lists features used:
- ✅ "Driver details for past years" (OK)
- ✅ "Details for past races" (OK)
- 🚨 **"Driver details for ALL years, ignored leakage"** (direct quote)
- 🚨 **"Details for FUTURE year races, ignored leakage"** (direct quote)

Ranking 17 reached their 0.953 score *knowing* their features encode information from the future. They "ignored leakage" because **on a leaderboard, leakage is a feature, not a bug.**

In production? That model would fail immediately when deployed to actual future data.

**My project's holdout improved by −0.0116** (better on unseen data than on CV). Their gap is unknown, but typical for leakage-based approaches is CV >> holdout (overfitting to the leaderboard).

### The Real Stopping Problem

Neither Rank 1, 2, 11, or 17 answers: **when do you decide you're done?**

The answer in leaderboard competitions: when the deadline arrives.

The answer in production ML: when the next model adds less value than its cost.

I documented this decision. They didn't. And I think that's worth more than 0.08 AUC points.

### What I Actually Learned

Winning a Kaggle leaderboard and building ML the right way are not the same game.

- Leaderboard game: more models, more features, leakage tolerance → higher score
- Production game: reproducible decisions, leakage audit, stopping rule → deployable model

My project chose the second path deliberately. So I added one more phase to prove it.

### Fase 14: Turning "when do I stop" into a real experiment

I picked the three algorithms both Kaggle winners leaned on heavily —
XGBoost, CatBoost, LightGBM — and threw them at my exact same feature set
(the 10 leakage-vetted columns from Fase 6) and my exact same
group-aware CV. No individual tuning. No cherry-picking. Just: does a
default version of a "bigger" algorithm beat my already-tuned
HistGradientBoosting?

| Model | ROC-AUC (CV) | Fit time/fold |
|---|---|---|
| **HistGradientBoosting (E20, tuned)** | **0.8611 ± 0.0250** | 4.4s |
| XGBoost (default) | 0.8590 ± 0.0229 | 2.2s |
| CatBoost (default) | 0.8606 ± 0.0216 | **153.6s** |
| LightGBM (default) | 0.8593 ± 0.0272 | 1.5s |
| Ensemble of all 4 (logistic stacking) | 0.8615 ± 0.0246 | — |

None of the three new algorithms beat my incumbent. The ensemble edged
ahead by +0.0004 — a number *sixty times smaller* than its own standard
deviation. Statistically, that's not a win. It's noise wearing a
costume.

I also tried two new engineered features — a 5-lap rolling average and a
recent pit-stop rate — the same way I validated everything else in this
project: leakage checklist first, adversarial test second, then measure
the actual effect. One made things dramatically worse (-0.039, the same
instability that killed my 3-lap rolling average back in Fase 6). The
other moved the needle by +0.0002 — again, noise.

**So I shipped nothing new.** The model from Fase 7 is still the model in
production.

### Why "I tried and it didn't help" is the actual result

Here's the part that surprised me: writing this down felt more valuable
than any of the individual experiments. Not because the numbers were
exciting — they weren't — but because now there's a document
(`model_selection_framework.md` in the repo) that says, with receipts:
*"here's what was tried, here's what it cost, here's why it wasn't
adopted."* That document is the actual deliverable of this phase, not
the (nonexistent) accuracy gain.

I even found a case where my own project's rules almost tripped me up. I
was tempted to double-check whether my model's feature importances stay
stable on the frozen holdout set — seemed harmless, "just diagnostics."
Except my own validation rules (written back in Fase 3, long before this
temptation existed) say the holdout gets touched *exactly once*, for the
one confirmatory evaluation in Fase 13. Running one more analysis on it —
even a read-only one — would break that rule. So I didn't do it, and
wrote down why. That's a more useful sentence for a portfolio than any
score.

### The Real Story: Context Matters

I'm not claiming my 0.87 ROC-AUC beats their 0.955. They won the game they chose to play. The real question is: **which game were we playing?**

**Their game (leaderboard):**
- Optimize for: Score
- Resources: GPU cluster, 48+ hours
- Constraints: None (leakage OK, reproducibility optional)
- Result: 0.955 ✅ Optimal for their context

**My game (production ML):**
- Optimize for: Interpretability + reproducibility + auditability
- Resources: Local CPU, documented decisions
- Constraints: No GPU, no leakage, auditable features
- Result: 0.8727 ✅ Optimal for my context

**The gap isn't "I was worse." It's "I chose different constraints."**

For a leaderboard? They were smarter. For a system that someone else has to maintain in production? I was smarter.

Staff/Senior engineers don't ask "who won." They ask: "**What constraints matter in my context, and did I optimize correctly for them?**"

This project documented that choice. That's worth more than any score.

---

*Full results, methodology, and code for Fase 14 are in the
[GitHub repo](https://github.com/aalopez76/ml-f1-pitstop), under
`artifacts/reports/model_selection_framework.md`.*
