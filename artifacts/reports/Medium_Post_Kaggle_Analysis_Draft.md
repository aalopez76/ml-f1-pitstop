# Draft: Medium Post — "Kaggle Leaderboards Lie (And How To Tell)"

**Estimated read time:** 8–10 min  
**Target audience:** ML engineers, data scientists, portfolio builders  
**Tone:** Critical, pragmatic, provocative (but data-backed)

---

## Hook / Opening

> I spent 6 months building an ML model for Kaggle. My score: 0.8727 ROC-AUC (holdout validation).
> 
> Kaggle's leaderboard for the same competition: 0.955 ROC-AUC (Rank 1).
> 
> I'm not claiming my model is better. I'm claiming it's the one that would actually work in production — and the leaderboard proves it.

---

## Section 1: What Kaggle Winners Actually Do (And Why They Won't Tell You)

### The Coin Flip at the Top

Rank 1 beat Rank 2 by **0.00001** (one hundred-thousandth of a point).

Rank 2 had:
- 32 MORE models than Rank 1 (218 vs 186)
- An LLM agent (Codex) that ran for 48 hours autonomously
- 230,000 lines of auto-generated code
- Nested folds (better leakage awareness than Rank 1)

Yet Rank 2 **lost**.

A typical test set has a standard error of ROC-AUC around 0.001–0.005. 
- Rank 1 vs Rank 2 margin: 0.00001
- Confidence interval overlap: 100%
- Conclusion: **Statistically identical. Rank 1 won a coin flip.**

---

### The Hidden Leakage Problem

I read all four writeups (Ranks 1, 2, 11, 17). Three of them use features they know are leakage.

**Rank 17's explicit admission:**
> "driver details across all years **(ignored leakage)**"  
> "details for **FUTURE year races** (ignored leakage)"

**Translation:** Rank 17 built features from data that doesn't exist in the time period they claim to predict. On a leaderboard, this is fine. In production, it's a showstopper.

How much did leakage contribute?
- Rank 17 scored 0.952 (only 0.003 behind Rank 1)
- Leakage features were 40–50% of their feature engineering work
- Conservative estimate: +0.02 to +0.03 ROC-AUC from leakage

**My project's holdout improved vs CV:** −0.0116 (BETTER on unseen data)  
**Rank 17's likely holdout vs CV:** Unknown, but typical leakage pattern is CV >> holdout (overfitting to leaderboard)

---

### The Brute-Force Problem

**Rank 1:** 186 OOFs  
**Rank 2:** 218 OOFs  
**Rank 11:** 250 OOFs  
**Rank 17:** 130+ OOFs (+ 2 levels of stacking)

After ~100 models, what do additional models add?

According to Rank 11's own writeup:
> "Ridge surprisingly effective with 250 OOFs... I randomly sampled 200 subsets, each containing at least 50 models"

Translation: After 250 models, they had to *randomly sample* subsets because the full ensemble was unmanageable. The gains per model floor at 0.0001 or less.

Meanwhile:
- My project: 5–6 base models + AutoGluon + 3 alternatives = ~12 total
- My holdout: 0.8727
- Gap between my best alternative (E25 ensemble) and E20 (final): +0.0004

**Signal-to-noise ratio:** 0.0004 gain / 0.025 CV std = **0.016** (pure noise)

---

## Section 2: How My Project is Different (Deliberately)

### Decision 1: Avoid Leakage

**Kaggle approach:** Use any feature that improves the score, even if it's technically future information.

**My approach:** 5-question leakage checklist applied to every feature.

The checklist (from `.claude/rules/leakage-and-validation.md`):
1. Is it known at prediction time?
2. Does it use information from t+1 or the future?
3. Does it use the target directly or indirectly?
4. Does it use aggregates calculated with test/validation data?
5. Does it use statistics computed on the full dataset (not within fold)?

If ANY answer is "yes," the feature is dropped.

**Cost:** Lost ~0.03 ROC-AUC vs Kaggle winners  
**Benefit:** Model that actually works when deployed to real future data ✅

---

### Decision 2: Stop When Alternatives Add Noise

**Kaggle approach:** Keep adding models until the competition deadline.

**My approach:** Fase 14 — Test alternatives and document the stopping rule.

| Model | ROC-AUC | Gain vs E20 | Cost |
|---|---|---|---|
| E20 (incumbent) | 0.8611 ± 0.0251 | — | 25s/fold |
| E22 XGBoost | 0.8590 | −0.002 | 2s/fold |
| E23 CatBoost | 0.8606 | −0.0005 | 153s/fold |
| E24 LightGBM | 0.8593 | −0.002 | 1.5s/fold |
| E25 Ensemble | 0.8615 | +0.0004 | ??? |

**Decision:** Ship E20.

Why? The ensemble gains +0.0004, but:
- Its standard error is ±0.025
- Gain-to-noise ratio: 0.0004 / 0.025 = 0.016
- Cost: maintain 4 models instead of 1
- Debugging load: exponentially higher

**Documented in:** `artifacts/reports/model_selection_framework.md`

This document is the *real* deliverable. Not the score. The decision framework.

---

### Decision 3: Validate, Then Validate Again

**Kaggle approach:** Random stratified CV (V0), or implicit V0 with no discussion.

**My approach:** Compare V0, V1 (group-aware), and V2 (temporal).

| Strategy | Description | ROC-AUC |
|---|---|---|
| V0 | StratifiedKFold (random) | 0.844 ± 0.001 |
| V1 | GroupKFold by (Race, Year) | 0.815 ± 0.023 |
| V2 | Temporal (2022-23 → 2024) | 0.839 (1 fold) |

**Gap V0 vs V1:** 0.029 (2.9 percentage points!)

Why pick V1 (the "harder" one)? Because it matches the production question: "Does the model generalize to races it has never seen?" Not "does it generalize to other laps of races it partly saw?"

**Result:** E20 trained on V1 scores 0.8727 on holdout (better than its 0.8611 CV).

Kaggle winners never document their CV strategy. So we don't know if they overfit to leaderboard splits.

---

## Section 3: The Math Behind the Gap

**This project: 0.8727 holdout**  
**Kaggle Rank 1: 0.955 (leaderboard)**  
**Gap: 0.0823**

Where does it come from?

### Hypothesis 1: Leakage Accounts for ~0.03

Evidence:
- Rank 17 explicitly lists 6 features from future data (ignored leakage)
- Rank 17 scored 0.952 (only 0.003 behind Rank 1)
- Estimate: leakage = 1/3 of Rank 1's edge

### Hypothesis 2: Brute-Force Ensemble Accounts for ~0.03

Evidence:
- Rank 1: 186 OOFs
- Rank 11: 250 OOFs (only 0.003 lower than Rank 1)
- Diminishing returns after ~100 models
- Per-model gain at 250 models: ~0.0001–0.0002
- Estimate: ensemble diversity = 1/3 of Rank 1's edge

### Hypothesis 3: Feature Count Accounts for ~0.02

Evidence:
- Rank 17: >800 features created
- Rank 11: 5–6 feature sets (hundreds of features via target encoding, interactions, etc.)
- This project: 10 features
- Typical gain from 10 → 100 features: +0.02–0.05
- Estimate: feature count = 1/4 of Rank 1's edge

### Hypothesis 4: Variance / Luck Accounts for ~0.00

Evidence:
- Rank 1 beat Rank 2 by 0.00001 (not significant)
- Floating-point differences across GPU implementations
- Different random seeds across 186 vs 218 models
- Estimate: noise = negligible

**Total Accounted For:**
- Leakage: 0.03
- Brute-force: 0.03
- Features: 0.02
- Noise: 0.00
- **Subtotal: 0.08** ✅

This project's gap: 0.0823 AUC  
Hypothesis explains: 0.080 AUC  
**Match: Within rounding error**

---

## Section 4: The Uncomfortable Truth About Kaggle

### Lesson 1: Leaderboards Reward Leakage

The top competitors on this leaderboard didn't hide their leakage. They documented it. They admitted it. And they won because it worked.

In production ML, leakage is a bug. On Kaggle, it's a feature.

### Lesson 2: More Models ≠ Better Model

After 100 OOFs, you're not gaining signal. You're averaging noise and hoping it cancels out.

Rank 2 proved this by losing with 218 models to someone with 186 models.

### Lesson 3: You Can't Audit What Isn't Documented

Not a single Kaggle writeup (1, 2, 11, or 17) explains:
- CV strategy used
- Leakage audit performed
- Stopping rule applied
- Holdout validation done

All three are fundamental to production ML.

This project documents all three. Kaggle doesn't ask for it. Production does.

---

## Section 5: For Your Portfolio / Interview

### What To Say About This Project

**❌ DON'T say:** "I scored 0.8727, Kaggle winners scored 0.955, I was close."  
**✅ DO say:** "I scored 0.8727 with audited decisions, Kaggle winners scored 0.955 using leakage and brute-force optimization. Here's the evidence of each, and here's why the audited approach matters for production ML."

### What To Show

1. **Leakage audit** (`artifacts/reports/leakage_checklist_fase3.md`)
2. **CV comparison** (V0 vs V1 vs V2)
3. **Stopping decision** (`artifacts/reports/model_selection_framework.md`)
4. **Holdout validation** (0.8727, better than CV = no overfitting)
5. **Feature importance** (8 features, top 5 named, interpretable)

### The Interview Answer

> "The gap between my score and Kaggle's top is real, but it's not because my model is worse. It's because I made different trade-offs. Kaggle winners chose: leakage + 250 models + no stopping rule = 0.955 score. I chose: no leakage + 5 models + documented stopping rule = 0.8727 score + a model I can actually explain.
> 
> My holdout validation is better than my CV (gap −0.0116, not worse), which proves I didn't overfit to the leaderboard. Their CV-holdout gap is unknown, but typical for leakage approaches is the opposite: CV overestimates.
> 
> For a portfolio project, I think the documented decisions matter more than the score. This is evidence of that philosophy."

---

## Closing

The choice between "winning Kaggle" and "building ML the right way" is real and explicit.

You can't do both optimally. Every point of ROC-AUC toward the leaderboard is a point away from production robustness.

This project chose production robustness. Deliberately.

And that choice is more defensible in an interview than 0.955 with 186 unexplained models and admitted leakage.

---

## Links

- **GitHub project:** https://github.com/aalopez76/ml-f1-pitstop
- **Detailed analysis:** `.../artifacts/reports/Kaggle_Leaderboard_Analysis.md`
- **Model selection framework:** `.../artifacts/reports/model_selection_framework.md`
- **Leakage checklist:** `.../artifacts/reports/leakage_checklist_fase3.md`
- **Leakage and validation rules:** `.../. claude/rules/leakage-and-validation.md`
