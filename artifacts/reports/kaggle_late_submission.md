# Kaggle Late Submission — Traceability Record

**Status:** prepared, **not yet submitted**.

This record exists because the project previously compared its internal holdout
score against publicly reported leaderboard scores as if the two were
commensurable. They are not, and worse, no submission had ever been made — so
there was no measurement of this model under the competition protocol at all.
`kaggle competitions submissions -c playground-series-s6e5` returns
*"No submissions found"*, confirming it.

The purpose of a late submission is **not** to show that the internal holdout
score "corresponds to" some leaderboard number. It is to obtain one additional
observation under a different evaluation protocol, recorded as such.

---

## Why the record is split in two

`retraining_after_score_observation: none` cannot honestly be asserted *before*
the score has been observed. The record therefore has two blocks, written at
two different moments:

```
freeze -> hash -> submit -> observe -> no reaction
```

Block A fixes what was sent and proves the artifact was frozen beforehand.
Block B is appended only after the score comes back.

---

## Block A — before submission

| Field | Value |
|---|---|
| `frozen_model` | `E20_hist_gradient_boosting` |
| `model_artifact` | `models/sklearn/e20_final.skops` |
| `model_sha256` | `a86a1b0e02ed9aee8e768c52c2dc3383b866bd10474683a6fe3e22c23395a44f` |
| `submission_file` | `artifacts/submission.csv` (188,165 rows + header) |
| `submission_sha256` | `32c7f12e76f25bc38488d560aaf615492ac5ffe9200d980f62c744c0a1471df8` |
| `submission_type` | post-hoc late submission |
| `model_frozen_before_submission` | yes — model serialized 2026-09-01 (Phase 12), predictions generated in Phase 13, both long before this record |
| `repository_commit` | *(fill at submission time — see below)* |
| `submission_timestamp` | *(fill at submission time)* |

## Block B — after the score is observed

| Field | Value |
|---|---|
| `score_observed_at` | *(pending)* |
| `late_submission_score` | *(pending)* |
| `retraining_after_score_observation` | *(pending — must be `none`)* |

---

## How to complete this record

Run from the repository root:

```bash
# 1. Verify the file is the one hashed above
sha256sum artifacts/submission.csv
# expected: 32c7f12e76f25bc38488d560aaf615492ac5ffe9200d980f62c744c0a1471df8

# 2. Record the commit you are submitting from
git rev-parse HEAD

# 3. Submit
uv run kaggle competitions submit \
  -c playground-series-s6e5 \
  -f artifacts/submission.csv \
  -m "late submission, frozen model E20_hist_gradient_boosting"

# 4. Read the score back
uv run kaggle competitions submissions -c playground-series-s6e5
```

Then fill Blocks A and B above.

---

## How the score must be reported

Wherever it appears:

> The frozen model was subsequently evaluated through a Kaggle late submission.
> This score is reported as an additional external reference point, not as a
> directly comparable estimate of the group-aware holdout performance. No model
> selection or retraining was performed after observing it.

**What must not be written:** that the difference between this score and the
internal holdout "measures how much of the score depends on the protocol". The
two numbers differ for several reasons at once — train/test distribution,
group structure, class prevalence, artifacts of the synthetic test set, and the
protocol itself — and this project has no experiment that separates them.

---

## The three quantities, kept apart

| Quantity | Value | Protocol |
|---|---|---|
| Group-aware CV | 0.8611 +/- 0.0251 | 5 folds over `dev`, grouped by `(Race, Year)` |
| Reserved internal holdout | 0.8727 | `Year == 2025`, evaluated exactly once (Phase 13) |
| Kaggle late submission | *(pending)* | Competition test set, row-level split |

They come from different evaluation protocols and are **never subtracted from
one another**, nor arranged in a single ranked row.

---

## Compatibility with project rules

Observing this score does not violate `CLAUDE.md` rule 8 or
`.claude/rules/leakage-and-validation.md` §8. Those rules prohibit *reacting*
to a leaderboard score — retraining or reselecting after seeing it — not
observing one with the model already frozen. The one-touch internal holdout
policy is likewise untouched: this submission uses the Kaggle test set, not the
reserved holdout.

If the score comes back well below the internal holdout, the default hypothesis
remains H4 of the spec — drift between the synthetic and original datasets —
and **not** a newly discovered bug to be fixed by adjusting the pipeline.
Reacting to it is precisely what the rule forbids.
