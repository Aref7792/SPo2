# train3.py

Standalone, LOSO-honest analysis pipeline built on top of `train2.py`'s
trained SpO2 estimator. It does not modify how the model is trained —
Step 1 reruns the unchanged 11-fold leave-one-subject-out (LOSO)
protocol from `train2.py` and saves per-window results to CSV; every
step after that is pure post-processing of those saved results (Step 6
is the one exception: it reuses the checkpoints Step 1 already trained
to run additional inference, never retraining).

Each step is independently re-runnable from its saved CSV without
repeating the (expensive) training step. All entry points are listed
under **Re-running a step** below.

---

## Step 1 — Data collection (`run_loso_and_collect`)

Reruns `train2.py`'s 11-fold LOSO training exactly as-is and, for
every held-out window, saves:

- `y_true`, `y_hat` (final model prediction)
- `y_linear` (the physics ratio-of-ratios baseline)
- `delta` (the learned correction, `y_hat - y_linear`)
- `subject_id`, `fold`

**Output:** `loso_fold_predictions.csv`

## Step 2 — Subject-level conformal calibration

Builds a distribution-free prediction interval around `y_hat`, with
one hard rule enforced throughout: the calibration set is the **11
subjects**, never the ~200 windows. Pooling windows would silently
inflate the sample size and produce a falsely narrow, invalid
interval — `window_level_pooling_pitfall()` is a deliberate,
labeled demonstration of that mistake, kept in the file so the
reasoning is visible, not just asserted.

Uses the exact finite-sample quantile (Vovk et al., 2005): with only
11 subjects, achievable coverage levels are multiples of 1/12
(~8.3 pts), and requests above ~91.7% are correctly reported as
unachievable with a finite interval rather than silently answered.

Key functions: `collect_subject_scores`, `conformal_quantile`,
`build_conformal_report`.

## Step 3 — Bounded-authority analysis

Per-subject summary (mean / p95 / max) of four quantities: physical
error, hybrid error, `|delta|` (authority used), and normalized
authority `rho = |delta| / max_residual`. Also reports how often the
correction sits near its architectural ceiling (`rho >= 0.8` / `0.95`).

Key functions: `per_subject_bounded_authority_stats`,
`print_near_bound_frequency`, `bounded_authority_report`.

## Step 4 — Authority vs. benefit (window-level)

Defines `gain = |physical error| - |hybrid error|` (positive =
correction helped) and plots it against `rho`. Finding: two clean
linear fans, not noise — right-direction corrections gain with `rho`,
wrong-direction corrections lose with `rho`, at roughly the same
rate. Pooled correlation between `rho` and `gain` is near zero only
because the two fans cancel when pooled — **`rho` predicts the
*size* of the outcome given the correction's direction, not the
direction itself.** That distinction motivates every step after this
one.

Key functions: `window_level_authority_benefit_table`,
`plot_rho_vs_gain`, `plot_rho_bin_mean_gain`,
`authority_vs_benefit_report`.

## Step 5 — Can a single model predict its own direction? (CSV-only)

Tests whether any scalar already in the plain CSV — `rho`, `|delta|`,
signed `delta`, `y_linear`, `y_hat`, or a handful of combinations —
predicts `direction_correct = sign(delta) == sign(y_true - y_linear)`.

Every candidate is scored with the same LOSO discipline used
throughout: pooled ROC-AUC/PR-AUC, an 11× leave-one-subject-excluded
AUC stability check, and — the check that actually matters — a
threshold selected on 10 subjects and scored only on the 11th.
**Verdict: none of the candidates is a usable monitor** — the best
one had AUC nominally above chance but held-out threshold accuracy
*below* the majority-class baseline.

Key functions: `build_direction_reliability_table`,
`evaluate_direction_candidate`, `direction_reliability_report`.

## Step 6 — Enriched runtime signals (the one step that runs fresh inference)

Extends the inference pass to capture what `RawSelfAttentionSpO2Net.forward()`
normally computes and discards: the pre-tanh correction logit
(`raw_delta`), the pooled attention embeddings (`ppg_feature`,
`r_feature`), MC-dropout uncertainty (with BatchNorm correctly pinned
in eval mode — the same freeze fix used elsewhere in this project),
and simple raw-PPG signal-quality measures. Reuses the 11 already-
trained checkpoints; **no retraining**.

Also introduces the one signal that actually works: disagreement
across a **leakage-free 3-model "clean ensemble"** per subject — that
subject's own held-out fold model plus the exactly two other fold
models whose training set also excluded it (`multi_subject_split`
excludes both val and test subjects from training). `cross_model_std_delta`
is the first candidate in the whole investigation to clear both bars:
LOSO AUC 0.676 (stable), held-out threshold accuracy meaningfully
above baseline.

**Outputs:** `loso_enriched.csv`, `loso_enriched_embeddings.npz`

Key functions: `forward_with_internals`, `enable_mc_dropout`,
`clean_ensemble_fold_indices`, `run_enriched_loso_inference`,
`enriched_direction_reliability_report`, `loso_logistic_combination`
(a fitted combination of the top signals — reported honestly as
*worse* than the best single signal, not spun as an improvement).

## Step 7 — Closing the loop: does it improve the actual estimate?

Uses `cross_model_std_delta`/`cross_model_sign_agreement` to gate or
attenuate the correction — hard monitor (accept/reject by threshold),
sign-agreement monitor (full authority only on ensemble-unanimous
sign, else attenuate by a LOSO-selected `beta`), and a soft
continuous version — and checks whether that changes the actual MAE/
RMSE, not just a classification metric. Every threshold/`beta` is
selected on the other 10 subjects only, never the one it's scored on.

Result: real, positive, but modest and uneven — pooled soft-monitor
beats both physics and the uncontrolled hybrid; the sign-monitor
turns Subject1 from the worst hybrid failure into a result that beats
physics outright; but a few subjects (Subject3, Subject4) do
slightly worse under every monitor.

Key functions: `hard_monitor_predict`, `sign_monitor_predict`,
`soft_monitor_predict`, `run_monitored_estimator_loso`,
`monitored_estimator_report`.

## Step 8 — Statistical validation & ablation

Formal validation of Step 7's results:

- Subject-wise paired MAE/RMSE differences for all five comparisons.
- Subject-level bootstrap 95% CIs (resamples **subjects**, never
  windows) on the pooled improvement numbers.
- Paired Wilcoxon + sign tests, with the small-*n* limitation stated
  explicitly (only one comparison's CI excludes zero; most tests
  aren't powered to resolve an effect at n=11).
- Fraction of windows accepted / attenuated / rejected, and error
  conditioned on that decision.
- **The decisive ablation:** sign-monitor vs. random rejection at the
  *same* per-subject rate, and vs. a magnitude-threshold rule on the
  same ensemble signal at the same rate. Sign-monitor beats random on
  most subjects but not all (Subject3, Subject12 do worse with the
  deliberate selection than with random rejection), and only beats
  the matched magnitude-threshold rule on 4/11 subjects — so the
  gain is real but not cleanly attributable to the sign-based
  mechanism specifically.
- An offline oracle upper bound (uses ground truth — not deployable,
  context only): the theoretical ceiling for any accept/reject
  monitor is only ~12% better than physics alone, which reframes how
  much headroom existed in the first place.

Key functions: `bootstrap_pooled_improvement`,
`print_significance_tests`, `matched_random_rejection`,
`matched_std_threshold`, `print_oracle`,
`statistical_validation_report`.

---

## Re-running a step without retraining

```python
import train3 as t3

# Step 2 (from Step 1's CSV)
fold_results = t3.load_fold_results_csv("loso_fold_predictions.csv")
t3.build_conformal_report(fold_results, coverage=0.90)

# Steps 3–5 (same fold_results)
t3.bounded_authority_report(fold_results, max_residual=2.0)
t3.authority_vs_benefit_report(fold_results, max_residual=2.0)
t3.direction_reliability_report(fold_results, max_residual=2.0)

# Steps 6b–8 (from Step 6's enriched CSV)
enriched_df = t3.load_enriched_results("loso_enriched.csv")
t3.enriched_direction_reliability_report(enriched_df)

per_subject = t3.run_monitored_estimator_loso(enriched_df)
t3.monitored_estimator_report(per_subject)
t3.statistical_validation_report(enriched_df, per_subject)
```

Step 1 (training) and Step 6 (fresh inference over the trained
checkpoints) are the only steps that need `train2.py`, a GPU, or the
raw PPG data — everything else runs on the saved CSVs alone.

## What downstream files reuse

`train4.py` imports `train3.py`'s training and enriched-inference
functions unchanged to (a) sweep `max_residual` across
`{0, 1, 2, 3, 4, 5}` and (b) compare the learned residual against the
ideal residual `delta_star = y_true - y_linear` — both are separate
experiments layered on top of this file, not modifications to it.
