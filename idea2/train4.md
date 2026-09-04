# train4.py

Two experiments layered on top of `train2.py`/`train3.py`'s existing,
unmodified training and inference code. Unlike `train3.py` (mostly
post-processing of one saved run), `train4.py` trains fresh models
where the question requires it — but only where it requires it: it
reuses or mathematically shortcuts every case it safely can.

---

## Part 1 — The `max_residual` architectural sweep (Steps A–C)

**Question:** how does the residual-bound hyperparameter itself trade
off accuracy, worst-subject robustness, and the value of the
ensemble-disagreement monitor?

Repeats the *exact* 11-fold LOSO training protocol from `train2.py`
at `max_residual ∈ {0, 1, 2, 3, 4, 5}`, holding every other choice
fixed — same splits, same seed, same optimizer, same epoch counts,
same architecture, same physics calibration.

- **`mr=0` needs no training at all.** `delta_spo2 = max_residual *
  tanh(raw_delta)` is identically zero when `max_residual=0`, which
  zeroes the gradient to the *entire* learned pathway (encoder,
  attention, correction head) — verified, not assumed:
  `y_hat == y_linear` for every window regardless of what an
  untrained head would have output. `physics_only_table()` builds
  this bound directly from an existing run's `y_true`/`y_linear`.
- **`mr=2`** (the bound used everywhere else in this project) reuses
  the checkpoints and enriched data already on disk rather than
  retraining a redundant, nominally-identical copy.
- **`mr ∈ {1, 3, 4, 5}`** are genuinely retrained via
  `train3.run_loso_and_collect` / `run_enriched_loso_inference` — no
  new training code, just a different `cfg["max_residual"]`.

`run_sweep()` orchestrates all six bounds and saves
`loso_enriched_mr{N}.csv` (+ embeddings `.npz`) for each.

`analyze_bound(mr, df)` then computes, per bound:

- pooled + per-subject MAE / RMSE / max absolute error
- residual behavior: mean `|delta|`, saturation rate (`rho >= 0.8`
  and `>= 0.95`)
- correction quality: fraction that helps / harms physics,
  wrong-direction rate
- subject robustness: subjects improved vs. worsened, worst-subject
  error, Subject1 tracked specifically
- **the established sign-agreement monitor's gain at that bound**
  (re-applies `train3.run_monitored_estimator_loso` unchanged — no
  new monitor mechanism is introduced here)

`plot_sweep()` produces the five requested figures, all vs.
`max_residual`: MAE/RMSE, worst-subject error, harmful-correction
rate, saturation rate, monitor gain.

**What it found:** pooled accuracy is nearly flat across mr=1–5, all
slightly worse than physics regardless of bound. Saturation is
*highest at the smallest bound* (mr=1: 29% of windows near-saturated)
and collapses as the bound grows — counterintuitive, since `rho =
|tanh(raw_delta)|` doesn't depend on `max_residual` at all;
training itself adapts, pushing `raw_delta` harder when the bound is
tight. Subject robustness is non-monotonic (mr=3 best at 6/11
subjects improved, mr=4 worst at 3/11). Monitor gain is positive at
every bound except mr=4 (slightly negative). Subject1's
wrong-direction rate holds at a constant 63.6% across every nonzero
bound — its failure doesn't depend on how much authority it's given.

## Part 2 — Learned residual vs. ideal residual (Step D)

**Question:** is subject-shift failure fundamentally a
direction/generalization problem, or an authority-magnitude problem?

Pure post-processing of the `loso_enriched_mr{1,2,3,4,5}.csv` files
Part 1 already produced — **no training**. `mr=0` is excluded (its
`delta` is identically zero, making direction comparisons
degenerate).

Defines, per window, `delta_star = y_true - y_linear` — the
correction that would make the physics baseline exactly right — and
compares it against the model's actual `delta`:

1. sign-correctness rate: `sign(delta) == sign(delta_star)`
2. direction-failure rate: `delta * delta_star < 0`
3. under-correction rate: right direction, `|delta| < |delta_star|`
4. over-correction rate: right direction, `|delta| > |delta_star|`
5. Pearson / Spearman correlation between `delta` and `delta_star`
6. magnitude error: `|delta| - |delta_star|`
7. residual MSE: `(delta - delta_star)^2`

computed per subject and per bound (`per_subject_mr_metrics`,
`pooled_mr_metrics`), plus a pooled-per-bound version for a more
stable reference than the noisy 11–24-window per-subject estimates.

**Window matching across bounds** (`sign_stability_across_mr`) joins
windows by `(subject_id, window_idx)` — verified safe beforehand:
`y_true` and `y_linear` are bit-identical for the same window across
every mr run, confirming the windowing and physics calibration never
changed, only the model.

`diagnose_subject1()` breaks down which failure mode dominates
Subject1's windows at each bound and votes across bounds for an
overall diagnosis.

Three plots: a subject × `max_residual` heatmap of sign-correctness,
Subject1's failure-mode breakdown per bound, and per-subject sign-
stability across bounds.

**What it found — the decisive result:** pooled `delta`/`delta_star`
correlation is ~0 at every bound (Pearson −0.07 to +0.06). Subject1's
dominant failure mode is **direction failure** at all 5 bounds
(63.6%, exactly constant) — not magnitude. Subject2 turned out to be
directionally *worse* than Subject1 (9% sign-correct vs. 36%), which
hadn't been visible in earlier MAE-based rankings. Sign stability
across bounds splits subjects into two regimes: some (Subject1,
Subject2, Subject4, Subject7) get a *reproducibly wrong* sign
regardless of bound (a genuine generalization failure), while others
(Subject9: 95.7% sign changes) are essentially noise-driven and
sensitive to arbitrary training configuration. Conclusion: subject-
shift failure here is fundamentally a **direction/generalization**
problem, not an authority-magnitude one.

---

## Re-running without retraining

Only Part 1's `mr ∈ {1,3,4,5}` require GPU training. Everything else
is replayable from the CSVs already on disk:

```python
import train4 as t4

# Part 1 analysis + plots, from already-saved loso_enriched_mr{N}.csv
tables = t4.load_sweep()
analyses = t4.run_full_analysis(tables)
t4.plot_sweep(analyses)

# Part 2, from the same saved files - no training at all
result = t4.run_delta_star_analysis()
```

## Outputs

- `loso_enriched_mr{0,1,2,3,4,5}.csv` (+ embeddings `.npz` for the
  retrained bounds)
- `loso_checkpoints_mr{1,3,4,5}/` (fresh checkpoints; mr=2 reuses
  `loso_checkpoints_train3/`)
- Plots: `sweep_mae_rmse.png`, `sweep_worst_subject.png`,
  `sweep_harmful_rate.png`, `sweep_saturation.png`,
  `sweep_monitor_gain.png`, `delta_star_sign_correct_heatmap.png`,
  `subject1_failure_mode.png`, `sign_stability_per_subject.png`
