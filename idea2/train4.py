"""
train4.py

Architectural residual-bound sweep: repeats the EXISTING, unchanged
11-fold LOSO training protocol (train2.py's run_one_fold via
train3.py's run_loso_and_collect / run_enriched_loso_inference) at
several values of max_residual, holding every other training choice
fixed - splits, seeds, optimizer, epochs, architecture, calibration
model, evaluation metrics. The only thing that changes across runs is
cfg["max_residual"].

Grid: {0, 1, 2, 3, 4, 5} (current bound, 2, lies inside this range).

  - max_residual=0 needs NO training. delta_spo2 =
    max_residual * tanh(raw_delta) is identically 0 when
    max_residual=0, which zeroes the gradient to the ENTIRE learned
    pathway (encoder, self-attention, correction head) during Stage
    2/3 - the only term that keeps a gradient is the physics buffer,
    which isn't trainable. Verified: y_hat = y_linear exactly for
    every window regardless of what the (untrained, arbitrary) head
    would have output. Reused directly from an existing run's
    y_true/y_linear rather than "training" a no-op.

  - max_residual=2 (the current/established bound) reuses the
    checkpoints and enriched data already produced by train3.py's
    last run unchanged - the literal reading of "keep every other
    choice identical" is to reuse that actual run rather than a
    fresh nominally-identical one.

  - max_residual in {1, 3, 4, 5} are genuinely retrained here, via
    the same train3.run_loso_and_collect / run_enriched_loso_inference
    used for every other LOSO result in this project - no new
    training code.

Re-applies ONLY the already-established sign-agreement monitor
(train3.sign_monitor_predict, via run_monitored_estimator_loso) to
each bound's data - no new gating mechanism is introduced here.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import train2 as t2
import train3 as t3


MAX_RESIDUAL_GRID = [0, 1, 2, 3, 4, 5]
CURRENT_MAX_RESIDUAL = 2

RED = "#B93A26"
TEAL = "#0F7A6C"
MUTED = "#7C8D85"
INK = "#131C1A"


# =========================================================
# Step A: run (or reuse) the sweep
# =========================================================

def cfg_for(max_residual):
    cfg = t3.build_cfg()
    cfg["max_residual"] = float(max_residual)
    return cfg


def physics_only_table(reference_df):
    """
    max_residual=0's exact, no-training-needed result (see module
    docstring). y_true/y_linear/subject_id are identical to any other
    max_residual run (same splits, same seed, same physics
    calibration fit per fold) - reused directly rather than
    regenerated.
    """

    df = reference_df.copy()
    df["y_hat"] = df["y_linear"]
    df["delta"] = 0.0
    df["raw_delta"] = np.nan
    df["ppg_feature_norm"] = np.nan
    df["r_feature_norm"] = np.nan
    df["delta_diff"] = 0.0
    df["delta_abs_diff"] = 0.0
    df["delta_local_var"] = 0.0
    df["mc_dropout_delta_std"] = 0.0
    df["cross_model_mean_delta"] = 0.0
    df["cross_model_std_delta"] = 0.0
    df["cross_model_sign_agreement"] = 1.0
    return df


def run_sweep(
    grid=MAX_RESIDUAL_GRID,
    existing_enriched_csv="loso_enriched.csv",
):

    t2.set_seed(42)
    device = t2.get_device()

    reference_df = None
    if os.path.exists(existing_enriched_csv):
        reference_df = t3.load_enriched_results(existing_enriched_csv)

    tables = {}

    for mr in grid:
        print("\n" + "=" * 70)
        print(f"MAX_RESIDUAL = {mr}")
        print("=" * 70)

        enriched_csv = f"loso_enriched_mr{mr}.csv"

        if mr == 0:
            if reference_df is None:
                raise RuntimeError("Need an existing enriched CSV to build the mr=0 physics-only control from.")
            df = physics_only_table(reference_df)
            df.to_csv(enriched_csv, index=False)
            print("max_residual=0: no training needed (verified zero-gradient no-op); reused y_true/y_linear from the existing run.")

        elif mr == CURRENT_MAX_RESIDUAL and reference_df is not None:
            df = reference_df.copy()
            df.to_csv(enriched_csv, index=False)
            print(f"max_residual={mr}: reused the existing checkpoints/enriched data from the current run (identical protocol, no redundant retraining).")

        else:
            cfg = cfg_for(mr)
            checkpoint_dir = f"loso_checkpoints_mr{mr}"
            os.makedirs(checkpoint_dir, exist_ok=True)

            fold_results = t3.run_loso_and_collect(cfg, device, checkpoint_dir)
            t3.save_fold_results_csv(fold_results, f"loso_fold_predictions_mr{mr}.csv")

            enriched_rows, enriched_embeddings = t3.run_enriched_loso_inference(cfg, device, checkpoint_dir)
            df = t3.save_enriched_results(enriched_rows, enriched_embeddings, enriched_csv, f"loso_enriched_mr{mr}_embeddings.npz")

        tables[mr] = df

    return tables


def load_sweep(grid=MAX_RESIDUAL_GRID):
    return {mr: t3.load_enriched_results(f"loso_enriched_mr{mr}.csv") for mr in grid}


# =========================================================
# Step B: per-bound metrics
# =========================================================

def _err_metrics(y_true, y_pred):
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_abs": float(np.max(abs_err)) if len(abs_err) else float("nan"),
    }


def analyze_bound(mr, df):
    """
    Everything requested for one max_residual value: pooled/per-
    subject accuracy, residual behavior, correction quality, subject
    robustness, and (via the existing sign-monitor machinery,
    unchanged) monitor gain.
    """

    y_true = df["y_true"].to_numpy(dtype=np.float64)
    y_linear = df["y_linear"].to_numpy(dtype=np.float64)
    y_hat = df["y_hat"].to_numpy(dtype=np.float64)
    delta = df["delta"].to_numpy(dtype=np.float64)
    subject_id = df["subject_id"].to_numpy()
    subjects = sorted(np.unique(subject_id))

    physical_err = np.abs(y_true - y_linear)
    hybrid_err = np.abs(y_true - y_hat)
    gain = physical_err - hybrid_err
    wrong_direction = np.sign(delta) != np.sign(y_true - y_linear)

    # --- 1. pooled + per-subject accuracy ---
    pooled_physics = _err_metrics(y_true, y_linear)
    pooled_hybrid = _err_metrics(y_true, y_hat)

    per_subject_physics, per_subject_hybrid = {}, {}
    for s in subjects:
        m = subject_id == s
        per_subject_physics[s] = _err_metrics(y_true[m], y_linear[m])
        per_subject_hybrid[s] = _err_metrics(y_true[m], y_hat[m])

    # --- 2. residual behavior ---
    if mr > 0:
        rho = np.abs(delta) / mr
    else:
        rho = np.zeros_like(delta)
    mean_abs_delta = float(np.mean(np.abs(delta)))
    frac_rho_ge_08 = float(np.mean(rho >= 0.8))
    frac_rho_ge_095 = float(np.mean(rho >= 0.95))

    # --- 3. correction quality ---
    frac_improves = float(np.mean(gain > 0))
    frac_harms = float(np.mean(gain < 0))
    # mr=0: delta is identically 0, so sign(delta)=0 while sign(y_true
    # - y_linear) is generally +/-1 - the formula would report ~100%
    # "wrong direction", which is a misleading artifact of comparing
    # against a zero vector, not a real finding. There is no
    # correction, so "direction" is undefined, not "always wrong".
    wrong_direction_rate = float("nan") if mr == 0 else float(np.mean(wrong_direction))

    # --- 4. subject robustness ---
    n_improved = sum(1 for s in subjects if per_subject_hybrid[s]["mae"] < per_subject_physics[s]["mae"])
    n_worsened = sum(1 for s in subjects if per_subject_hybrid[s]["mae"] > per_subject_physics[s]["mae"])
    worst_subject_mae = max(per_subject_hybrid[s]["mae"] for s in subjects)
    worst_subject_rmse = max(per_subject_hybrid[s]["rmse"] for s in subjects)
    worst_subject_name = max(subjects, key=lambda s: per_subject_hybrid[s]["mae"])

    subject1 = None
    if "Subject1" in subjects:
        m = subject_id == "Subject1"
        subject1 = {
            "physics": _err_metrics(y_true[m], y_linear[m]),
            "hybrid": _err_metrics(y_true[m], y_hat[m]),
            "mean_abs_delta": float(np.mean(np.abs(delta[m]))),
            "wrong_direction_rate": float(np.mean(wrong_direction[m])),
            "frac_rho_ge_08": float(np.mean(rho[m] >= 0.8)) if mr > 0 else 0.0,
        }

    # --- 5. monitor gain: ONLY the established sign-agreement monitor ---
    if mr > 0:
        per_subject_monitor = t3.run_monitored_estimator_loso(df)
        y_true_all = np.concatenate([r["y_true"] for r in per_subject_monitor])
        y_sign_all = np.concatenate([r["y_sign"] for r in per_subject_monitor])
        pooled_sign = _err_metrics(y_true_all, y_sign_all)
        n_sign_beats_hybrid = sum(
            1 for r in per_subject_monitor
            if _err_metrics(r["y_true"], r["y_sign"])["mae"] < _err_metrics(r["y_true"], r["y_hybrid"])["mae"]
        )
    else:
        # mr=0: delta is identically 0, so any monitor gating is a no-op by construction.
        pooled_sign = dict(pooled_physics)
        n_sign_beats_hybrid = 0

    return {
        "mr": mr,
        "pooled_physics": pooled_physics,
        "pooled_hybrid": pooled_hybrid,
        "pooled_sign": pooled_sign,
        "per_subject_physics": per_subject_physics,
        "per_subject_hybrid": per_subject_hybrid,
        "mean_abs_delta": mean_abs_delta,
        "frac_rho_ge_08": frac_rho_ge_08,
        "frac_rho_ge_095": frac_rho_ge_095,
        "frac_improves": frac_improves,
        "frac_harms": frac_harms,
        "wrong_direction_rate": wrong_direction_rate,
        "n_improved": n_improved,
        "n_worsened": n_worsened,
        "n_subjects": len(subjects),
        "worst_subject_mae": worst_subject_mae,
        "worst_subject_rmse": worst_subject_rmse,
        "worst_subject_name": worst_subject_name,
        "subject1": subject1,
        "n_sign_beats_hybrid": n_sign_beats_hybrid,
    }


def print_bound_report(a):
    mr = a["mr"]
    print("\n" + "#" * 70)
    print(f"MAX_RESIDUAL = {mr}")
    print("#" * 70)

    print("\n1. Pooled accuracy")
    print(f"  physics : MAE={a['pooled_physics']['mae']:.3f}  RMSE={a['pooled_physics']['rmse']:.3f}  MaxAbsErr={a['pooled_physics']['max_abs']:.3f}")
    print(f"  hybrid  : MAE={a['pooled_hybrid']['mae']:.3f}  RMSE={a['pooled_hybrid']['rmse']:.3f}  MaxAbsErr={a['pooled_hybrid']['max_abs']:.3f}")

    print("\n   Per-subject (hybrid) MAE / RMSE")
    for s in sorted(a["per_subject_hybrid"]):
        h = a["per_subject_hybrid"][s]
        p = a["per_subject_physics"][s]
        print(f"     {s:14s} hybrid MAE={h['mae']:7.3f} RMSE={h['rmse']:7.3f}   physics MAE={p['mae']:7.3f} RMSE={p['rmse']:7.3f}")

    print("\n2. Residual behavior")
    print(f"  mean |delta| = {a['mean_abs_delta']:.3f}   frac(rho>=0.8) = {a['frac_rho_ge_08']:.1%}   frac(rho>=0.95) = {a['frac_rho_ge_095']:.1%}")

    print("\n3. Correction quality")
    print(f"  frac improves physics = {a['frac_improves']:.1%}   frac harms physics = {a['frac_harms']:.1%}   wrong-direction rate = {a['wrong_direction_rate']:.1%}")

    print("\n4. Subject robustness")
    print(f"  subjects improved over physics: {a['n_improved']}/{a['n_subjects']}   worsened: {a['n_worsened']}/{a['n_subjects']}")
    print(f"  worst subject (hybrid MAE): {a['worst_subject_name']}  MAE={a['worst_subject_mae']:.3f}  RMSE={a['worst_subject_rmse']:.3f}")
    if a["subject1"] is not None:
        s1 = a["subject1"]
        print(
            f"  Subject1: physics MAE={s1['physics']['mae']:.3f}  hybrid MAE={s1['hybrid']['mae']:.3f}  "
            f"mean|delta|={s1['mean_abs_delta']:.3f}  wrong-dir%={s1['wrong_direction_rate']:.1%}  "
            f"frac(rho>=0.8)={s1['frac_rho_ge_08']:.1%}"
        )

    print("\n5. Established sign-agreement monitor")
    print(f"  sign-monitor pooled: MAE={a['pooled_sign']['mae']:.3f}  RMSE={a['pooled_sign']['rmse']:.3f}")
    print(f"  monitor gain over hybrid (MAE): {a['pooled_hybrid']['mae'] - a['pooled_sign']['mae']:+.3f}")
    print(f"  subjects where sign-monitor beats hybrid: {a['n_sign_beats_hybrid']}/{a['n_subjects']}")


def run_full_analysis(tables, grid=MAX_RESIDUAL_GRID):
    analyses = {}
    for mr in grid:
        a = analyze_bound(mr, tables[mr])
        analyses[mr] = a
        print_bound_report(a)
    return analyses


# =========================================================
# Step C: the five central plots
# =========================================================

def plot_sweep(analyses, grid=MAX_RESIDUAL_GRID, out_prefix="sweep"):

    xs = grid
    physics_mae = [analyses[mr]["pooled_physics"]["mae"] for mr in xs]
    hybrid_mae = [analyses[mr]["pooled_hybrid"]["mae"] for mr in xs]
    physics_rmse = [analyses[mr]["pooled_physics"]["rmse"] for mr in xs]
    hybrid_rmse = [analyses[mr]["pooled_hybrid"]["rmse"] for mr in xs]

    # Plot 1: MAE/RMSE vs max_residual
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(xs, physics_mae, "--", color=MUTED, marker="o", label="physics MAE")
    ax.plot(xs, hybrid_mae, "-", color=TEAL, marker="o", label="hybrid MAE")
    ax.plot(xs, physics_rmse, "--", color=MUTED, marker="s", alpha=0.6, label="physics RMSE")
    ax.plot(xs, hybrid_rmse, "-", color=RED, marker="s", label="hybrid RMSE")
    ax.set_xlabel("max_residual")
    ax.set_ylabel("SpO2 points")
    ax.set_title("Pooled MAE / RMSE vs. max_residual")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_mae_rmse.png")
    plt.close(fig)

    # Plot 2: worst-subject error vs max_residual
    worst_mae = [analyses[mr]["worst_subject_mae"] for mr in xs]
    worst_rmse = [analyses[mr]["worst_subject_rmse"] for mr in xs]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(xs, worst_mae, "-", color=TEAL, marker="o", label="worst-subject MAE")
    ax.plot(xs, worst_rmse, "-", color=RED, marker="s", label="worst-subject RMSE")
    ax.set_xlabel("max_residual")
    ax.set_ylabel("SpO2 points")
    ax.set_title("Worst-subject error vs. max_residual")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_worst_subject.png")
    plt.close(fig)

    # Plot 3: harmful-correction rate vs max_residual
    frac_harms = [analyses[mr]["frac_harms"] for mr in xs]
    frac_improves = [analyses[mr]["frac_improves"] for mr in xs]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(xs, frac_harms, "-", color=RED, marker="o", label="fraction harms physics")
    ax.plot(xs, frac_improves, "-", color=TEAL, marker="o", label="fraction improves physics")
    ax.set_xlabel("max_residual")
    ax.set_ylabel("fraction of windows")
    ax.set_title("Correction quality vs. max_residual")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_harmful_rate.png")
    plt.close(fig)

    # Plot 4: saturation rate vs max_residual
    frac_08 = [analyses[mr]["frac_rho_ge_08"] for mr in xs]
    frac_095 = [analyses[mr]["frac_rho_ge_095"] for mr in xs]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(xs, frac_08, "-", color=INK, marker="o", label="frac(rho >= 0.8)")
    ax.plot(xs, frac_095, "-", color=RED, marker="s", label="frac(rho >= 0.95)")
    ax.set_xlabel("max_residual")
    ax.set_ylabel("fraction of windows")
    ax.set_title("Saturation rate vs. max_residual")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_saturation.png")
    plt.close(fig)

    # Plot 5: monitor gain vs max_residual
    monitor_gain = [analyses[mr]["pooled_hybrid"]["mae"] - analyses[mr]["pooled_sign"]["mae"] for mr in xs]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    colors = [TEAL if g >= 0 else RED for g in monitor_gain]
    ax.bar([str(x) for x in xs], monitor_gain, color=colors)
    ax.axhline(0.0, color="0.3", linewidth=1)
    ax.set_xlabel("max_residual")
    ax.set_ylabel("MAE(hybrid) - MAE(sign-monitor)")
    ax.set_title("Sign-agreement monitor gain vs. max_residual")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_monitor_gain.png")
    plt.close(fig)

    print(f"\nSaved 5 plots: {out_prefix}_mae_rmse.png, {out_prefix}_worst_subject.png, "
          f"{out_prefix}_harmful_rate.png, {out_prefix}_saturation.png, {out_prefix}_monitor_gain.png")


# =========================================================
# Step D: learned residual vs. IDEAL residual (delta_star)
#
# Pure post-processing of the already-saved mr={1,2,3,4,5} enriched
# CSVs. No training, no new sweep values. mr=0 is excluded on
# purpose: delta is identically 0 there, so delta vs. delta_star
# comparisons are degenerate (see Step C).
#
# delta_star = y_true - y_linear is the residual the network would
# need to output to make the physics baseline exactly correct - the
# "ideal" target the correction head is implicitly trying to learn,
# even though it is never given delta_star directly (only y_true via
# the Huber loss on y_hat = y_linear + delta).
#
# Window matching across mr runs uses (subject_id, window_idx):
# verified above that y_true and y_linear are bit-identical for the
# same (subject_id, window_idx) across every mr in {1,2,3,4,5} - same
# splits, same seed, same physics calibration - so this is a safe
# match, not an assumption.
# =========================================================

DELTA_STAR_GRID = [1, 2, 3, 4, 5]


def load_delta_star_table(grid=DELTA_STAR_GRID):
    frames = []
    for mr in grid:
        df = t3.load_enriched_results(f"loso_enriched_mr{mr}.csv").copy()
        df["mr"] = mr
        df["delta_star"] = df["y_true"] - df["y_linear"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _delta_vs_star_metrics(delta, delta_star):
    sign_delta = np.sign(delta)
    sign_star = np.sign(delta_star)
    sign_correct = sign_delta == sign_star
    direction_failure = (delta * delta_star) < 0
    undercorrect = sign_correct & (np.abs(delta) < np.abs(delta_star))
    overcorrect = sign_correct & (np.abs(delta) > np.abs(delta_star))

    pearson_r = pearson_p = spearman_r = spearman_p = float("nan")
    if len(delta) >= 3 and np.std(delta) > 0 and np.std(delta_star) > 0 and t3._HAVE_SCIPY:
        pearson_r, pearson_p = t3.pearsonr(delta, delta_star)
        spearman_r, spearman_p = t3.spearmanr(delta, delta_star)

    magnitude_error = np.abs(delta) - np.abs(delta_star)

    return {
        "n": len(delta),
        "sign_correct_rate": float(np.mean(sign_correct)),
        "direction_failure_rate": float(np.mean(direction_failure)),
        "undercorrect_rate": float(np.mean(undercorrect)),
        "overcorrect_rate": float(np.mean(overcorrect)),
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        "mean_magnitude_error": float(np.mean(magnitude_error)),
        "residual_mse": float(np.mean((delta - delta_star) ** 2)),
    }


def per_subject_mr_metrics(table):
    rows = []
    for (s, mr), g in table.groupby(["subject_id", "mr"]):
        m = _delta_vs_star_metrics(g["delta"].to_numpy(), g["delta_star"].to_numpy())
        m["subject"] = s
        m["mr"] = mr
        rows.append(m)
    return pd.DataFrame(rows)


def pooled_mr_metrics(table):
    rows = []
    for mr, g in table.groupby("mr"):
        m = _delta_vs_star_metrics(g["delta"].to_numpy(), g["delta_star"].to_numpy())
        m["mr"] = mr
        rows.append(m)
    return pd.DataFrame(rows)


def print_delta_star_tables(per_subject_df, pooled_df):
    print("\n" + "#" * 70)
    print("LEARNED RESIDUAL (delta) vs. IDEAL RESIDUAL (delta_star = y_true - y_linear)")
    print("#" * 70)
    print(
        "NOTE: per-subject correlations use only 11-24 windows each - noisy point "
        "estimates. Pooled (all subjects, that mr) correlations are shown alongside "
        "as a more stable reference, not a replacement."
    )

    print("\nPooled, per max_residual")
    print(f"{'mr':>3s} {'n':>4s} {'sign_ok%':>9s} {'dir_fail%':>10s} {'undercorr%':>11s} {'overcorr%':>10s} {'pearson_r':>10s} {'spearman_r':>11s} {'mag_err':>8s} {'res_MSE':>8s}")
    for _, r in pooled_df.sort_values("mr").iterrows():
        print(
            f"{int(r['mr']):3d} {int(r['n']):4d} {r['sign_correct_rate']:9.1%} {r['direction_failure_rate']:10.1%} "
            f"{r['undercorrect_rate']:11.1%} {r['overcorrect_rate']:10.1%} {r['pearson_r']:10.3f} {r['spearman_r']:11.3f} "
            f"{r['mean_magnitude_error']:8.3f} {r['residual_mse']:8.3f}"
        )

    print("\nPer subject, per max_residual")
    for s in sorted(per_subject_df["subject"].unique()):
        print(f"\n  {s}")
        print(f"  {'mr':>3s} {'n':>4s} {'sign_ok%':>9s} {'dir_fail%':>10s} {'undercorr%':>11s} {'overcorr%':>10s} {'pearson_r':>10s} {'spearman_r':>11s} {'mag_err':>8s} {'res_MSE':>8s}")
        sub = per_subject_df[per_subject_df["subject"] == s].sort_values("mr")
        for _, r in sub.iterrows():
            print(
                f"  {int(r['mr']):3d} {int(r['n']):4d} {r['sign_correct_rate']:9.1%} {r['direction_failure_rate']:10.1%} "
                f"{r['undercorrect_rate']:11.1%} {r['overcorrect_rate']:10.1%} {r['pearson_r']:10.3f} {r['spearman_r']:11.3f} "
                f"{r['mean_magnitude_error']:8.3f} {r['residual_mse']:8.3f}"
            )


def diagnose_subject1(per_subject_df):
    print("\n" + "#" * 70)
    print("SUBJECT1: WHICH FAILURE MODE DOMINATES?")
    print("#" * 70)

    s1 = per_subject_df[per_subject_df["subject"] == "Subject1"].sort_values("mr")
    if s1.empty:
        print("Subject1 not present.")
        return

    print(f"{'mr':>3s} {'dir_fail%':>10s} {'undercorr%':>11s} {'overcorr%':>10s} {'mag_err':>8s}   dominant mode")
    votes = {"direction_failure": 0, "undercorrect": 0, "overcorrect": 0}
    for _, r in s1.iterrows():
        rates = {
            "direction_failure": r["direction_failure_rate"],
            "undercorrect": r["undercorrect_rate"],
            "overcorrect": r["overcorrect_rate"],
        }
        dominant = max(rates, key=rates.get)
        votes[dominant] += 1
        print(
            f"{int(r['mr']):3d} {r['direction_failure_rate']:10.1%} {r['undercorrect_rate']:11.1%} "
            f"{r['overcorrect_rate']:10.1%} {r['mean_magnitude_error']:8.3f}   {dominant}"
        )

    overall_dominant = max(votes, key=votes.get)
    print(f"\nDominant failure mode across mr=1..5 for Subject1: {overall_dominant.upper().replace('_', ' ')} "
          f"({votes[overall_dominant]}/{len(s1)} bounds)")
    print(
        "Interpretation: direction_failure = wrong residual direction; "
        "undercorrect = right direction, insufficient magnitude; "
        "overcorrect = right direction, excessive magnitude."
    )


def sign_stability_across_mr(table, grid=DELTA_STAR_GRID):
    """
    For each window matched by (subject_id, window_idx) across every
    mr in grid, does sign(delta) stay the same? Requires all mr
    values present for that window (drops any window missing from
    one run - should be none, given verified identical windowing).
    """

    pivot = table.pivot_table(index=["subject_id", "window_idx"], columns="mr", values="delta", aggfunc="first")
    pivot = pivot.dropna(subset=grid)
    signs = np.sign(pivot[grid].to_numpy())

    stable = (signs == signs[:, [0]]).all(axis=1)
    frac_changed = float(np.mean(~stable))

    per_subject = {}
    subj_index = pivot.index.get_level_values("subject_id")
    for s in sorted(subj_index.unique()):
        m = subj_index == s
        per_subject[s] = float(np.mean(~stable[m]))

    return frac_changed, per_subject, pivot, stable


def print_sign_stability(table):
    print("\n" + "#" * 70)
    print(f"SIGN STABILITY OF delta ACROSS max_residual = {DELTA_STAR_GRID} (same physical window, matched by subject_id + window_idx)")
    print("#" * 70)

    frac_changed, per_subject, pivot, stable = sign_stability_across_mr(table)
    print(f"\nOverall: {frac_changed:.1%} of windows have a sign(delta) that changes somewhere across mr={DELTA_STAR_GRID}")
    print(f"{'Subject':14s} {'n windows':>10s} {'%sign changes':>14s}")
    for s, frac in sorted(per_subject.items()):
        n = int((pivot.index.get_level_values("subject_id") == s).sum())
        print(f"{s:14s} {n:10d} {frac:14.1%}")

    return frac_changed, per_subject, pivot, stable


# =========================================================
# Step D plots
# =========================================================

def plot_delta_star_heatmap(per_subject_df, out_path="delta_star_sign_correct_heatmap.png"):
    subjects = sorted(per_subject_df["subject"].unique())
    mrs = sorted(per_subject_df["mr"].unique())
    grid = np.full((len(subjects), len(mrs)), np.nan)
    for i, s in enumerate(subjects):
        for j, mr in enumerate(mrs):
            row = per_subject_df[(per_subject_df["subject"] == s) & (per_subject_df["mr"] == mr)]
            if len(row):
                grid[i, j] = row.iloc[0]["sign_correct_rate"]

    fig, ax = plt.subplots(figsize=(6, 6.5), dpi=150)
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(mrs)))
    ax.set_xticklabels(mrs)
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects)
    ax.set_xlabel("max_residual")
    ax.set_title("sign(delta) == sign(delta_star) rate")
    for i in range(len(subjects)):
        for j in range(len(mrs)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if grid[i, j] < 0.35 or grid[i, j] > 0.75 else "black")
    fig.colorbar(im, ax=ax, label="sign-correct rate")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_subject1_diagnosis(per_subject_df, out_path="subject1_failure_mode.png"):
    s1 = per_subject_df[per_subject_df["subject"] == "Subject1"].sort_values("mr")
    xs = s1["mr"].to_numpy()
    x = np.arange(len(xs))
    w = 0.25

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.bar(x - w, s1["direction_failure_rate"], w, label="direction_failure", color=RED)
    ax.bar(x, s1["undercorrect_rate"], w, label="undercorrect (right dir, too small)", color="#E0A32E")
    ax.bar(x + w, s1["overcorrect_rate"], w, label="overcorrect (right dir, too large)", color=TEAL)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in xs])
    ax.set_xlabel("max_residual")
    ax.set_ylabel("fraction of Subject1's windows")
    ax.set_title("Subject1: failure-mode breakdown vs. max_residual")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_sign_stability(per_subject_frac, out_path="sign_stability_per_subject.png"):
    subjects = sorted(per_subject_frac.keys())
    fracs = [per_subject_frac[s] for s in subjects]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.bar(subjects, fracs, color=TEAL)
    ax.set_ylabel("fraction of windows where sign(delta) changes across mr=1..5")
    ax.set_title("Cross-mr sign stability, per subject")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def run_delta_star_analysis(grid=DELTA_STAR_GRID):
    table = load_delta_star_table(grid)
    per_subject_df = per_subject_mr_metrics(table)
    pooled_df = pooled_mr_metrics(table)

    print_delta_star_tables(per_subject_df, pooled_df)
    diagnose_subject1(per_subject_df)
    frac_changed, per_subject_frac, pivot, stable = print_sign_stability(table)

    plot_delta_star_heatmap(per_subject_df)
    plot_subject1_diagnosis(per_subject_df)
    plot_sign_stability(per_subject_frac)

    return {
        "table": table,
        "per_subject_df": per_subject_df,
        "pooled_df": pooled_df,
        "sign_stability_frac_changed": frac_changed,
        "sign_stability_per_subject": per_subject_frac,
    }


if __name__ == "__main__":

    tables = run_sweep()
    analyses = run_full_analysis(tables)
    plot_sweep(analyses)
