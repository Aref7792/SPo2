"""
train2.py

Encoder-decoder pretraining variant of train1.py's
RawSelfAttentionSpO2Net.

Idea
----
1. STAGE 1 (self-supervised):
   Build an autoencoder around the exact same ConvTokenEncoder used
   inside RawSelfAttentionSpO2Net. Train it to reconstruct its own
   4-channel input (RED, IR, RED-IR, RED*IR) from the token
   embeddings it produces. This pretrains the encoder on raw signal
   structure without needing SpO2 labels.

2. STAGE 2 (frozen encoder):
   Drop the decoder. Plug the pretrained ConvTokenEncoder into
   RawSelfAttentionSpO2Net, freeze it, and train everything
   downstream of it (positional encoding, self-attention, pooling,
   R-embedding, residual correction head) against SpO2 labels.

3. STAGE 3 (fine-tune):
   Unfreeze the encoder and continue training end-to-end, using a
   smaller learning rate for the pretrained encoder than for the
   rest of the network.

Everything else (data loading, windowing, deterministic subject
split, physics ratio-of-ratios calibration, the model architecture
itself, the training/validation loops) is reused unchanged from
train1.py so the three stages stay directly comparable to the
single-stage baseline there.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train1 import (
    set_seed,
    get_device,
    load_all_files,
    create_windowed_dataset,
    RawSpO2WindowDataset,
    ConvTokenEncoder,
    RawSelfAttentionSpO2Net,
    RunningAverage,
    compute_regression_metrics,
    train_one_epoch,
    validate_one_epoch,
)


# =========================================================
# Utilities
# =========================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_interaction_channels(x_seq: torch.Tensor) -> torch.Tensor:
    """
    x_seq: [B, 2, T] raw synchronized (RED, IR)

    Returns [B, 4, T]: RED, IR, RED-IR, RED*IR.

    Mirrors the channel construction inside
    RawSelfAttentionSpO2Net.forward, so the autoencoder pretrains on
    exactly the input distribution ConvTokenEncoder will see once it
    is plugged into the full model.
    """
    red = x_seq[:, 0:1, :]
    ir = x_seq[:, 1:2, :]

    red_minus_ir = red - ir
    red_times_ir = red * ir

    return torch.cat(
        [red, ir, red_minus_ir, red_times_ir],
        dim=1,
    )


# =========================================================
# Stage 1:
# Token encoder-decoder (pretraining)
# =========================================================

class ConvTokenDecoder(nn.Module):
    """
    Mirrors ConvTokenEncoder, decoding token embeddings back into
    the reconstructed input channels.

    Input:
        [B, T, D]

    Output:
        [B, C, T]

    ConvTokenEncoder does not change sequence length (stride=1
    throughout), so the decoder does not need any upsampling either
    - it is a plain mirrored conv stack.
    """

    def __init__(
        self,
        out_channels: int = 4,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(
                hidden_dim,
                32,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(
                32,
                out_channels,
                kernel_size=7,
                padding=3,
            ),
        )

    def forward(self, tokens):
        # tokens: [B, T, D]
        x = tokens.transpose(1, 2)   # [B, D, T]
        x = self.net(x)              # [B, C, T]
        return x


class PPGTokenAutoencoder(nn.Module):
    """
    Stage 1 pretraining model.

    Reuses ConvTokenEncoder exactly as RawSelfAttentionSpO2Net uses
    it, so encoder.state_dict() from this module can be loaded
    directly into RawSelfAttentionSpO2Net.ppg_encoder once
    pretraining is done.

    The encoder still consumes the full 4-channel engineered input
    (RED, IR, RED-IR, RED*IR) - that is the input distribution it
    will see downstream. But the decoder only reconstructs the 2
    true underlying signals (RED, IR): RED-IR and RED*IR are
    deterministic functions of them, so reconstructing those too is
    redundant - and RED*IR in particular has ~300-600x the variance
    of the other three channels (measured on real data), which would
    dominate an unweighted multi-channel MSE loss and swamp the
    gradient signal for the two channels that actually matter.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 2,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = ConvTokenEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.decoder = ConvTokenDecoder(
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x):
        tokens = self.encoder(x)      # [B, T, D]
        x_hat = self.decoder(tokens)  # [B, out_channels, T]
        return x_hat


def train_autoencoder_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):
    model.train()

    loss_meter = RunningAverage()

    for x_seq, _, _ in loader:

        x_seq = x_seq.to(device)
        x_in = build_interaction_channels(x_seq)

        optimizer.zero_grad()

        x_hat = model(x_in)

        # Reconstruct only the 2 true underlying channels (RED, IR),
        # not the derived RED-IR / RED*IR channels in x_in.
        loss = criterion(x_hat, x_seq)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        bs = x_seq.size(0)
        loss_meter.update(loss.item(), bs)

    return loss_meter.avg


@torch.no_grad()
def validate_autoencoder(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    loss_meter = RunningAverage()

    for x_seq, _, _ in loader:

        x_seq = x_seq.to(device)
        x_in = build_interaction_channels(x_seq)

        x_hat = model(x_in)

        loss = criterion(x_hat, x_seq)

        bs = x_seq.size(0)
        loss_meter.update(loss.item(), bs)

    return loss_meter.avg


def pretrain_autoencoder(
    train_loader,
    val_loader,
    device,
    hidden_dim,
    dropout,
    lr,
    weight_decay,
    num_epochs,
    save_path,
):
    print("\n")
    print("=" * 70)
    print("STAGE 1: PPG TOKEN ENCODER-DECODER PRETRAINING")
    print("=" * 70)

    model = PPGTokenAutoencoder(
        in_channels=4,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
    )

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):

        train_loss = train_autoencoder_one_epoch(
            model, train_loader, optimizer, criterion, device,
        )

        val_loss = validate_autoencoder(
            model, val_loader, criterion, device,
        )

        scheduler.step(val_loss)

        print(
            f"AE Epoch [{epoch:03d}/{num_epochs:03d}] | "
            f"Train MSE: {train_loss:.6f} | "
            f"Val MSE: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                    "val_loss": best_val_loss,
                    "hidden_dim": hidden_dim,
                },
                save_path,
            )

            print(f"Saved best autoencoder: {save_path}")

    print(f"\nBest AE validation MSE: {best_val_loss:.6f}")

    checkpoint = torch.load(save_path, map_location=device)

    return checkpoint["encoder_state_dict"]


# =========================================================
# Stage 2 / Stage 3:
# SpO2 estimator with a pretrained, freezable encoder
# =========================================================

class PretrainedRawSelfAttentionSpO2Net(RawSelfAttentionSpO2Net):
    """
    Same architecture and forward() as RawSelfAttentionSpO2Net, plus
    the ability to load a pretrained ppg_encoder and freeze/unfreeze
    it in stages.

    Freezing sets requires_grad=False on the encoder's parameters
    AND forces the encoder into eval() mode even when the rest of
    the model is in train() mode. This matters because
    ConvTokenEncoder uses BatchNorm1d: requires_grad=False alone
    does not stop BatchNorm's running_mean/running_var buffers from
    drifting away from the pretrained checkpoint, since those are
    buffers, not parameters, and BatchNorm normalizes with
    per-batch statistics (not the running stats) whenever the module
    is in train mode. Without the explicit eval() override below,
    the "frozen" encoder would not actually stay fixed during Stage
    2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._encoder_trainable = True

    def load_pretrained_encoder(self, encoder_state_dict):
        self.ppg_encoder.load_state_dict(encoder_state_dict)

    def set_encoder_trainable(self, trainable: bool):
        self._encoder_trainable = trainable

        for p in self.ppg_encoder.parameters():
            p.requires_grad = trainable

        self.ppg_encoder.train(trainable)

    def train(self, mode: bool = True):
        super().train(mode)

        if mode and not self._encoder_trainable:
            self.ppg_encoder.eval()

        return self


def run_supervised_stage(
    stage_name,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    y_mean,
    y_std,
    residual_lambda,
    num_epochs,
    save_path,
    print_every: int = 1,
):
    """
    Shared training loop for Stage 2 (frozen encoder) and Stage 3
    (fine-tune). Both stages use the exact same train_one_epoch /
    validate_one_epoch functions from train1.py - only the
    optimizer and the encoder's trainability differ between stages.

    print_every > 1 throttles the per-epoch log line (useful when
    this runs inside a cross-validation loop over many folds);
    checkpoint saves are always printed regardless.
    """

    print("\n")
    print("=" * 70)
    print(stage_name)
    print("=" * 70)

    print("Trainable parameters:", count_parameters(model))

    best_val_rmse = float("inf")

    for epoch in range(1, num_epochs + 1):

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            y_mean=y_mean,
            y_std=y_std,
            residual_lambda=residual_lambda,
        )

        val_metrics = validate_one_epoch(
            model, val_loader, criterion, device,
            y_mean=y_mean, y_std=y_std,
        )

        scheduler.step(val_metrics["loss"])

        if epoch % print_every == 0 or epoch == num_epochs:
            print(
                f"[{stage_name}] Epoch [{epoch:03d}/{num_epochs:03d}] | "
                f"Train RMSE: {train_metrics['rmse']:.4f} | "
                f"Val RMSE: {val_metrics['rmse']:.4f} | "
                f"Linear Val RMSE: {val_metrics['linear_rmse']:.4f} | "
                f"Delta mean: {val_metrics['delta_mean']:.4f} | "
                f"Delta std: {val_metrics['delta_std']:.4f}"
            )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_rmse": best_val_rmse,
                    "val_mse": val_metrics["mse"],
                    "linear_val_rmse": val_metrics["linear_rmse"],
                    "y_mean": y_mean,
                    "y_std": y_std,
                },
                save_path,
            )

            print(f"[{stage_name}] Saved best model to {save_path} (Val RMSE={best_val_rmse:.4f})")

    print(f"\nBest {stage_name} validation RMSE: {best_val_rmse:.4f}")

    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, best_val_rmse


# =========================================================
# One leave-one-subject-out fold
# =========================================================

def multi_subject_split(
    windowed_df,
    subject_col,
    val_subjects,
    test_subjects,
):
    """
    Same no-leakage contract as train1.subject_wise_split, but
    allows more than one validation subject (still exactly one
    held-out test subject).

    Diagnosing the shrunk-model LOSO run showed the losing folds had
    validation RMSE tracking BELOW the linear baseline for the
    entire run while test RMSE ended up above it - i.e. the model
    was fitting the single validation subject's idiosyncratic bias,
    not overfitting the training data. Averaging checkpoint
    selection over 2 validation subjects instead of 1 should make
    that early-stopping signal less tied to any one subject.
    """

    all_subjects = set(windowed_df[subject_col].unique())
    held_out = set(val_subjects) | set(test_subjects)

    missing_subjects = held_out - all_subjects
    if missing_subjects:
        raise ValueError(
            f"These held-out subjects are not in the dataset: {sorted(missing_subjects)}."
        )

    train_subjects = sorted(all_subjects - held_out)
    if len(train_subjects) == 0:
        raise ValueError("No training subjects remain after validation/test split.")

    train_df = windowed_df[windowed_df[subject_col].isin(train_subjects)].reset_index(drop=True)
    val_df = windowed_df[windowed_df[subject_col].isin(val_subjects)].reset_index(drop=True)
    test_df = windowed_df[windowed_df[subject_col].isin(test_subjects)].reset_index(drop=True)

    return train_df, val_df, test_df


def run_one_fold(
    cfg,
    windowed_df,
    val_subjects,
    test_subjects,
    fold_tag,
    device,
    checkpoint_dir,
):
    """
    Runs the full 3-stage pipeline (pretrain -> freeze -> fine-tune)
    for one train/val/test subject split and returns a dict of
    summary metrics.

    Stage 1 is re-run from scratch on THIS fold's train+val windows
    only, so no information about the held-out test subject - not
    even unlabeled raw-signal structure - leaks into the pretrained
    encoder. The held-out val subject(s) are used only for early
    stopping / checkpoint selection in every stage, never for
    gradient updates or for picking the test-time model.
    """

    set_seed(cfg["seed"])

    train_df, val_df, test_df = multi_subject_split(
        windowed_df=windowed_df,
        subject_col=cfg["subject_col"],
        val_subjects=val_subjects,
        test_subjects=test_subjects,
    )

    print(
        f"[{fold_tag}] train={sorted(train_df[cfg['subject_col']].unique())} "
        f"val={val_subjects} test={test_subjects}"
    )
    print(
        f"[{fold_tag}] windows -> "
        f"train:{len(train_df)} val:{len(val_df)} test:{len(test_df)}"
    )

    # -----------------------------------------------------
    # Classical ratio-of-ratios calibration (train subjects only)
    # -----------------------------------------------------

    R_train = train_df["r_val"].values.astype(np.float64)
    SpO2_train = train_df[cfg["target_col"]].values.astype(np.float64)
    linear_b, linear_a = np.polyfit(R_train, SpO2_train, deg=1)

    R_test = test_df["r_val"].values.astype(np.float64)
    SpO2_test = test_df[cfg["target_col"]].values.astype(np.float64)
    test_linear_pred = linear_a + linear_b * R_test
    linear_test_metrics = compute_regression_metrics(SpO2_test, test_linear_pred)

    # -----------------------------------------------------
    # Target normalization (train subjects only)
    # -----------------------------------------------------

    y_train = train_df[cfg["target_col"]].values.astype(np.float32)
    y_mean = float(y_train.mean())
    y_std = max(float(y_train.std()), 1e-6)

    # -----------------------------------------------------
    # Datasets / loaders
    # -----------------------------------------------------

    def make_loader(df, shuffle):
        ds = RawSpO2WindowDataset(
            df, cfg["seq_feature_cols"], target_col=cfg["target_col"],
            seq_len=cfg["window_size"], normalize_seq=cfg["normalize_seq"],
            y_mean=y_mean, y_std=y_std,
        )
        return DataLoader(
            ds, batch_size=cfg["batch_size"], shuffle=shuffle,
            num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available(),
        )

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df, shuffle=False)
    test_loader = make_loader(test_df, shuffle=False)

    # -----------------------------------------------------
    # STAGE 1: pretrain the token encoder-decoder
    # -----------------------------------------------------

    ae_save_path = os.path.join(checkpoint_dir, f"{fold_tag}_autoencoder.pt")

    encoder_state_dict = pretrain_autoencoder(
        train_loader=train_loader, val_loader=val_loader, device=device,
        hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
        lr=cfg["ae_lr"], weight_decay=cfg["ae_weight_decay"],
        num_epochs=cfg["ae_epochs"], save_path=ae_save_path,
    )

    # -----------------------------------------------------
    # STAGE 2: drop the decoder, freeze the encoder
    # -----------------------------------------------------

    model = PretrainedRawSelfAttentionSpO2Net(
        hidden_dim=cfg["hidden_dim"], num_heads=cfg["num_heads"], dropout=cfg["dropout"],
        linear_a=linear_a, linear_b=linear_b, max_seq_len=cfg["window_size"],
        max_residual=cfg["max_residual"],
    ).to(device)

    model.load_pretrained_encoder(encoder_state_dict)
    model.set_encoder_trainable(False)

    criterion = nn.HuberLoss(delta=1.0)

    frozen_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["frozen_lr"], weight_decay=cfg["frozen_weight_decay"],
    )
    frozen_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        frozen_optimizer, mode="min", factor=0.5, patience=10,
    )

    frozen_save_path = os.path.join(checkpoint_dir, f"{fold_tag}_frozen.pt")

    model, frozen_val_rmse = run_supervised_stage(
        stage_name=f"[{fold_tag}] STAGE 2 (frozen encoder)",
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=frozen_optimizer, scheduler=frozen_scheduler, criterion=criterion,
        device=device, y_mean=y_mean, y_std=y_std,
        residual_lambda=cfg["residual_lambda"],
        num_epochs=cfg["frozen_epochs"], save_path=frozen_save_path,
        print_every=cfg["print_every"],
    )

    frozen_test_metrics = validate_one_epoch(
        model=model, loader=test_loader, criterion=criterion,
        device=device, y_mean=y_mean, y_std=y_std,
    )

    # -----------------------------------------------------
    # STAGE 3: unfreeze the encoder, fine-tune end-to-end
    # -----------------------------------------------------

    model.set_encoder_trainable(True)

    encoder_param_ids = {id(p) for p in model.ppg_encoder.parameters()}

    encoder_params = []
    head_params = []

    for p in model.parameters():
        if id(p) in encoder_param_ids:
            encoder_params.append(p)
        else:
            head_params.append(p)

    finetune_optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg["encoder_lr"]},
            {"params": head_params, "lr": cfg["head_lr"]},
        ],
        weight_decay=cfg["finetune_weight_decay"],
    )
    finetune_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        finetune_optimizer, mode="min", factor=0.5, patience=10,
    )

    finetune_save_path = os.path.join(checkpoint_dir, f"{fold_tag}_finetuned.pt")

    model, finetuned_val_rmse = run_supervised_stage(
        stage_name=f"[{fold_tag}] STAGE 3 (fine-tune)",
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=finetune_optimizer, scheduler=finetune_scheduler, criterion=criterion,
        device=device, y_mean=y_mean, y_std=y_std,
        residual_lambda=cfg["residual_lambda"],
        num_epochs=cfg["finetune_epochs"], save_path=finetune_save_path,
        print_every=cfg["print_every"],
    )

    finetuned_test_metrics = validate_one_epoch(
        model=model, loader=test_loader, criterion=criterion,
        device=device, y_mean=y_mean, y_std=y_std,
    )

    result = {
        "fold": fold_tag,
        "val_subjects": list(val_subjects),
        "test_subject": test_subjects[0],
        "linear_test_rmse": linear_test_metrics["rmse"],
        "linear_test_mae": linear_test_metrics["mae"],
        "frozen_val_rmse": frozen_val_rmse,
        "frozen_test_rmse": frozen_test_metrics["rmse"],
        "finetuned_val_rmse": finetuned_val_rmse,
        "finetuned_test_rmse": finetuned_test_metrics["rmse"],
        "finetuned_test_mae": finetuned_test_metrics["mae"],
    }

    print(
        f"\n[{fold_tag} SUMMARY] test={test_subjects[0]} | "
        f"Linear RMSE: {result['linear_test_rmse']:.4f} | "
        f"Frozen RMSE: {result['frozen_test_rmse']:.4f} | "
        f"Finetuned RMSE: {result['finetuned_test_rmse']:.4f}"
    )

    return result


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    device = get_device()

    # =====================================================
    # CONFIG
    # =====================================================

    cfg = dict(
        seed=42,

        data_dir="RValues_RawPPG_Aref_",
        pattern="*.h5",

        subject_col="SubjectID",
        time_col="timestamp",
        target_col="SpO2_Rad",
        seq_feature_cols=["red_win_filtered", "ir_win_filtered"],

        window_size=400,
        stride=400,

        batch_size=16,
        num_workers=0,
        normalize_seq=False,

        # Shrunk from hidden_dim=32/num_heads=4/dropout=0.2: two
        # rounds of LOSO showed regularizing the residual head's
        # OUTPUT (max_residual, residual_lambda) barely moved
        # generalization (fine-tuned mean RMSE 3.2947 -> 3.2809,
        # still worse than the 3.1913 linear baseline). That points
        # at raw model capacity, not output scale, as the actual
        # problem - with ~9 training subjects and ~200 windows total
        # per fold, a 32-wide attention stack has far more capacity
        # than the data can constrain. Halving hidden_dim (and heads
        # with it) roughly quarters the attention projection
        # parameter count; the extra dropout adds further capacity
        # control.
        hidden_dim=16,
        num_heads=2,
        dropout=0.3,

        # Left at the values from the previous (capacity-neutral)
        # regularization attempt - that change was harmless, just
        # not sufficient on its own.
        max_residual=2.0,
        residual_lambda=0.01,

        # STAGE 1: encoder-decoder pretraining.
        # NOTE: this dataset is small, so extra epochs are cheap.
        # Watch the AE val MSE curve for a given fold - if it is
        # still falling steeply at ae_epochs, raise ae_epochs
        # rather than trusting the final number.
        ae_lr=1e-3,
        ae_weight_decay=1e-4,
        ae_epochs=150,

        # STAGE 2: frozen pretrained encoder.
        frozen_lr=1e-3,
        frozen_weight_decay=1e-3,
        frozen_epochs=60,

        # STAGE 3: end-to-end fine-tuning.
        encoder_lr=1e-4,   # pretrained encoder: smaller LR
        head_lr=5e-4,      # attention / residual head: larger LR
        finetune_weight_decay=1e-3,
        finetune_epochs=60,

        # See "Leave-one-subject-out cross-validation" comment below
        # for why this was raised from 1.
        n_val_subjects=2,

        # Throttle the per-epoch log line inside each fold so an
        # 11-fold LOSO run stays readable. Set to 1 for full detail
        # on a single fold.
        print_every=10,
    )

    checkpoint_dir = "loso_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # =====================================================
    # Load + window data once - only the subject split
    # changes per fold, not the windowing.
    # =====================================================

    df = load_all_files(data_dir=cfg["data_dir"], pattern=cfg["pattern"])

    print("Raw dataframe shape:", df.shape)
    print("Subjects:", df[cfg["subject_col"]].nunique())

    windowed_df = create_windowed_dataset(
        df=df,
        subject_col=cfg["subject_col"],
        time_col=cfg["time_col"],
        target_col=cfg["target_col"],
        seq_feature_cols=cfg["seq_feature_cols"],
        window_size=cfg["window_size"],
        stride=cfg["stride"],
        drop_incomplete=True,
    )

    print("Windowed dataframe shape:", windowed_df.shape)

    # =====================================================
    # Leave-one-subject-out cross-validation
    #
    # For each subject, hold it out as TEST. The next n_val_subjects
    # subjects in the (sorted, wrapping) rotation become VAL, used
    # only for early stopping / checkpoint selection - never for
    # gradient updates or for choosing the test-time model. Every
    # remaining subject is TRAIN. Every subject gets exactly one
    # turn as TEST, so the aggregate result below is not tied to any
    # one lucky/unlucky split.
    #
    # n_val_subjects > 1: a single validation subject turned out to
    # be a noisy early-stopping / checkpoint-selection signal - on
    # the losing folds, val RMSE tracked BELOW the linear baseline
    # for the entire run while test RMSE ended up above it, meaning
    # the model was fitting that one subject's idiosyncratic bias
    # rather than a generalizable correction. Averaging the
    # selection signal over more than one held-out subject should
    # make it less tied to any single subject's quirks.
    # =====================================================

    subjects = sorted(windowed_df[cfg["subject_col"]].unique())
    n_folds = len(subjects)
    n_val_subjects = cfg["n_val_subjects"]

    print(f"\nRunning leave-one-subject-out CV over {n_folds} subjects:")
    print(subjects)

    fold_results = []

    for i, test_subject in enumerate(subjects):

        val_subjects = [
            subjects[(i + 1 + k) % n_folds]
            for k in range(n_val_subjects)
        ]
        fold_tag = f"fold{i + 1:02d}_{test_subject}"

        print("\n")
        print("#" * 70)
        print(f"# FOLD {i + 1}/{n_folds}  |  test={test_subject}  val={val_subjects}")
        print("#" * 70)

        result = run_one_fold(
            cfg=cfg,
            windowed_df=windowed_df,
            val_subjects=val_subjects,
            test_subjects=[test_subject],
            fold_tag=fold_tag,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )

        fold_results.append(result)

    # =====================================================
    # Aggregate LOSO summary
    # =====================================================

    def summarize(key):
        vals = np.array([r[key] for r in fold_results], dtype=np.float64)
        return float(vals.mean()), float(vals.std())

    print("\n")
    print("=" * 70)
    print("LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    header = (
        f"{'test_subject':14s} {'linear_rmse':>12s} "
        f"{'frozen_rmse':>12s} {'finetuned_rmse':>15s}"
    )
    print(header)
    print("-" * len(header))

    for r in fold_results:
        print(
            f"{r['test_subject']:14s} "
            f"{r['linear_test_rmse']:12.4f} "
            f"{r['frozen_test_rmse']:12.4f} "
            f"{r['finetuned_test_rmse']:15.4f}"
        )

    linear_mean, linear_std = summarize("linear_test_rmse")
    frozen_mean, frozen_std = summarize("frozen_test_rmse")
    finetuned_mean, finetuned_std = summarize("finetuned_test_rmse")
    finetuned_mae_mean, finetuned_mae_std = summarize("finetuned_test_mae")

    print("-" * len(header))
    print(
        f"{'MEAN':14s} "
        f"{linear_mean:12.4f} "
        f"{frozen_mean:12.4f} "
        f"{finetuned_mean:15.4f}"
    )
    print(
        f"{'STD':14s} "
        f"{linear_std:12.4f} "
        f"{frozen_std:12.4f} "
        f"{finetuned_std:15.4f}"
    )

    print(f"\nFine-tuned test MAE (mean +/- std): {finetuned_mae_mean:.4f} +/- {finetuned_mae_std:.4f}")

    n_better = sum(
        1 for r in fold_results if r["finetuned_test_rmse"] < r["linear_test_rmse"]
    )
    print(
        f"\nFine-tuned model beat the linear baseline on "
        f"{n_better}/{n_folds} held-out test subjects."
    )

    print(f"\nPer-fold checkpoints saved under: {checkpoint_dir}/")
