import os
import glob
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Utilities
# =========================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using device: cuda:0")
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using device: cpu")

    return device


# =========================================================
# Data loading
# =========================================================

def load_all_files(data_dir: str, pattern: str = "*.h5") -> pd.DataFrame:
    file_paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No files found in {data_dir} matching {pattern}")

    dfs = []
    for fp in file_paths:
        temp_df = pd.read_hdf(fp).copy()
        if "source_file" not in temp_df.columns:
            temp_df["source_file"] = os.path.basename(fp)
        dfs.append(temp_df)

    return pd.concat(dfs, ignore_index=True)


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    col_types = {}
    for col in df.columns:
        val = df[col].iloc[0]
        if isinstance(val, (list, tuple, np.ndarray)):
            col_types[col] = "vector"
        elif pd.api.types.is_numeric_dtype(df[col]):
            col_types[col] = "scalar"
        else:
            col_types[col] = "other"
    return col_types


def create_windowed_dataset(
    df: pd.DataFrame,
    subject_col: str = "SubjectID",
    time_col: str = "timestamp",
    target_col: str = "SpO2_Rad",
    seq_feature_cols: List[str] = None,
    window_size: int = 400,
    stride: int = 400,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """
    Builds one row per window.

    For each window:
    - seq_feature_cols become length-400 arrays
    - target_col becomes one scalar label for the window
    - metadata is taken from the first row
    """
    if seq_feature_cols is None:
        seq_feature_cols = ['red_win_filtered', 'ir_win_filtered']

    required_cols = seq_feature_cols + [
        target_col, subject_col, time_col, "r_val"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    all_samples = []

    for subject_id, group in df.groupby(subject_col):
        group = group.sort_values(time_col).reset_index(drop=True)
        n = len(group)

        starts = range(0, n - window_size + 1, stride) if drop_incomplete else range(0, n, stride)

        for start in starts:
            end = start + window_size
            window = group.iloc[start:end]

            if len(window) < window_size and drop_incomplete:
                continue

            sample = {
                "SubjectID": subject_id,
                "window_start_idx": start,
                "window_end_idx": min(end - 1, n - 1),
                "source_file": window["source_file"].iloc[0] if "source_file" in window.columns else "",
                "r_val": np.float32(window["r_val"].iloc[0]),
            }

            # sequence inputs
            for col in seq_feature_cols:
                arr = window[col].to_numpy(dtype=np.float32)
                if arr.ndim != 1 or len(arr) != window_size:
                    raise ValueError(
                        f"Column {col} did not form a 1D window of length {window_size}. "
                        f"Got shape {arr.shape}."
                    )
                sample[col] = arr

            # scalar label per window
            sample[target_col] = np.float32(window[target_col].iloc[0])

            all_samples.append(sample)

    return pd.DataFrame(all_samples)


def subject_wise_split(
    windowed_df: pd.DataFrame,
    subject_col: str = "SubjectID",
    val_subjects: List = None,
    test_subjects: List = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Deterministic subject-wise train/validation/test split.

    Exactly one subject is reserved for validation and one different subject
    is reserved for testing. All remaining subjects are used for training.
    This avoids subject leakage across splits.
    """
    if val_subjects is None or len(val_subjects) != 1:
        raise ValueError("Provide exactly one validation subject, e.g. val_subjects=['Subject11'].")

    if test_subjects is None or len(test_subjects) != 1:
        raise ValueError("Provide exactly one test subject, e.g. test_subjects=['Subject12'].")

    if val_subjects[0] == test_subjects[0]:
        raise ValueError("Validation and test subjects must be different.")

    all_subjects = np.array(sorted(windowed_df[subject_col].unique()))
    held_out = set(val_subjects) | set(test_subjects)

    missing_subjects = held_out - set(all_subjects)
    if missing_subjects:
        raise ValueError(
            f"These held-out subjects are not in the dataset: {sorted(missing_subjects)}. "
            f"Available subjects are: {all_subjects.tolist()}"
        )

    train_subjects = np.array([s for s in all_subjects if s not in held_out])

    if len(train_subjects) == 0:
        raise ValueError("No training subjects remain after validation/test split.")

    train_df = windowed_df[windowed_df[subject_col].isin(train_subjects)].reset_index(drop=True)
    val_df = windowed_df[windowed_df[subject_col].isin(val_subjects)].reset_index(drop=True)
    test_df = windowed_df[windowed_df[subject_col].isin(test_subjects)].reset_index(drop=True)

    print("\nDeterministic subject split:")
    print("Train subjects:", train_subjects.tolist())
    print("Validation subject:", val_subjects)
    print("Test subject:", test_subjects)

    return train_df, val_df, test_df


# =========================================================
# Dataset
# =========================================================

class RawSpO2WindowDataset(Dataset):
    def __init__(
        self,
        windowed_df: pd.DataFrame,
        seq_feature_cols: List[str],
        target_col: str = "SpO2_Rad",
        seq_len: int = 400,
        normalize_seq: bool = True,
        y_mean: float = None,
        y_std: float = None,
    ):
        self.df = windowed_df.reset_index(drop=True).copy()
        self.seq_feature_cols = seq_feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.normalize_seq = normalize_seq
        self.y_mean = y_mean
        self.y_std = y_std

        required_cols = seq_feature_cols + [target_col, "r_val"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing column: {col}")

    def __len__(self):
        return len(self.df)

    def _to_1d_float_array(self, x, col_name: str, expected_len: int) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"Column '{col_name}' is not 1D. Got shape {arr.shape}.")
        if len(arr) != expected_len:
            raise ValueError(f"Column '{col_name}' length mismatch. Expected {expected_len}, got {len(arr)}.")
        return arr

    def _normalize_signal(self, x: np.ndarray) -> np.ndarray:
        std = x.std()
        if std < 1e-8:
            std = 1.0
        return (x - x.mean()) / std

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        seq_features = []
        for col in self.seq_feature_cols:
            arr = self._to_1d_float_array(row[col], col, self.seq_len)
            if self.normalize_seq:
                arr = self._normalize_signal(arr)
            seq_features.append(arr)

        x_seq = np.stack(seq_features, axis=0).astype(np.float32)  # [2, 400]
        x_r = np.float32(row["r_val"])

        y = np.float32(row[self.target_col])

        if self.y_mean is not None and self.y_std is not None:
            y = (y - self.y_mean) / self.y_std

        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(x_r, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


# =========================================================
# Model blocks
# =========================================================

class ConvTokenEncoder(nn.Module):
    """
    Joint temporal encoder for RED/IR interaction channels.

    Expected input:
        [B, C, T]

    For this model:
        C = 4
        channel 0 = RED
        channel 1 = IR
        channel 2 = RED - IR
        channel 3 = RED * IR
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                32,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(
                32,
                hidden_dim,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: [B, C, T]
        x = self.net(x)          # [B, D, T]
        x = x.transpose(1, 2)    # [B, T, D]
        return x


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional embedding added to temporal tokens.
    """

    def __init__(
        self,
        max_len: int,
        dim: int,
    ):
        super().__init__()

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_len, dim)
        )

        nn.init.normal_(
            self.pos_embedding,
            mean=0.0,
            std=0.02,
        )

    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)

        if T > self.pos_embedding.size(1):
            raise ValueError(
                f"Sequence length {T} exceeds positional encoding "
                f"capacity {self.pos_embedding.size(1)}."
            )

        return x + self.pos_embedding[:, :T, :]


class TemporalSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.Tanh(),
            nn.Linear(2 * dim, 1)
        )

    def forward(self, x):
        # x: [B, T, D]
        w = self.score(x)
        w = torch.softmax(w, dim=1)
        pooled = (x * w).sum(dim=1)
        return pooled


# =========================================================
# Joint RED+IR encoder with temporal self-attention
# =========================================================

class RawSelfAttentionSpO2Net(nn.Module):
    """
    Physics-guided residual SpO2 estimator.

    Architecture
    ------------
    RED
    IR
    RED - IR
    RED * IR
        -> joint temporal CNN
        -> learnable positional encoding
        -> temporal self-attention
        -> attention pooling

    R
        -> small embedding

    [PPG feature, R feature]
        -> residual head
        -> bounded Delta_SpO2

    Final prediction:
        SpO2_hat = a + bR + Delta_SpO2

    Skin tone is not used.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.2,
        linear_a: float = 0.0,
        linear_b: float = 0.0,
        max_seq_len: int = 400,
        max_residual: float = 5.0,
    ):
        super().__init__()

        self.max_residual = float(max_residual)

        # ----------------------------------------------------
        # 4-channel PPG encoder:
        #
        # 0: RED
        # 1: IR
        # 2: RED - IR
        # 3: RED * IR
        # ----------------------------------------------------
        self.ppg_encoder = ConvTokenEncoder(
            in_channels=4,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # ----------------------------------------------------
        # Explicit temporal position information
        # ----------------------------------------------------
        self.positional_encoding = LearnablePositionalEncoding(
            max_len=max_seq_len,
            dim=hidden_dim,
        )

        # ----------------------------------------------------
        # Temporal self-attention
        # ----------------------------------------------------
        self.self_attn = TemporalSelfAttentionBlock(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ----------------------------------------------------
        # Window-level pooling
        # ----------------------------------------------------
        self.pool = AttentionPooling(
            dim=hidden_dim
        )

        # ----------------------------------------------------
        # Explicit R embedding
        # ----------------------------------------------------
        self.r_embed = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
        )

        # ----------------------------------------------------
        # Residual correction head
        # ----------------------------------------------------
        correction_input_dim = hidden_dim + 8

        self.correction_head = nn.Sequential(
            nn.Linear(
                correction_input_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                16,
            ),
            nn.ReLU(),

            nn.Linear(
                16,
                1,
            ),
        )

        # ----------------------------------------------------
        # Classical ratio-of-ratios calibration
        # ----------------------------------------------------
        self.register_buffer(
            "linear_a",
            torch.tensor(
                float(linear_a),
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "linear_b",
            torch.tensor(
                float(linear_b),
                dtype=torch.float32,
            ),
        )

        # Start exactly from the classical baseline.
        nn.init.zeros_(
            self.correction_head[-1].weight
        )

        nn.init.zeros_(
            self.correction_head[-1].bias
        )

    def forward(self, x_seq, x_r):

        # ----------------------------------------------------
        # Classical physics baseline
        # ----------------------------------------------------
        spo2_linear = (
            self.linear_a
            + self.linear_b * x_r
        )

        # ----------------------------------------------------
        # Raw synchronized signals
        #
        # x_seq:
        #     [B, 2, T]
        # ----------------------------------------------------
        red = x_seq[:, 0:1, :]
        ir = x_seq[:, 1:2, :]

        # ----------------------------------------------------
        # Explicit RED/IR interaction channels
        # ----------------------------------------------------
        red_minus_ir = red - ir
        red_times_ir = red * ir

        ppg_input = torch.cat(
            [
                red,
                ir,
                red_minus_ir,
                red_times_ir,
            ],
            dim=1,
        )
        # [B, 4, T]

        # ----------------------------------------------------
        # Joint temporal feature extraction
        # ----------------------------------------------------
        ppg_tokens = self.ppg_encoder(
            ppg_input
        )
        # [B, T, D]

        # ----------------------------------------------------
        # Positional information
        # ----------------------------------------------------
        ppg_tokens = self.positional_encoding(
            ppg_tokens
        )

        # ----------------------------------------------------
        # Temporal self-attention
        # ----------------------------------------------------
        ppg_tokens = self.self_attn(
            ppg_tokens
        )

        # ----------------------------------------------------
        # Window-level PPG representation
        # ----------------------------------------------------
        ppg_feature = self.pool(
            ppg_tokens
        )
        # [B, D]

        # ----------------------------------------------------
        # R embedding
        # ----------------------------------------------------
        if x_r.ndim == 1:
            r_input = x_r.unsqueeze(-1)
        else:
            r_input = x_r

        r_feature = self.r_embed(
            r_input
        )
        # [B, 8]

        # ----------------------------------------------------
        # Residual feature
        # ----------------------------------------------------
        residual_feature = torch.cat(
            [
                ppg_feature,
                r_feature,
            ],
            dim=-1,
        )

        raw_delta = self.correction_head(
            residual_feature
        ).squeeze(-1)

        # ----------------------------------------------------
        # Bounded correction
        #
        # Prevent the NN from completely overriding the
        # physics-based estimate on unseen subjects.
        # ----------------------------------------------------
        delta_spo2 = (
            self.max_residual
            * torch.tanh(raw_delta)
        )

        # ----------------------------------------------------
        # Final prediction
        # ----------------------------------------------------
        y_hat = (
            spo2_linear
            + delta_spo2
        )

        return (
            y_hat,
            spo2_linear,
            delta_spo2,
        )


# =========================================================
# Training helpers
# =========================================================

class RunningAverage:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(1, self.count)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    bias = float(np.mean(y_pred - y_true))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    y_mean,
    y_std,
    residual_lambda=0.0,
):
    """
    Network outputs SpO2 directly in ORIGINAL SpO2 units.

    Dataset target y is normalized, so convert it back before computing
    the residual-learning loss.

    Optional residual regularization:
        residual_lambda * mean(delta^2)

    This discourages the network from unnecessarily overriding the
    strong linear ratio-of-ratios baseline.
    """

    model.train()

    loss_meter = RunningAverage()

    preds_all = []
    targets_all = []
    linear_all = []
    delta_all = []

    for x_seq, x_r, y_norm in loader:

        x_seq = x_seq.to(device)
        x_r = x_r.to(device).float()
        y_norm = y_norm.to(device).float()

        # Convert normalized target back to original SpO2 units.
        y_true = (
            y_norm * y_std
            + y_mean
        )

        optimizer.zero_grad()

        y_hat, y_linear, delta = model(
            x_seq,
            x_r,
        )

        prediction_loss = criterion(
            y_hat,
            y_true,
        )

        residual_penalty = (
            delta.pow(2).mean()
        )

        loss = (
            prediction_loss
            + residual_lambda * residual_penalty
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        bs = y_true.size(0)

        loss_meter.update(
            loss.item(),
            bs,
        )

        preds_all.append(
            y_hat.detach().cpu().numpy()
        )
        targets_all.append(
            y_true.detach().cpu().numpy()
        )
        linear_all.append(
            y_linear.detach().cpu().numpy()
        )
        delta_all.append(
            delta.detach().cpu().numpy()
        )

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    linear_all = np.concatenate(linear_all)
    delta_all = np.concatenate(delta_all)

    metrics = compute_regression_metrics(
        targets_all,
        preds_all,
    )

    linear_metrics = compute_regression_metrics(
        targets_all,
        linear_all,
    )

    metrics["loss"] = loss_meter.avg
    metrics["linear_rmse"] = linear_metrics["rmse"]
    metrics["delta_mean"] = float(delta_all.mean())
    metrics["delta_std"] = float(delta_all.std())

    return metrics


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    y_mean=None,
    y_std=None,
):
    model.eval()

    loss_meter = RunningAverage()

    preds_all = []
    targets_all = []
    linear_all = []
    delta_all = []

    for x_seq, x_r, y_norm in loader:

        x_seq = x_seq.to(device)
        x_r = x_r.to(device).float()
        y_norm = y_norm.to(device).float()

        if y_mean is not None and y_std is not None:
            y_true = (
                y_norm * y_std
                + y_mean
            )
        else:
            y_true = y_norm

        y_hat, y_linear, delta = model(
            x_seq,
            x_r,
        )

        loss = criterion(
            y_hat,
            y_true,
        )

        bs = y_true.size(0)

        loss_meter.update(
            loss.item(),
            bs,
        )

        preds_all.append(
            y_hat.cpu().numpy()
        )
        targets_all.append(
            y_true.cpu().numpy()
        )
        linear_all.append(
            y_linear.cpu().numpy()
        )
        delta_all.append(
            delta.cpu().numpy()
        )

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    linear_all = np.concatenate(linear_all)
    delta_all = np.concatenate(delta_all)

    metrics = compute_regression_metrics(
        targets_all,
        preds_all,
    )

    linear_metrics = compute_regression_metrics(
        targets_all,
        linear_all,
    )

    metrics["loss"] = loss_meter.avg

    metrics["linear_mse"] = (
        linear_metrics["mse"]
    )

    metrics["linear_rmse"] = (
        linear_metrics["rmse"]
    )

    metrics["linear_mae"] = (
        linear_metrics["mae"]
    )

    metrics["delta_mean"] = float(
        delta_all.mean()
    )

    metrics["delta_std"] = float(
        delta_all.std()
    )

    metrics["delta_abs_mean"] = float(
        np.abs(delta_all).mean()
    )

    return metrics


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    set_seed(42)
    device = get_device()

    # -----------------------------
    # Config
    # -----------------------------
    data_dir = "RValues_RawPPG_Aref_"
    pattern = "*.h5"

    subject_col = "SubjectID"
    time_col = "timestamp"
    target_col = "SpO2_Rad"

    seq_feature_cols = ['red_win_filtered', 'ir_win_filtered']

    window_size = 400
    stride = 400
    # Hold out exactly one patient for validation and one for testing.
    # Change these IDs to subjects that exist in your dataset.
    val_subjects = ["Subject11"]
    test_subjects = ["Subject2"]
    batch_size = 16
    num_workers = 0
    normalize_seq = False

    hidden_dim = 32
    num_heads = 4
    dropout = 0.2
    lr = 1e-3
    weight_decay = 1e-3
    num_epochs = 40
    max_residual = 5.0

    # Small penalty keeps the learned correction conservative.
    # Start with 0.01. You can also test 0.0 and 0.1.
    residual_lambda = 0.001

    save_path = "best_interaction_self_attention_spo2_model.pt"

    # -----------------------------
    # Load data
    # -----------------------------
    df = load_all_files(data_dir=data_dir, pattern=pattern)

    print("Raw dataframe shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Subjects:", df[subject_col].nunique())
    print("\nRaw target stats:")
    print(df[target_col].describe())

    windowed_df = create_windowed_dataset(
        df=df,
        subject_col=subject_col,
        time_col=time_col,
        target_col=target_col,
        seq_feature_cols=seq_feature_cols,
        window_size=window_size,
        stride=stride,
        drop_incomplete=True,
    )

    print("\nWindowed dataframe shape:", windowed_df.shape)
    print("Windowed target example:", windowed_df[target_col].iloc[0])

    print("\nAvailable subjects:")
    print(sorted(windowed_df[subject_col].unique()))

    train_df, val_df, test_df = subject_wise_split(
        windowed_df=windowed_df,
        subject_col=subject_col,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
    )

    print("\nTrain windows:", len(train_df))
    print("Validation windows:", len(val_df))
    print("Test windows:", len(test_df))

    # -----------------------------
    # Fit classical ratio-of-ratios calibration
    # USING TRAIN SUBJECTS ONLY
    #
    # SpO2 = a + b * R
    # -----------------------------
    R_train = train_df["r_val"].values.astype(np.float64)
    SpO2_train = train_df[target_col].values.astype(np.float64)

    linear_b, linear_a = np.polyfit(
        R_train,
        SpO2_train,
        deg=1,
    )

    print("\nPhysics calibration from TRAIN subjects only:")
    print(
        f"SpO2_linear = {linear_a:.6f} "
        f"+ ({linear_b:.6f}) * R"
    )

    # Verify the classical validation baseline before training NN.
    R_val = val_df["r_val"].values.astype(np.float64)
    SpO2_val = val_df[target_col].values.astype(np.float64)

    val_linear_pred = (
        linear_a
        + linear_b * R_val
    )

    linear_val_metrics = compute_regression_metrics(
        SpO2_val,
        val_linear_pred,
    )

    print(
        "Linear ratio-of-ratios validation | "
        f"MSE: {linear_val_metrics['mse']:.4f} | "
        f"RMSE: {linear_val_metrics['rmse']:.4f} | "
        f"MAE: {linear_val_metrics['mae']:.4f}"
    )

    R_test = test_df["r_val"].values.astype(np.float64)
    SpO2_test = test_df[target_col].values.astype(np.float64)

    test_linear_pred = linear_a + linear_b * R_test
    linear_test_metrics = compute_regression_metrics(
        SpO2_test,
        test_linear_pred,
    )

    print(
        "Linear ratio-of-ratios test | "
        f"MSE: {linear_test_metrics['mse']:.4f} | "
        f"RMSE: {linear_test_metrics['rmse']:.4f} | "
        f"MAE: {linear_test_metrics['mae']:.4f}"
    )

    # -----------------------------
    # Normalize target from train only
    # -----------------------------
    y_train = train_df[target_col].values.astype(np.float32)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    y_std = max(y_std, 1e-6)

    print("\nTarget normalization stats:")
    print("y_mean:", y_mean)
    print("y_std :", y_std)

    # -----------------------------
    # Datasets / loaders
    # -----------------------------
    train_dataset = RawSpO2WindowDataset(
        windowed_df=train_df,
        seq_feature_cols=seq_feature_cols,
        target_col=target_col,
        seq_len=window_size,
        normalize_seq=normalize_seq,
        y_mean=y_mean,
        y_std=y_std,
    )

    val_dataset = RawSpO2WindowDataset(
        windowed_df=val_df,
        seq_feature_cols=seq_feature_cols,
        target_col=target_col,
        seq_len=window_size,
        normalize_seq=normalize_seq,
        y_mean=y_mean,
        y_std=y_std,
    )

    test_dataset = RawSpO2WindowDataset(
        windowed_df=test_df,
        seq_feature_cols=seq_feature_cols,
        target_col=target_col,
        seq_len=window_size,
        normalize_seq=normalize_seq,
        y_mean=y_mean,
        y_std=y_std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    x_seq_sample, x_r_sample, y_sample = next(iter(train_loader))
    print("\nBatch shapes:")
    print("x_seq shape:", x_seq_sample.shape)    # [B,2,400]
    print("x_r shape:", x_r_sample.shape)        # [B]
    print("y shape:", y_sample.shape)            # [B]

    # -----------------------------
    # Model
    # -----------------------------
    model = RawSelfAttentionSpO2Net(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        linear_a=linear_a,
        linear_b=linear_b,
        max_seq_len=window_size,
        max_residual=max_residual,
    ).to(device)

    print("\nModel architecture:")
    print(
        "RED, IR, RED-IR, RED*IR -> Joint CNN -> "
        "Positional Encoding -> Self-Attention -> Attention Pooling; "
        "R -> Embedding; then bounded residual correction"
    )

    criterion = nn.HuberLoss(delta=1.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
    )

    # -----------------------------
    # Train
    # -----------------------------
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
            model,
            val_loader,
            criterion,
            device,
            y_mean=y_mean,
            y_std=y_std,
        )

        scheduler.step(
            val_metrics["loss"]
        )

        train_eval = validate_one_epoch(
            model,
            train_loader,
            criterion,
            device,
            y_mean=y_mean,
            y_std=y_std,
        )

        print(
            f"Epoch [{epoch:03d}/{num_epochs:03d}] | "
            f"Train MSE: {train_eval['mse']:.4f} | "
            f"Train RMSE: {train_eval['rmse']:.4f} | "
            f"Val MSE: {val_metrics['mse']:.4f} | "
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
                    "linear_a": linear_a,
                    "linear_b": linear_b,
                    "residual_lambda": residual_lambda,
                    "max_residual": max_residual,
                    "hidden_dim": hidden_dim,
                    "num_heads": num_heads,
                    "dropout": dropout,
                    "weight_decay": weight_decay,
                    "seq_feature_cols": seq_feature_cols,
                    "val_subjects": val_subjects,
                    "test_subjects": test_subjects,
                    "window_size": window_size,
                    "y_mean": y_mean,
                    "y_std": y_std,
                },
                save_path,
            )
            print(f"Saved best model to {save_path} (Val RMSE={best_val_rmse:.4f})")

    print(f"\nTraining complete. Best Val RMSE: {best_val_rmse:.4f}")

    # -----------------------------
    # Final evaluation on TEST subject
    # The test subject was not used for training, scheduler updates,
    # checkpoint selection, or any other model-selection decision.
    # -----------------------------
    checkpoint = torch.load(
        save_path,
        map_location=device,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_metrics = validate_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        y_mean=y_mean,
        y_std=y_std,
    )

    print(
        "\nFINAL TEST RESULTS | "
        f"MSE: {test_metrics['mse']:.4f} | "
        f"RMSE: {test_metrics['rmse']:.4f} | "
        f"MAE: {test_metrics['mae']:.4f} | "
        f"Linear Test RMSE: {test_metrics['linear_rmse']:.4f} | "
        f"Delta mean: {test_metrics['delta_mean']:.4f} | "
        f"Delta std: {test_metrics['delta_std']:.4f}"
    )
