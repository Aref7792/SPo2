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
    - skintone becomes one scalar per window
    - metadata is taken from the first row
    """
    if seq_feature_cols is None:
        seq_feature_cols = ['red_win_filtered', 'ir_win_filtered']

    required_cols = seq_feature_cols + [
        target_col, subject_col, time_col, "skintone", "r_val"
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
                "skintone": np.float32(window["skintone"].iloc[0]),
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deterministic subject-wise split.

    You explicitly choose which patients/subjects are used for validation.
    This prevents random validation selection and avoids subject leakage.
    """
    if val_subjects is None or len(val_subjects) == 0:
        raise ValueError(
            "You must provide val_subjects, for example: val_subjects = [3, 7, 12]"
        )

    all_subjects = np.array(sorted(windowed_df[subject_col].unique()))
    val_subjects = np.array(val_subjects)

    missing_subjects = set(val_subjects) - set(all_subjects)
    if len(missing_subjects) > 0:
        raise ValueError(
            f"These validation subjects are not in the dataset: {sorted(missing_subjects)}. "
            f"Available subjects are: {all_subjects.tolist()}"
        )

    val_subjects_set = set(val_subjects)
    train_subjects = np.array([s for s in all_subjects if s not in val_subjects_set])

    if len(train_subjects) == 0:
        raise ValueError("All subjects were assigned to validation. No training subjects remain.")

    train_df = windowed_df[windowed_df[subject_col].isin(train_subjects)].reset_index(drop=True)
    val_df = windowed_df[windowed_df[subject_col].isin(val_subjects)].reset_index(drop=True)

    print("\nDeterministic subject split:")
    print("Train subjects:", train_subjects.tolist())
    print("Validation subjects:", val_subjects.tolist())

    return train_df, val_df


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

        required_cols = seq_feature_cols + [target_col, "skintone", "r_val"]
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
        x_skin = np.float32(row["skintone"])
        x_r = np.float32(row["r_val"])

        y = np.float32(row[self.target_col])

        if self.y_mean is not None and self.y_std is not None:
            y = (y - self.y_mean) / self.y_std

        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(x_skin, dtype=torch.float32),
            torch.tensor(x_r, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


# =========================================================
# Model blocks
# =========================================================

class ConvTokenEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: [B, 1, T]
        x = self.net(x)          # [B, D, T]
        x = x.transpose(1, 2)    # [B, T, D]
        return x


class CrossAttentionBlock(nn.Module):
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

    def forward(self, q, kv):
        # q, kv: [B, T, D]
        attn_out, _ = self.attn(q, kv, kv)
        x = self.norm1(q + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


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


class SkinFiLM(nn.Module):
    """
    Produces FiLM parameters for red and IR token streams from scalar skin tone.

    Output:
        gamma_r, beta_r, gamma_i, beta_i
    each with shape [B, 1, D]
    """
    def __init__(self, token_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 4 * token_dim),
        )
        self.token_dim = token_dim

    def forward(self, x_skin):
        # x_skin: [B] or [B,1]
        if x_skin.ndim == 1:
            x_skin = x_skin.unsqueeze(-1)  # [B,1]

        params = self.net(x_skin)          # [B,4D]
        gamma_r, beta_r, gamma_i, beta_i = torch.chunk(params, 4, dim=-1)
        #gamma_r, beta_r = torch.chunk(params, 2, dim=-1)

        gamma_r = gamma_r.unsqueeze(1)     # [B,1,D]
        beta_r = beta_r.unsqueeze(1)
        #gamma_i = gamma_i.unsqueeze(1)
        #beta_i = beta_i.unsqueeze(1)

        return gamma_r, beta_r, gamma_i, beta_i


# =========================================================
# Raw-only cross-attention model + SkinFiLM
# =========================================================

class RawCrossAttentionSpO2Net(nn.Module):
    """
    Physics-guided residual SpO2 estimator.

    Final prediction:
        SpO2_hat = SpO2_linear(R) + Delta_NN

    where:
        SpO2_linear(R) = linear_a + linear_b * R

    The neural network learns only the residual correction Delta_NN.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        linear_a: float = 0.0,
        linear_b: float = 0.0,
    ):
        super().__init__()

        self.red_encoder = ConvTokenEncoder(
            in_channels=1,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.ir_encoder = ConvTokenEncoder(
            in_channels=1,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.skin_film = SkinFiLM(
            token_dim=hidden_dim,
            hidden_dim=16,
        )

        self.red_to_ir = CrossAttentionBlock(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.ir_to_red = CrossAttentionBlock(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.temporal_attn = TemporalSelfAttentionBlock(
            hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.pool = AttentionPooling(
            hidden_dim * 2
        )

        # Small skin embedding used at the correction stage too.
        self.skin_embed = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
        )

        # Neural correction head.
        correction_input_dim = hidden_dim * 2 + 8

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

        # Store classical calibration as non-trainable buffers.
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

        # VERY IMPORTANT:
        # initialize residual correction to exactly zero.
        #
        # Therefore before any training:
        #     Delta_NN = 0
        # and
        #     SpO2_hat = a + bR
        #
        nn.init.zeros_(
            self.correction_head[-1].weight
        )
        nn.init.zeros_(
            self.correction_head[-1].bias
        )

    def forward(self, x_seq, x_skin, x_r):

        # ----------------------------------------------------
        # Classical physics-based estimate
        # ----------------------------------------------------
        spo2_linear = (
            self.linear_a
            + self.linear_b * x_r
        )

        # ----------------------------------------------------
        # Learned PPG pathway
        # ----------------------------------------------------
        red = x_seq[:, 0:1, :]
        ir = x_seq[:, 1:2, :]

        red = self.red_encoder(red)
        ir = self.ir_encoder(ir)

        gamma_r, beta_r, gamma_i, beta_i = (
            self.skin_film(x_skin)
        )
        # gamma_r, beta_r = (
        #             self.skin_film(x_skin)
        #         )
        # Preserve original baseline behavior:
        # only RED receives FiLM conditioning.
        red = red * (1.0 + gamma_r) + beta_r

        # Bidirectional Red/IR cross-attention
        red_cross = self.red_to_ir(
            red,
            ir,
        )

        ir_cross = self.ir_to_red(
            ir,
            red,
        )

        fused = torch.cat(
            [red_cross, ir_cross],
            dim=-1,
        )

        # Temporal self-attention
        fused = self.temporal_attn(
            fused
        )

        # Window-level representation
        ppg_feature = self.pool(
            fused
        )

        # Explicit skin feature for residual correction
        if x_skin.ndim == 1:
            skin_input = x_skin.unsqueeze(-1)
        else:
            skin_input = x_skin

        skin_feature = self.skin_embed(
            skin_input
        )

        residual_feature = torch.cat(
            [
                ppg_feature,
                skin_feature,
            ],
            dim=-1,
        )

        delta_spo2 = self.correction_head(
            residual_feature
        ).squeeze(-1)

        # ----------------------------------------------------
        # Physics baseline + learned residual
        # ----------------------------------------------------
        y_hat = (
            spo2_linear
            + delta_spo2
        )

        return y_hat, spo2_linear, delta_spo2


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

    for x_seq, x_skin, x_r, y_norm in loader:

        x_seq = x_seq.to(device)
        x_skin = x_skin.to(device).float()
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
            x_skin,
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

    for x_seq, x_skin, x_r, y_norm in loader:

        x_seq = x_seq.to(device)
        x_skin = x_skin.to(device).float()
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
            x_skin,
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
    # Choose exactly which patient/subject IDs are used for validation.
    # Change this after checking the printed available subjects.
    val_subjects = ["Subject11"]  # <-- CHANGE THIS, e.g., [3, 7, 12]
    batch_size = 16
    num_workers = 0
    normalize_seq = False

    hidden_dim = 64
    num_heads = 4
    dropout = 0.1
    lr = 1e-3
    weight_decay = 1e-4
    num_epochs = 40

    # Small penalty keeps the learned correction conservative.
    # Start with 0.01. You can also test 0.0 and 0.1.
    residual_lambda = 0.01

    save_path = "best_ratio_residual_spo2_model.pt"

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
    print("Windowed skintone example:", windowed_df["skintone"].iloc[0])

    print("\nAvailable subjects:")
    print(sorted(windowed_df[subject_col].unique()))

    train_df, val_df = subject_wise_split(
        windowed_df=windowed_df,
        subject_col=subject_col,
        val_subjects=val_subjects,
    )

    print("\nTrain windows:", len(train_df))
    print("Validation windows:", len(val_df))

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

    x_seq_sample, x_skin_sample, x_r_sample, y_sample = next(iter(train_loader))
    print("\nBatch shapes:")
    print("x_seq shape:", x_seq_sample.shape)    # [B,2,400]
    print("x_skin shape:", x_skin_sample.shape)  # [B]
    print("x_r shape:", x_r_sample.shape)        # [B]
    print("y shape:", y_sample.shape)            # [B]

    # -----------------------------
    # Model
    # -----------------------------
    model = RawCrossAttentionSpO2Net(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        linear_a=linear_a,
        linear_b=linear_b,
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
                    "seq_feature_cols": seq_feature_cols,
                    "window_size": window_size,
                    "y_mean": y_mean,
                    "y_std": y_std,
                },
                save_path,
            )
            print(f"Saved best model to {save_path} (Val RMSE={best_val_rmse:.4f})")

    print(f"\nTraining complete. Best Val RMSE: {best_val_rmse:.4f}")