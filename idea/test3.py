import os
import glob
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def chop_all_columns_smart(
    df: pd.DataFrame,
    subject_col: str = "SubjectID",
    time_col: str = "timestamp",
    window_size: int = 400,
    stride: int = 400,
    static_cols: List[str] = None,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """
    Chop all columns into windows.

    Rules:
    - static cols: keep one value per window
    - vector cols: keep the first row's vector as-is
    - scalar cols: convert 400 successive rows into a length-400 vector
    - other cols: keep first row value
    """
    if static_cols is None:
        static_cols = ["SubjectID", "source_file", "skintone"]

    col_types = detect_column_types(df)
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
                "window_start_idx": start,
                "window_end_idx": min(end - 1, n - 1),
            }

            for col in group.columns:
                if col in static_cols:
                    sample[col] = window[col].iloc[0]
                elif col_types[col] == "vector":
                    sample[col] = window[col].iloc[0]
                elif col_types[col] == "scalar":
                    sample[col] = window[col].to_numpy(dtype=np.float32)
                else:
                    sample[col] = window[col].iloc[0]

            all_samples.append(sample)

    return pd.DataFrame(all_samples)


def subject_wise_split(
    windowed_df: pd.DataFrame,
    subject_col: str = "SubjectID",
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subjects = windowed_df[subject_col].unique()
    train_subjects, val_subjects = train_test_split(
        subjects,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
    )

    train_df = windowed_df[windowed_df[subject_col].isin(train_subjects)].reset_index(drop=True)
    val_df = windowed_df[windowed_df[subject_col].isin(val_subjects)].reset_index(drop=True)

    return train_df, val_df


def select_sequence_feature_columns(
    windowed_df: pd.DataFrame,
    target_col: str,
    static_cols: List[str],
    seq_len: int = 400,
) -> List[str]:
    excluded = set(static_cols + [target_col, "window_start_idx", "window_end_idx", "timestamp", "sample_idx", "window_start_timestamp"])
    seq_cols = []

    for col in windowed_df.columns:
        if col in excluded:
            continue

        val = windowed_df[col].iloc[0]
        arr = np.asarray(val)

        if arr.ndim == 1 and len(arr) == seq_len:
            seq_cols.append(col)

    return seq_cols


def select_static_feature_columns(
    windowed_df: pd.DataFrame,
    target_col: str,
    static_cols: List[str],
) -> List[str]:
    excluded = set([target_col, "window_start_idx", "window_end_idx"])
    out = []

    for col in static_cols:
        if col in windowed_df.columns and col not in excluded:
            val = windowed_df[col].iloc[0]
            if np.isscalar(val) or isinstance(val, str):
                out.append(col)

    return out


class SpO2WindowDataset(Dataset):
    def __init__(
        self,
        windowed_df: pd.DataFrame,
        seq_feature_cols: List[str],
        static_feature_cols: List[str],
        target_col: str = "SpO2_Rad",
        seq_len: int = 400,
        normalize_seq: bool = False,
        target_mode: str = "last",  # "last" or "sequence"
    ):
        self.df = windowed_df.reset_index(drop=True).copy()
        self.seq_feature_cols = seq_feature_cols
        self.static_feature_cols = static_feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.normalize_seq = normalize_seq
        self.target_mode = target_mode

        for col in seq_feature_cols + static_feature_cols + [target_col]:
            if col not in self.df.columns:
                raise ValueError(f"Missing column: {col}")

    def __len__(self) -> int:
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

        x_seq = np.stack(seq_features, axis=0).astype(np.float32) if len(seq_features) > 0 else np.empty((0, self.seq_len), dtype=np.float32)

        static_features = []
        for col in self.static_feature_cols:
            val = row[col]
            if isinstance(val, str):
                # skip string metadata like SubjectID/source_file from numeric model input
                continue
            static_features.append(np.float32(val))

        x_static = np.asarray(static_features, dtype=np.float32)

        target_val = row[self.target_col]
        target_arr = self._to_1d_float_array(target_val, self.target_col, self.seq_len)

        if self.target_mode == "last":
            y = np.float32(target_arr[-1])
        elif self.target_mode == "sequence":
            y = target_arr
        else:
            raise ValueError("target_mode must be 'last' or 'sequence'")

        return (
            torch.tensor(x_seq, dtype=torch.float32),      # [num_seq_features, 400]
            torch.tensor(x_static, dtype=torch.float32),   # [num_static_features]
            torch.tensor(y, dtype=torch.float32),
        )


if __name__ == "__main__":
    set_seed(42)

    data_dir = "RValues_RawPPG_Aref"
    pattern = "*.h5"
    subject_col = "SubjectID"
    time_col = "timestamp"
    target_col = "SpO2_Rad"

    # these are one value per window
    static_cols = [
        "SubjectID",
        "source_file",
        "skintone",
        "r_motion",
        "ac_red",
        "dc_red",
        "ac_ir",
        "dc_ir",
        "r_val",
    ]

    window_size = 400
    stride = 400
    val_size = 0.2
    batch_size = 32
    target_mode = "last"
    num_workers = 0

    df = load_all_files(data_dir=data_dir, pattern=pattern)

    print("Raw dataframe shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Subjects:", df[subject_col].nunique())

    windowed_df = chop_all_columns_smart(
        df=df,
        subject_col=subject_col,
        time_col=time_col,
        window_size=window_size,
        stride=stride,
        static_cols=static_cols,
        drop_incomplete=True,
    )

    print("\nWindowed dataframe shape:", windowed_df.shape)

    seq_feature_cols = select_sequence_feature_columns(
        windowed_df=windowed_df,
        target_col=target_col,
        static_cols=static_cols,
        seq_len=window_size,
    )

    static_feature_cols = select_static_feature_columns(
        windowed_df=windowed_df,
        target_col=target_col,
        static_cols=static_cols,
    )

    print("\nSequence feature columns:")
    print(seq_feature_cols)

    print("\nStatic feature columns:")
    print(static_feature_cols)

    train_df, val_df = subject_wise_split(
        windowed_df=windowed_df,
        subject_col=subject_col,
        val_size=val_size,
        random_state=42,
    )

    print("\nTrain windows:", len(train_df))
    print("Validation windows:", len(val_df))

    train_dataset = SpO2WindowDataset(
        windowed_df=train_df,
        seq_feature_cols=seq_feature_cols,
        static_feature_cols=static_feature_cols,
        target_col=target_col,
        seq_len=window_size,
        normalize_seq=False,
        target_mode=target_mode,
    )

    val_dataset = SpO2WindowDataset(
        windowed_df=val_df,
        seq_feature_cols=seq_feature_cols,
        static_feature_cols=static_feature_cols,
        target_col=target_col,
        seq_len=window_size,
        normalize_seq=False,
        target_mode=target_mode,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    x_seq, x_static, y = next(iter(train_loader))

    print("\nBatch shapes:")
    print("x_seq shape:", x_seq.shape)        # [B, num_seq_features, 400]
    print("x_static shape:", x_static.shape)  # [B, num_static_features]
    print("y shape:", y.shape)                # [B] if target_mode='last'

    