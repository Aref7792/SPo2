# Physics-Guided Multimodal SpO₂ Estimation

This repository implements a **physics-guided deep learning framework for non-invasive SpO₂ estimation** from photoplethysmography (PPG) signals.

The model combines:

- Red PPG waveform
- Infrared (IR) PPG waveform
- Ratio-of-ratios (`R`) feature
- Skin-tone information

with a classical pulse-oximetry calibration model and a neural residual correction network.

Instead of asking the neural network to estimate SpO₂ entirely from scratch, the architecture starts from a conventional ratio-of-ratios estimate


$\hat{SpO}_2^{\text{linear}} = a + bR$

and learns only a residual correction:

\[
\hat{SpO}_2 =
\hat{SpO}_2^{\text{linear}}
+
\Delta_{\text{NN}}.
\]

This design provides an explicit physics-based baseline while allowing the neural network to model nonlinear effects that are not captured by the conventional calibration equation.

---

## Overview

The complete pipeline is:

```text
                 Ratio of Ratios (R)
                         │
                         ▼
                Linear Calibration
                  SpO₂ = a + bR
                         │
                         ├───────────────────────────────┐
                         │                               │
Red PPG ──► CNN Encoder ──► Skin FiLM ──┐              │
                                         │              │
                                         ▼              │
                               Red → IR Cross Attention │
                                         │              │
                                         ├──────┐       │
                                         │      │       │
IR PPG ───► CNN Encoder ─────────────────┘      │       │
                                                │       │
                           IR → Red Cross Attention     │
                                                │       │
                                                ▼       │
                                         Concatenation  │
                                                │       │
                                                ▼       │
                                     Temporal Self-Attention
                                                │
                                                ▼
                                       Attention Pooling
                                                │
Skin Tone ─────────────────► Skin Embedding ────┤
                                                │
                                                ▼
                                      Residual MLP Head
                                                │
                                                ▼
                                           ΔSpO₂
                                                │
                                                ▼
                         Final Prediction = a + bR + ΔSpO₂
```

---

## Key Features

### Physics-Guided Residual Learning

A conventional ratio-of-ratios calibration is fitted using only the training subjects:

```python
SpO2_linear = a + b * R
```

The neural network then learns a correction term:

```python
SpO2_pred = SpO2_linear + delta_spo2
```

This formulation encourages the network to improve upon the conventional pulse-oximetry model instead of replacing it completely.

### Independent Red and IR PPG Encoders

The Red and IR waveforms are processed using separate 1D convolutional encoders.

Each encoder contains:

```text
Conv1D
  ↓
BatchNorm
  ↓
ReLU
  ↓
Dropout
  ↓
Conv1D
  ↓
BatchNorm
  ↓
ReLU
  ↓
Dropout
  ↓
Conv1D
  ↓
BatchNorm
  ↓
ReLU
```

For the default configuration:

```text
Input PPG window: 400 samples
Hidden dimension: 64
```

The resulting token representations have the form:

```text
[B, T, D]
```

where:

- `B` = batch size
- `T` = temporal dimension
- `D` = feature dimension

### Bidirectional Red–IR Cross-Attention

Red and IR PPG contain complementary physiological information. The model therefore performs cross-modal attention in both directions:

```text
Red queries IR
IR queries Red
```

using PyTorch multi-head attention.

### Temporal Self-Attention

Following cross-modal fusion, Red and IR representations are concatenated and processed by a temporal self-attention block:

```text
Red/IR fused feature
        ↓
Temporal Self-Attention
```

This allows the model to capture dependencies across different locations within each PPG window.

### Attention Pooling

Instead of averaging all temporal features, the network learns an attention weight for each temporal token.

For token representations \(x_t\),

\[
\alpha_t =
\frac{\exp(s(x_t))}
{\sum_j \exp(s(x_j))}
\]

and the final PPG representation is

\[
z =
\sum_t \alpha_t x_t.
\]

### Skin-Tone Conditioning with FiLM

Skin tone can influence optical PPG measurements.

The architecture introduces skin-tone information directly into the learned PPG representation using **Feature-wise Linear Modulation (FiLM)**.

The skin-tone network generates modulation parameters:

\[
\gamma,\beta = f_{\text{skin}}(s)
\]

which modify the Red PPG representation as

\[
x_{\text{red}}'
=
(1+\gamma)x_{\text{red}}+\beta.
\]

In the current implementation, FiLM conditioning is applied to the **Red PPG branch**. Skin tone is also independently encoded and supplied to the final residual prediction head.

---

## Dataset Format

The code expects one or more HDF5 files:

```text
*.h5
```

located inside the configured dataset directory.

By default:

```python
data_dir = "RValues_RawPPG_Aref_"
pattern = "*.h5"
```

Each file is loaded using:

```python
pd.read_hdf(...)
```

and all files are concatenated into a single Pandas DataFrame.

### Required Columns

| Column | Description |
|---|---|
| `SubjectID` | Subject/patient identifier |
| `timestamp` | Temporal ordering |
| `red_win_filtered` | Red PPG signal |
| `ir_win_filtered` | Infrared PPG signal |
| `SpO2_Rad` | Ground-truth SpO₂ |
| `skintone` | Skin-tone scalar |
| `r_val` | Ratio-of-ratios feature |

---

## Window Generation

The raw data are converted into fixed-length temporal windows.

Default parameters:

```python
window_size = 400
stride = 400
```

Each model sample contains:

```text
Red PPG:       400 samples
IR PPG:        400 samples
Skin tone:     1 scalar
R value:       1 scalar
SpO₂ target:   1 scalar
```

The resulting PPG input tensor has shape:

```text
[B, 2, 400]
```

where channel `0` corresponds to Red PPG and channel `1` corresponds to IR PPG.

---

## Subject-Wise Train/Validation Split

To prevent leakage between windows belonging to the same person, the repository performs a **subject-wise split** rather than randomly splitting individual windows.

Validation subjects are manually specified:

```python
val_subjects = ["Subject11"]
```

All windows from these subjects are assigned to validation. All remaining subjects are used for training.

---

## Ratio-of-Ratios Baseline

Before neural-network training, the code fits a linear pulse-oximetry calibration:

\[
SpO_2 = a+bR.
\]

The coefficients are estimated using:

```python
linear_b, linear_a = np.polyfit(
    R_train,
    SpO2_train,
    deg=1,
)
```

Importantly, the calibration is fitted using **training subjects only**.

The resulting linear model is also evaluated independently on the validation subject before neural-network training.

---

## Neural Network Architecture

The main model is:

```python
RawCrossAttentionSpO2Net
```

with the default configuration:

```python
hidden_dim = 64
num_heads = 4
dropout = 0.1
```

### Model Components

```text
RawCrossAttentionSpO2Net
│
├── Red ConvTokenEncoder
│
├── IR ConvTokenEncoder
│
├── SkinFiLM
│
├── Red → IR CrossAttentionBlock
│
├── IR → Red CrossAttentionBlock
│
├── TemporalSelfAttentionBlock
│
├── AttentionPooling
│
├── Skin Embedding MLP
│
└── Residual Correction MLP
```

---

## Residual Prediction Head

After Red/IR fusion and temporal pooling, the resulting PPG feature is concatenated with an explicit skin-tone embedding.

The correction network predicts

\[
\Delta SpO_2.
\]

The final estimate is

\[
\boxed{
\hat{SpO}_2
=
a+bR+\Delta SpO_2
}
\]

The final layer of the residual network is initialized to zero.

Therefore, before neural-network training,

\[
\Delta SpO_2 = 0
\]

and

\[
\hat{SpO}_2 = a+bR.
\]

Training therefore starts exactly from the conventional ratio-of-ratios solution.

---

## Residual Regularization

The training objective contains two terms:

\[
\mathcal{L}
=
\mathcal{L}_{MSE}
+
\lambda_{\text{res}}
\mathbb{E}
\left[
\Delta SpO_2^2
\right].
\]

The second term discourages unnecessarily large corrections to the physics-based baseline.

The default value is:

```python
residual_lambda = 0.01
```

---

## Training

The model uses:

```text
Optimizer: AdamW
Loss: MSE
Initial learning rate: 1e-3
Weight decay: 1e-4
Epochs: 40
Batch size: 16
Gradient clipping: 5.0
```

A `ReduceLROnPlateau` scheduler is used:

```python
torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=20,
)
```

The best model is selected according to validation RMSE.

---

## Evaluation Metrics

The following regression metrics are reported:

### Mean Squared Error

\[
MSE =
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat y_i)^2
\]

### Root Mean Squared Error

\[
RMSE =
\sqrt{MSE}
\]

### Mean Absolute Error

\[
MAE =
\frac{1}{N}
\sum_{i=1}^{N}
|y_i-\hat y_i|
\]

### Bias

\[
Bias =
\frac{1}{N}
\sum_{i=1}^{N}
(\hat y_i-y_i)
\]

The implementation additionally reports statistics of the learned residual:

```text
Mean ΔSpO₂
Standard deviation ΔSpO₂
Mean absolute ΔSpO₂
```

---

## Installation

Clone the repository:

```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY>
```

Create a Python environment:

```bash
conda create -n spo2 python=3.10
conda activate spo2
```

Install the required packages:

```bash
pip install numpy pandas torch tables
```

`tables` is required by Pandas for reading HDF5 files.

For CUDA-enabled PyTorch, install the appropriate PyTorch build for your CUDA environment.

---

## Running the Code

Place the dataset inside:

```text
RValues_RawPPG_Aref_/
```

For example:

```text
project/
│
├── train.py
├── README.md
└── RValues_RawPPG_Aref_/
    ├── subject01.h5
    ├── subject02.h5
    ├── subject03.h5
    └── ...
```

Set the validation subject:

```python
val_subjects = ["Subject11"]
```

Then run:

```bash
python train.py
```

The script automatically uses CUDA when available and otherwise falls back to CPU.

---

## Example Training Output

The training log has the following form:

```text
Epoch [001/040] |
Train MSE: ... |
Train RMSE: ... |
Val MSE: ... |
Val RMSE: ... |
Linear Val RMSE: ... |
Delta mean: ... |
Delta std: ...
```

This enables direct comparison between the learned estimator and the conventional ratio-of-ratios baseline during training.

---

## Saved Checkpoint

The best model is saved as:

```text
best_ratio_residual_spo2_model.pt
```

The checkpoint contains:

```python
{
    "epoch": ...,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "val_rmse": ...,
    "val_mse": ...,
    "linear_val_rmse": ...,
    "linear_a": ...,
    "linear_b": ...,
    "residual_lambda": ...,
    "seq_feature_cols": ...,
    "window_size": ...,
    "y_mean": ...,
    "y_std": ...
}
```

---

## Important Configuration Parameters

The main experimental parameters are located near the bottom of the training script:

```python
window_size = 400
stride = 400

val_subjects = ["Subject11"]

batch_size = 16
num_workers = 0

normalize_seq = False

hidden_dim = 64
num_heads = 4
dropout = 0.1

lr = 1e-3
weight_decay = 1e-4
num_epochs = 40

residual_lambda = 0.01
```

---

## Signal Normalization

The dataset class supports optional per-window normalization:

```python
normalize_seq = True
```

which applies

$x' =\frac{x-\mu_x}{\sigma_x}$.


However, the current default configuration is:

```python
normalize_seq = False
```

so Red and IR PPG signals are supplied to the neural network without this normalization.

---

## Why Physics-Guided Residual Learning?

A purely data-driven model learns:

```text
PPG → SpO₂
```

without explicitly exploiting the established relationship between Red/IR absorption and oxygen saturation.

A conventional pulse-oximetry model instead uses:

```text
R → SpO₂
```

but is limited by its calibration model.

This approach combines both:

```text
Known physiological relationship
              +
Learned signal representation
              =
Physics-guided SpO₂ estimator
```

The neural network can therefore focus on effects that are not adequately represented by the linear calibration, including:

- waveform morphology,
- temporal characteristics,
- Red/IR interactions,
- subject-dependent optical variation,
- skin-tone-related effects,
- nonlinear deviations from ratio-of-ratios calibration.

---

## Recommended Experimental Comparisons

The implementation supports comparison among:

### Classical Baseline

```text
R
↓
Linear Regression
↓
SpO₂
```

### Neural PPG Model

```text
Red PPG + IR PPG
↓
Deep Network
↓
SpO₂
```

### Physics-Guided Model

```text
                 R ──► Linear SpO₂
                         │
Red + IR + Skin ──► NN ──┤
                         ▼
                    Final SpO₂
```

---

## Recommended Repository Structure

```text
.
├── train.py
├── README.md
├── requirements.txt
├── models/
│   └── .gitkeep
└── data/
    └── README.md
```

Because physiological datasets may contain sensitive or restricted data, raw HDF5 datasets should generally **not** be committed to a public repository unless redistribution is explicitly permitted.

---

## Requirements

Example `requirements.txt`:

```text
numpy
pandas
torch
tables
```

---

## Reproducibility

The code initializes random seeds for Python, NumPy, PyTorch, and CUDA using:

```python
set_seed(42)
```

This improves reproducibility across experiments, although GPU operations may still exhibit some nondeterminism depending on the PyTorch/CUDA configuration.

---

## Future Work



---

## Disclaimer


## Citation



## License


or the license required by the associated dataset, institution, or publication.
