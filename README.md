# Physics-Guided SpO₂ Estimation from Red/IR PPG

This repository implements a **physics-guided residual learning framework for SpO₂ estimation** using:

- Red PPG
- Infrared (IR) PPG
- Skin tone
- Ratio-of-ratios (`R`)

The model combines a conventional ratio-of-ratios calibration with a neural correction network. Instead of replacing the classical SpO₂ model, the neural network learns only a residual correction.

## Core Idea

The final prediction is

```text
SpO2_hat = SpO2_linear + Delta_SpO2_NN
```

where

```text
SpO2_linear = a + bR
```

The coefficients `a` and `b` are fitted using only the training subjects.

The neural network then learns

```text
Delta_SpO2_NN
```

from Red PPG, IR PPG, and skin-tone information.

---

## Pipeline

```text
                           r_val (R)
                              │
                              ▼
                 Linear ratio-of-ratios model
                       SpO₂ = a + bR
                              │
                              ▼
                        SpO₂_linear
                              │
                              │
                              └───────────────────────────────┐
                                                              │
Red PPG [B,1,400]                                             │
      │                                                       │
      ▼                                                       │
Red ConvTokenEncoder                                          │
      │                                                       │
      ▼                                                       │
Red tokens                                                    │
      │                                                       │
      │         Skin tone                                     │
      │             │                                         │
      │             ▼                                         │
      │          SkinFiLM                                     │
      │             │                                         │
      │        γ_red, β_red                                   │
      │             │                                         │
      ▼             ▼                                         │
FiLM-modulated Red tokens                                     │
      │                                                       │
      │                    IR PPG [B,1,400]                    │
      │                          │                             │
      │                          ▼                             │
      │                 IR ConvTokenEncoder                    │
      │                          │                             │
      │                          ▼                             │
      │                      IR tokens                         │
      │                          │                             │
      ├──────────────┐           │                             │
      │              │           │                             │
      ▼              │           ▼                             │
Red → IR             │       IR → Red                         │
Cross-Attention      │       Cross-Attention                  │
Q = Red              │       Q = IR                           │
K,V = IR             │       K,V = Red                        │
      │              │           │                             │
      └──────────────┴───────────┘                             │
                     │                                        │
                     ▼                                        │
                Concatenation                                  │
           [red_cross, ir_cross]                               │
                     │                                        │
                     ▼                                        │
          Temporal Self-Attention                              │
                     │                                        │
                     ▼                                        │
             Attention Pooling                                 │
                     │                                        │
                     ▼                                        │
               PPG feature                                    │
                     │                                        │
                     │              Skin tone                  │
                     │                  │                       │
                     │                  ▼                       │
                     │              Skin MLP                   │
                     │                  │                       │
                     │                  ▼                       │
                     │            Skin feature [8]             │
                     │                  │                       │
                     └──────────┬───────┘                       │
                                ▼                               │
                         Concatenation                          │
                 [PPG feature, skin feature]                    │
                                │                               │
                                ▼                               │
                       Correction MLP                           │
                                │                               │
                                ▼                               │
                           ΔSpO₂_NN                             │
                                │                               │
                                └───────────────┬───────────────┘
                                                ▼
                         Final SpO₂ prediction
                   SpO₂_hat = SpO₂_linear + ΔSpO₂_NN
```

---

## Model Architecture

### 1. Red and IR PPG Inputs

Each sample contains two PPG channels:

```text
x_seq shape = [B, 2, 400]
```

with

```text
Channel 0 = Red PPG
Channel 1 = IR PPG
```

The Red and IR signals are processed by separate 1D convolutional encoders.

### 2. ConvTokenEncoder

Each PPG encoder consists of three 1D convolutional stages:

```text
Conv1D
→ BatchNorm
→ ReLU
→ Dropout
→ Conv1D
→ BatchNorm
→ ReLU
→ Dropout
→ Conv1D
→ BatchNorm
→ ReLU
```

The default hidden dimension is

```text
hidden_dim = 64
```

The encoder output is transposed into token form:

```text
[B, T, D]
```

for attention processing.

### 3. Skin-Tone FiLM Conditioning

Skin tone is passed through a small FiLM network.

The FiLM block produces modulation parameters for the PPG representation. In the current implementation, only the **Red PPG tokens** are modulated:

```text
red = red * (1 + gamma_red) + beta_red
```

Although the FiLM module computes parameters for both Red and IR streams, the IR modulation parameters are not applied in the current model.

### 4. Bidirectional Cross-Attention

The model uses cross-attention in both directions.

#### Red → IR

```text
Query       = Red tokens
Key, Value  = IR tokens
```

#### IR → Red

```text
Query       = IR tokens
Key, Value  = FiLM-conditioned Red tokens
```

The two outputs are

```text
red_cross
ir_cross
```

and are concatenated along the feature dimension.

### 5. Temporal Self-Attention

The concatenated Red/IR representation is passed through a temporal self-attention block:

```text
[red_cross, ir_cross]
        │
        ▼
Temporal Self-Attention
```

This captures relationships across temporal positions within the PPG window.

### 6. Attention Pooling

The temporal representation is converted into one window-level PPG feature using learned attention pooling.

The pooling block learns a score for every temporal token and computes a weighted sum:

```text
Temporal tokens
      │
      ▼
Attention scores
      │
      ▼
Softmax
      │
      ▼
Weighted sum
      │
      ▼
PPG feature
```

### 7. Explicit Skin Embedding

Skin tone is also processed through a separate MLP:

```text
Skin tone
   │
   ▼
Linear(1, 8)
   │
  ReLU
   │
   ▼
Linear(8, 8)
   │
  ReLU
   │
   ▼
Skin feature
```

This skin feature is concatenated with the pooled PPG feature.

### 8. Neural Residual Correction

The combined representation is passed through the correction head:

```text
PPG feature + Skin feature
          │
          ▼
       Linear
          │
         ReLU
          │
       Dropout
          │
       Linear
          │
         ReLU
          │
       Linear
          │
          ▼
      ΔSpO₂_NN
```

The neural network therefore predicts only the correction term.

---

## Physics-Guided Residual Formulation

The classical ratio-of-ratios calibration is

```text
SpO2_linear = a + bR
```

The final model prediction is

```text
SpO2_hat = SpO2_linear + Delta_SpO2_NN
```

The ratio-of-ratios value `R` is **not passed into the neural correction network**. It is used only by the classical linear branch.

This separation allows the model to retain the conventional pulse-oximetry estimate while using PPG morphology and skin-tone information to learn a data-driven correction.

---

## Zero-Initialized Residual Head

The last layer of the residual correction head is initialized with zero weights and zero bias.

Therefore, before training:

```text
Delta_SpO2_NN = 0
```

and the full model initially behaves exactly like the classical linear calibration:

```text
SpO2_hat = a + bR
```

The neural network gradually learns a correction only when supported by the training data.

---

## Dataset

The script expects one or more HDF5 files:

```text
*.h5
```

inside the configured data directory.

Default:

```python
data_dir = "RValues_RawPPG_Aref_"
pattern = "*.h5"
```

The files are loaded with:

```python
pandas.read_hdf(...)
```

and concatenated into a single DataFrame.

### Required Columns

| Column | Description |
|---|---|
| `SubjectID` | Subject identifier |
| `timestamp` | Time index used for sorting |
| `red_win_filtered` | Filtered Red PPG |
| `ir_win_filtered` | Filtered IR PPG |
| `SpO2_Rad` | Ground-truth SpO₂ |
| `skintone` | Skin-tone scalar |
| `r_val` | Ratio-of-ratios value |

---

## Window Generation

The raw data are converted into non-overlapping windows by default.

```python
window_size = 400
stride = 400
```

Each training sample contains:

```text
Red PPG        : 400 samples
IR PPG         : 400 samples
Skin tone      : 1 scalar
Ratio R        : 1 scalar
SpO₂ target    : 1 scalar
```

Incomplete windows are discarded.

---

## Subject-Wise Data Split

The model uses a deterministic subject-wise train/validation split.

Example:

```python
val_subjects = ["Subject11"]
```

All windows belonging to the selected validation subject are placed in the validation set. The remaining subjects are used for training.

This avoids subject leakage between training and validation data.

---

## Ratio-of-Ratios Calibration

The classical linear model is fitted only using the training subjects:

```python
linear_b, linear_a = np.polyfit(
    R_train,
    SpO2_train,
    deg=1,
)
```

which gives

```text
SpO2_linear = linear_a + linear_b * R
```

The validation performance of this classical baseline is calculated before neural-network training.

This provides a direct reference for determining whether the neural correction improves over the ratio-of-ratios model.

---

## Training Objective

The main prediction loss is mean squared error:

```text
MSE(SpO2_hat, SpO2_true)
```

An optional residual regularization term is also added:

```text
residual_lambda * mean(Delta_SpO2_NN²)
```

The full training objective is therefore:

```text
Loss =
    MSE(SpO2_hat, SpO2_true)
    +
    residual_lambda * mean(Delta_SpO2_NN²)
```

Default:

```python
residual_lambda = 0.01
```

This discourages the neural network from making unnecessarily large corrections to the physics-based baseline.

---

## Default Training Configuration

```python
window_size = 400
stride = 400

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

The optimizer is:

```text
AdamW
```

and the learning-rate scheduler is:

```text
ReduceLROnPlateau
```

with:

```python
factor = 0.5
patience = 20
```

Gradient clipping is applied with:

```text
max_norm = 5.0
```

---

## Evaluation Metrics

The implementation reports:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Bias
- Linear baseline RMSE
- Mean residual correction
- Standard deviation of the residual correction
- Mean absolute residual correction

A typical training log looks like:

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

---

## Installation

### Option 1: Conda Environment

Create a new environment:

```bash
conda create -n spo2 python=3.10 -y
conda activate spo2
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Option 2: Python Virtual Environment

Create the environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Then install:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## PyTorch and CUDA

The code automatically selects:

```text
cuda:0
```

when CUDA is available. Otherwise, it runs on CPU.

For GPU training, make sure your installed PyTorch version is compatible with your CUDA driver.

If needed, install the appropriate PyTorch build from the official PyTorch installation instructions, and then install the remaining packages with:

```bash
pip install numpy pandas tables
```

---

## Running the Model

A simple repository structure is:

```text
project/
│
├── train.py
├── README.md
├── requirements.txt
│
└── RValues_RawPPG_Aref_/
    ├── subject_01.h5
    ├── subject_02.h5
    ├── subject_03.h5
    └── ...
```

Set the validation subject in `train.py`:

```python
val_subjects = ["Subject11"]
```

Then run:

```bash
python train.py
```

---

## Saved Model

The best checkpoint is saved as:

```text
best_ratio_residual_spo2_model.pt
```

The checkpoint stores:

```text
epoch
model_state_dict
optimizer_state_dict
val_rmse
val_mse
linear_val_rmse
linear_a
linear_b
residual_lambda
seq_feature_cols
window_size
y_mean
y_std
```

This preserves both the neural-network parameters and the fitted classical calibration parameters.

---

## Important Implementation Details

### Ratio-of-Ratios Branch

`r_val` is used only in:

```text
SpO2_linear = a + bR
```

It is not concatenated with the neural features.

### Skin Tone Is Used Twice

Skin tone contributes through two separate paths:

```text
1. Skin tone → SkinFiLM → Red token modulation
2. Skin tone → Skin MLP → final residual feature
```

### Only Red PPG Is FiLM-Modulated

The current FiLM implementation computes parameters corresponding to both wavelengths, but only:

```text
gamma_red
beta_red
```

are applied.

### No PPG Normalization by Default

The dataset class supports window normalization, but the current configuration uses:

```python
normalize_seq = False
```

---

## Reproducibility

The script sets the random seed to:

```python
42
```

for:

- Python
- NumPy
- PyTorch
- CUDA

using:

```python
set_seed(42)
```

---

## Potential Extensions

Possible future experiments include:

- Leave-one-subject-out cross-validation
- Multi-subject validation
- Applying FiLM to both Red and IR streams
- Comparing Red-only and IR-only models
- Removing skin-tone conditioning
- Removing cross-attention
- Removing temporal self-attention
- Removing residual regularization
- Comparing against a pure neural SpO₂ estimator
- Nonlinear calibration of the ratio-of-ratios branch
- Explicit AC/DC signal decomposition
- Skin-tone subgroup evaluation
- Uncertainty estimation

---

## Disclaimer

This implementation is intended for **research purposes only**.

It has not been clinically validated and should not be used for medical diagnosis, patient monitoring, or treatment decisions without appropriate clinical and regulatory validation.
