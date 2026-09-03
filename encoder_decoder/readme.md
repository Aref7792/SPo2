# SpO2 Encoder-Decoder Pipeline

**Engineering Report · PPG-Based SpO2 Estimation**  
**Files:** `train1.py` → `train2.py`  
**Dataset:** 11 subjects · `RValues_RawPPG_Aref_`  
**Signal:** RED / IR PPG, 400-sample windows  
**Date:** 2026-09-01

---

## Bottom Line

The pipeline works: the encoder pretrains cleanly, freezing is genuinely frozen, and the residual head trains stably.

However, across a proper 11-fold leave-one-subject-out (LOSO) evaluation, the learned correction does **not reliably beat** the classical ratio-of-ratios physics baseline:

$$
SpO_2 = a + bR
$$

The best configuration gets within about **0.8%** of the baseline's mean RMSE and wins on **6 of 11** held-out subjects. This makes the learned model competitive, but not a clear improvement.

Two substantive implementation issues were identified and fixed. The remaining performance gap appears more consistent with a **data-quantity ceiling** than with a remaining tuning issue.

---

# 1. Architecture

## 1.1 Physics baseline with a bounded learned correction

Each sample contains:

- A synchronized RED and IR PPG window
- Window length: **400 samples**
- Duration: **16 s at 25 Hz**
- A ratio-of-ratios value \(R\)
- A reference `SpO2_Rad` label

The network does **not** predict SpO2 entirely from scratch.

Instead, the classical pulse-oximetry relationship is first fitted on the training subjects:

$$
SpO_{2,\mathrm{linear}} = a + bR
$$

The learned network predicts only a bounded residual correction:

$$
\widehat{SpO_2}
=
SpO_{2,\mathrm{linear}}
+
\Delta
$$

where

$$
\Delta
=
	ext{max\_residual}
\cdot
	anh\left(
	ext{correction\_head}
(
	ext{ppg\_feature},
	ext{r\_feature}
)

ight)
$$

The final layer of the correction head is initialized to zero. Therefore, at initialization,

$$
\widehat{SpO_2}
=
SpO_{2,\mathrm{linear}}
$$

and the learned model must earn any deviation from the physics baseline through optimization.

---

## 1.2 Forward-pass structure

```text
RED, IR window
[2 x 400]
      |
      v
Interaction channels
RED, IR, RED-IR, RED*IR
[4 x 400]
      |
      v
ConvTokenEncoder
tokens [400 x D]
      |
      v
Positional Encoding
+ Self-Attention
+ Attention Pooling
      |
      v
ppg_feature [D]
      |
      +--------------------+
                           |
R                          |
ratio-of-ratios            |
|                          |
v                          |
r_embed MLP                |
r_feature [8]              |
|                          |
+------------+-------------+
             |
             v
      Residual correction head
             |
             v
          raw_delta
             |
             v
Delta = max_residual * tanh(raw_delta)
             |
             v
SpO2_hat = SpO2_linear + Delta
```

The learned and fixed paths meet only at the final sum. The network can shift the physics prediction but cannot replace the baseline outright.

---

## 1.3 Module reference

| Module | Role | Output shape |
|---|---|---:|
| `ConvTokenEncoder` | 3 × Conv1d with kernels 7, 5, 3; no temporal downsampling | `[400, D]` |
| `LearnablePositionalEncoding` | Learned additive positional embedding | `[400, D]` |
| `TemporalSelfAttentionBlock` | Multi-head self-attention + FFN, post-norm | `[400, D]` |
| `AttentionPooling` | Learned softmax weighting over 400 tokens | `[D]` |
| `r_embed` | 2-layer MLP on scalar \(R\) | `[8]` |
| `correction_head` | MLP with zero-initialized final layer | `[1]` |

---

# 2. Three-Stage Training Procedure

The same `ConvTokenEncoder` is used across all three stages.

## Stage 1 — Self-supervised pretraining

The encoder is trained together with a decoder to reconstruct the true RED and IR waveforms.

```text
ConvTokenEncoder
    trainable
       |
       v
ConvTokenDecoder
    trainable
       |
       v
Reconstruct RED and IR
```

No SpO2 labels are used.

### Objective

$$
\mathcal{L}_{\mathrm{pretrain}}=\mathrm{MSE}(
[\widehat{RED},\widehat{IR}],
[RED,IR])
$$

### Configuration

- Learning rate: `1e-3`
- Epochs: `150`

---

## Stage 2 — Frozen encoder, train estimator head

The pretrained decoder is discarded.

The encoder weights are transferred into the full SpO2 estimator and frozen.

```text
Pretrained ConvTokenEncoder
       |
       | weights carried forward
       v
ConvTokenEncoder
frozen + .eval() pinned

Decoder discarded

Attention + Residual Head
trainable
```

The estimator head is trained against SpO2 while the encoder remains fixed.

### Objective

Huber loss against reference SpO2.

### Configuration

- Encoder: frozen
- Head learning rate: `1e-3`
- Epochs: `60`

---

## Stage 3 — End-to-end fine-tuning

The encoder is unfrozen and jointly fine-tuned with the attention and residual head.

A lower learning rate is applied to the pretrained encoder.

### Configuration

- Encoder learning rate: `1e-4`
- Head learning rate: `5e-4`
- Epochs: `60`
- Loss: Huber loss
- Training: end-to-end

This differential learning rate is intended to preserve the pretrained representation while still allowing task-specific adaptation.

---

# 3. Problems Found During Development

## 3.1 Freezing did not initially freeze BatchNorm behavior

The first implementation set

```python
requires_grad = False
```

on the encoder parameters and treated the encoder as frozen.

That was insufficient because `BatchNorm1d` maintains running statistics as **buffers**, not trainable parameters.

Therefore, under

```python
model.train()
```

the BatchNorm running statistics continued to change even though gradients were disabled.

### Fix

`set_encoder_trainable()` was changed so that it also switches the encoder mode:

```python
encoder.train(trainable)
```

When frozen, the encoder is explicitly pinned to:

```python
encoder.eval()
```

The network's `train()` method is also overridden so that calling `model.train()` does not accidentally return the frozen encoder to training mode.

This ensures both:

- Parameters stay fixed.
- BatchNorm running statistics stay fixed.

---

## 3.2 Reconstruction loss exploded because of `RED * IR`

The first Stage 1 implementation attempted to reconstruct all four encoder input channels:

- RED
- IR
- RED - IR
- RED × IR

Validation MSE became extremely large, approximately:

```text
600,000 – 700,000
```

For comparison, predicting the channel mean produced an MSE of only about:

```text
302
```

### Measured per-channel statistics on Subject8

| Channel | Standard deviation | Variance |
|---|---:|---:|
| RED | 13.57 | 184 |
| IR | 20.48 | 419 |
| RED - IR | 9.39 | 88 |
| RED × IR | 352.66 | 124,371 |

The product channel has roughly **300–600×** the variance of the other channels.

Under a single unweighted reconstruction MSE, the RED×IR term dominated both the objective and its gradients.

The derived channels are also redundant because:

$$
RED-IR = f(RED,IR)
$$

and

$$
RED 	imes IR = g(RED,IR)
$$

### Fix

Keep all four channels as **encoder inputs**, because the downstream encoder is expected to receive them:

```text
[RED, IR, RED-IR, RED*IR]
```

but make the decoder reconstruct only the two underlying signals:

```text
[RED, IR]
```

After this change, validation reconstruction MSE converged to approximately:

```text
65 – 70
```

which is about **4.5× better** than the mean-prediction baseline.

---

## 3.3 A single validation/test split was misleading

One deterministic split used:

```text
Validation = Subject11
Test       = Subject2
```

The fine-tuned model looked successful on validation:

| Model | Validation RMSE |
|---|---:|
| Physics baseline | 5.19 |
| Fine-tuned model | 4.56 |

But on the held-out test subject:

| Model | Test RMSE |
|---|---:|
| Physics baseline | 6.25 |
| Fine-tuned model | 6.93 |

The learned correction had therefore adapted to the validation subject without generalizing to the test subject.

With only 11 subjects, a one-subject validation set has very high variance.

This motivated switching to **leave-one-subject-out cross-validation**.

---

# 4. LOSO Cross-Validation

For every fold:

1. One subject is used as the held-out test subject.
2. Validation subject(s) are selected only from the remaining subjects.
3. Stage 1 is pretrained from scratch using only training/validation data.
4. The held-out test subject's waveform is never used during pretraining.
5. The ratio-of-ratios linear baseline is fitted using training subjects only.
6. Stage 2 and Stage 3 are then trained for that fold.

This prevents unlabeled test-signal leakage into the pretrained encoder.

---

# 5. Four LOSO Experiments

## Mean test RMSE across 11 subjects

| Round | Change | Linear | Fine-tuned | Wins |
|---|---|---:|---:|---:|
| 1 | Baseline: `hidden_dim=32`, 1 validation subject | 3.191 | 3.295 | 4 / 11 |
| 2 | `max_residual: 5 -> 2`, `lambda: 0.001 -> 0.01` | 3.191 | 3.281 | 5 / 11 |
| 3 | Smaller model: `hidden_dim: 32 -> 16`, heads `4 -> 2` | 3.191 | 3.216 | 6 / 11 |
| 4 | 2 validation subjects instead of 1 | 3.047 | 3.141 | 6 / 11 |

Round 3 produced the closest match to the classical baseline.

---

## 5.1 Round 2 — Stronger output regularization

Changes:

```text
max_residual:     5 -> 2
residual_lambda:  0.001 -> 0.01
```

Result:

```text
Linear RMSE:      3.191
Fine-tuned RMSE:  3.281
Wins:             5 / 11
```

This helped slightly.

---

## 5.2 Round 3 — Reduce model capacity

Changes:

```text
hidden_dim: 32 -> 16
attention heads: 4 -> 2
```

Trainable parameter count dropped approximately from:

```text
34,842 -> 14,394
```

Result:

```text
Linear RMSE:      3.191
Fine-tuned RMSE:  3.216
Wins:             6 / 11
```

The capacity reduction improved the model substantially more than stronger output regularization.

This suggests that approximately 200 training windows were not sufficient to reliably constrain the larger model.

---

## 5.3 Round 4 — Two validation subjects

Round 3 suggested that single-subject validation might be producing unstable checkpoint selection.

On some losing folds, validation RMSE remained better than the physics baseline throughout training even though test RMSE became worse.

Round 4 therefore averaged checkpoint selection across two validation subjects.

Result:

```text
Linear RMSE:      3.047
Fine-tuned RMSE:  3.141
Wins:             6 / 11
```

This fixed one specific failure:

```text
Subject3: loss -> win
```

but introduced regressions for:

```text
Subject5
Subject8
```

The aggregate gap therefore remained similar.

---

# 6. Best Configuration: Round 3

Round 3 used:

- `hidden_dim = 16`
- `attention_heads = 2`
- One validation subject
- Pretrained ConvTokenEncoder
- Frozen-head training
- Differential-rate end-to-end fine-tuning

## RMSE by held-out test subject

| Test subject | Linear | Frozen | Fine-tuned | vs. baseline |
|---|---:|---:|---:|---|
| Subject1 | 3.871 | 4.117 | 4.371 | loses |
| Subject2 | 6.653 | 6.717 | 6.748 | loses |
| Subject3 | 1.995 | 2.471 | 2.434 | loses |
| Subject4 | 4.257 | 4.255 | 4.249 | wins |
| Subject5 | 2.183 | 2.184 | 2.185 | loses |
| Subject6 | 2.309 | 2.750 | 2.587 | loses |
| Subject7 | 2.845 | 2.682 | 2.599 | wins |
| Subject8 | 1.148 | 0.998 | 1.003 | wins |
| Subject9 | 2.451 | 2.072 | 2.018 | wins |
| Subject11 | 5.586 | 5.470 | 5.444 | wins |
| Subject12 | 1.807 | 1.765 | 1.737 | wins |

The fine-tuned model wins on:

```text
Subject4
Subject7
Subject8
Subject9
Subject11
Subject12
```

for a total of:

```text
6 / 11 subjects
```

The losses do not appear to cluster clearly by obvious properties such as recording length or \(R\)-range.

---

# 7. Main Conclusions

## 7.1 The training pipeline is technically sound

The corrected implementation behaves as intended:

- Encoder self-supervised pretraining converges.
- Decoder reconstructs only the real RED/IR signals.
- The frozen encoder is genuinely frozen.
- BatchNorm statistics remain fixed in Stage 2.
- Stage 3 uses a sensible lower encoder learning rate.
- Held-out test data are excluded from pretraining.

---

## 7.2 The learned correction does not yet outperform the physics model

The strongest tested configuration remains approximately tied with the ratio-of-ratios baseline.

The report's best model reaches:

```text
Linear mean RMSE:      3.191
Fine-tuned mean RMSE:  3.216
```

Thus, the learned model is close, but does not establish a meaningful average improvement.

---

## 7.3 Model capacity mattered more than residual regularization

Reducing:

```text
hidden_dim: 32 -> 16
heads:      4 -> 2
```

closed substantially more of the performance gap than tightening:

```text
max_residual
residual_lambda
```

This is consistent with an over-capacity problem relative to the small dataset.

---

## 7.4 Single-subject validation is unstable

With only 11 subjects, using one subject for early stopping creates a noisy checkpoint-selection signal.

Adding a second validation subject corrected one diagnosed overfitting case, but produced new regressions on other folds.

Therefore, the main issue is not simply the choice of one specific validation subject.

---

## 7.5 The likely next bottleneck is dataset size

Two different interventions were tested:

1. Stronger residual regularization
2. Lower model capacity / broader validation

Both plateaued at approximately the same subject-level win rate.

The strongest interpretation is that the remaining limitation is primarily **subject diversity and sample count**, rather than another obvious architectural bug.

---

# 8. Pipeline Summary

```text
                     STAGE 1
                     -------
Raw RED/IR
    |
Construct:
RED, IR, RED-IR, RED*IR
    |
ConvTokenEncoder
    |
ConvTokenDecoder
    |
Reconstruct RED/IR
    |
Save pretrained encoder
             |
             v
                     STAGE 2
                     -------
Load encoder weights
    |
Freeze parameters
Pin encoder to eval()
    |
Self-attention
Attention pooling
    |
Fuse with R embedding
    |
Residual correction head
    |
SpO2_hat = a + bR + Delta
    |
Train head only
             |
             v
                     STAGE 3
                     -------
Unfreeze encoder
    |
Encoder LR = 1e-4
Head LR    = 5e-4
    |
End-to-end fine-tuning
    |
Evaluate held-out subject
```

---

# 9. Associated Files

```text
train1.py
train2.py
enc_dec.py
```

Preserved LOSO checkpoints:

```text
loso_checkpoints/
```
