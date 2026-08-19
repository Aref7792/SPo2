# Leave-One-Subject-Out SpO2 Estimation Results

## 1. Experimental Overview

This report summarizes the subject-wise validation results for the physics-guided ratio-residual SpO2 model. Each run uses one subject as the validation subject and all remaining subjects for training. Physics calibration is fitted using **training subjects only**, avoiding leakage from the held-out validation subject.

### Common setup

- PyTorch: `2.5.1+cu121`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 4060`
- Raw dataframe shape: `(79600, 20)`
- Number of subjects: `11`
- Windowed dataframe shape: `(199, 9)`
- Sequence input shape per batch: `[16, 2, 400]`
- Skin-tone input shape per batch: `[16]`
- Ratio-of-ratios input shape per batch: `[16]`
- Target shape per batch: `[16]`
- Training epochs per run: `40`
- Model checkpoint: `best_ratio_residual_spo2_model.pt`

Available subjects: `Subject1, Subject2, Subject3, Subject4, Subject5, Subject6, Subject7, Subject8, Subject9, Subject11, Subject12`.

> Note: `Subject10` is not present in the reported dataset.

### Dataset target statistics

| Statistic | SpO2_Rad |
|---|---:|
| Count | 79,600 |
| Mean | 86.9095 |
| Std. dev. | 8.3252 |
| Min | 72 |
| 25th percentile | 79 |
| Median | 87 |
| 75th percentile | 95 |
| Max | 100 |

## 2. Subject-Wise Summary

| Validation subject | Val. windows | Linear RMSE | Best residual RMSE | ΔRMSE | Improvement | Best epoch | Final val. RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Subject1 | 11 | 4.1865 | **2.7744** | +1.4121 | +33.73% | 19 | 3.9031 |
| Subject2 | 11 | 6.8132 | **6.8012** | +0.0120 | +0.18% | 1 | 8.4756 |
| Subject3 | 24 | 1.9809 | **1.4522** | +0.5287 | +26.69% | 10 | 1.9352 |
| Subject4 | 16 | 4.3162 | **3.6001** | +0.7161 | +16.59% | 30 | 4.0615 |
| Subject5 | 17 | 2.1774 | **1.6968** | +0.4806 | +22.07% | 40 | 1.6968 |
| Subject6 | 13 | 2.2288 | **2.1799** | +0.0489 | +2.19% | 6 | 3.1881 |
| Subject7 | 18 | 2.7673 | **2.6079** | +0.1594 | +5.76% | 7 | 2.8800 |
| Subject8 | 20 | 1.3167 | **0.9764** | +0.3403 | +25.84% | 18 | 2.1787 |
| Subject9 | 23 | 2.1817 | **1.7545** | +0.4272 | +19.58% | 21 | 2.8584 |
| Subject11 | 22 | 5.4885 | **5.4975** | -0.0090 | -0.16% | 1 | 6.9560 |
| Subject12 | 24 | 1.8490 | **1.4286** | +0.4204 | +22.74% | 8 | 3.9324 |

Positive `ΔRMSE` means that the learned residual model improves on the linear ratio-of-ratios baseline.

## 3. Aggregate Results

- Subjects improved over the linear baseline: **10/11**
- Mean subject-wise linear RMSE: **3.2097**
- Mean subject-wise best residual-model RMSE: **2.7972**
- Mean subject-wise RMSE reduction: **0.4124** (12.85%)
- Window-weighted pooled linear RMSE: **3.3756**
- Window-weighted pooled best-model RMSE: **3.0805**
- Window-weighted pooled RMSE reduction: **0.2952** (8.74%)
- Best held-out subject: **Subject8**, RMSE = **0.9764**
- Most difficult held-out subject: **Subject2**, RMSE = **6.8012**

## 4. Physics Calibration and Normalization by Fold

For each validation fold, the baseline is

`SpO2_linear = a + b * R`

| Validation subject | a | b | Linear MSE | Linear RMSE | Linear MAE | y_mean | y_std |
|---|---:|---:|---:|---:|---:|---:|---:|
| Subject1 | 110.766313 | -30.923059 | 17.5269 | 4.1865 | 3.5282 | 87.0319 | 8.3786 |
| Subject2 | 111.789313 | -32.396271 | 46.4195 | 6.8132 | 5.7464 | 87.1011 | 8.2620 |
| Subject3 | 110.068238 | -29.822906 | 3.9241 | 1.9809 | 1.6370 | 86.6571 | 8.2930 |
| Subject4 | 110.776140 | -30.465843 | 18.6294 | 4.3162 | 3.1703 | 86.6066 | 8.2924 |
| Subject5 | 110.606306 | -30.636493 | 4.7412 | 2.1774 | 1.7026 | 87.1264 | 8.2602 |
| Subject6 | 110.355550 | -30.156608 | 4.9677 | 2.2288 | 1.7996 | 86.8817 | 8.3110 |
| Subject7 | 110.504980 | -30.163680 | 7.6581 | 2.7673 | 2.2007 | 86.8950 | 8.3735 |
| Subject8 | 110.595910 | -30.359187 | 1.7336 | 1.3167 | 1.0393 | 86.8548 | 8.3707 |
| Subject9 | 111.273729 | -31.508580 | 4.7596 | 2.1817 | 1.6389 | 86.8182 | 8.3012 |
| Subject11 | 110.258300 | -29.665703 | 30.1241 | 5.4885 | 4.5907 | 87.0113 | 8.2428 |
| Subject12 | 110.989991 | -30.756924 | 3.4189 | 1.8490 | 1.6489 | 87.0629 | 8.4374 |

## 5. Per-Subject Results

### Subject1

- Validation windows: **11**
- Physics model: `SpO2_linear = 110.766313 + (-30.923059) * R`
- Linear baseline: MSE = **17.5269**, RMSE = **4.1865**, MAE = **3.5282**
- Best learned-model validation RMSE: **2.7744** at epoch **19**
- Change relative to baseline: **+1.4121 RMSE (+33.73%)**; the learned residual model improved the baseline.
- Epoch-40 train RMSE: **1.4583**
- Epoch-40 validation RMSE: **3.9031**

### Subject2

- Validation windows: **11**
- Physics model: `SpO2_linear = 111.789313 + (-32.396271) * R`
- Linear baseline: MSE = **46.4195**, RMSE = **6.8132**, MAE = **5.7464**
- Best learned-model validation RMSE: **6.8012** at epoch **1**
- Change relative to baseline: **+0.0120 RMSE (+0.18%)**; the learned residual model improved the baseline.
- Epoch-40 train RMSE: **1.4272**
- Epoch-40 validation RMSE: **8.4756**

### Subject3

- Validation windows: **24**
- Physics model: `SpO2_linear = 110.068238 + (-29.822906) * R`
- Linear baseline: MSE = **3.9241**, RMSE = **1.9809**, MAE = **1.6370**
- Best learned-model validation RMSE: **1.4522** at epoch **10**
- Change relative to baseline: **+0.5287 RMSE (+26.69%)**; the learned residual model improved the baseline.
- Epoch-40 train RMSE: **1.3668**
- Epoch-40 validation RMSE: **1.9352**

### Subject4

- Validation windows: **16**
- Physics model: `SpO2_linear = 110.776140 + (-30.465843) * R`
- Linear baseline: MSE = **18.6294**, RMSE = **4.3162**, MAE = **3.1703**
- Best learned-model validation RMSE: **3.6001** at epoch **30**
- Change relative to baseline: **+0.7161 RMSE (+16.59%)**; the learned residual model improved the baseline.
- Epoch-40 train RMSE: **2.3374**
- Epoch-40 validation RMSE: **4.0615**

### Subject5

- Validation windows: **17**
- Physics model: `SpO2_linear = 110.606306 + (-30.636493) * R`
- Linear baseline: MSE = **4.7412**, RMSE = **2.1774**, MAE = **1.7026**
- Best learned-model validation RMSE: **1.6968** at epoch **40**
- Change relative to baseline: **+0.4806 RMSE (+22.07%)**; the learned residual model improved the baseline.
- Epoch-40 train RMSE: **2.1549**
- Epoch-40 validation RMSE: **1.6968**

### Subject6

- Validation windows: **13**
- Physics model: `SpO2_linear = 110.355550 + (-30.156608) * R`
- Linear baseline: MSE = **4.9677**, RMSE = **2.2288**, MAE = **1.7996**
- Best learned-model validation RMSE: **2.1799** at epoch **6**
- Change relative to baseline: **+0.0489 RMSE (+2.19%)**; the learned residual model improved the baseline.
- Epoch-40 train RMSE: **1.4776**
- Epoch-40 validation RMSE: **3.1881**

### Subject7

- Validation windows: **18**
- Physics model: `SpO2_linear = 110.504980 + (-30.163680) * R`
- Linear baseline: MSE = **7.6581**, RMSE = **2.7673**, MAE = **2.2007**
- Best learned-model validation RMSE: **2.6079** at epoch **7**
- Change relative to baseline: **+0.1594 RMSE (+5.76%)**; the learned residual model improved the baseline.
- Epoch-40 train RMSE: **1.9011**
- Epoch-40 validation RMSE: **2.8800**

### Subject8

- Validation windows: **20**
- Physics model: `SpO2_linear = 110.595910 + (-30.359187) * R`
- Linear baseline: MSE = **1.7336**, RMSE = **1.3167**, MAE = **1.0393**
- Best learned-model validation RMSE: **0.9764** at epoch **18**
- Change relative to baseline: **+0.3403 RMSE (+25.84%)**; the learned residual model improved the baseline.
- Epoch-40 train RMSE: **2.2988**
- Epoch-40 validation RMSE: **2.1787**

### Subject9

- Validation windows: **23**
- Physics model: `SpO2_linear = 111.273729 + (-31.508580) * R`
- Linear baseline: MSE = **4.7596**, RMSE = **2.1817**, MAE = **1.6389**
- Best learned-model validation RMSE: **1.7545** at epoch **21**
- Change relative to baseline: **+0.4272 RMSE (+19.58%)**; the learned residual model improved the baseline.
- Epoch-40 train RMSE: **2.1059**
- Epoch-40 validation RMSE: **2.8584**

### Subject11

- Validation windows: **22**
- Physics model: `SpO2_linear = 110.258300 + (-29.665703) * R`
- Linear baseline: MSE = **30.1241**, RMSE = **5.4885**, MAE = **4.5907**
- Best learned-model validation RMSE: **5.4975** at epoch **1**
- Change relative to baseline: **-0.0090 RMSE (-0.16%)**; the learned residual model did not improve the baseline.
- Epoch-40 train RMSE: **1.3420**
- Epoch-40 validation RMSE: **6.9560**

### Subject12

- Validation windows: **24**
- Physics model: `SpO2_linear = 110.989991 + (-30.756924) * R`
- Linear baseline: MSE = **3.4189**, RMSE = **1.8490**, MAE = **1.6489**
- Best learned-model validation RMSE: **1.4286** at epoch **8**
- Change relative to baseline: **+0.4204 RMSE (+22.74%)**; the learned residual model improved the baseline.
- Epoch-40 train RMSE: **1.2943**
- Epoch-40 validation RMSE: **3.9324**

## 6. Main Observations

1. **The residual correction is beneficial for most held-out subjects.** The best checkpoint beats the physics-only ratio-of-ratios baseline for 10 of the 11 validation subjects.

2. **Generalization is strongly subject dependent.** Best validation RMSE ranges from **0.9764** for Subject8 to **6.8012** for Subject2. This spread is much larger than the within-fold training error and indicates substantial inter-subject variability.

3. **Subject2 is a difficult fold.** The linear RMSE is 6.8132 and the learned model only reduces it to 6.8012, a negligible improvement of about 0.18%. The best checkpoint occurs at epoch 1, while later training sharply worsens validation performance.

4. **Subject11 is the only fold where the learned residual is worse than the physics baseline.** Its linear RMSE is 5.4885 versus a best learned-model RMSE of 5.4975. The best checkpoint is also at epoch 1, suggesting that subject-specific residual patterns learned from the other subjects do not transfer well to Subject11.

5. **Several folds show clear overfitting after the best epoch.** For example, Subject1 reaches a best validation RMSE of 2.7744 at epoch 19 but ends at 3.9031; Subject8 reaches 0.9764 at epoch 18 but ends at 2.1787; Subject12 reaches 1.4286 at epoch 8 but ends at 3.9324. Therefore, reporting the best validation checkpoint is essential.

6. **The strongest folds are Subject8, Subject12, Subject3, Subject5, and Subject9.** Their best validation RMSE values are 0.9764, 1.4286, 1.4522, 1.6968, and 1.7545, respectively.

7. **The physics baseline itself varies greatly across subjects.** Linear RMSE ranges from 1.3167 to 6.8132, indicating that a single ratio-of-ratios relationship estimated from the remaining subjects does not generalize uniformly to every held-out individual.

## 7. Ranking by Best Validation RMSE

| Rank | Subject | Best RMSE | Linear RMSE | Improvement |
|---:|---|---:|---:|---:|
| 1 | Subject8 | **0.9764** | 1.3167 | +0.3403 |
| 2 | Subject12 | **1.4286** | 1.8490 | +0.4204 |
| 3 | Subject3 | **1.4522** | 1.9809 | +0.5287 |
| 4 | Subject5 | **1.6968** | 2.1774 | +0.4806 |
| 5 | Subject9 | **1.7545** | 2.1817 | +0.4272 |
| 6 | Subject6 | **2.1799** | 2.2288 | +0.0489 |
| 7 | Subject7 | **2.6079** | 2.7673 | +0.1594 |
| 8 | Subject1 | **2.7744** | 4.1865 | +1.4121 |
| 9 | Subject4 | **3.6001** | 4.3162 | +0.7161 |
| 10 | Subject11 | **5.4975** | 5.4885 | -0.0090 |
| 11 | Subject2 | **6.8012** | 6.8132 | +0.0120 |

## 8. Runtime Note

Every run printed the following preload warning:

```text
ERROR: ld.so: object '/ag4247/lib/x86_64-linux-gnu/libGLEW.so' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
```

This warning did not prevent PyTorch from detecting CUDA or using the NVIDIA GeForce RTX 4060 in the reported runs.

## 9. Overall Conclusion

The physics-guided residual model generally improves subject-independent SpO2 estimation relative to the linear ratio-of-ratios calibration, but the benefit is heterogeneous across subjects. The aggregate subject-wise mean RMSE decreases from **3.2097** to **2.7972**, while the window-weighted pooled RMSE decreases from **3.3756** to **3.0805**. The large degradation after the best epoch in several folds highlights the need for checkpoint selection/early stopping and suggests that subject-independent residual learning remains sensitive to inter-subject distribution shift, particularly for Subjects 2 and 11.