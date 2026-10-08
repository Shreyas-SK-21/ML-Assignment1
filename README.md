# Machine Learning Assignment 1: Polynomial Regression Analysis
**Roll Number:** IMT2024045  
**Course:** Machine Learning (SEM 5)  

---

## 📌 Repository Overview

This repository contains the complete implementation and empirical analysis for **Assignment 1 (Polynomial Regression)** built entirely from scratch using NumPy.

Two separate predictive scenarios are solved:
1. **Problem 1: Steam Turbine Optimization (`var1`)** — Predicting the Net Power Score ($y$) based on 6 operational parameters ($x_1, \dots, x_6$).
2. **Problem 2: Subterranean Thermal Reservoir Mapping (`var2`)** — Predicting the Thermal Anomaly Score ($y$) based on 3 spatial coordinates ($x_1, x_2, x_3$).

---

## 📁 Repository Structure

```text
ML-Assignment1/
│
├── IMT2024045/                     # Original training and test datasets
│   ├── IMT2024045_train_var1.csv
│   ├── IMT2024045_test_var1.csv
│   ├── IMT2024045_train_var2.csv
│   └── IMT2024045_test_var2.csv
│
├── 80_20_split/                    # Standard 80/20 train-validation holdout pipeline
│   ├── problem1/                   # Modular pipeline for Problem 1 (train, evaluate, predict)
│   ├── problem2/                   # Modular pipeline for Problem 2 (train, evaluate, predict)
│   ├── exhaustive_sweep.py         # Full feature-subset × degree sweep (80/20)
│   ├── exhaustive_sweep_results/   # Output metrics across all combinations
│   ├── IMT2024045_pred_var1.csv    # 80/20 test predictions for var1 (1000 rows, format verified)
│   └── IMT2024045_pred_var2.csv    # 80/20 test predictions for var2 (1000 rows, format verified)
│
├── kfold_split/                    # 10-Fold Cross-Validation pipeline
│   ├── kfold_exhaustive_sweep.py   # Full 10-fold CV sweep across all feature subsets × degrees
│   ├── kfold_results/              # Output CV metrics, averages, and fold standard deviations
│   ├── IMT2024045_pred_var1.csv    # 10-Fold CV test predictions for var1 (format verified)
│   └── IMT2024045_pred_var2.csv    # 10-Fold CV test predictions for var2 (format verified)
│
├── resources/                      # Report and plotting assets (Overleaf-ready)
│   ├── report.tex                  # Comprehensive LaTeX report
│   ├── var1_sweep.png              # Problem 1 validation MSE progression plot
│   ├── var1_residuals.png          # Problem 1 residual analysis plot
│   ├── var2_sweep.png              # Problem 2 validation MSE progression plot
│   ├── var2_residuals.png          # Problem 2 residual analysis plot
│   └── draft1.pdf                  # Compiled PDF report
│
├── IMT2024045_pred_var1.csv        # Final submission prediction file for Problem 1 (1000 rows)
├── IMT2024045_pred_var2.csv        # Final submission prediction file for Problem 2 (1000 rows)
├── sample_submission.csv           # Reference sample submission provided by course staff
└── ML_Assignment_1.pdf             # Original assignment specification document
```

---

## 🏆 Optimal Configurations & Findings

| Problem | Features Used | Optimal Degree | 80/20 Val MSE | 80/20 Val $R^2$ | 10-Fold Val MSE | 10-Fold Std Dev ($\sigma$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Problem 1 (`var1`)** | All 6 ($x_1 \dots x_6$) | **Degree 4** | **0.835** | **0.928** | **0.833** | **0.144** |
| **Problem 2 (`var2`)** | All 3 ($x_1 \dots x_3$) | **Degree 8** | **0.324** | **0.994** | **0.273** | **0.027** |

* **Feature Selection Justification:** Exhaustive powerset sweeps demonstrated that omitting features severely degrades performance due to loss of physical information (e.g., in Problem 1, best 3-feature MSE is 5.560 vs. 0.835 for 6 features; in Problem 2, 1-feature MSE is 38.028 vs. 0.324 for 3 features).
* **Stability:** 10-Fold CV verified that the chosen degrees exhibit the lowest standard deviation across folds, preventing high-degree variance explosion.

---

## 🚀 How to Reproduce

### 1. 80/20 Split Pipeline
To run the exhaustive sweep across all feature permutations and polynomial degrees under an 80/20 split:
```bash
cd 80_20_split
python exhaustive_sweep.py
```
To run the individual modular training/prediction pipelines:
```bash
cd 80_20_split/problem1
python train.py
python predict.py
```

### 2. 10-Fold Cross-Validation Pipeline
To reproduce the 10-Fold Cross-Validation exhaustive sweep:
```bash
cd kfold_split
python kfold_exhaustive_sweep.py
```

---

## 📄 Submission Format Verification
All output prediction files (`IMT2024045_pred_var1.csv`, `IMT2024045_pred_var2.csv` across root, `80_20_split/`, and `kfold_split/`) strictly match `sample_submission.csv`:
- **Rows:** Exactly 1000 test predictions (+1 header row)
- **Column:** Single column named `y`
- **Integrity:** Zero NaN or infinite values
