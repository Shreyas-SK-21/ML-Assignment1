import sys
sys.path.insert(0, '.')
import os
import numpy as np
import matplotlib.pyplot as plt
import data_loader, model as M, evaluate as E, train as T, predict as P

# 1. Setup Output Directory
out_dir = "report_results_deg3_6feat"
os.makedirs(out_dir, exist_ok=True)

# 2. Configure Parameters
features = ["x1", "x2", "x3", "x4", "x5", "x6"]
degree = 3

# 3. Load Data
X_train, y_train, X_val, y_val, X_test, feature_names = data_loader.load_data(features=features)

# 4. Train Model
w, powers, tm, vm = T.train_model(X_train, y_train, X_val, y_val, degree=degree, feature_names=feature_names, verbose=False)

# 5. Run Degree Sweep (for plot)
results = T.sweep_degrees(X_train, y_train, X_val, y_val, degree_min=1, degree_max=6, feature_names=feature_names)
degrees = [r['degree'] for r in results]
train_mse = [r['train_mse'] for r in results]
val_mse = [r['val_mse'] for r in results]
train_r2 = [r['train_r2'] for r in results]
val_r2 = [r['val_r2'] for r in results]

# --- PLOT 1: Degree Sweep ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(degrees, train_mse, 'o-', label='Train MSE', color='steelblue')
axes[0].plot(degrees, val_mse, 's-', label='Val MSE', color='tomato')
axes[0].set_xlabel('Polynomial Degree')
axes[0].set_ylabel('MSE')
axes[0].set_title('MSE vs Degree (6 features)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(degrees, train_r2, 'o-', label='Train R²', color='steelblue')
axes[1].plot(degrees, val_r2, 's-', label='Val R²', color='tomato')
axes[1].set_xlabel('Polynomial Degree')
axes[1].set_ylabel('R² Score')
axes[1].set_title('R² vs Degree (6 features)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out_dir}/plot_degree_sweep.png", dpi=150)
plt.close()

# --- PLOT 2: Residuals ---
Phi_train, _ = M.expand_features(X_train, degree=degree)
Phi_val, _ = M.expand_features(X_val, degree=degree)
y_pred_train = M.predict(Phi_train, w)
y_pred_val = M.predict(Phi_val, w)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_pred_train, y_train, alpha=0.5, s=15, color='steelblue')
lims = [min(y_train.min(), y_pred_train.min()), max(y_train.max(), y_pred_train.max())]
axes[0].plot(lims, lims, 'r--', label='Perfect fit')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title(f'Train: Predicted vs Actual (Deg {degree})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_pred_val, y_val, alpha=0.5, s=15, color='tomato')
lims = [min(y_val.min(), y_pred_val.min()), max(y_val.max(), y_pred_val.max())]
axes[1].plot(lims, lims, 'b--', label='Perfect fit')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title(f'Val: Predicted vs Actual (Deg {degree})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out_dir}/plot_residuals.png", dpi=150)
plt.close()

# --- SAVE SUMMARY TEXT ---
labels = M.get_term_labels(feature_names, powers)
order = np.argsort(np.abs(w))[::-1][:10]

summary = f"=== PROBLEM 1 REPORT RESULTS ===\n"
summary += f"Features Used: {', '.join(features)}\n"
summary += f"Polynomial Degree: {degree}\n"
summary += f"Total Terms: {len(powers)}\n"
summary += f"--------------------------------\n"
summary += f"Train MSE: {tm['mse']:.6f}\n"
summary += f"Train R2:  {tm['r2']:.6f}\n"
summary += f"Val MSE:   {vm['mse']:.6f}\n"
summary += f"Val R2:    {vm['r2']:.6f}\n"
summary += f"--------------------------------\n"
summary += f"Top 10 Terms by Magnitude:\n"
for i in order:
    summary += f"  {labels[i]:20s} w = {w[i]:+.4f}\n"

with open(f"{out_dir}/metrics_summary.txt", "w", encoding='utf-8') as f:
    f.write(summary)

# --- SAVE PREDICTIONS ---
P.generate_predictions(X_test, w, degree=degree, output_csv="../../IMT2024045_pred_var1.csv", save=True)
P.generate_predictions(X_test, w, degree=degree, output_csv=f"{out_dir}/IMT2024045_pred_var1_copy.csv", save=True)

print(f"Success! Report files saved to {out_dir}")
