"""
full_sweep.py
-------------
Comprehensive sweep across all feature sets and polynomial degrees
for BOTH Problem 1 (var1) and Problem 2 (var2).

Results saved to:  full_sweep_results/
  - metrics_var1.csv          : all var1 runs (MSE, RMSE, R2, MAE — train & val)
  - metrics_var2.csv          : all var2 runs
  - summary_report.md         : human-readable markdown table of all results
  - plots/var1_sweep.png      : MSE & R2 vs degree for each feature set (var1)
  - plots/var2_sweep.png      : MSE & R2 vs degree for each feature set (var2)
  - plots/var1_residuals_<tag>.png  : residual plots for every run
  - plots/var2_residuals_<tag>.png  : residual plots for every run
  - predictions/              : prediction CSVs for every run
"""

import sys, os
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')       # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ──────────────────────────────────────────────────────────────────
BASE     = 'C:/Users/shrey/Desktop/SEM5/ML/Assss1/ML-Assignment1'
DATA_DIR = f'{BASE}/IMT2024045'
OUT_DIR  = f'{BASE}/full_sweep_results'

os.makedirs(f'{OUT_DIR}/plots',       exist_ok=True)
os.makedirs(f'{OUT_DIR}/predictions', exist_ok=True)

# ── Shared helpers ──────────────────────────────────────────────────────────
def mse(yt, yp):  return float(np.mean((yt - yp)**2))
def rmse(yt, yp): return float(np.sqrt(mse(yt, yp)))
def r2(yt, yp):
    ss_res = np.sum((yt - yp)**2); ss_tot = np.sum((yt - np.mean(yt))**2)
    return float(1.0 - ss_res / ss_tot)
def mae(yt, yp):  return float(np.mean(np.abs(yt - yp)))

def load(train_csv, test_csv, features, val_ratio=0.20, seed=42):
    trdf = pd.read_csv(train_csv); tedf = pd.read_csv(test_csv)
    X_all  = trdf[features].values.astype(float)
    y_all  = trdf['y'].values.astype(float)
    X_test = tedf[features].values.astype(float)
    rng = np.random.default_rng(seed); idx = rng.permutation(len(X_all))
    n_val = int(np.floor(val_ratio * len(X_all)))
    return (X_all[idx[n_val:]], y_all[idx[n_val:]],
            X_all[idx[:n_val]], y_all[idx[:n_val]],
            X_test)

def poly_expand(X, degree):
    from itertools import combinations_with_replacement
    n, p = X.shape; powers = []
    for d in range(degree + 1):
        for combo in combinations_with_replacement(range(p), d):
            exps = [0]*p
            for i in combo: exps[i] += 1
            powers.append(tuple(exps))
    Phi = np.ones((n, len(powers)))
    for ci, exps in enumerate(powers):
        for fi, e in enumerate(exps):
            if e > 0: Phi[:, ci] *= X[:, fi]**e
    return Phi, powers

def term_label(fname, powers):
    labels = []
    for exps in powers:
        parts = [f'{f}^{e}' if e > 1 else f for f, e in zip(fname, exps) if e > 0]
        labels.append('*'.join(parts) if parts else '1')
    return labels

def fit_model(X_train, y_train, X_val, y_val, X_test, degree, feature_names):
    Phi_tr, powers = poly_expand(X_train, degree)
    Phi_va, _      = poly_expand(X_val,   degree)
    Phi_te, _      = poly_expand(X_test,  degree)
    w, _, _, _ = np.linalg.lstsq(Phi_tr, y_train, rcond=None)
    yp_tr = Phi_tr @ w; yp_va = Phi_va @ w; yp_te = Phi_te @ w
    return {
        'w': w, 'powers': powers,
        'yp_train': yp_tr, 'yp_val': yp_va, 'yp_test': yp_te,
        'train_mse': mse(y_train,yp_tr), 'val_mse': mse(y_val,yp_va),
        'train_rmse':rmse(y_train,yp_tr),'val_rmse':rmse(y_val,yp_va),
        'train_r2':  r2(y_train,yp_tr),  'val_r2':  r2(y_val,yp_va),
        'train_mae': mae(y_train,yp_tr),  'val_mae': mae(y_val,yp_va),
        'n_terms': len(powers),
    }

def save_residual_plot(y_train, yp_train, y_val, yp_val, degree, tag, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    lim_tr = [min(y_train.min(), yp_train.min()), max(y_train.max(), yp_train.max())]
    axes[0].scatter(yp_train, y_train, s=8, alpha=0.4, color='steelblue')
    axes[0].plot(lim_tr, lim_tr, 'r--', lw=1.5, label='Perfect fit')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
    axes[0].set_title(f'Train  |  {tag}  |  Deg {degree}')
    axes[0].legend(fontsize=8)
    lim_va = [min(y_val.min(), yp_val.min()), max(y_val.max(), yp_val.max())]
    axes[1].scatter(yp_val, y_val, s=8, alpha=0.45, color='tomato')
    axes[1].plot(lim_va, lim_va, 'b--', lw=1.5, label='Perfect fit')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
    axes[1].set_title(f'Val  |  {tag}  |  Deg {degree}  |  Val R²={r2(y_val,yp_val):.3f}')
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/plots/residuals_{tag}_deg{degree}.png', dpi=120)
    plt.close()

def run_sweep(problem, train_csv, test_csv, feature_sets, degrees):
    """Run all combinations and collect results."""
    all_rows = []
    sweep_data = {}   # keyed by feature-set label for the sweep plot

    total = sum(len(degrees) for _ in feature_sets)
    done  = 0

    for feat_label, features in feature_sets.items():
        print(f'\n  [{problem}] Feature set: {feat_label}')
        X_train, y_train, X_val, y_val, X_test = load(train_csv, test_csv, features)
        sweep_data[feat_label] = {'degrees': [], 'train_mse': [], 'val_mse': [],
                                  'train_r2': [], 'val_r2': []}
        for deg in degrees:
            done += 1
            print(f'    Deg {deg:2d}  ({done}/{total})  ... ', end='', flush=True)
            res = fit_model(X_train, y_train, X_val, y_val, X_test, deg, features)

            tag = f'{problem}_{feat_label}'
            # Save residual plot
            save_residual_plot(y_train, res['yp_train'],
                               y_val,   res['yp_val'],
                               deg, tag, OUT_DIR)

            # Save prediction CSV
            pred_path = f'{OUT_DIR}/predictions/{problem}_{feat_label}_deg{deg}.csv'
            pd.DataFrame({'y': res['yp_test']}).to_csv(pred_path, index=False)

            # Collect row
            all_rows.append({
                'problem': problem,
                'features': feat_label,
                'degree': deg,
                'n_terms': res['n_terms'],
                'train_mse':  round(res['train_mse'],  6),
                'val_mse':    round(res['val_mse'],    6),
                'train_rmse': round(res['train_rmse'], 6),
                'val_rmse':   round(res['val_rmse'],   6),
                'train_r2':   round(res['train_r2'],   6),
                'val_r2':     round(res['val_r2'],     6),
                'train_mae':  round(res['train_mae'],  6),
                'val_mae':    round(res['val_mae'],    6),
            })
            sweep_data[feat_label]['degrees'].append(deg)
            sweep_data[feat_label]['train_mse'].append(res['train_mse'])
            sweep_data[feat_label]['val_mse'].append(res['val_mse'])
            sweep_data[feat_label]['train_r2'].append(res['train_r2'])
            sweep_data[feat_label]['val_r2'].append(res['val_r2'])

            print(f'Val MSE={res["val_mse"]:.4f}  Val R2={res["val_r2"]:.4f}')

    return all_rows, sweep_data


def save_sweep_plot(problem, sweep_data, out_dir):
    n_sets = len(sweep_data)
    fig, axes = plt.subplots(2, n_sets, figsize=(7*n_sets, 10))
    if n_sets == 1:
        axes = axes.reshape(2, 1)
    colors = ['steelblue', 'tomato', 'seagreen', 'darkorchid']
    for col, (label, data) in enumerate(sweep_data.items()):
        degs = data['degrees']
        ax_mse = axes[0, col]
        ax_mse.plot(degs, data['train_mse'], 'o-', color='steelblue', label='Train MSE')
        ax_mse.plot(degs, data['val_mse'],   's-', color='tomato',    label='Val MSE')
        ax_mse.set_xlabel('Degree'); ax_mse.set_ylabel('MSE')
        ax_mse.set_title(f'{problem} | {label}\nMSE vs Degree')
        ax_mse.set_xticks(degs); ax_mse.legend(fontsize=9); ax_mse.grid(alpha=0.3)

        ax_r2 = axes[1, col]
        ax_r2.plot(degs, data['train_r2'], 'o-', color='steelblue', label='Train R2')
        ax_r2.plot(degs, data['val_r2'],   's-', color='tomato',    label='Val R2')
        ax_r2.set_xlabel('Degree'); ax_r2.set_ylabel('R²')
        ax_r2.set_title(f'{problem} | {label}\nR² vs Degree')
        ax_r2.set_xticks(degs); ax_r2.legend(fontsize=9); ax_r2.grid(alpha=0.3)

    plt.suptitle(f'{problem} — Full Degree Sweep', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/plots/{problem}_sweep.png', dpi=130)
    plt.close()
    print(f'  Sweep plot saved -> {out_dir}/plots/{problem}_sweep.png')


def write_summary(all_rows, path):
    df = pd.DataFrame(all_rows)
    df.to_csv(path.replace('.md', '.csv'), index=False)

    with open(path, 'w', encoding='utf-8') as f:
        f.write('# Full Sweep Results — IMT2024045\n\n')
        for prob in df['problem'].unique():
            f.write(f'## {prob}\n\n')
            sub = df[df['problem'] == prob]
            for feat in sub['features'].unique():
                fsub = sub[sub['features'] == feat].sort_values('degree')
                f.write(f'### Feature set: `{feat}`\n\n')
                f.write('| Degree | Terms | Train MSE | Val MSE | Train R² | Val R² | Train RMSE | Val RMSE |\n')
                f.write('|--------|-------|-----------|---------|----------|--------|------------|----------|\n')
                for _, row in fsub.iterrows():
                    f.write(f"| {int(row['degree'])} | {int(row['n_terms'])} | "
                            f"{row['train_mse']:.4f} | {row['val_mse']:.4f} | "
                            f"{row['train_r2']:.4f} | {row['val_r2']:.4f} | "
                            f"{row['train_rmse']:.4f} | {row['val_rmse']:.4f} |\n")
                # best row
                best = fsub.loc[fsub['val_mse'].idxmin()]
                f.write(f"\n**Best by Val MSE: Degree {int(best['degree'])}** "
                        f"(Val MSE={best['val_mse']:.4f}, Val R²={best['val_r2']:.4f})\n\n")
    print(f'\n  Summary saved -> {path}')

# ═══════════════════════════════════════════════════════════════════
# PROBLEM 1 — var1
# ═══════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('  PROBLEM 1 — var1')
print('='*60)

VAR1_TRAIN = f'{DATA_DIR}/IMT2024045_train_var1.csv'
VAR1_TEST  = f'{DATA_DIR}/IMT2024045_test_var1.csv'

FEAT_SETS_VAR1 = {
    '3feat': ['x1','x2','x3'],
    '6feat': ['x1','x2','x3','x4','x5','x6'],
}
DEGREES_VAR1 = list(range(1, 11))   # 1 to 10

rows_var1, sweep_var1 = run_sweep('var1', VAR1_TRAIN, VAR1_TEST, FEAT_SETS_VAR1, DEGREES_VAR1)
save_sweep_plot('var1', sweep_var1, OUT_DIR)

# ═══════════════════════════════════════════════════════════════════
# PROBLEM 2 — var2
# ═══════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('  PROBLEM 2 — var2')
print('='*60)

VAR2_TRAIN = f'{DATA_DIR}/IMT2024045_train_var2.csv'
VAR2_TEST  = f'{DATA_DIR}/IMT2024045_test_var2.csv'

FEAT_SETS_VAR2 = {
    '1feat_x1':    ['x1'],
    '3feat':       ['x1','x2','x3'],
}
DEGREES_VAR2 = list(range(1, 11))   # 1 to 10

rows_var2, sweep_var2 = run_sweep('var2', VAR2_TRAIN, VAR2_TEST, FEAT_SETS_VAR2, DEGREES_VAR2)
save_sweep_plot('var2', sweep_var2, OUT_DIR)

# ═══════════════════════════════════════════════════════════════════
# SAVE ALL RESULTS
# ═══════════════════════════════════════════════════════════════════
all_rows = rows_var1 + rows_var2
write_summary(all_rows, f'{OUT_DIR}/summary_report.md')

# Also separate CSVs per problem
pd.DataFrame(rows_var1).to_csv(f'{OUT_DIR}/metrics_var1.csv', index=False)
pd.DataFrame(rows_var2).to_csv(f'{OUT_DIR}/metrics_var2.csv', index=False)

print('\n' + '='*60)
print('  ALL DONE!')
print(f'  Results in: {OUT_DIR}')
print('='*60)
