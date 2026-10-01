import sys, os, itertools
import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────
K_FOLDS = 10
RANDOM_SEED = 42

BASE = 'C:/Users/shrey/Desktop/SEM5/ML/Assss1/ML-Assignment1'
DATA_DIR = f'{BASE}/IMT2024045'
OUT_DIR = f'{BASE}/kfold_split/kfold_results'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helper Functions ───────────────────────────────────────────────────────
def get_k_folds(n, k=10, seed=42):
    """Returns a list of k arrays, each containing indices for a validation fold."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, k)

def load_all_data(train_csv, features):
    """Loads training data without a train/val split (for K-Fold)."""
    trdf = pd.read_csv(train_csv)
    X_all = trdf[list(features)].values.astype(float)
    y_all = trdf['y'].values.astype(float)
    return X_all, y_all

def poly_expand(X, degree):
    """Expands features to polynomial terms up to `degree`."""
    from itertools import combinations_with_replacement
    n, p = X.shape
    powers = []
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

def run_kfold_combinations(problem, train_csv, all_features, degrees):
    # Generate all non-empty subsets of features
    subsets = []
    for r in range(1, len(all_features)+1):
        subsets.extend(list(itertools.combinations(all_features, r)))
    
    results = []
    total = len(subsets) * len(degrees)
    done = 0
    print(f"Starting {problem} {K_FOLDS}-Fold sweep ({total} configurations)...")
    
    for feats in subsets:
        X_all, y_all = load_all_data(train_csv, feats)
        feat_name = "_".join(feats)
        n_samples = len(X_all)
        folds = get_k_folds(n_samples, k=K_FOLDS, seed=RANDOM_SEED)
        
        for deg in degrees:
            # We only need to generate the powers structure once per (feats, deg)
            Phi_all, powers = poly_expand(X_all, deg)
            n_terms = len(powers)
            
            fold_t_mse = []
            fold_v_mse = []
            fold_t_r2  = []
            fold_v_r2  = []
            
            for i in range(K_FOLDS):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(K_FOLDS) if j != i])
                
                Phi_tr, y_tr = Phi_all[train_idx], y_all[train_idx]
                Phi_va, y_va = Phi_all[val_idx], y_all[val_idx]
                
                # Fit model
                w, _, _, _ = np.linalg.lstsq(Phi_tr, y_tr, rcond=None)
                
                # Predict
                yp_tr = Phi_tr @ w
                yp_va = Phi_va @ w
                
                # Metrics
                t_mse = float(np.mean((y_tr - yp_tr)**2))
                v_mse = float(np.mean((y_va - yp_va)**2))
                
                ss_tot_t = np.sum((y_tr - np.mean(y_tr))**2)
                ss_res_t = np.sum((y_tr - yp_tr)**2)
                t_r2 = 1.0 - (ss_res_t / ss_tot_t) if ss_tot_t != 0 else 0
                
                ss_tot_v = np.sum((y_va - np.mean(y_va))**2)
                ss_res_v = np.sum((y_va - yp_va)**2)
                v_r2 = 1.0 - (ss_res_v / ss_tot_v) if ss_tot_v != 0 else 0
                
                fold_t_mse.append(t_mse)
                fold_v_mse.append(v_mse)
                fold_t_r2.append(t_r2)
                fold_v_r2.append(v_r2)
            
            # Aggregate metrics across the K folds
            results.append({
                'features': feat_name,
                'num_features': len(feats),
                'degree': deg,
                'terms': n_terms,
                'train_mse_avg': round(np.mean(fold_t_mse), 6),
                'val_mse_avg': round(np.mean(fold_v_mse), 6),
                'train_r2_avg': round(np.mean(fold_t_r2), 6),
                'val_r2_avg': round(np.mean(fold_v_r2), 6),
                'val_mse_std': round(np.std(fold_v_mse), 6) # Helpful to see how stable the model is
            })
            
            done += 1
            if done % 50 == 0:
                print(f"  {problem}: {done}/{total} done")
                
    df = pd.DataFrame(results)
    df.to_csv(f'{OUT_DIR}/{problem}_{K_FOLDS}fold_exhaustive_metrics.csv', index=False)
    
    # Save top 20 by average validation MSE
    best = df.sort_values('val_mse_avg').head(20)
    best.to_csv(f'{OUT_DIR}/{problem}_top20_best_{K_FOLDS}fold.csv', index=False)
    print(f"Finished {problem}. Results saved to {OUT_DIR}.\n")

# ── Execute Sweeps ─────────────────────────────────────────────────────────

# var1: 6 features -> 63 combinations * 10 degrees = 630 runs (each doing 10 folds = 6300 models)
run_kfold_combinations('var1', f'{DATA_DIR}/IMT2024045_train_var1.csv', 
                       ['x1','x2','x3','x4','x5','x6'], range(1, 11))

# var2: 3 features -> 7 combinations * 10 degrees = 70 runs (each doing 10 folds = 700 models)
run_kfold_combinations('var2', f'{DATA_DIR}/IMT2024045_train_var2.csv', 
                       ['x1','x2','x3'], range(1, 11))
