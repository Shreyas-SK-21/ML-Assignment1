import sys, os, itertools
import numpy as np
import pandas as pd

BASE = 'C:/Users/shrey/Desktop/SEM5/ML/Assss1/ML-Assignment1'
DATA_DIR = f'{BASE}/IMT2024045'
OUT_DIR = f'{BASE}/exhaustive_sweep_results'
os.makedirs(OUT_DIR, exist_ok=True)

def load(train_csv, test_csv, features, val_ratio=0.20, seed=42):
    trdf = pd.read_csv(train_csv)
    tedf = pd.read_csv(test_csv)
    X_all = trdf[list(features)].values.astype(float)
    y_all = trdf['y'].values.astype(float)
    X_test = tedf[list(features)].values.astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_all))
    n_val = int(np.floor(val_ratio * len(X_all)))
    return (X_all[idx[n_val:]], y_all[idx[n_val:]],
            X_all[idx[:n_val]], y_all[idx[:n_val]],
            X_test)

def poly_expand(X, degree):
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

def run_all_combinations(problem, train_csv, test_csv, all_features, degrees):
    subsets = []
    for r in range(1, len(all_features)+1):
        subsets.extend(list(itertools.combinations(all_features, r)))
    
    results = []
    total = len(subsets) * len(degrees)
    done = 0
    print(f"Starting {problem} exhaustive sweep ({total} configurations)...")
    
    for feats in subsets:
        X_train, y_train, X_val, y_val, X_test = load(train_csv, test_csv, feats)
        feat_name = "_".join(feats)
        for deg in degrees:
            Phi_tr, powers = poly_expand(X_train, deg)
            Phi_va, _ = poly_expand(X_val, deg)
            w, _, _, _ = np.linalg.lstsq(Phi_tr, y_train, rcond=None)
            
            yp_tr = Phi_tr @ w
            yp_va = Phi_va @ w
            
            t_mse = float(np.mean((y_train - yp_tr)**2))
            v_mse = float(np.mean((y_val - yp_va)**2))
            
            ss_tot_t = np.sum((y_train - np.mean(y_train))**2)
            ss_res_t = np.sum((y_train - yp_tr)**2)
            t_r2 = 1.0 - (ss_res_t / ss_tot_t) if ss_tot_t != 0 else 0
            
            ss_tot_v = np.sum((y_val - np.mean(y_val))**2)
            ss_res_v = np.sum((y_val - yp_va)**2)
            v_r2 = 1.0 - (ss_res_v / ss_tot_v) if ss_tot_v != 0 else 0
            
            results.append({
                'features': feat_name,
                'num_features': len(feats),
                'degree': deg,
                'terms': len(powers),
                'train_mse': round(t_mse, 6),
                'val_mse': round(v_mse, 6),
                'train_r2': round(t_r2, 6),
                'val_r2': round(v_r2, 6)
            })
            done += 1
            if done % 100 == 0:
                print(f"  {problem}: {done}/{total} done")
                
    df = pd.DataFrame(results)
    df.to_csv(f'{OUT_DIR}/{problem}_exhaustive_metrics.csv', index=False)
    
    best = df.sort_values('val_mse').head(20)
    best.to_csv(f'{OUT_DIR}/{problem}_top20_best.csv', index=False)
    print(f"Finished {problem}. Results saved to {OUT_DIR}.\n")

# Run var1 (63 feature combinations * 10 degrees = 630 runs)
run_all_combinations('var1', f'{DATA_DIR}/IMT2024045_train_var1.csv', f'{DATA_DIR}/IMT2024045_test_var1.csv', ['x1','x2','x3','x4','x5','x6'], range(1, 11))

# Run var2 (7 feature combinations * 10 degrees = 70 runs)
run_all_combinations('var2', f'{DATA_DIR}/IMT2024045_train_var2.csv', f'{DATA_DIR}/IMT2024045_test_var2.csv', ['x1','x2','x3'], range(1, 11))
