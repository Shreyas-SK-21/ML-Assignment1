"""
smoke_test.py — Problem 2 (var2)
Quick end-to-end verification that all modules work correctly.
"""
import sys
sys.path.insert(0, '.')

import data_loader, model as M, evaluate as E, train as T, predict as P
import numpy as np

TRAIN_CSV = 'C:/Users/shrey/Desktop/SEM5/ML/Assss1/ML-Assignment1/IMT2024045/IMT2024045_train_var2.csv'
TEST_CSV  = 'C:/Users/shrey/Desktop/SEM5/ML/Assss1/ML-Assignment1/IMT2024045/IMT2024045_test_var2.csv'

# Load data
X_train, y_train, X_val, y_val, X_test, feature_names = data_loader.load_data(
    train_csv=TRAIN_CSV, test_csv=TEST_CSV
)

# Degree sweep
results = T.sweep_degrees(X_train, y_train, X_val, y_val, feature_names=feature_names)
print('\nDegree | Train MSE | Val MSE  | Train R2 | Val R2')
print('-' * 65)
for r in results:
    print('  {}    | {:.4f}    | {:.4f}   | {:.4f}   | {:.4f}'.format(
        r['degree'], r['train_mse'], r['val_mse'], r['train_r2'], r['val_r2']
    ))

# Final model at degree 4
w, powers, tm, vm = T.train_model(X_train, y_train, X_val, y_val, degree=4, feature_names=feature_names)

# Predict
y_pred = P.generate_predictions(
    X_test, w, degree=4,
    output_csv='C:/Users/shrey/Desktop/SEM5/ML/Assss1/IMT2024045_pred_var2.csv',
    save=True
)
print('DONE')
