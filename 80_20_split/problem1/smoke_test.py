import sys
sys.path.insert(0, '.')

import data_loader, model as M, evaluate as E, train as T, predict as P
import numpy as np

# Load data
X_train, y_train, X_val, y_val, X_test, feature_names = data_loader.load_data()

# Degree sweep
results = T.sweep_degrees(X_train, y_train, X_val, y_val, feature_names=feature_names)
print('\nDegree | Train MSE | Val MSE  | Train R2 | Val R2')
print('-' * 60)
for r in results:
    print(f"  {r['degree']}    | {r['train_mse']:.4f}    | {r['val_mse']:.4f}   | {r['train_r2']:.4f}   | {r['val_r2']:.4f}")

# Final model at degree 3
w, powers, tm, vm = T.train_model(X_train, y_train, X_val, y_val, degree=3, feature_names=feature_names)

# Predict
y_pred = P.generate_predictions(X_test, w, degree=3, save=True)
print('DONE')
