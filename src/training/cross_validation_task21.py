import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing.dataloader_task21 import load_and_preprocess
from src.models.model_task21 import build_model
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

TIME_STEP = 60

X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_preprocess(time_step=TIME_STEP)

X_cv = np.concatenate([X_train, X_val], axis=0)
y_cv = np.concatenate([y_train, y_val], axis=0)

tscv = TimeSeriesSplit(n_splits=5)

mae_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv)):
    print(f"\nFold {fold+1}/5")

    X_fold_train, X_fold_val = X_cv[train_idx], X_cv[val_idx]
    y_fold_train, y_fold_val = y_cv[train_idx], y_cv[val_idx]

    model = build_model(time_step=TIME_STEP, n_features=5)
    model.fit(
        X_fold_train, y_fold_train,
        epochs=5,
        batch_size=32,
        validation_data=(X_fold_val, y_fold_val),
        verbose=0
    )

    preds = model.predict(X_fold_val)
    mae = np.mean(np.abs(y_fold_val - preds.flatten()))
    mae_scores.append(mae)
    print(f"Fold {fold+1} MAE: {mae:.4f}")

print(f"\nMean MAE across 5 folds: {np.mean(mae_scores):.4f}")
print(f"Std  MAE across 5 folds: {np.std(mae_scores):.4f}")