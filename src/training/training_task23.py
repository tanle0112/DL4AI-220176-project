import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing.dataloader_task23 import load_and_preprocess
from src.models.model_task23 import build_model
from tensorflow.keras.callbacks import EarlyStopping

K = 7

X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_preprocess(
    time_step=60, forecast_day=K
)

model = build_model(time_step=60, n_features=5, forecast_day=K)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

model.save(f"stock_model_task23_HPG_day{K}.h5")
print(f"Model saved: stock_model_task23_HPG_day{K}.h5")