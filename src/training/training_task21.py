import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing.dataloader_task21 import load_and_preprocess
from src.models.model_task21 import build_model
from tensorflow.keras.callbacks import EarlyStopping

X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_preprocess()

model = build_model(time_step=60, n_features=5)

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

model.save("stock_model_task21_HPG.h5")
print("Model saved!")