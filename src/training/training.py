import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing.dataloader import load_and_preprocess
from src.models.model import build_model


X_train, y_train, X_test, y_test, scaler = load_and_preprocess(forecast_day=3)

model = build_model()

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

model.save("stock_model_day3.h5")