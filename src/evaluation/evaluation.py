import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing.dataloader import load_and_preprocess


X_train, y_train, X_test, y_test, scaler = load_and_preprocess(forecast_day=3)

model = load_model("stock_model_day3.h5")

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("MAE :", mae)
print("RMSE:", rmse)

plt.figure(figsize=(12,6))
plt.plot(y_test, label="Actual")
plt.plot(predictions, label="Predicted")
plt.title("AAPL Stock Price Prediction (3-Day Ahead)")
plt.xlabel("Time")
plt.ylabel("Scaled Close Price")
plt.legend()
plt.show()