import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing.dataloader_task13 import load_and_preprocess


X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_preprocess()

model = load_model("stock_model_task13_day7.h5")

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test.flatten(), predictions.flatten())
rmse = np.sqrt(mean_squared_error(y_test.flatten(), predictions.flatten()))

print("MAE :", mae)
print("RMSE:", rmse)

plt.figure(figsize=(12,6))
plt.plot(y_test[:,0], label="Actual Day+1")
plt.plot(predictions[:,0], label="Predicted Day+1")
plt.title("Task 1.3 - 7 Day Forecast (First Day Horizon)")
plt.xlabel("Samples")
plt.ylabel("Scaled Close Price")
plt.legend()
plt.show()