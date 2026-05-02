import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


def load_raw_data():
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    file_path = os.path.join(base_path, "/Users/tanle/Downloads/Spring2026/CS313/DL4AI-220176-project/data/AAPL.csv")
    df = pd.read_csv(file_path)
    return df

def create_dataset(data, time_step=60, forecast_day=7):
    X, y = [], []

    for i in range(time_step, len(data) - forecast_day + 1):
        X.append(data[i-time_step:i])
        y.append(data[i:i+forecast_day, 3])

    return np.array(X), np.array(y)


def load_and_preprocess(time_step=60, forecast_day=7):
    df = load_raw_data()

    features = ['Low', 'High', 'Open', 'Close', 'Adjusted Close', 'Volume']
    data = df[features].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = create_dataset(data_scaled, time_step, forecast_day)

    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]

    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_preprocess()
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)