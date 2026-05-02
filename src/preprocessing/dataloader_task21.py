import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


def load_raw_data():
    file_path = "/Users/tanle/Downloads/Spring2026/CS313/DL4AI-220176-project/data-vn-20230228/stock-historical-data/HPG-VNINDEX-History.csv"
    df = pd.read_csv(file_path)
    return df

def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i])
        y.append(data[i, 2])  
    return np.array(X), np.array(y)


def load_and_preprocess(time_step=60, min_data_points=120):
    df = load_raw_data()

    if len(df) < min_data_points:
        raise ValueError(f"Không đủ data: {len(df)} < {min_data_points}")
    print(f"HPG có {len(df)} data points - đủ điều kiện")

    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = create_dataset(data_scaled, time_step)

    train_size = int(len(X) * 0.70)
    val_size   = int(len(X) * 0.15)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val,   y_val   = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test,  y_test  = X[train_size + val_size:], y[train_size + val_size:]

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_preprocess()
    print("X_train shape:", X_train.shape)
    print("y_test shape:", y_test.shape)