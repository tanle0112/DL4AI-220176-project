#load dataset 
#chia train, val, test 
#normalize 0,1 
#resize (if need) 
#def get_dataloader():
#    return train_loader, val_loader, test_loader

#Task 1.1 use multi features rather than just only Open 
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
def load_raw_data():
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    file_path = os.path.join(base_path, "data", "AAPL.csv")

    df = pd.read_csv(file_path)
    return df

def create_dataset(data, time_step=60, forecast_day=1):
    X, y = [], []

    for i in range(time_step, len(data) - forecast_day + 1):
        X.append(data[i-time_step:i])
        y.append(data[i + forecast_day - 1, 3])

    return np.array(X), np.array(y)

def load_and_preprocess(time_step=60, forecast_day=1):
    df = load_raw_data()

    features = ['Low', 'High', 'Open', 'Close', 'Adjusted Close', 'Volume']
    data = df[features].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = create_dataset(data_scaled, time_step, forecast_day)

    split = int(len(X) * 0.8)

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]

    return X_train, y_train, X_test, y_test, scaler
if __name__ == "__main__":

    X_train, y_train, X_test, y_test, scaler = load_and_preprocess(forecast_day=3)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_test shape :", y_test.shape)