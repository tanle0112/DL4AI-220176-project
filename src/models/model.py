# src/models/model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_model(time_step=60, n_features=6):
    model = Sequential()

    model.add(LSTM(64, return_sequences=True, input_shape=(time_step, n_features)))
    model.add(Dropout(0.2))

    model.add(LSTM(64))
    model.add(Dropout(0.2))

    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model