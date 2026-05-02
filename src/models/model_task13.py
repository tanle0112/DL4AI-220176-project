from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_model(time_step=60, n_features=6, forecast_day=7):
    model = Sequential()

    model.add(LSTM(128, return_sequences=True, input_shape=(time_step, n_features))) 
    model.add(Dropout(0.2))

    model.add(LSTM(128))  
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(forecast_day))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model