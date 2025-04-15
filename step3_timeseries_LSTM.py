import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Load the dataset
df = pd.read_csv("combined_data.csv")

# Step 2: Split the data
benign_data = df[df['attack_type'] == 'benign'].reset_index(drop=True)
mirai_data = df[df['attack_type'].str.contains('mirai', case=False)].reset_index(drop=True)
gafgyt_data = df[df['attack_type'].str.contains('gafgyt', case=False)].reset_index(drop=True)

# Step 3: Define the best feature
feature_column = 'HH_L0.01_mean'  # This was found to be informative


def create_lstm_dataset(series, time_step=1):
    """
    Converts the time series data into a format suitable for LSTM input.
    """
    data = series.values
    data = data.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled_data) - time_step - 1):
        X.append(scaled_data[i:(i + time_step), 0])
        y.append(scaled_data[i + time_step, 0])

    X = np.array(X)
    y = np.array(y)

    # Reshaping X for LSTM [samples, time_steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y, scaler


def apply_lstm(series, label, time_step=100, steps=10, last_n=1000):
    """
    Applies LSTM model for time series forecasting.
    """
    # Select last N points
    subset = series[-last_n:]

    # Prepare the data for LSTM
    X, y, scaler = create_lstm_dataset(subset, time_step)

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(time_step, 1)))
    model.add(Dense(units=1))  # Output layer

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # Make predictions
    last_data = subset[-time_step:].values.reshape(-1, 1)
    last_data_scaled = scaler.transform(last_data)
    last_data_scaled = last_data_scaled.reshape(1, time_step, 1)

    forecast = []
    for _ in range(steps):
        predicted = model.predict(last_data_scaled, verbose=0)
        forecast.append(predicted[0][0])
        last_data_scaled = np.append(last_data_scaled[:, 1:, :], predicted.reshape(1, 1, 1), axis=1)

    # Inverse scaling of forecast
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(subset)), subset, label='Original Data', color='blue')
    plt.plot(range(len(subset), len(subset) + steps), forecast, label='Forecasted Data', color='red')
    plt.title(f'{label} Data Forecast using LSTM (Last {last_n} Points)')
    plt.xlabel("Time Index")
    plt.ylabel("Feature Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Apply LSTM to each category
apply_lstm(benign_data[feature_column], "Benign")
apply_lstm(mirai_data[feature_column], "Mirai")
apply_lstm(gafgyt_data[feature_column], "Gafgyt")
