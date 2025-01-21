# Original Hyper.py Code with Trend Integration
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
data = pd.read_csv("merged_data_py.csv")


# Prepare the dataset with additional trend feature
def prepare_data(data, seq_length):
    # Adding trend as an additional feature
    data['Annual_Avg'] = data['Energy_Consumption'].rolling(window=12).mean()
    data = data.dropna()  # Drop rows with NaN values caused by rolling mean

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Energy_Consumption', 'Annual_Avg']])
    
    x, y = [], []
    for i in range(len(data_scaled) - seq_length):
        x.append(data_scaled[i:i+seq_length, :])  # Include both features (raw and trend)
        y.append(data_scaled[i+seq_length, 0])  # Predict the raw value (Energy_Consumption)
    
    x = np.array(x)
    y = np.array(y)
    return x, y, scaler

# Build the model
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Main function
def main():
    file_name = 'your_data.csv'
    data = load_data(file_name)
    
    seq_length = 12
    x, y, scaler = prepare_data(data[['Energy_Consumption']], seq_length)
    
    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = build_model((seq_length, x.shape[2]))
    model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))
    
    predictions = model.predict(x_test)

    # Reverse scaling
    predicted_scaled = np.zeros((len(predictions), 2))  # Create a placeholder for inverse transform
    predicted_scaled[:, 0] = predictions[:, 0]
    predicted = scaler.inverse_transform(predicted_scaled)[:, 0]
    y_test_scaled = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), 1)))))[:, 0]
    
    mse = mean_squared_error(y_test_scaled, predicted)
    print(f"Mean Squared Error: {mse}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_scaled, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.legend()
    plt.title('LSTM Predictions with Trend Input')
    plt.show()

if __name__ == '__main__':
    main()
