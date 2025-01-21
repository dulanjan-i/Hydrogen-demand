import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
data = pd.read_csv("merged_data_with_t45.csv")

# Normalize numeric features
scaler_energy = MinMaxScaler()
scaler_co2 = MinMaxScaler()
data['Energy_Consumption'] = scaler_energy.fit_transform(data[['Energy_Consumption']])
data['CO2_Balances'] = scaler_co2.fit_transform(data[['CO2_Balances']])
data[['t45_electricity', 't45_hydrogen', 't45_red_eff']] = MinMaxScaler().fit_transform(
    data[['t45_electricity', 't45_hydrogen', 't45_red_eff']]
)

# One-hot encode 'State'
encoder = OneHotEncoder(sparse_output=False)
state_encoded = encoder.fit_transform(data[['State']])
state_feature_names = encoder.get_feature_names_out(['State'])
state_encoded_df = pd.DataFrame(state_encoded, columns=state_feature_names)
data = pd.concat([data.reset_index(drop=True), state_encoded_df], axis=1)

# Define features and target
features = ['Year', 'Energy_Consumption', 't45_electricity', 't45_hydrogen', 't45_red_eff'] + list(state_feature_names)
target_energy = 'Energy_Consumption'
target_co2 = 'CO2_Balances'

# Split data by time periods
train_data = data[data['Year'] <= 2015]  # Training data: 1990–2015
val_data = data[(data['Year'] > 2015) & (data['Year'] <= 2022)]  # Validation data: 2016–2022
test_data = data[data['Year'].isin([2030, 2037, 2045])].copy()  # Test data: Future years with T45 scenarios

# Prepare features and target for training and validation
X_train, y_train_energy = train_data[features], train_data[target_energy]
X_val, y_val_energy = val_data[features], val_data[target_energy]
X_train_co2, y_train_co2 = train_data[features], train_data[target_co2]
X_val_co2, y_val_co2 = val_data[features], val_data[target_co2]

# Reshape data for LSTM
def reshape_lstm_data(X, y=None):
    X_reshaped = X.values.reshape(X.shape[0], 1, X.shape[1])  # Reshape to (samples, timesteps, features)
    if y is not None:
        y_reshaped = y.values
        return X_reshaped, y_reshaped
    return X_reshaped

X_train, y_train_energy = reshape_lstm_data(X_train, y_train_energy)
X_val, y_val_energy = reshape_lstm_data(X_val, y_val_energy)
X_train_co2, y_train_co2 = reshape_lstm_data(X_train_co2, y_train_co2)
X_val_co2, y_val_co2 = reshape_lstm_data(X_val_co2, y_val_co2)

# Build LSTM models for energy and CO2
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=100, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Single output for regression
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Energy Model
energy_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
energy_model.fit(X_train, y_train_energy, epochs=50, batch_size=32, 
                 validation_data=(X_val, y_val_energy), callbacks=[early_stopping], verbose=1)

# CO2 Model
co2_model = build_lstm_model((X_train_co2.shape[1], X_train_co2.shape[2]))
co2_model.fit(X_train_co2, y_train_co2, epochs=50, batch_size=32, 
              validation_data=(X_val_co2, y_val_co2), callbacks=[early_stopping], verbose=1)

# Prepare test data for predictions
X_test = reshape_lstm_data(test_data[features])

# Predict for each scenario
for scenario in ['t45_electricity', 't45_hydrogen', 't45_red_eff']:
    # Predict energy consumption for the scenario
    energy_preds = energy_model.predict(X_test)

    # Replace actual energy consumption with predicted values in test data
    test_data[f'Predicted_Energy_Consumption_{scenario}'] = energy_preds.flatten()
    
    # Construct new inputs for CO2 prediction by replacing Energy_Consumption with predictions
    co2_input_features = test_data[features].copy()
    co2_input_features['Energy_Consumption'] = test_data[f'Predicted_Energy_Consumption_{scenario}']
    co2_input_features = co2_input_features[['Year', 'Energy_Consumption', 't45_electricity', 't45_hydrogen', 't45_red_eff'] + list(state_feature_names)]
    co2_input_reshaped = reshape_lstm_data(co2_input_features)

    # Predict CO2 balances for the scenario
    co2_preds = co2_model.predict(co2_input_reshaped)
    test_data[f'Predicted_CO2_Balances_{scenario}'] = co2_preds.flatten()

# Inverse transform predictions
for scenario in ['t45_electricity', 't45_hydrogen', 't45_red_eff']:
    # Energy inverse transform
    pred_energy = test_data[f'Predicted_Energy_Consumption_{scenario}'].values.reshape(-1, 1)
    test_data[f'Predicted_Energy_Consumption_{scenario}'] = scaler_energy.inverse_transform(pred_energy).flatten()

    # CO2 inverse transform
    pred_co2 = test_data[f'Predicted_CO2_Balances_{scenario}'].values.reshape(-1, 1)
    test_data[f'Predicted_CO2_Balances_{scenario}'] = scaler_co2.inverse_transform(pred_co2).flatten()

# Save predictions
output_cols = ['State', 'Year'] + [
    f'Predicted_Energy_Consumption_{scenario}' for scenario in ['t45_electricity', 't45_hydrogen', 't45_red_eff']
] + [
    f'Predicted_CO2_Balances_{scenario}' for scenario in ['t45_electricity', 't45_hydrogen', 't45_red_eff']
]
test_data[output_cols].to_csv("lstm_t45_predictions.csv", index=False)

# Print sample of results
print("\nSample Results:")
print(test_data[['State', 'Year', 'Predicted_Energy_Consumption_t45_electricity', 'Predicted_CO2_Balances_t45_electricity']].head())
