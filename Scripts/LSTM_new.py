import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# Load historical and scenario datasets
historical_data = pd.read_csv("merged_data_py.csv")
scenario_data = pd.read_csv("t45_predictions.csv")

# Step 1: Normalize the data
state_scalers = {}
for state in historical_data['State'].unique():
    state_data = historical_data[historical_data['State'] == state]
    scaler_energy = MinMaxScaler()
    scaler_co2 = MinMaxScaler()
    state_scalers[state] = {
        'energy': scaler_energy.fit(state_data[['Energy_Consumption']]),
        'co2': scaler_co2.fit(state_data[['CO2_Balances']]),
    }
    historical_data.loc[historical_data['State'] == state, 'Energy_Consumption'] = scaler_energy.transform(state_data[['Energy_Consumption']])
    historical_data.loc[historical_data['State'] == state, 'CO2_Balances'] = scaler_co2.transform(state_data[['CO2_Balances']])

# Normalize 'Year'
year_scaler = MinMaxScaler()
historical_data['Year'] = year_scaler.fit_transform(historical_data[['Year']])

# One-hot encode 'State'
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
state_encoded = encoder.fit_transform(historical_data[['State']])
state_feature_names = encoder.get_feature_names_out(['State'])
state_encoded_df = pd.DataFrame(state_encoded, columns=state_feature_names)
historical_data = pd.concat([historical_data.reset_index(drop=True), state_encoded_df], axis=1)

# Step 2: Define features and targets for energy and CO2 predictions
features_energy = ['Year', 'Energy_Consumption'] + list(state_feature_names)
features_co2_historical = ['Year', 'CO2_Balances'] + list(state_feature_names)
features_co2_energy = ['Year', 'Energy_Consumption'] + list(state_feature_names)
target_energy = 'Energy_Consumption'
target_co2 = 'CO2_Balances'

# Split the data
train_data = historical_data[historical_data['Year'] <= year_scaler.transform(pd.DataFrame([[2015]], columns=['Year']))[0][0]]
val_data = historical_data[historical_data['Year'] > year_scaler.transform(pd.DataFrame([[2015]], columns=['Year']))[0][0]]

X_train_energy, y_train_energy = train_data[features_energy], train_data[target_energy]
X_val_energy, y_val_energy = val_data[features_energy], val_data[target_energy]

X_train_co2_historical, y_train_co2_historical = train_data[features_co2_historical], train_data[target_co2]
X_val_co2_historical, y_val_co2_historical = val_data[features_co2_historical], val_data[target_co2]

X_train_co2_energy, y_train_co2_energy = train_data[features_co2_energy], train_data[target_co2]
X_val_co2_energy, y_val_co2_energy = val_data[features_co2_energy], val_data[target_co2]

# Reshape data for LSTM
def reshape_lstm_data(X, y=None):
    X_reshaped = X.to_numpy().reshape(X.shape[0], 1, X.shape[1])
    if y is not None:
        y_reshaped = y.to_numpy()
        return X_reshaped, y_reshaped
    return X_reshaped

X_train_energy, y_train_energy = reshape_lstm_data(X_train_energy, y_train_energy)
X_val_energy, y_val_energy = reshape_lstm_data(X_val_energy, y_val_energy)

X_train_co2_historical, y_train_co2_historical = reshape_lstm_data(X_train_co2_historical, y_train_co2_historical)
X_val_co2_historical, y_val_co2_historical = reshape_lstm_data(X_val_co2_historical, y_val_co2_historical)

X_train_co2_energy, y_train_co2_energy = reshape_lstm_data(X_train_co2_energy, y_train_co2_energy)
X_val_co2_energy, y_val_co2_energy = reshape_lstm_data(X_val_co2_energy, y_val_co2_energy)

# Step 3: Define and train LSTM models
def build_lstm_model(units=50, dropout_rate=0.2, input_shape=(1, 1)):
    model = Sequential()
    model.add(LSTM(units=units, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units // 2, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train energy model
energy_model = build_lstm_model(input_shape=(1, X_train_energy.shape[2]))
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
energy_model.fit(X_train_energy, y_train_energy, epochs=50, batch_size=32, validation_data=(X_val_energy, y_val_energy), callbacks=[early_stopping], verbose=1)

# Train CO2 model for historical predictions
historical_co2_model = build_lstm_model(input_shape=(1, X_train_co2_historical.shape[2]))
historical_co2_model.fit(
    X_train_co2_historical, y_train_co2_historical,
    epochs=50, batch_size=32,
    validation_data=(X_val_co2_historical, y_val_co2_historical),
    callbacks=[early_stopping],
    verbose=1
)

# Train CO2 model for energy-based predictions
energy_based_co2_model = build_lstm_model(input_shape=(1, X_train_co2_energy.shape[2]))
energy_based_co2_model.fit(
    X_train_co2_energy, y_train_co2_energy,
    epochs=50, batch_size=32,
    validation_data=(X_val_co2_energy, y_val_co2_energy),
    callbacks=[early_stopping],
    verbose=1
)

# Step 4: Forecast for future years (BAU)
future_years = [2030, 2037, 2045]
# Create future_df with all combinations of states and years
future_df = pd.DataFrame([
    {'State': state, 'Year': year}
    for year in future_years
    for state in historical_data['State'].unique()
])

# Scale 'Year' column
future_df['Year'] = year_scaler.transform(future_df[['Year']])

# One-hot encode state features for future_df
state_encoded = encoder.transform(future_df[['State']])
state_encoded_df = pd.DataFrame(state_encoded, columns=state_feature_names)
future_df = pd.concat([future_df.reset_index(drop=True), state_encoded_df], axis=1)

# Predict energy consumption if 'Forecasted_Energy_Consumption' is not in future_df
if 'Forecasted_Energy_Consumption' not in future_df.columns:
    X_temp = reshape_lstm_data(future_df[['Year'] + list(state_feature_names)])
    future_df['Forecasted_Energy_Consumption'] = energy_model.predict(X_temp).flatten()

# Add 'Energy_Consumption' to future_df
future_df['Energy_Consumption'] = future_df['Forecasted_Energy_Consumption']

# Ensure all necessary features are included before reshaping
missing_features = set(features_energy) - set(future_df.columns)
if missing_features:
    raise KeyError(f"Missing features for energy model: {missing_features}")

# Prepare features for energy-based prediction
X_future_energy = reshape_lstm_data(future_df[features_energy])

# Use historical CO2 model for historical predictions
X_future_co2_historical = reshape_lstm_data(future_df[['Year'] + state_feature_names])
historical_co2_forecast = historical_co2_model.predict(X_future_co2_historical).flatten()
future_df['CO2_Predicted_from_Historical'] = historical_co2_forecast

# Use energy-based CO2 model for predictions based on forecasted energy consumption
X_future_co2_energy = reshape_lstm_data(future_df[features_co2_energy])
co2_forecast_bau = energy_based_co2_model.predict(X_future_co2_energy).flatten()
future_df['CO2_Predicted_from_BAU'] = co2_forecast_bau

# Compare the two CO2 predictions
mape = mean_absolute_percentage_error(
    future_df['CO2_Predicted_from_Historical'],
    future_df['CO2_Predicted_from_BAU']
)
rmse = mean_squared_error(
    future_df['CO2_Predicted_from_Historical'],
    future_df['CO2_Predicted_from_BAU'],
    squared=False
)
r2 = r2_score(
    future_df['CO2_Predicted_from_Historical'],
    future_df['CO2_Predicted_from_BAU']
)

# Print comparison metrics
print(f"Comparison of CO2 Models:")
print(f"  - Mean Absolute Percentage Error (MAPE): {mape:.4f}")
print(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"  - R² Score: {r2:.4f}")

# Step 6: Integrate T45 scenarios
scenario_values = scenario_data.set_index(['State', 'Year']).reset_index()

# Adjust 'Year' type in both dataframes to match if needed
scenario_data['Year'] = scenario_data['Year'].astype(int)
future_df['Year'] = future_df['Year'].astype(int)

# Merge with alignment
future_df = future_df.merge(
    scenario_values,
    on=['State', 'Year'],
    how='left'
)

# Compute Remaining Energy Consumption
for scenario in ['t45_electricity', 't45_hydrogen', 't45_red_eff']:
    future_df[f'Remaining_Energy_Consumption_{scenario}'] = (
        future_df['Forecasted_Energy_Consumption'] - future_df[scenario]
    )

# Recalculate CO2 balances explicitly for remaining energy consumption
for scenario in ['t45_electricity', 't45_hydrogen', 't45_red_eff']:
    future_df[f'Remaining_CO2_Balance_{scenario}'] = energy_based_co2_model.predict(reshape_lstm_data(
        future_df[features_co2_energy].assign(Energy_Consumption=future_df[f'Remaining_Energy_Consumption_{scenario}'])
    )).flatten()

# Inverse transform results
for state in historical_data['State'].unique():
    idx = future_df['State'] == state
    future_df.loc[idx, 'Forecasted_Energy_Consumption'] = state_scalers[state]['energy'].inverse_transform(
        future_df.loc[idx, ['Forecasted_Energy_Consumption']].values
    )
    future_df.loc[idx, 'CO2_Predicted_from_BAU'] = state_scalers[state]['co2'].inverse_transform(
        future_df.loc[idx, ['CO2_Predicted_from_BAU']].values
    )
    future_df.loc[idx, 'CO2_Predicted_from_Historical'] = state_scalers[state]['co2'].inverse_transform(
        future_df.loc[idx, ['CO2_Predicted_from_Historical']].values
    )
    for scenario in ['t45_electricity', 't45_hydrogen', 't45_red_eff']:
        future_df.loc[idx, f'Remaining_Energy_Consumption_{scenario}'] = state_scalers[state]['energy'].inverse_transform(
            future_df.loc[idx, [f'Remaining_Energy_Consumption_{scenario}']].values
        )
        future_df.loc[idx, f'Remaining_CO2_Balance_{scenario}'] = state_scalers[state]['co2'].inverse_transform(
            future_df.loc[idx, [f'Remaining_CO2_Balance_{scenario}']].values
        )

# Inverse transform Year
future_df['Year'] = year_scaler.inverse_transform(future_df[['Year']])

# Save results
output_cols = ['State', 'Year', 'Forecasted_Energy_Consumption', 'CO2_Predicted_from_Historical', 'CO2_Predicted_from_BAU'] + \
    [f'Remaining_Energy_Consumption_{scenario}' for scenario in ['t45_electricity', 't45_hydrogen', 't45_red_eff']] + \
    [f'Remaining_CO2_Balance_{scenario}' for scenario in ['t45_electricity', 't45_hydrogen', 't45_red_eff']]

future_df[output_cols].to_csv('forecasted_t45_results.csv', index=False)
print("Predictions saved to 'forecasted_t45_results.csv'.")
