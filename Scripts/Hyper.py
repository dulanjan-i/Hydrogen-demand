import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# Step 1: Data Preparation
# Load historical and scenario datasets
historical_data = pd.read_csv("merged_data_py.csv")
scenario_data = pd.read_csv("t45_predictions.csv")

# Normalize numerical data (Energy Consumption and CO2 Balances) by state
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

# Step 2: Define Features and Targets for Models
features_energy = ['Year'] + list(state_feature_names)
features_co2_historical = ['Year'] + list(state_feature_names)
features_co2_energy = ['Year', 'Energy_Consumption'] + list(state_feature_names)

target_energy = 'Energy_Consumption'
target_co2 = 'CO2_Balances'

# Split data into training (1990-2015) and validation (2015-2022)
train_data = historical_data[historical_data['Year'] <= year_scaler.transform(pd.DataFrame([[2015]], columns=['Year']))[0][0]]
val_data = historical_data[historical_data['Year'] > year_scaler.transform(pd.DataFrame([[2015]], columns=['Year']))[0][0]]

X_train_energy, y_train_energy = train_data[features_energy], train_data[target_energy]
X_val_energy, y_val_energy = val_data[features_energy], val_data[target_energy]

X_train_co2_historical, y_train_co2_historical = train_data[features_co2_historical], train_data[target_co2]
X_val_co2_historical, y_val_co2_historical = val_data[features_co2_historical], val_data[target_co2]

X_train_co2_energy, y_train_co2_energy = train_data[features_co2_energy], train_data[target_co2]
X_val_co2_energy, y_val_co2_energy = val_data[features_co2_energy], val_data[target_co2]

# Step 3: LSTM Model Definitions
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

# Build LSTM model
def build_lstm_model(units=8, dropout_rate=0.3, input_shape=(1, 1)):
    model = Sequential()
    model.add(LSTM(units=units, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units // 1, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train energy model
energy_model = build_lstm_model(input_shape=(1, X_train_energy.shape[2]))
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
energy_model.fit(X_train_energy, y_train_energy, epochs=5, batch_size=4, validation_data=(X_val_energy, y_val_energy), callbacks=[early_stopping], verbose=1)

# Train CO2 models
historical_co2_model = build_lstm_model(input_shape=(1, X_train_co2_historical.shape[2]))
historical_co2_model.fit(
    X_train_co2_historical, y_train_co2_historical,
    epochs=10, batch_size=32,
    validation_data=(X_val_co2_historical, y_val_co2_historical),
    callbacks=[early_stopping], verbose=1
)

energy_based_co2_model = build_lstm_model(input_shape=(1, X_train_co2_energy.shape[2]))
energy_based_co2_model.fit(
    X_train_co2_energy, y_train_co2_energy,
    epochs=8, batch_size=8,
    validation_data=(X_val_co2_energy, y_val_co2_energy),
    callbacks=[early_stopping], verbose=1
)

# Step 4: Forecast Energy Consumption
future_years = [2030, 2037, 2045]
future_df = pd.DataFrame([
    {'State': state, 'Year': year}
    for year in future_years
    for state in historical_data['State'].unique()
])

# Normalize and encode future_df
future_df['Year'] = year_scaler.transform(future_df[['Year']])
state_encoded = encoder.transform(future_df[['State']])
state_encoded_df = pd.DataFrame(state_encoded, columns=state_feature_names)
future_df = pd.concat([future_df.reset_index(drop=True), state_encoded_df], axis=1)

# Predict energy consumption
X_future_energy = reshape_lstm_data(future_df[features_energy])
future_df['Forecasted_Energy_Consumption'] = energy_model.predict(X_future_energy).flatten()

# Step 5: Predict CO2 Balances
# Historical CO2 predictions
X_future_co2_historical = reshape_lstm_data(future_df[['Year'] + list(state_feature_names)])
future_df['CO2_Predicted_from_Historical'] = historical_co2_model.predict(X_future_co2_historical).flatten()

# Energy-based CO2 predictions
future_df['Energy_Consumption'] = future_df['Forecasted_Energy_Consumption']
X_future_co2_energy = reshape_lstm_data(future_df[features_co2_energy])
future_df['CO2_Predicted_from_BAU'] = energy_based_co2_model.predict(X_future_co2_energy).flatten()

#Inverse transform forecasted values to original scale
for state in historical_data['State'].unique():
    idx = future_df['State'] == state
    # Inverse transform energy consumption and CO₂ balances
    future_df.loc[idx, 'Forecasted_Energy_Consumption'] = state_scalers[state]['energy'].inverse_transform(
        future_df.loc[idx, ['Forecasted_Energy_Consumption']].values
    )
    future_df.loc[idx, 'CO2_Predicted_from_BAU'] = state_scalers[state]['co2'].inverse_transform(
        future_df.loc[idx, ['CO2_Predicted_from_BAU']].values
    )
    future_df.loc[idx, 'CO2_Predicted_from_Historical'] = state_scalers[state]['co2'].inverse_transform(
        future_df.loc[idx, ['CO2_Predicted_from_Historical']].values
    )

# Inverse transform Year
future_df['Year'] = year_scaler.inverse_transform(future_df[['Year']])

# Export forecasted BAU data
forecasted_bau = future_df[['State', 'Year', 'Forecasted_Energy_Consumption', 'CO2_Predicted_from_Historical', 'CO2_Predicted_from_BAU']]
forecasted_bau.to_csv('forecasted_BAU_5.csv', index=False)
print("Exported forecasted BAU data to 'forecasted_BAU_5.csv'")

###############################################################################

# Re-import forecasted BAU data and t45 predictions
forecasted_bau = pd.read_csv('forecasted_BAU_5.csv')
t45_predictions = pd.read_csv('t45_predictions.csv')

# Merge the two DataFrames
merged_df = forecasted_bau.merge(
    t45_predictions,
    on=['State', 'Year'],
    how='left'
)
print("Merged forecasted BAU with t45 predictions.")


# Perform subtraction with real-scale t45 values
merged_df['Remaining_Energy_Consumption_t45_electricity'] = (
    merged_df['Forecasted_Energy_Consumption'] - merged_df['t45_electricity']
).clip(lower=0)

merged_df['Remaining_Energy_Consumption_t45_hydrogen'] = (
    merged_df['Forecasted_Energy_Consumption'] - merged_df['t45_hydrogen']
).clip(lower=0)

merged_df['Remaining_Energy_Consumption_t45_red_eff'] = (
    merged_df['Forecasted_Energy_Consumption'] - merged_df['t45_red_eff']
).clip(lower=0)

# Save intermediate results for debugging
merged_df.to_csv('debug_t45_energy_subtraction.csv', index=False)

# Step 7: Normalize Remaining Energy Consumption Columns in merged_df
# Transform the remaining energy consumption columns back to scaled values
# Step 3: Transform Remaining Energy Consumption columns back to scaled values

# Transform 'Remaining_Energy_Consumption_t45_electricity'
for state in historical_data['State'].unique():
    idx = merged_df['State'] == state
    merged_df.rename(columns={'Remaining_Energy_Consumption_t45_electricity': 'Energy_Consumption'}, inplace=True)
    merged_df.loc[idx, 'Energy_Consumption'] = state_scalers[state]['energy'].transform(
        merged_df.loc[idx, ['Energy_Consumption']]
    )
    merged_df.rename(columns={'Energy_Consumption': 'Remaining_Energy_Consumption_t45_electricity'}, inplace=True)

# Transform 'Remaining_Energy_Consumption_t45_hydrogen'
for state in historical_data['State'].unique():
    idx = merged_df['State'] == state
    merged_df.rename(columns={'Remaining_Energy_Consumption_t45_hydrogen': 'Energy_Consumption'}, inplace=True)
    merged_df.loc[idx, 'Energy_Consumption'] = state_scalers[state]['energy'].transform(
        merged_df.loc[idx, ['Energy_Consumption']]
    )
    merged_df.rename(columns={'Energy_Consumption': 'Remaining_Energy_Consumption_t45_hydrogen'}, inplace=True)

# Transform 'Remaining_Energy_Consumption_t45_red_eff'
for state in historical_data['State'].unique():
    idx = merged_df['State'] == state
    merged_df.rename(columns={'Remaining_Energy_Consumption_t45_red_eff': 'Energy_Consumption'}, inplace=True)
    merged_df.loc[idx, 'Energy_Consumption'] = state_scalers[state]['energy'].transform(
        merged_df.loc[idx, ['Energy_Consumption']]
    )
    merged_df.rename(columns={'Energy_Consumption': 'Remaining_Energy_Consumption_t45_red_eff'}, inplace=True)

# Scale 'Year' column in merged_df
merged_df['Year'] = year_scaler.transform(merged_df[['Year']])

# Scale one-hot encoded 'State' columns in merged_df
state_encoded = encoder.transform(merged_df[['State']])
state_encoded_df = pd.DataFrame(state_encoded, columns=state_feature_names)
merged_df = pd.concat([merged_df.reset_index(drop=True), state_encoded_df], axis=1)


# Step 8: Predict CO2 Balances for Scenarios from merged_df
# Predict CO2 balances for each scenario using scaled inputs

# Predict CO2 Balances for t45_electricity
merged_df.rename(columns={'Remaining_Energy_Consumption_t45_electricity': 'Energy_Consumption'}, inplace=True)
X_electricity_co2 = merged_df[['Year'] + list(state_feature_names) + ['Energy_Consumption']]
merged_df['Remaining_CO2_Balance_t45_electricity'] = energy_based_co2_model.predict(
    X_electricity_co2.values.reshape(-1, 1, X_electricity_co2.shape[1])
).flatten()
merged_df.rename(columns={'Energy_Consumption': 'Remaining_Energy_Consumption_t45_electricity'}, inplace=True)

# Predict CO2 Balances for t45_hydrogen
merged_df.rename(columns={'Remaining_Energy_Consumption_t45_hydrogen': 'Energy_Consumption'}, inplace=True)
X_hydrogen_co2 = merged_df[['Year'] + list(state_feature_names) + ['Energy_Consumption']]
merged_df['Remaining_CO2_Balance_t45_hydrogen'] = energy_based_co2_model.predict(
    X_hydrogen_co2.values.reshape(-1, 1, X_hydrogen_co2.shape[1])
).flatten()
merged_df.rename(columns={'Energy_Consumption': 'Remaining_Energy_Consumption_t45_hydrogen'}, inplace=True)

# Predict CO2 Balances for t45_red_eff
merged_df.rename(columns={'Remaining_Energy_Consumption_t45_red_eff': 'Energy_Consumption'}, inplace=True)
X_red_eff_co2 = merged_df[['Year'] + list(state_feature_names) + ['Energy_Consumption']]
merged_df['Remaining_CO2_Balance_t45_red_eff'] = energy_based_co2_model.predict(
    X_red_eff_co2.values.reshape(-1, 1, X_red_eff_co2.shape[1])
).flatten()
merged_df.rename(columns={'Energy_Consumption': 'Remaining_Energy_Consumption_t45_red_eff'}, inplace=True)

# Inverse transform Year
merged_df['Year'] = year_scaler.inverse_transform(merged_df[['Year']])

# Inverse Transform State and All Relevant Columns
# Inverse transform the one-hot encoded 'State' column
# merged_df['State'] = encoder.inverse_transform(merged_df[state_feature_names].to_numpy())
# Drop the one-hot encoded columns as they are no longer needed
merged_df.drop(columns=state_feature_names, inplace=True)
 


# Inverse transform Remaining Energy Consumption and CO2 Balance columns
for state in historical_data['State'].unique():
    idx = merged_df['State'] == state
    
    # Inverse transform remaining energy consumption columns
    for col in ['Remaining_Energy_Consumption_t45_electricity', 
                'Remaining_Energy_Consumption_t45_hydrogen', 
                'Remaining_Energy_Consumption_t45_red_eff']:
        merged_df.loc[idx, col] = state_scalers[state]['energy'].inverse_transform(
            merged_df.loc[idx, [col]]
        )

    # Inverse transform CO2 balance columns
    for col in ['Remaining_CO2_Balance_t45_electricity', 
                'Remaining_CO2_Balance_t45_hydrogen', 
                'Remaining_CO2_Balance_t45_red_eff']:
        merged_df.loc[idx, col] = state_scalers[state]['co2'].inverse_transform(
            merged_df.loc[idx, [col]]
        )



# Step 9: Save Final Results to CSV
# Save the updated merged_df with all predictions
merged_df.to_csv('final_t45_results_5.csv', index=False)

print("Predictions saved to 'final_t45_results_5.csv'.")


#==================== VALIDATION CHECKS ======================
num_features = len(['Year'] + list(state_feature_names) + ['Energy_Consumption'])  # Update with correct features
test_input = np.random.rand(32, 1, num_features)  # Replace with synthetic data
test_output = energy_based_co2_model.predict(test_input)
print(test_output)

# Evaluate Models: RMSE, R², and MAPE
def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"Evaluation for {model_name}:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.4f}")
    print("-" * 40)

# Historical CO2 Model Evaluation
historical_co2_y_true = merged_df['CO2_Predicted_from_Historical']  # Replace with true values if available
historical_co2_y_pred = merged_df['Remaining_CO2_Balance_t45_electricity']  # Adjust as needed
evaluate_model(historical_co2_y_true, historical_co2_y_pred, "Historical CO2 Model")

# Energy-Based CO2 Model Evaluation
energy_based_co2_y_true = merged_df['CO2_Predicted_from_BAU']  # Replace with true values if available
energy_based_co2_y_pred = merged_df['Remaining_CO2_Balance_t45_hydrogen']  # Adjust as needed
evaluate_model(energy_based_co2_y_true, energy_based_co2_y_pred, "Energy-Based CO2 Model")

# T45 Scenario Evaluation (Example: t45_electricity)
t45_scenario_y_true = merged_df['CO2_Predicted_from_Historical']  # Replace with true values if available
t45_scenario_y_pred = merged_df['Remaining_CO2_Balance_t45_red_eff']  # Adjust as needed
evaluate_model(t45_scenario_y_true, t45_scenario_y_pred, "T45 Red_Eff CO2 Model")


print(merged_df[['Remaining_Energy_Consumption_t45_electricity', 
                 'Remaining_Energy_Consumption_t45_hydrogen', 
                 'Remaining_Energy_Consumption_t45_red_eff']].describe())
print(merged_df[['Remaining_CO2_Balance_t45_electricity', 
                 'Remaining_CO2_Balance_t45_hydrogen', 
                 'Remaining_CO2_Balance_t45_red_eff']].describe())

import matplotlib.pyplot as plt

for scenario in ['t45_electricity', 't45_hydrogen', 't45_red_eff']:
    plt.figure()
    plt.scatter(merged_df[f'Remaining_Energy_Consumption_{scenario}'], 
                merged_df[f'Remaining_CO2_Balance_{scenario}'], label=scenario)
    plt.xlabel('Remaining Energy Consumption')
    plt.ylabel('Remaining CO2 Balance')
    plt.title(f'Relationship for {scenario}')
    plt.legend()
    plt.show()









