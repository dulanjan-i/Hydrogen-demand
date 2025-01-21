import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('final_t45_results_4.csv')

# Aggregate data for Germany-wide view
aggregated_data = data.groupby('Year').sum().reset_index()

# Variables to plot
variables = [
    "Forecasted_Energy_Consumption",
    "CO2_Predicted_from_BAU",
    "Remaining_Energy_Consumption_t45_electricity",
    "Remaining_Energy_Consumption_t45_hydrogen",
    "Remaining_Energy_Consumption_t45_red_eff",
    "Remaining_CO2_Balance_t45_electricity",
    "Remaining_CO2_Balance_t45_hydrogen",
    "Remaining_CO2_Balance_t45_red_eff"
]

# Function to plot individual variables
for variable in variables:
    plt.figure(figsize=(10, 6))
    plt.plot(aggregated_data['Year'], aggregated_data[variable], marker='o')
    plt.title(f'{variable} - Germany-wide Aggregated')
    plt.xlabel('Year')
    plt.ylabel(variable)
    plt.grid()
    plt.savefig(f'{variable}_Germany_wide.png')
    plt.show()

# Define the variables for each category
energy_consumption_vars = [
    'Forecasted_Energy_Consumption',
    'Remaining_Energy_Consumption_t45_electricity',
    'Remaining_Energy_Consumption_t45_hydrogen',
    'Remaining_Energy_Consumption_t45_red_eff'
]

co2_balance_vars = [
    'CO2_Predicted_from_BAU',
    'Remaining_CO2_Balance_t45_electricity',
    'Remaining_CO2_Balance_t45_hydrogen',
    'Remaining_CO2_Balance_t45_red_eff'
]

# Generate the first plot for energy consumption
plt.figure(figsize=(12, 8))
for variable in energy_consumption_vars:
    plt.plot(aggregated_data['Year'], aggregated_data[variable], marker='o', label=variable)
plt.title('Germany-wide Aggregated Energy Consumption Variables')
plt.xlabel('Year')
plt.ylabel('Energy Consumption (Units)')
plt.legend()
plt.grid()
plt.savefig('Germany_wide_Energy_Consumption.png')
plt.show()

# Generate the second plot for CO2 balances
plt.figure(figsize=(12, 8))
for variable in co2_balance_vars:
    plt.plot(aggregated_data['Year'], aggregated_data[variable], marker='o', label=variable)
plt.title('Germany-wide Aggregated CO2 Balances Variables')
plt.xlabel('Year')
plt.ylabel('CO2 Balances (Units)')
plt.legend()
plt.grid()
plt.savefig('Germany_wide_CO2_Balances.png')
plt.show()

# Plot 12 small graphs in one image (4x3 grid)
fig, axes = plt.subplots(4, 3, figsize=(20, 15), sharex=True)
axes = axes.flatten()
for i, variable in enumerate(variables):
    axes[i].plot(aggregated_data['Year'], aggregated_data[variable], marker='o')
    axes[i].set_title(variable)
    axes[i].grid()
plt.tight_layout()
plt.savefig('12_Variables_Small_Graphs.png')
plt.show()

# Separate BAU and T45 energy scenarios
energy_scenarios = [
    "Forecasted_Energy_Consumption",
    "Remaining_Energy_Consumption_t45_electricity",
    "Remaining_Energy_Consumption_t45_hydrogen",
    "Remaining_Energy_Consumption_t45_red_eff",
]

for scenario in energy_scenarios:
    plt.figure(figsize=(10, 6))
    plt.plot(aggregated_data['Year'], aggregated_data[scenario], marker='o')
    plt.title(f'{scenario} - Germany-wide')
    plt.xlabel('Year')
    plt.ylabel('Energy Consumption')
    plt.grid()
    plt.savefig(f'{scenario}_Energy_Scenario.png')
    plt.show()

# Separate BAU and T45 CO₂ scenarios
co2_scenarios = [
    "CO2_Predicted_from_BAU",
    "Remaining_CO2_Balance_t45_electricity",
    "Remaining_CO2_Balance_t45_hydrogen",
    "Remaining_CO2_Balance_t45_red_eff"
]

for scenario in co2_scenarios:
    plt.figure(figsize=(10, 6))
    plt.plot(aggregated_data['Year'], aggregated_data[scenario], marker='o')
    plt.title(f'{scenario} - Germany-wide')
    plt.xlabel('Year')
    plt.ylabel('CO₂ Balances')
    plt.grid()
    plt.savefig(f'{scenario}_CO2_Scenario.png')
    plt.show()
