#!/usr/bin/env python3
# -*- coding: utf-8 -*-


###### CALCULATING THE KEY METRICS ########
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('merged_data_py.csv')

# Ensure data is sorted by state and year
df = df.sort_values(by=["State", "Year"])

# Check the first few rows
print(df.head())

### 1. Average Annual CO2 Reduction Rates ###
# Calculate year-on-year differences for CO2 balances
df['CO2_Change'] = df.groupby('State')['CO2_Balances'].diff()

# Calculate annual reduction rates for each state
df['Annual_Reduction_Rate'] = df['CO2_Change'] / df.groupby('State')['CO2_Balances'].shift(1)

# Aggregate: Calculate the mean reduction rate per state and overall
statewise_reduction = df.groupby('State')['Annual_Reduction_Rate'].mean()
overall_reduction = statewise_reduction.mean()

print("Average Annual CO2 Reduction Rates by State:")
print(statewise_reduction)
print(f"Overall Average Annual Reduction Rate: {overall_reduction:.2%}")

### 2. Energy Efficiency Improvements ###
# Calculate CO2 intensity (CO2 per unit of energy consumption)
df['CO2_Intensity'] = df['CO2_Balances'] / df['Energy_Consumption']

# Aggregate: Mean CO2 intensity per state
statewise_efficiency = df.groupby('State')['CO2_Intensity'].mean()

print("Energy Efficiency Improvements by State (Mean CO2 Intensity):")
print(statewise_efficiency)

### 3. Variability of Emissions Across States ###
# Calculate standard deviation of CO2 balances for each year
variability = df.groupby('Year')['CO2_Balances'].std()

print("Yearly Variability of CO2 Emissions Across States:")
print(variability)

# Optional: Plot variability over time
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
variability.plot(kind='line', marker='o', title="Variability of CO2 Emissions Across States")
plt.xlabel("Year")
plt.ylabel("Standard Deviation of CO2 Emissions")
plt.grid()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Save Metrics as CSV Files
# Save statewise reductions
statewise_reduction.to_csv('statewise_reduction.csv', header=["Average_Annual_Reduction_Rate"])
# Save overall efficiency
statewise_efficiency.to_csv('statewise_efficiency.csv', header=["Mean_CO2_Intensity"])
# Save yearly variability
variability.to_csv('yearly_variability.csv', header=["CO2_Emissions_STD"])

print("Metrics saved as CSV files successfully!")

# 2. Plot Statewide Metrics and Variability
fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # Create a 2x2 grid of subplots

# (a) Statewise Reductions
statewise_reduction.plot(kind='bar', ax=axes[0, 0], color='skyblue', title='Statewise Average CO2 Reduction Rates')
axes[0, 0].set_ylabel('Reduction Rate')
axes[0, 0].grid(axis='y')

# (b) Overall Annual Reduction Rates
df.groupby('Year')['Annual_Reduction_Rate'].mean().plot(ax=axes[0, 1], color='green', marker='o', title='Overall Annual CO2 Reduction Rates')
axes[0, 1].set_ylabel('Reduction Rate')
axes[0, 1].set_xlabel('Year')
axes[0, 1].grid()

# (b) Overall Annual Reduction Rates - Time Series Scatterplot
mean_annual_reduction = df.groupby('Year')['Annual_Reduction_Rate'].mean()

# Plot the time series scatterplot
mean_annual_reduction.plot(
    ax=axes[0, 1],
    kind='line',
    color='green',
    marker='o',
    title='Overall Annual CO2 Reduction Rates'
)
axes[0, 1].set_ylabel('Reduction Rate')
axes[0, 1].set_xlabel('Year')
axes[0, 1].grid()



# (c) Statewise Efficiency (CO2 Intensity)
statewise_efficiency.plot(kind='bar', ax=axes[1, 0], color='orange', title='Statewise Energy Efficiency (Mean CO2 Intensity)')
axes[1, 0].set_ylabel('CO2 Intensity')
axes[1, 0].grid(axis='y')

# (d) Variability of CO2 Emissions
variability.plot(ax=axes[1, 1], color='purple', marker='o', title='Yearly Variability of CO2 Emissions')
axes[1, 1].set_ylabel('Standard Deviation')
axes[1, 1].set_xlabel('Year')
axes[1, 1].grid()

# Adjust layout for better readability
plt.tight_layout()

# Save the combined plot as an image
plt.savefig('metrics_visualization.png', dpi=300)
print("Plots saved as 'metrics_visualization.png'!")

# Show the plot
plt.show()

