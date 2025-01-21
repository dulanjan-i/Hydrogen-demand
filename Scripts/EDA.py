import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm


data = pd.read_csv("merged_data_py.csv")

# VISUALIZATIONS 

# Aggregate data to the federal level
federal_data = data.groupby('Year').agg({
    'Energy_Consumption': 'sum',
    'CO2_Balances': 'sum'
}).reset_index()

# 1. Annual trend of energy consumption (Federal level)
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Energy_Consumption', data=federal_data, marker='o', label='Energy Consumption')
plt.title('Annual Trend of Energy Consumption (Federal Level)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Energy Consumption', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()

# 2. Annual trend of CO2 balance (Federal level)
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='CO2_Balances', data=federal_data, marker='o', color='orange', label='CO2 Balances')
plt.title('Annual Trend of CO2 Balances (Federal Level)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('CO2 Balances', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()

# Linear regression with added constant
x = federal_data['Energy_Consumption']
y = federal_data['CO2_Balances']

x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())

# 3. Relationship between CO2 balances and energy consumption
plt.figure(figsize=(10, 6))
sns.regplot(x='Energy_Consumption', y='CO2_Balances', data=federal_data, line_kws={"color": "red"})
plt.title('Relationship between CO2 Balances and Energy Consumption', fontsize=14)
plt.xlabel('Energy Consumption', fontsize=12)
plt.ylabel('CO2 Balances', fontsize=12)
plt.grid(True)
plt.show()

# Correlation coefficient
correlation = federal_data[['Energy_Consumption', 'CO2_Balances']].corr().iloc[0, 1]
correlation

# Calculate CO2 emitted per unit of energy consumption (Efficiency metric)
data['CO2_per_Energy'] = data['CO2_Balances'] / data['Energy_Consumption']

# Aggregate efficiency at federal level
federal_efficiency = data.groupby('Year').agg({
    'CO2_Balances': 'sum',
    'Energy_Consumption': 'sum'
}).reset_index()
federal_efficiency['CO2_per_Energy'] = (
    federal_efficiency['CO2_Balances'] / federal_efficiency['Energy_Consumption']
)

CO2_per_Energy = federal_efficiency['CO2_per_Energy'] = (
    federal_efficiency['CO2_Balances'] / federal_efficiency['Energy_Consumption'])

# Plot federal-level efficiency over time
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='CO2_per_Energy', data=federal_efficiency, marker='o', color='green')
plt.title('Federal-Level Efficiency: CO2 Balance per Unit of Energy Consumption', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('CO2 per Energy Unit', fontsize=12)
plt.grid(True)
plt.show()

# Calculate state-level average efficiency across all years
state_efficiency = data.groupby('State').agg({
    'CO2_Balances': 'sum',
    'Energy_Consumption': 'sum'
}).reset_index()
state_efficiency['CO2_per_Energy'] = (
    state_efficiency['CO2_Balances'] / state_efficiency['Energy_Consumption']
)
colors = sns.color_palette("hsv", len(state_efficiency))

# Visualize state-level efficiency
plt.figure(figsize=(12, 6))
sns.barplot(x='CO2_per_Energy', y='State', data=state_efficiency, palette='viridis')
plt.title('State-Level Efficiency: CO2 Balance per Unit of Energy Consumption', fontsize=14)
plt.xlabel('CO2 per Energy Unit', fontsize=12)
plt.ylabel('State', fontsize=12)
plt.grid(axis='x')
plt.show()

# Aggregate data by state for pie charts
state_data = data.groupby('State').agg({
    'Energy_Consumption': 'sum',
    'CO2_Balances': 'sum'
}).reset_index()

# Define a set of distinct colors for better contrast
distinct_colors1 = sns.color_palette("tab20", len(state_data))
distinct_colors2 = sns.color_palette("tab20c", len(state_data))

# Pie chart for CO2 Balances 
plt.figure(figsize=(10, 8))
plt.pie(
    state_data['CO2_Balances'],
    labels=None,
    autopct='%1.1f%%',
    startangle=140,
    colors=distinct_colors1
)
plt.legend(state_data['State'], loc='upper left', bbox_to_anchor=(-0.2, 1), fontsize=10, title='States')
plt.title('CO2 Balances by State (Aggregate)', fontsize=14)
plt.tight_layout()
plt.show()

# Pie chart for Energy Consumption 
plt.figure(figsize=(10, 8))
plt.pie(
    state_data['Energy_Consumption'],
    labels=None,
    autopct='%1.1f%%',
    startangle=140,
    colors=distinct_colors2
)
plt.legend(state_data['State'], loc='upper left', bbox_to_anchor=(-0.2, 1), fontsize=10, title='States')
plt.title('Energy Consumption by State (Aggregate)', fontsize=14)
plt.tight_layout()
plt.show()
