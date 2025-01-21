# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'merged_data_py.csv'  
data = pd.read_csv(file_path)

# Display the first few rows to understand the structure
print("Dataset preview:")
print(data.head())

# Check for column names, data types, and missing values
print("\nDataset info:")
print(data.info())
print("\nMissing values:")
print(data.isnull().sum())

# Convert Year to integer 
data['Year'] = data['Year'].astype(int)

# Check for duplicate rows and drop them if necessary
print(f"\nNumber of duplicate rows: {data.duplicated().sum()}")
data = data.drop_duplicates()

# Handle missing values (e.g., fill, drop, or interpolate)
# Fill missing CO2 balances and energy consumption using linear interpolation
data['CO2_Balances'] = data['CO2_Balances'].interpolate()
data['Energy_Consumption'] = data['Energy_Consumption'].interpolate()

# Verify if all missing values are handled
print("\nMissing values after cleaning:")
print(data.isnull().sum())

#====== START OF EDA ==============
# Time series plots for CO2 balances and energy consumption for each state
states = data['State'].unique()
plt.figure(figsize=(15, 8))
for state in states:
    state_data = data[data['State'] == state]
    plt.plot(state_data['Year'], state_data['CO2_Balances'], label=f"{state} - CO2")
plt.title('CO2 Balances by State')
plt.xlabel('Year')
plt.ylabel('CO2 Balances')
plt.legend()
plt.show()

# Aggregate trends across all states
agg_data = data.groupby('Year')[['CO2_Balances', 'Energy_Consumption']].sum().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(agg_data['Year'], agg_data['CO2_Balances'], label='Total CO2 Balances', color='blue')
plt.plot(agg_data['Year'], agg_data['Energy_Consumption'], label='Total Energy Consumption', color='red')
plt.title('Aggregate CO2 Balances and Energy Consumption Trends')
plt.xlabel('Year')
plt.ylabel('Values')
plt.legend()
plt.show()

# Compute descriptive statistics
print("\nDescriptive statistics for CO2 Balances:")
print(data['CO2_Balances'].describe())
print("\nDescriptive statistics for Energy Consumption:")
print(data['Energy_Consumption'].describe())

# Save the cleaned dataset
cleaned_file_path = 'cleaned_data.csv'
data.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned dataset saved to {cleaned_file_path}")

