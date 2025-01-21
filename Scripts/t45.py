import pandas as pd

# Load your dataset
data = pd.read_csv("merged_data_py.csv")

# Step 1: Calculate State Shares for 2022
data_2022 = data[data['Year'] == 2022]
total_energy_2022 = data_2022['Energy_Consumption'].sum()
print(total_energy_2022)

# Calculate state-level shares of energy consumption
data_2022 = data_2022.copy()  # Avoid SettingWithCopyWarning
data_2022['state_share'] = data_2022['Energy_Consumption'] / total_energy_2022
print(data_2022)

# Step 2: Convert T45 values from TWh to TJ
t45_electricity = {2030: 15 * 3600, 2037: 75 * 3600, 2045: 289 * 3600}
t45_hydrogen = {2030: 26 * 3600, 2037: 191 * 3600, 2045: 437 * 3600}
t45_red_eff = {2030: 15 * 3600, 2037: 81 * 3600, 2045: 315 * 3600}

print(t45_electricity, 
          t45_hydrogen, 
          t45_red_eff
          )
# Step 3: Create new rows for each T45 year across all states
years_to_add = [2030, 2037, 2045]
print(years_to_add)
new_rows = []

for year in years_to_add:
    for _, row in data_2022.iterrows():
        new_rows.append({
            'State': row['State'],
            'Year': year,
            'scenario': 't45_electricity',
            'energy_tj': t45_electricity[year] * row['state_share']
        })
        new_rows.append({
            'State': row['State'],
            'Year': year,
            'scenario': 't45_hydrogen',
            'energy_tj': t45_hydrogen[year] * row['state_share']
        })
        new_rows.append({
            'State': row['State'],
            'Year': year,
            'scenario': 't45_red_eff',
            'energy_tj': t45_red_eff[year] * row['state_share']
        })

# Convert to DataFrame
new_data = pd.DataFrame(new_rows)
print(new_data)

# Step 4: Pivot the data to have a cleaner format (each scenario in its own column)
t45_pivot = new_data.pivot_table(index=['State', 'Year'], columns='scenario', values='energy_tj').reset_index()
print(t45_pivot)

# Step 5: Extend the main dataset to include the missing years
states = data['State'].unique()
years = sorted(data['Year'].unique().tolist() + years_to_add)

# Create a full grid of states and years
state_year_grid = pd.DataFrame([(State, year) for State in states for year in years], columns=['State', 'Year'])
print(state_year_grid)
state_year_grid.to_csv("extended.csv")
# Merge the state-year grid with the main dataset
data_extended = pd.merge(state_year_grid, data, on=['State', 'Year'], how='left')
print(data_extended)

# Step 6: Merge the extended dataset with T45 values
final_data = pd.merge(data_extended, t45_pivot, on=['State', 'Year'], how='left')

# Check for missing values
print(final_data.isnull().sum())

# Save the updated dataset
final_data.to_csv("merged_data_with_t45.csv", index=False)
