import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS, RandomEffects, compare
from scipy.stats import chi2
from statsmodels.tools.tools import add_constant

# Load the data
df = pd.read_csv('merged_data_py.csv', parse_dates=['Year'])

# Set a multi-index for panel data (State and Year)
df = df.set_index(['State', 'Year'])

# Ensure all variables are float/int and handle missing values
df = df.dropna()

# Define the dependent and independent variables
dependent = df['CO2_Balances']
independent = df[['Energy_Consumption']]

# Add a constant (if needed)
# independent = add_constant(independent)

# Fixed Effects Model
fe_model = PanelOLS(dependent, independent, entity_effects=True)
fe_results = fe_model.fit()
print(fe_results)

# Random Effects Model
re_model = RandomEffects(dependent, independent)
re_results = re_model.fit()
print(re_results)

# SAVING MODEL RESULTS
# Define file paths for saving the summaries
fixed_effects_file = "fixed_effects_summary.txt"
random_effects_file = "random_effects_summary.txt"

# Save the summaries
with open(fixed_effects_file, "w") as fe_file:
    fe_file.write(str(fe_results.summary))

with open(random_effects_file, "w") as re_file:
    re_file.write(str(re_results.summary))

print("Summaries saved successfully!")

# HAUSMAN TEST TO SEE WHICH MODEL BEST DESCRIBES THE EFFECTS
# Extracting model parameters
beta_fe = fe_results.params  # Fixed effects coefficients
beta_re = re_results.params  # Random effects coefficients

# Extracting covariance matrices
cov_fe = fe_results.cov  # Fixed effects covariance matrix
cov_re = re_results.cov  # Random effects covariance matrix

# Difference in coefficients
diff = beta_fe - beta_re

# Variance of the difference
diff_var = cov_fe - cov_re

# Hausman statistic
hausman_stat = float(diff.T @ np.linalg.inv(diff_var) @ diff)
p_value = 1 - chi2.cdf(hausman_stat, len(diff))

# Output the results
print(f"Hausman Test Statistic: {hausman_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Compare Fixed Effects and Random Effects using linearmodels.compare
comparison = compare({"Fixed Effects": fe_results, "Random Effects": re_results})
print(comparison)
