library(ggplot2)
library(tidyverse)
library(dplyr)
library(readr)

#=========================VISUALIZATIONS=================================
#Visualizing Time Series for Each State
final <- read.csv("final_t45_results_0.csv")

# Plot CO2 balances over time
ggplot(final, aes(x = Year, y = CO2_Predicted_from_Historical, color = State)) +
  geom_line() +
  labs(title = "CO2 Balances Over Time by State (2030-2045)",
       x = "Year", y = "CO2 Emissions (1000T CO2)") +
  theme_minimal()

# Plot energy consumption over time
ggplot(final, aes(x = Year, y = Forecasted_Energy_Consumption, color = State)) +
  geom_line() +
  labs(title = "Industrial Energy Consumption Over Time by State",
       x = "Year", y = "Energy Consumption (TJ)") +
  theme_minimal()


# CO2 balances by state
ggplot(final, aes(x = Year, y = Remaining_CO2_Balance_t45_hydrogen)) +
  geom_line(color = "blue") +
  facet_wrap(~ State, scales = "free_y") +
  labs(title = "CO2 Balances Over Time by State",
       x = "Year", y = "CO2 Emissions (1000T CO2)") +
  theme_minimal()

# Energy consumption by state
ggplot(clean_data, aes(x = Year, y = Energy_Consumption)) +
  geom_line(color = "green") +
  facet_wrap(~ State, scales = "free_y") +
  labs(title = "Industrial Energy Consumption Over Time by State",
       x = "Year", y = "Energy Consumption (TJ)") +
  theme_minimal()

###forecasting for the next 10 years based on the current trend - did just for fun so
### in other script named ARIMA 

# Implementing Vector Autoregression (VAR) to analyze the interactions between energy consumption and CO2 balances

data <- pdata.frame(clean_data, index = c("State", "Year")) #converting the current dataset into a panel dataset for future analysis
lapply(data[, c("CO2_Emissions", "Energy_Consumption")], adf.test)

#differencing to make the data stationary ( ADF Test Pval: 0.3> 0.05 ; meaning it is non stationary)
data$CO2_diff <- diff(data$CO2_Emissions)
data$Energy_diff <- diff(data$Energy_Consumption)

data$Year <- as.numeric(as.character(data$Year))

# Plotting time series for CO2 Balances for each state
# Folder for CO2 Balances plots
co2_folder <- "/Users/dulanjanwijenayake/Library/CloudStorage/OneDrive-HiroshimaUniversity/Energiewende-Dulanjan’s Mac/Energy/CO2_bal_state"

# Create the folder if it doesn't exist
if (!dir.exists(co2_folder)) {
  dir.create(co2_folder, recursive = TRUE)
}

# Generate and save plots for CO2 Balances
lapply(unique(data$State), function(state) {
  state_data <- subset(data, State == state)
  ts_data <- ts(state_data$CO2_Emissions, start = c(min(state_data$Year)), frequency = 1)
  
  # Save plot as PNG
  file_name <- paste0(co2_folder, "/", state, "_CO2_Balances.png")
  png(file_name, width = 800, height = 600)
  plot(ts_data, col = "green", main = paste("CO2 Balances -", state), 
       ylab = "CO2 Emissions", xlab = "Year")
  dev.off()  # Close the device
})

# Plotting time series for CO2 Balances for each state
# Folder for Energy Consumption plots
energy_folder <- "/Users/dulanjanwijenayake/Library/CloudStorage/OneDrive-HiroshimaUniversity/Energiewende-Dulanjan’s Mac/Energy/Energy_con_state"

# Create the folder if it doesn't exist
if (!dir.exists(energy_folder)) {
  dir.create(energy_folder, recursive = TRUE)
}

# Generate and save plots for Energy Consumption
lapply(unique(data$State), function(state) {
  state_data <- subset(data, State == state)
  ts_data <- ts(state_data$Energy_Consumption, start = c(min(state_data$Year)), frequency = 1)
  
  # Save plot as PNG
  file_name <- paste0(energy_folder, "/", state, "_Energy_Consumption.png")
  png(file_name, width = 800, height = 600)
  plot(ts_data, col = "blue", main = paste("Energy Consumption -", state), 
       ylab = "Energy Consumption", xlab = "Year")
  dev.off()  # Close the device
})

# Combined Plot for All States
# Reshape the data for easier plotting
data_long <- melt(data, id.vars = c("State", "Year"), 
                  measure.vars = c("CO2_Emissions", "Energy_Consumption"),
                  variable.name = "Metric", value.name = "Value")

# Plot using ggplot2
facet_plot <- ggplot(data_long, aes(x = Year, y = Value, color = Metric)) +
  geom_line(size = 0.8) +
  facet_wrap(~ State, scales = "free_y") +
  labs(title = "CO2 Balances and Energy Consumption by State", 
       x = "Year", y = "Value") +
  scale_color_manual(values = c("CO2_Emissions" = "blue", "Energy_Consumption" = "red"),
                     labels = c("CO2 Balances", "Energy Consumption")) +
  theme_minimal() +
  theme(strip.text = element_text(size = 10),
        legend.position = "bottom",
        legend.title = element_blank())

# Save the plot to a file
ggsave("facet_wrap_states.png", plot = facet_plot, width = 16, height = 12, dpi = 300)

# Display the plot
print(facet_plot)

# Summary statistics for CO2 and Energy Consumption
summary_stats <- data %>%
  group_by(State) %>%
  summarise(
    Mean_CO2 = mean(CO2_Emissions, na.rm = TRUE),
    SD_CO2 = sd(CO2_Emissions, na.rm = TRUE),
    Mean_Energy = mean(Energy_Consumption, na.rm = TRUE),
    SD_Energy = sd(Energy_Consumption, na.rm = TRUE)
  )
print(summary_stats)
write_csv(summary_stats, "summary_CO2_and_Energy.csv")


#================NEXT STEPS TOWARDS THE SCENARIO TESTING========================

#Granger causality test: to determine whether one variable can predict another
# to test interactions between CO2 balances and energy consumption


# Granger causality test for each state
# Initialize a list to store results
results_list <- lapply(split(data, data$State), function(sub_data) {
  # Ensure sub_data is sorted by Year
  sub_data <- sub_data[order(sub_data$Year), ]
  
  # Check if sub_data has enough rows for testing
  if (nrow(sub_data) > 2) {
    # Granger causality tests
    CO2_to_Energy <- grangertest(CO2_Emissions ~ Energy_Consumption, order = 2, data = sub_data)
    Energy_to_CO2 <- grangertest(Energy_Consumption ~ CO2_Emissions, order = 2, data = sub_data)
    
    # Extracting p-values and F-statistics
    return(data.frame(
      State = unique(sub_data$State),
      CO2_to_Energy_F = CO2_to_Energy$F[2],
      CO2_to_Energy_p = CO2_to_Energy$`Pr(>F)`[2],
      Energy_to_CO2_F = Energy_to_CO2$F[2],
      Energy_to_CO2_p = Energy_to_CO2$`Pr(>F)`[2]
    ))
  } else {
    # Return NA for insufficient data
    return(data.frame(
      State = unique(sub_data$State),
      CO2_to_Energy_F = NA,
      CO2_to_Energy_p = NA,
      Energy_to_CO2_F = NA,
      Energy_to_CO2_p = NA
    ))
  }
})

# Combine all results into a single data frame
granger_table <- bind_rows(results_list)

# View the combined results
print(granger_table)
write_csv(granger_table, "granger_table.csv")

pdata <- pdata.frame(data, index = c("State", "Year"))
pdata$State <- as.factor(pdata$State)
pdata$Year <- as.integer(pdata$Year)
pdata$CO2_Emissions <- as.numeric(pdata$CO2_Emissions)
pdata$Energy_Consumption <- as.numeric(pdata$Energy_Consumption)

# Fit Panel VAR Model
pvar_model <- pvarfeols(
  dependent_vars = c("CO2_Emissions", "Energy_Consumption"),
  lags = 1,
  transformation = "demean",
  data = pdata,
  panel_identifier = c("State", "Year")
)

