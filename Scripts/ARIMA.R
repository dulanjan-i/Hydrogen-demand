library(forecast)

# Filter data for one state (e.g., Baden-Württemberg)
BW <- clean_data %>% filter(State == "Baden-Wuttemberg")
BW_ts <- ts(BW$CO2_Emissions, start = c(1990), frequency = 1)  # Replace with actual start year

# Fit ARIMA on CO2 emissions
fit <- auto.arima(BW$CO2_Emissions, seasonal = FALSE)

# Summarize the ARIMA model
summary(fit)

# Plot the forecast
forecast_plot <- forecast(fit, h = 10)  # Forecast next 10 years
autoplot(forecast_plot) +
  labs(title = "ARIMA Forecast for CO2 Emissions in Baden-Württemberg",
       x = "Year", y = "CO2 Emissions (1000T CO2)")


# Filter data for one state (e.g., Baden-Württemberg)
BW <- clean_data %>% filter(State == "Baden-Wuttemberg")
BW_ts <- ts(BW$Energy_Consumption, start = c(1990), frequency = 1)  # Replace with actual start year

# Fit ARIMA on CO2 emissions
fit2 <- auto.arima(BW$Energy_Consumption, seasonal = FALSE)

# Summarize the ARIMA model
summary(fit2)

# Plot the forecast
forecast_plot <- forecast(fit2, h = 10)  # Forecast next 10 years
autoplot(forecast_plot) +
  labs(title = "ARIMA Forecast for Energy_Consumption in Baden-Württemberg",
       x = "Year", y = "CO2 Emissions (1000T CO2)")
# Ensure Year is numeric
data$Year <- as.numeric(as.character(data$Year))

# Split data by State
state_split <- split(data, data$State)

# Create a time series for each state
state_ts <- lapply(state_split, function(df) {
  ts(df$CO2_Emissions, start = min(df$Year), frequency = 1)
})

# Determine the range of years
years <- seq(min(data$Year), max(data$Year))

# Plot the first state as the base plot
plot(years, state_ts[[1]], type = "l", col = 1, lty = 1, 
     xlab = "Year", ylab = "CO2 Emissions", main = "CO2 Balances by State",
     ylim = range(data$CO2_Emissions, na.rm = TRUE))

# Add other states
colors <- rainbow(length(state_ts))  # Unique colors for each state
for (i in seq_along(state_ts)) {
  lines(years, state_ts[[i]], col = colors[i], lty = 1)
}

# Add legend
legend("topright", legend = names(state_ts), col = colors, lty = 1, cex = 0.8)
