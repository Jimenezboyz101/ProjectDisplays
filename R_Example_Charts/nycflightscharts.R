# Load libraries
library(tidyverse)
library(openintro)

# Load data
data(nycflights)
view(nycflights)

# Filter for LAX flights and plot
onlylax <- nycflights %>% 
  filter(dest == "LAX")

ggplot(data = onlylax, aes(x = dep_delay)) +
  geom_histogram(binwidth = 20, fill = "steelblue", color = "black") +
  labs(title = "Departure Delays for Flights to LAX", x = "Departure Delay (minutes)", y = "Count")

# Summarize LAX flight delays
onlylax %>% 
  summarize(avg_dep_delay = mean(dep_delay, na.rm=TRUE), 
            med_dep_delay = median(dep_delay, na.rm=TRUE), 
            frequency = n())

# Filter for SFO flights in February
sfo_feb_flights <- nycflights %>% 
  filter(dest == "SFO" & month == 2)

num_sfo_feb_flights <- nrow(sfo_feb_flights)

# Median arrival delay by carrier
carrier_medians <- sfo_feb_flights %>% 
  group_by(carrier) %>% 
  summarize(arr_delay_median = median(arr_delay, na.rm = TRUE))

# Plot median arrival delay by carrier
ggplot(data = carrier_medians, aes(x = carrier, y = arr_delay_median, fill = carrier)) +
  geom_col() +
  labs(title = "Median Arrival Delay by Carrier (SFO Flights in February)", x = "Carrier", y = "Median Arrival Delay (minutes)") +
  theme_minimal()

# IQR of arrival delay by carrier
carrier_iqr <- sfo_feb_flights %>% 
  group_by(carrier) %>% 
  summarize(arr_delay_IQR = IQR(arr_delay, na.rm = TRUE))

# Plot IQR of arrival delays by carrier
ggplot(data = carrier_iqr, aes(x = carrier, y = arr_delay_IQR, fill = carrier)) +
  geom_col() +
  labs(title = "IQR of Arrival Delay by Carrier (SFO Flights in February)", x = "Carrier", y = "Arrival Delay IQR (minutes)") +
  theme_minimal()

# Boxplot of arrival delays by carrier
ggplot(data = sfo_feb_flights, aes(x = carrier, y = arr_delay, fill = carrier)) +
  geom_boxplot() +
  labs(title = "Arrival Delay Distribution by Carrier (SFO Flights in February)", x = "Carrier", y = "Arrival Delay (minutes)") +
  theme_minimal()

