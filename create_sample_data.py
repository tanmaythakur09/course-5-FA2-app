import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate date range
start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 1, 1)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Create sample data
n_records = 5658
atm_ids = np.random.choice(range(1, 51), n_records)  # 50 unique ATMs
dates = np.random.choice(date_range, n_records)

# Generate distributions based on project findings
total_withdrawals = np.random.normal(49808, 14904, n_records)
total_withdrawals = np.clip(total_withdrawals, 1380, 107790)

total_deposits = np.random.normal(10129, 4879, n_records)
total_deposits = np.clip(total_deposits, 0, 32395)

previous_day_cash = np.random.normal(100000, 50000, n_records)
previous_day_cash = np.clip(previous_day_cash, 10000, 300000)

cash_demand_next_day = total_withdrawals * np.random.uniform(0.9, 1.1, n_records)

# Day of week
day_of_week_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                   4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
dates_pd = pd.to_datetime(dates)
day_of_week = [day_of_week_map[d.weekday()] for d in dates_pd]

# Time of day
time_choices = ['Morning', 'Afternoon', 'Evening', 'Night']
time_of_day = np.random.choice(time_choices, n_records)

# Location type
location_choices = ['Standalone', 'Supermarket', 'Mall', 'Bank Branch', 'Gas Station']
location_type = np.random.choice(location_choices, n_records)

# Weather condition
weather_choices = ['Clear', 'Cloudy', 'Rainy', 'Snowy']
weather_condition = np.random.choice(weather_choices, n_records)

# Nearby competitors
nearby_competitors = np.random.randint(0, 10, n_records)

# Holiday flag (1% holidays)
holiday_flag = np.random.choice([0, 1], n_records, p=[0.99, 0.01])

# Special event flag (10% events)
special_event_flag = np.random.choice([0, 1], n_records, p=[0.90, 0.10])

# Create dataframe
df = pd.DataFrame({
    'Date': dates,
    'ATM_ID': atm_ids,
    'Total_Withdrawals': total_withdrawals.round(2),
    'Total_Deposits': total_deposits.round(2),
    'Previous_Day_Cash_Level': previous_day_cash.round(2),
    'Cash_Demand_Next_Day': cash_demand_next_day.round(2),
    'Day_of_Week': day_of_week,
    'Time_of_Day': time_of_day,
    'Location_Type': location_type,
    'Weather_Condition': weather_condition,
    'Nearby_Competitor_ATMs': nearby_competitors,
    'Holiday_Flag': holiday_flag,
    'Special_Event_Flag': special_event_flag
})

# Save to CSV
df.to_csv('atm_cash_management_dataset.csv', index=False)
print(f"✓ Sample dataset created: {len(df)} records")
print(f"✓ File saved: atm_cash_management_dataset.csv")
