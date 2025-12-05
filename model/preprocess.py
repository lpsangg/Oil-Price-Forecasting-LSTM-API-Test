import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter, MonthLocator # Thêm MonthLocator vào để sửa lỗi


# read data
df = pd.read_csv('./data/QDL-OPEC.csv')
print(df)
print(df.describe())
print(df.isna().sum())

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Set 'date' as the index
df.set_index('date', inplace=True)

# Create a continuous daily time series
continuous_time_series = df.resample('D').asfreq()

# Create a new column containing the rolling average of the previous 5 days and the next 5 days
window_size = 5
continuous_time_series['rolling_avg'] = continuous_time_series['value'].rolling(
    window=window_size*2 + 1,
    min_periods=1,
    center=True
).mean()

# Fill missing values using the computed rolling average
continuous_time_series['value'].fillna(continuous_time_series['rolling_avg'], inplace=True)

# Remove the auxiliary 'rolling_avg' column
continuous_time_series.drop(columns=['rolling_avg'], inplace=True)

# Display the result
print(continuous_time_series)

# reset index
continuous_time_series = continuous_time_series.reset_index()

# Convert 'date' column to datetime
continuous_time_series['date'] = pd.to_datetime(continuous_time_series['date'])

# Extract year from the 'date' column
continuous_time_series['Year'] = continuous_time_series['date'].dt.year

# Sort data by date
continuous_time_series.sort_values(by='date', inplace=True)

# Plot oil price over the years
plt.figure(figsize=(10, 5))
plt.plot(continuous_time_series['date'], continuous_time_series['value'], label='Value', color='red')
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Average Oil Price of OPEC Member Countries')
plt.legend()

# Format x-axis: major ticks by year, minor ticks by month
years = YearLocator()
years_fmt = DateFormatter('%Y')
months = MonthLocator()

ax = plt.gca()
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

plt.tight_layout()
plt.show()

# save data
continuous_time_series.to_csv('./data/preprocess-QDL-OPEC.csv', index=True)



