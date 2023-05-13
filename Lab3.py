import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Read the dataset into a pandas dataframe
data = pd.read_csv('tv_shows.csv')
print(data)
# Scale numerical data using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Convert the scaled data back to a pandas dataframe
clean_data = pd.DataFrame(scaled_data, columns=data.columns)

# Save the cleaned data to a new CSV file
#clean_data.to_csv('cleaned_lab3.csv', index=False)