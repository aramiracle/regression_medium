import pandas as pd

# Function to convert a number string with comma decimal separator to float
def convert_to_float(num_str):
    if isinstance(num_str, str) and ',' in num_str:  # Check if the input is a string containing a comma
        num_parts = num_str.split(',')  # Split the string by comma
        if len(num_parts) == 2:  # Check if there are two parts after splitting
            try:
                return float('.'.join(num_parts))  # Join the parts with a dot and convert to float
            except ValueError:  # Catch ValueError if conversion fails
                pass
    return num_str  # Return the original input if it doesn't match the conditions

# Function to replace negative values in a series with the median of non-negative values
def replace_negative_with_median(series):
    series_float = series.apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
    # Convert string numbers to float, replacing comma with dot
    median = series_float[series_float >= 0].median()  # Calculate median of non-negative values
    for i in range(len(series_float)):
        if isinstance(series_float.iloc[i], float) and series_float.iloc[i] < 0:
            series_float.iloc[i] = median  # Replace negative values with median
    return series_float

# Function to convert all entries in a DataFrame to float
def convert_df_float(df):
    df = df.applymap(convert_to_float)  # Apply convert_to_float function to every element in DataFrame
    df = df.apply(replace_negative_with_median, axis=0)  # Apply replace_negative_with_median function to each column
    return df  # Return the converted DataFrame

if __name__=='__main__':
    dataset_dir = 'dataset/AirQuality.csv'

    df = pd.read_csv(dataset_dir, delimiter=';')
    df = df.iloc[:, 2:15]  # Select columns 2 through 14 (0-based indexing)

    df_converted = convert_df_float(df)  # Convert DataFrame to float
    df_converted = df_converted.dropna()  # Drop rows with NaN values
    df_converted.to_csv('dataset/air_quality_converted.csv', index=False)  # Save converted DataFrame to CSV
