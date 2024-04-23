import pandas as pd

def convert_to_float(num_str):
    if isinstance(num_str, str) and ',' in num_str:
        num_parts = num_str.split(',')
        if len(num_parts) == 2:
            try:
                return float('.'.join(num_parts))
            except ValueError:
                pass
    return num_str

def replace_negative_with_median(series):
    series_float = series.apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
    median = series_float[series_float >= 0].median()
    for i in range(len(series_float)):
        if isinstance(series_float.iloc[i], float) and series_float.iloc[i] < 0:
            series_float.iloc[i] = median
    return series_float

def convert_df_float(df):
    df = df.applymap(convert_to_float)
    df = df.apply(replace_negative_with_median, axis=0)
    return df

if __name__=='__main__':
    dataset_dir = 'dataset/AirQuality.csv'

    df = pd.read_csv(dataset_dir, delimiter=';')
    df = df.iloc[:, 2:15]

    df_converted = convert_df_float(df)
    df_converted = df_converted.dropna()
    df_converted.to_csv('dataset/air_quality_converted.csv', index=False)

    print(df)
    print(df_converted)
