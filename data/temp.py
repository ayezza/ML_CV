import pandas as pd

if __name__ == "__main__":
    # Create a sample DataFrame
    # df = pd.read_csv("ENB_data.csv")
    # df['eucliean_agg'] = ((df['heating_load'] ** 2 + df['cooling_load'] ** 2) ** 0.5)
    # get the min and max of the new column
    # min_value = df['eucliean_agg'].min()
    # max_value = df['eucliean_agg'].max()
    # print(f"Min value of eucliean_agg: {min_value}")
    # print(f"Max value of eucliean_agg: {max_value}")

    df = pd.read_csv("weather_data.csv")
    print(df.head())
    print(f"columns:\n {list(df.columns)}")
    print(f"Summary:\n{df.describe()}")
    print(f"\nNumber of missing values in each column:\n{df.isnull().sum()}")
    print(f"Min value of Temperature (C): {df['Temperature (C)'].min()}")
    print(f"Max value of Temperature (C): {df['Temperature (C)'].max()}")
    print(f"Mean value of Temperature (C): {df['Temperature (C)'].mean()}")
    print(f"Standard deviation of Temperature (C): {df['Temperature (C)'].std()}")
    print(f"Skewness of Temperature (C): {df['Temperature (C)'].skew()}")
    print(f"Kurtosis of Temperature (C): {df['Temperature (C)'].kurt()}")

    print(f"\nUnique values in 'Summary': {df['Summary'].unique()}\n{df['Summary'].value_counts()}")
    print(f"\nUnique values in counts in 'Precip Type': {df['Precip Type'].value_counts()}\n{df['Precip Type'].value_counts(normalize=True)}")
    print(f"\nUnique values in counts in 'Loud Cover': {df['Loud Cover'].value_counts()}\n{df['Loud Cover'].value_counts(normalize=True)}")

    df= df[df['Precip Type'].notna()]
    df.to_csv("weather_data_cleaned.csv", index=False)
    