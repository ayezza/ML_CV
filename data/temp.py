import pandas as pd



if __name__ == "__main__":
    # Create a sample DataFrame
    df = pd.read_csv("bike_sharing_day.csv")
    df = df.drop(columns=["cnt"])
    df.to_csv("bike_sharing_day.csv", index=False)