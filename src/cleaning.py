import pandas as pd


def load_data(file_path):
    return pd.read_csv(file_path)


def clean_data(df):
    # Remove rows with missing scores
    df = df.dropna(subset=["score"])

    # Keep only scores between 0 and 100
    df = df[(df["score"] >= 0) & (df["score"] <= 100)]

    return df


def get_scores(df):
    return df["score"].to_numpy()