import ast

import numpy as np
import pandas as pd


def extract_lat_lon(location):
    if isinstance(location, str):
        try:
            location_dict = ast.literal_eval(location)
            if "exact_data" in location_dict:
                return (
                    "exact",
                    location_dict["exact_data"]["point"]["latitude"],
                    location_dict["exact_data"]["point"]["longitude"],
                )
            elif "fuzzy_data" in location_dict:
                return (
                    "fuzzy",
                    location_dict["fuzzy_data"]["point"]["latitude"],
                    location_dict["fuzzy_data"]["point"]["longitude"],
                )
        except (SyntaxError, ValueError):
            return np.nan, np.nan, np.nan
    return np.nan, np.nan, np.nan


def process_features(column):
    if isinstance(column, str):
        try:
            return ast.literal_eval(column)
        except:
            return np.nan

    return np.nan


def format_million_billion(x):
    """Converts large numbers to M (Million) or B (Billion)."""
    if x >= 1e9:
        return f"{x / 1e9:.2f}B"
    elif x >= 1e6:
        return f"{x / 1e6:.2f}M"
    else:
        return f"{x:.2f}"


def process_data(df):
    df[["location_type", "latitude", "longitude"]] = df["location"].apply(
        lambda x: pd.Series(extract_lat_lon(x))
    )

    df.drop(columns=["price", "location"], inplace=True)
    df = pd.concat(
        [df, df["features"].apply(lambda x: pd.Series(process_features(x)))],
        axis=1,
    )
    df.drop(columns=["features"], inplace=True)

    df["price_h"] = df["price"].apply(format_million_billion)
    df["unit_price_h"] = df["unit_price"].apply(format_million_billion)

    return df
