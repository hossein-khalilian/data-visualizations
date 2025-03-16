import ast
import re

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


def convert_persian_digits(text):
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    english_digits = "0123456789"
    translation_table = str.maketrans(persian_digits, english_digits)
    return text.translate(translation_table)


def parse_floor(value):
    if isinstance(value, int):
        return value, None

    value = convert_persian_digits(str(value))
    match = re.match(r"(\d+) از (\d+)", value)

    if match:
        return int(match.group(1)), int(match.group(2))
    elif "زیرهمکف" in value:
        match = re.search(r"\d+", value)
        return -1, int(match.group()) if match else None
    elif "همکف" in value:
        match = re.search(r"\d+", value)
        return (0, int(match.group())) if match else (0, None)

    return None, None


def format_million_billion(x):
    """Converts large numbers to M (Million) or B (Billion)."""
    if x >= 1e9:
        return f"{x / 1e9:.1f}B"
    elif x >= 1e6:
        return f"{x / 1e6:.1f}M"
    else:
        return f"{x:.1f}"


def process_data(df):
    df[["location_type", "latitude", "longitude"]] = df["location"].apply(
        lambda x: pd.Series(extract_lat_lon(x))
    )

    features_df = df["features"].apply(lambda x: pd.Series(process_features(x)))
    features_df.replace({"ندارد": False}, inplace=True)
    features_df[["elevator", "parking", "warehouse", "balcony"]] = features_df[
        ["elevator", "parking", "warehouse", "balcony"]
    ].replace({"": True})
    features_df["are_images_valid"] = (
        features_df["are_images_valid"].replace({"بله": True}).fillna("False")
    )
    features_df["production_year"] = (
        features_df["production_year"].replace({"قبل از ۱۳۷۰": "1369"}).astype(int)
    )
    features_df["rooms"] = features_df["rooms"].replace({"+۴": "5"}).astype(int)
    features_df[["floor_number", "total_floors"]] = features_df["floor"].apply(
        lambda x: pd.Series(parse_floor(x))
    )
    features_df.drop(columns="floor", inplace=True)

    df = pd.concat(
        [
            df.drop(
                columns=[
                    "_id",
                    "price",
                    "location",
                    "features",
                    "brand_model",
                    "gender",
                    "originality",
                    "status",
                    "credit",
                    "rent",
                ]
            ),
            features_df,
        ],
        axis=1,
    )
    df["price_h"] = df["price"].apply(format_million_billion)
    df["unit_price_h"] = df["unit_price"].apply(format_million_billion)

    return df
