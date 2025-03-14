import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def location_outliers(df, eps_km=0.5, min_samples=10):
    """
    Detects outliers in geolocation data using DBSCAN.

    Args:
        df (DataFrame): DataFrame with 'latitude' and 'longitude' columns.
        eps_km (float): The maximum distance (in km) for two points to be considered neighbors.
        min_samples (int): Minimum number of points to form a cluster.

    Returns:
        Series: Boolean Series (True = outlier, False = normal).
        Series: String Series with "Outlier" for outliers, "Normal" for others.
    """
    kms_per_radian = 6371.0088
    eps = eps_km / kms_per_radian

    valid_coords = df[["latitude", "longitude"]].dropna()

    if valid_coords.empty:
        return pd.Series([np.nan] * len(df), index=df.index), pd.Series(
            [np.nan] * len(df), index=df.index
        )

    # Convert coordinates to radians
    radians_coords = np.radians(valid_coords)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(
        eps=eps, min_samples=min_samples, algorithm="ball_tree", metric="haversine"
    )
    labels = dbscan.fit_predict(radians_coords)

    outlier_series = (
        df["outlier"] if "outlier" in df else pd.Series(False, index=df.index)
    )
    outlier_type_series = (
        df["outlier_type"] if "outlier_type" in df else pd.Series("", index=df.index)
    )

    outlier_series[valid_coords.index] = labels == -1
    outlier_type_series[valid_coords.index] += np.where(labels == -1, "location:", "")

    return outlier_series, outlier_type_series


def price_outliers_iqr(df, coef=1.5):
    Q1 = df["unit_price"].quantile(0.25)
    Q3 = df["unit_price"].quantile(0.75)
    IQR = Q3 - Q1
    inner_lower_fence = Q1 - 1.5 * IQR
    outer_lower_fence = Q1 - 3 * IQR
    inner_upper_fence = Q3 + 1.5 * IQR
    outer_upper_fence = Q3 + 3 * IQR

    outlier_series = (
        df["outlier"].copy() if "outlier" in df else pd.Series(False, index=df.index)
    )
    outlier_type_series = (
        df["outlier_type"].copy()
        if "outlier_type" in df
        else pd.Series("", index=df.index)
    )

    outlier_series[df[df["unit_price"] < inner_lower_fence].index] = True
    outlier_series[df[df["unit_price"] > outer_upper_fence].index] = True
    outlier_type_series[
        df[df["unit_price"] < inner_lower_fence].index
    ] += "unit_price_iqr:"
    outlier_type_series[
        df[df["unit_price"] > outer_upper_fence].index
    ] += "unit_price_iqr:"

    return outlier_series, outlier_type_series
