"""Utilities for formatting pandas DataFrames for human readability."""

from __future__ import annotations

import pandas as pd


def round_continuous_features(
    df: pd.DataFrame,
    *,
    decimals: int = 1,
) -> pd.DataFrame:
    """Return a copy of ``df`` with float columns rounded to ``decimals`` places.

    Parameters
    ----------
    df:
        Input dataframe.
    decimals:
        Number of decimal places to round to. Must be non-negative.

    Returns
    -------
    pandas.DataFrame
        Rounding is applied only to floating-point columns; other columns are
        left unchanged.
    """
    if decimals < 0:
        raise ValueError("decimals must be non-negative")

    rounded_df = df.copy()
    float_cols = rounded_df.select_dtypes(include=["float", "float16", "float32", "float64"]).columns
    if not float_cols.empty:
        rounded_df.loc[:, float_cols] = rounded_df.loc[:, float_cols].round(decimals)
    return rounded_df
