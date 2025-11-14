"""Helpers for filtering player data."""

from __future__ import annotations

import pandas as pd


def filter_min_games_played(df: pd.DataFrame, min_games: int = 30) -> pd.DataFrame:
    """Return a dataframe with only players who have played at least ``min_games`` games."""
    if "G" not in df.columns:
        raise KeyError("Expected column 'G' to be present in dataframe.")

    # Ensure we work with a fresh frame to avoid chained assignment surprises.
    return df[df["G"] >= min_games].copy()


def filter_multi_team_stints(
    df: pd.DataFrame,
    *,
    player_col: str = "Player-additional",
    season_col: str = "season_start",
    team_col: str = "Team",
) -> pd.DataFrame:
    """Remove per-team rows for multi-team seasons while keeping aggregate rows (e.g. '2TM')."""
    required = {player_col, season_col, team_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {sorted(missing)}")

    team_values = df[team_col].astype(str)
    normalized = team_values.str.upper()
    aggregate_mask = normalized.str.endswith("TM") | (normalized == "TOT")

    multi_team_mask = (
        df.groupby([player_col, season_col], sort=False)[team_col]
        .transform("nunique")
        > 1
    )

    rows_to_drop = multi_team_mask & ~aggregate_mask
    return df.loc[~rows_to_drop].copy()


def filter_years_in_nba_nonzero(
    df: pd.DataFrame,
    *,
    years_col: str = "years_in_nba",
) -> pd.DataFrame:
    """Return rows where the years-in-league metric is non-zero."""
    if years_col not in df.columns:
        raise KeyError(f"DataFrame missing required column: '{years_col}'")

    return df[df[years_col] != 0].copy()
