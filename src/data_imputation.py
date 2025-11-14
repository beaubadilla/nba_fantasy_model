import pandas as pd
from typing import List

def impute_missing_seasons(df: pd.DataFrame, id_col: str = "Player-additional") -> pd.DataFrame:
    """
    Fills in missing seasons for players who skipped years (e.g. injuries).

    Args:
        df (pd.DataFrame): Raw player-season data with 'season_start' and id_col.
        id_col (str): Column name uniquely identifying players.

    Returns:
        pd.DataFrame: DataFrame with missing seasons imputed (G=0, did_play=False).
    """
    all_seasons = sorted(df['season_start'].unique())

    # Step 1: Determine range of seasons for each player
    player_ranges = (
        df.groupby(id_col)['season_start']
        .agg(['min', 'max'])
        .rename(columns={'min': 'rookie_year', 'max': 'final_year'})
    )

    # Step 2: Generate full player-season grid
    grid_rows = []
    for player, row in player_ranges.iterrows():
        for season in range(row['rookie_year'], row['final_year'] + 1):
            grid_rows.append({id_col: player, 'season_start': season})

    full_grid = pd.DataFrame(grid_rows)

    # Step 3: Merge with actual data
    merged = full_grid.merge(df, on=[id_col, 'season_start'], how='left')

    # Step 4: Mark did_play and fill G
    merged['G'] = merged['G'].fillna(0)
    merged['did_play'] = merged['G'] > 0

    return merged

def fill_static_columns(merged_df: pd.DataFrame, static_cols: List[str], id_col: str = "Player-additional") -> pd.DataFrame:
    """
    Safely fill static columns (e.g. Player name, Position) for imputed player-season rows.

    Args:
        df (pd.DataFrame): DataFrame with imputed rows (including id_col and static columns).
        static_cols (List[str]): List of static columns to forward/backward fill.
        id_col (str): Unique player identifier column name.

    Returns:
        pd.DataFrame: DataFrame with static columns filled and merged back.
    """
    # Step 1: Select only id_col and static_cols, apply ffill and bfill per player
    filled_static = (
        merged_df
        .sort_values('season_start')
        .loc[:, [id_col] + static_cols]
        .groupby(id_col)
        .transform(lambda x: x.ffill().bfill())
    )
    # Step 2: Re-attach id_col (transform drops it)
    filled_static[id_col] = merged_df[id_col]

    # Step 3: Take the latest info per player (last non-null)
    static_info = (
        filled_static
        .groupby(id_col)
        .last()
        .reset_index()
    )

    # Step 4: Drop static_cols from original merged_df and merge filled values back
    merged_df = merged_df.drop(columns=static_cols, errors='ignore')
    merged_df = merged_df.merge(static_info, on=id_col, how='left')
    return merged_df


def fill_age_column(df: pd.DataFrame, id_col: str = "Player-additional") -> pd.DataFrame:
    """
    Recalculates age per season based on rookie age and year difference.

    Args:
        df (pd.DataFrame): DataFrame containing at least 'season_start' and 'Age'.
        id_col (str): Unique player identifier.

    Returns:
        pd.DataFrame: DataFrame with corrected 'Age'.
    """
    rookie_info = (
        df.sort_values('season_start')
          .dropna(subset=['Age'])
          .groupby(id_col)
          .first()
          .reset_index()[[id_col, 'season_start', 'Age']]
          .rename(columns={'season_start': 'rookie_year', 'Age': 'rookie_age'})
    )

    df = df.merge(rookie_info, on=id_col, how='left')
    df['Age'] = df['rookie_age'] + (df['season_start'] - df['rookie_year'])
    return df.drop(columns=['rookie_year', 'rookie_age'])
