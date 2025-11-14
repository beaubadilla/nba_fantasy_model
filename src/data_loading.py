import pandas as pd
import os
import glob
from typing import List, Optional

def read_all_season_files(
    data_root: str,
    filename: str = "players.csv",
    season_format: str = "%y_%y"
) -> pd.DataFrame:
    """
    Reads all player CSVs across season folders, adds season_start, and combines into one DataFrame.

    Args:
        data_root (str): Root directory containing season subfolders like '09_10', '10_11', etc.
        filename (str): Name of the CSV file to read from each folder.
        season_format (str): Format used for season folder names.

    Returns:
        pd.DataFrame: Combined DataFrame with an added 'season_start' column.
    """
    all_rows = []

    # Grab all season folders
    folders = sorted(glob.glob(os.path.join(data_root, "*/")))

    for folder in folders:
        season_str = os.path.basename(os.path.normpath(folder))
        csv_path = os.path.join(folder, filename)

        if not os.path.exists(csv_path):
            print(f"Skipping missing file: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
            # Parse season start year from folder name (e.g. '09_10' -> 2009)
            start_year = int(season_str[:2])
            start_year += 2000 if start_year < 50 else 1900
            df['season_start'] = start_year
            all_rows.append(df)
        except Exception as e:
            print(f"Failed to read {csv_path}: {e}")

    combined_df = pd.concat(all_rows, ignore_index=True)
    return combined_df

def save_combined_df(df: pd.DataFrame, output_path: Optional[str] = None):
    """
    Saves the combined DataFrame to a CSV file if an output path is provided.

    Args:
        df (pd.DataFrame): The combined player-season DataFrame.
        output_path (Optional[str]): Destination path to save the CSV.
    """
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Combined dataset saved to {output_path}")
