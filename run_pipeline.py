import argparse
import os
from src.data_loading import read_all_season_files, save_combined_df
from src.data_imputation import (
    impute_missing_seasons,
    fill_static_columns,
    fill_age_column,
)
from src.feature_engineering import (
    add_new_coach_flag,
    add_new_team_flag,
    add_years_in_nba,
    add_career_per_game_features,
    add_previous_season_lag_features,
    add_fantasy_points_per_game,
)
from src.utils.logging_utils import get_logger
from src.filter import (
    filter_min_games_played,
    filter_multi_team_stints,
    filter_years_in_nba_nonzero,
)
from src.utils.dataframe_formatting import round_continuous_features

logger = get_logger(__name__)

# Update this mapping to rename columns for interpretability.
# Example: {"lag1_PTS": "pts_prev_season"}
COLUMN_RENAME_MAP = {
    "lag1_PTS": "ppg_prev_season",
    "lag1_AST": "apg_prev_season",
    "lag1_TRB": "rpg_prev_season",
    "lag1_BLK": "bpg_prev_season",
    "lag1_STL": "spg_prev_season",
    "lag1_TOV": "topg_prev_season",
    "lag1_FTA": "ftapg_prev_season",
    "lag1_G": "games_available_prev_season",
    "lag1_Age": "age_prev_season",
    "lag1_years_in_nba": "years_in_nba_prev_season",
    "career_PTS_pg": "roll_career_ppg",
    "career_AST_pg": "roll_career_apg",
    "career_TRB_pg": "roll_career_rpg",
    "career_BLK_pg": "roll_career_bpg",
    "career_STL_pg": "roll_career_spg",
    "career_TOV_pg": "roll_career_topg",
    "Pos": "position",
}


def rename_columns(df, *, dataset_label: str):
    """Rename columns using ``COLUMN_RENAME_MAP`` and log what changed."""
    if not COLUMN_RENAME_MAP:
        logger.info(
            "Skipping column rename for %s dataset; COLUMN_RENAME_MAP is empty.",
            dataset_label,
        )
        return df

    missing_columns = [col for col in COLUMN_RENAME_MAP if col not in df.columns]
    if missing_columns:
        raise KeyError(
            f"Cannot rename columns for {dataset_label}; missing columns: {missing_columns}"
        )

    renamed_df = df.rename(columns=COLUMN_RENAME_MAP)
    logger.info(
        "Renamed %d columns for %s dataset.", len(COLUMN_RENAME_MAP), dataset_label
    )
    return renamed_df


def main(data_root: str, output_path: str):
    base_dir, base_filename = os.path.split(output_path)
    name, ext = os.path.splitext(base_filename)
    if not ext:
        ext = ".csv"
    imputed_output_path = os.path.join(base_dir, f"{name}_imputed{ext}")
    feature_engineer_output_path = os.path.join(
        base_dir, f"{name}_feature_engineer{ext}"
    )
    filtered_output_path = os.path.join(base_dir, f"{name}_filtered{ext}")

    logger.info("📥 Loading raw player data...")
    df = read_all_season_files(data_root)

    logger.info("🔧 Imputing missing seasons...")
    df_imputed = impute_missing_seasons(df)

    logger.info("🧱 Filling static player columns...")
    static_cols = ["Player", "Pos"]
    df_imputed = fill_static_columns(df_imputed, static_cols)

    logger.info("🎂 Filling age column...")
    df_imputed = fill_age_column(df_imputed)

    logger.info("💾 Saving transformed dataset...")
    save_combined_df(df_imputed, imputed_output_path)

    # Feature engineering steps
    logger.info("Adding 'years in NBA' feature...")
    df_fe = add_years_in_nba(df_imputed)

    logger.info("Adding 'rolled up career per game' feature...")
    df_fe = add_career_per_game_features(
        df_fe, per_game_cols=["PTS", "AST", "TRB", "BLK", "STL", "TOV"]
    )  # add more as needed

    logger.info("Calculating fantasy points per game target...")
    df_fe = add_fantasy_points_per_game(df_fe)

    logger.info("Adding 'previous season lag' features...")
    lag_columns = [
        "FTA",
        "PTS",
        "AST",
        "TRB",
        "BLK",
        "STL",
        "TOV",
        "G",
        "Age",
        "years_in_nba",
    ]
    df_fe = add_previous_season_lag_features(df_fe, lag_cols=lag_columns)
    print(df_fe.columns)

    logger.info("Adding 'new coach' flag feature...")
    df_fe = add_new_coach_flag(df_fe)

    logger.info("Adding 'new team' flag feature...")
    df_fe = add_new_team_flag(df_fe)

    logger.info("📊 Saving feature engineering...")
    save_combined_df(df_fe, feature_engineer_output_path)

    # Preparing final dataset for modeling
    logger.info("Preparing final dataset for modeling...")

    logger.info("Removing single-team rows from multi-team seasons...")
    df_fe = filter_multi_team_stints(df_fe)

    logger.info("Filtering out seasons with zero years in NBA...")
    df_fe = filter_years_in_nba_nonzero(df_fe)

    logger.info("Filtering out players with fewer than 30 games played...")
    df_fe = filter_min_games_played(df_fe)

    # Select relevant columns for modeling
    filtered_cols = [
        "Player-additional",
        "season_start",
        "Team",
        "lag1_Age",
        "Pos",
        "lag1_G",
        "lag1_FTA",
        "lag1_PTS",
        "lag1_AST",
        "lag1_TRB",
        "lag1_BLK",
        "lag1_STL",
        "lag1_TOV",
        "career_PTS_pg",
        "career_AST_pg",
        "career_TRB_pg",
        "career_BLK_pg",
        "career_STL_pg",
        "career_TOV_pg",
        "fantasy_points_per_game",
        "lag1_years_in_nba",
        "new_coach",
        "new_team",
    ]
    df_filtered = df_fe[filtered_cols].copy()
    logger.info(f"Saving filtered dataset to {filtered_output_path}...")
    df_filtered_rounded = round_continuous_features(df_filtered, decimals=1)
    df_filtered_prepared = rename_columns(df_filtered_rounded, dataset_label="filtered")
    save_combined_df(df_filtered_prepared, filtered_output_path)

    modeling_cols = [
        "Player-additional",
        "season_start",
        "lag1_Age",
        "Pos",
        "lag1_G",
        "lag1_FTA",
        "lag1_PTS",
        "lag1_AST",
        "lag1_TRB",
        "lag1_BLK",
        "lag1_STL",
        "lag1_TOV",
        "career_PTS_pg",
        "career_AST_pg",
        "career_TRB_pg",
        "career_BLK_pg",
        "career_STL_pg",
        "career_TOV_pg",
        "fantasy_points_per_game",
        "lag1_years_in_nba",
        "new_coach",
        "new_team",
    ]
    df_modeling = df_fe[modeling_cols].copy()
    processed_dir = os.path.join(
        os.path.abspath(os.path.join(base_dir or ".", os.pardir)), "processed"
    )
    os.makedirs(processed_dir, exist_ok=True)
    processed_output_path = os.path.join(processed_dir, "data.csv")
    df_modeling_rounded = round_continuous_features(df_modeling, decimals=1)
    df_modeling_prepared = rename_columns(df_modeling_rounded, dataset_label="modeling")
    logger.info(f"Saving final dataset to {processed_output_path}...")
    save_combined_df(df_modeling_prepared, processed_output_path)

    logger.info("✅ Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NBA Fantasy Forecasting data pipeline"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to the raw data root directory",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the processed CSV"
    )

    args = parser.parse_args()
    main(args.data_root, args.output)
