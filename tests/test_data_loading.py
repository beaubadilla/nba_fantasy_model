import os
import pandas as pd
import pytest
import tempfile
from src.data_loading import read_all_season_files

@pytest.fixture
def mock_season_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock folders and CSVs
        seasons = ['09_10', '10_11']
        data = [
            {'Player': ['A', 'B'], 'FP': [20, 30]},
            {'Player': ['A', 'C'], 'FP': [25, 35]}
        ]

        for season, df_data in zip(seasons, data):
            folder_path = os.path.join(temp_dir, season)
            os.makedirs(folder_path, exist_ok=True)
            df = pd.DataFrame(df_data)
            df.to_csv(os.path.join(folder_path, 'players.csv'), index=False)

        yield temp_dir

def test_read_all_season_files(mock_season_data):
    df = read_all_season_files(data_root=mock_season_data)

    # Expect 4 rows total (2 per season)
    assert len(df) == 4
    # Check that season_start exists and is parsed correctly
    assert 'season_start' in df.columns
    assert df['season_start'].nunique() == 2
    assert sorted(df['season_start'].unique()) == [2009, 2010]
