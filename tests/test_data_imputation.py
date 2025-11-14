import pandas as pd
from src.data_imputation import impute_missing_seasons, fill_static_columns, fill_age_column

def test_impute_missing_seasons():
    df = pd.DataFrame({
        'Player-additional': ['A', 'A', 'B'],
        'season_start': [2010, 2012, 2011],
        'G': [80, 70, 50]
    })

    result = impute_missing_seasons(df)

    # Player A should have 2010, 2011, 2012 (3 rows); B should have just 2011 (1 row)
    a_rows = result[result['Player-additional'] == 'A']
    assert len(a_rows) == 3
    assert (a_rows['season_start'] == [2010, 2011, 2012]).all()
    # Check imputed row
    assert a_rows.loc[a_rows['season_start'] == 2011, 'G'].values[0] == 0
    assert not a_rows.loc[a_rows['season_start'] == 2011, 'did_play'].values[0]

def test_fill_static_columns():
    df = pd.DataFrame({
        'Player-additional': ['A', 'A', 'A'],
        'season_start': [2010, 2011, 2012],
        'Player': ['Alice', None, None],
        'Pos': ['PG', None, None],
    })

    filled = fill_static_columns(df, static_cols=['Player', 'Pos'])

    assert filled['Player'].tolist() == ['Alice', 'Alice', 'Alice']
    assert filled['Pos'].tolist() == ['PG', 'PG', 'PG']

def test_fill_static_columns_preserves_id_column():
    # Simulate input with one missing row and ensure Player-additional is preserved
    df = pd.DataFrame({
        'Player-additional': ['X', 'X', 'X'],
        'season_start': [2010, 2011, 2012],
        'Player': ['Xander', None, None],
        'Pos': ['SG', None, None],
        'Ht': ["6-6", None, None],
        'Wt': [200, None, None],
        'GP': [80, 0, 0],
    })

    result = fill_static_columns(df, static_cols=['Player', 'Pos', 'Ht', 'Wt'])

    assert 'Player-additional' in result.columns, "'Player-additional' should not be dropped"
    assert result.loc[result['season_start'] == 2011, 'Player'].values[0] == 'Xander', "Static field not forward filled"
    assert result.loc[result['season_start'] == 2012, 'Ht'].values[0] == "6-6", "Ht not forward filled"
    assert result.loc[result['season_start'] == 2012, 'Wt'].values[0] == 200, "Wt not forward filled"


def test_fill_age_column():
    df = pd.DataFrame({
        'Player-additional': ['A', 'A', 'A'],
        'season_start': [2010, 2011, 2012],
        'Age': [22, None, None]
    })

    result = fill_age_column(df)
    assert result['Age'].tolist() == [22, 23, 24]
