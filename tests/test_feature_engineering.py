import pandas as pd
import pytest

from src.feature_engineering import (
    _season_label_to_year,
    add_career_per_game_features,
    add_fantasy_points_per_game,
    add_new_coach_flag,
    add_new_team_flag,
    add_previous_season_lag_features,
)
from pandas.api.types import is_bool_dtype


def test_add_career_per_game_features_with_gap_season():
    df = pd.DataFrame(
        {
            "Player-additional": ["A", "A", "A", "A"],
            "season_start": [2010, 2011, 2012, 2013],
            "PTS": [10, 20, None, 40],  # Gap season in 2012
            "G": [10, 20, 0, 40],
        }
    )

    result = add_career_per_game_features(df, per_game_cols=["PTS"])

    # 2013: (10*10 + 20*20) / (10 + 20) = (100 + 400) / 30 = 16.67
    expected = (10 * 10 + 20 * 20) / (10 + 20)
    actual = result.loc[result["season_start"] == 2013, "career_PTS_pg"].values[0]
    assert abs(actual - expected) < 1e-2


def test_add_fantasy_points_per_game():
    df = pd.DataFrame(
        {
            "PTS": [30, 12],
            "AST": [10, 3],
            "TRB": [10, 4],
            "BLK": [1, 1],
            "STL": [1, 0],
            "TOV": [2, 5],
        }
    )

    result = add_fantasy_points_per_game(df)

    assert "fantasy_points_per_game" in result.columns
    assert pytest.approx(result.loc[0, "fantasy_points_per_game"], rel=1e-6) == 59
    manual = (12 * 1.0) + (3 * 1.5) + (4 * 1.2) + (1 * 2.0) + (0 * 2.0) - (5 * 1.0)
    assert pytest.approx(result.loc[1, "fantasy_points_per_game"], rel=1e-6) == manual


def test_add_previous_season_lag_features_handles_gap():
    df = pd.DataFrame(
        {
            "Player-additional": ["p1", "p1", "p1"],
            "season_start": [2019, 2021, 2024],
            "Team": ["AAA", "AAA", "AAA"],
            "PTS": [10, 20, 30],
            "FTA": [1, 2, 3],
        }
    )

    result = add_previous_season_lag_features(df, lag_cols=["PTS", "FTA"])

    lag_2019 = result.loc[result["season_start"] == 2019, "lag1_PTS"].iloc[0]
    lag_2021 = result.loc[result["season_start"] == 2021, "lag1_PTS"].iloc[0]
    lag_2024 = result.loc[result["season_start"] == 2024, "lag1_PTS"].iloc[0]

    assert pd.isna(lag_2019)
    assert lag_2021 == 10
    assert lag_2024 == 20


def test_add_previous_season_lag_features_prefers_multi_team_aggregate():
    df = pd.DataFrame(
        {
            "Player-additional": ["lagger"] * 8,
            "season_start": [2021, 2022, 2022, 2022, 2023, 2023, 2023, 2024],
            "Team": ["CHI", "ATL", "2TM", "PHI", "NYK", "2TM", "DAL", "LAL"],
            "PTS": [10, 40, 100, 60, 50, 120, 70, 80],
            "FTA": [1, 4, 10, 6, 5, 12, 7, 8],
        }
    )

    result = add_previous_season_lag_features(df, lag_cols=["PTS", "FTA"])

    lag_2023_pts = result.loc[result["season_start"] == 2023, "lag1_PTS"].unique()
    lag_2024_pts = result.loc[result["season_start"] == 2024, "lag1_PTS"].iloc[0]
    lag_2023_fta = result.loc[result["season_start"] == 2023, "lag1_FTA"].unique()
    lag_2024_fta = result.loc[result["season_start"] == 2024, "lag1_FTA"].iloc[0]

    assert lag_2023_pts.tolist() == [100]
    assert lag_2023_fta.tolist() == [10]
    assert lag_2024_pts == 120
    assert lag_2024_fta == 12


@pytest.mark.parametrize(
    "label,expected",
    [
        ("22_23", 2022),
        ("2021_22", 2021),
        ("99_00", 1999),
        ("2005-06", 2005),
    ],
)
def test_season_label_to_year(label, expected):
    assert _season_label_to_year(label) == expected


@pytest.mark.parametrize("label", ["", None, "_23"])
def test_season_label_to_year_invalid(label):
    with pytest.raises(ValueError):
        _season_label_to_year(label)


def test_add_new_coach_flag(tmp_path):
    df = pd.DataFrame(
        {
            "Team": ["ATL", "BOS", "CHI"],
            "season_start": [2022, 2022, 2022],
        }
    )

    coaches_max_path = tmp_path / "coaches_max.csv"
    pd.DataFrame(
        {
            "Tm": ["ATL", "BOS"],
            "Season": ["21_22", "21_22"],
            "Coach": ["Coach A", "Coach B"],
        }
    ).to_csv(coaches_max_path, index=False)

    coaches_first_path = tmp_path / "coaches_first.csv"
    pd.DataFrame(
        {
            "Tm": ["ATL", "BOS"],
            "Season": ["22_23", "22_23"],
            "Coach": ["Coach X", "Coach B"],
        }
    ).to_csv(coaches_first_path, index=False)

    result = add_new_coach_flag(
        df,
        team_col="Team",
        season_col="season_start",
        coaches_max_path=str(coaches_max_path),
        coaches_first_path=str(coaches_first_path),
    )

    atl_flag = result.loc[result["Team"] == "ATL", "new_coach"].iloc[0]
    bos_flag = result.loc[result["Team"] == "BOS", "new_coach"].iloc[0]
    chi_flag = result.loc[result["Team"] == "CHI", "new_coach"].iloc[0]

    assert bool(atl_flag)
    assert not bool(bos_flag)
    assert chi_flag is pd.NA
    assert is_bool_dtype(result["new_coach"])


def test_add_new_coach_flag_latest_season(tmp_path):
    df = pd.DataFrame(
        {
            "Team": ["GSW", "GSW", "GSW"],
            "season_start": [2021, 2022, 2023],
        }
    )

    coaches_max_path = tmp_path / "coaches_max.csv"
    pd.DataFrame(
        {
            "Tm": ["GSW", "GSW"],
            "Season": ["21_22", "22_23"],
            "Coach": ["Kerr", "Kerr"],
        }
    ).to_csv(coaches_max_path, index=False)

    coaches_first_path = tmp_path / "coaches_first.csv"
    pd.DataFrame(
        {
            "Tm": ["GSW", "GSW", "GSW"],
            "Season": ["21_22", "22_23", "23_24"],
            "Coach": ["Kerr", "Kerr", "Kerr"],
        }
    ).to_csv(coaches_first_path, index=False)

    result = add_new_coach_flag(
        df,
        team_col="Team",
        season_col="season_start",
        coaches_max_path=str(coaches_max_path),
        coaches_first_path=str(coaches_first_path),
    )

    gsw_22_flag = result.loc[
        (result["Team"] == "GSW") & (result["season_start"] == 2022), "new_coach"
    ].iloc[0]
    gsw_23_flag = result.loc[
        (result["Team"] == "GSW") & (result["season_start"] == 2023), "new_coach"
    ].iloc[0]

    assert not bool(gsw_22_flag)
    assert not bool(gsw_23_flag)
    assert is_bool_dtype(result["new_coach"])


def test_add_new_coach_flag_with_player_gap(tmp_path):
    df = pd.DataFrame(
        {
            "Team": ["LAL", "GSW", "LAL", "GSW"],
            "season_start": [2019, 2019, 2021, 2021],
        }
    )

    coaches_max_path = tmp_path / "coaches_max.csv"
    pd.DataFrame(
        {
            "Tm": ["LAL", "LAL", "LAL", "GSW", "GSW", "GSW"],
            "Season": ["19_20", "20_21", "21_22", "19_20", "20_21", "21_22"],
            "Coach": [
                "Old Coach",
                "New Coach",
                "New Coach",
                "Same Coach",
                "Same Coach",
                "Same Coach",
            ],
        }
    ).to_csv(coaches_max_path, index=False)

    coaches_first_path = tmp_path / "coaches_first.csv"
    pd.DataFrame(
        {
            "Tm": ["LAL", "LAL", "LAL", "GSW", "GSW", "GSW"],
            "Season": ["19_20", "20_21", "21_22", "19_20", "20_21", "21_22"],
            "Coach": [
                "Old Coach",
                "New Coach",
                "New Coach",
                "Same Coach",
                "Same Coach",
                "Same Coach",
            ],
        }
    ).to_csv(coaches_first_path, index=False)

    result = add_new_coach_flag(
        df,
        team_col="Team",
        season_col="season_start",
        coaches_max_path=str(coaches_max_path),
        coaches_first_path=str(coaches_first_path),
    )

    flag_lal_2019 = result.loc[
        (result["Team"] == "LAL") & (result["season_start"] == 2019), "new_coach"
    ].iloc[0]
    flag_lal_2021 = result.loc[
        (result["Team"] == "LAL") & (result["season_start"] == 2021), "new_coach"
    ].iloc[0]
    flag_gsw_2019 = result.loc[
        (result["Team"] == "GSW") & (result["season_start"] == 2019), "new_coach"
    ].iloc[0]
    flag_gsw_2021 = result.loc[
        (result["Team"] == "GSW") & (result["season_start"] == 2021), "new_coach"
    ].iloc[0]

    assert flag_lal_2019 is pd.NA
    assert bool(flag_lal_2021)
    assert flag_gsw_2019 is pd.NA
    assert not bool(flag_gsw_2021)
    assert is_bool_dtype(result["new_coach"])


# New test ensures multi-team seasons and aggregate rows are handled correctly
def test_add_new_coach_flag_handles_multiteam_season(tmp_path):
    df = pd.DataFrame(
        {
            "Player-additional": ["duranke01"] * 5,
            "Team": ["BRK", "2TM", "BRK", "PHO", "PHO"],
            "season_start": [2021, 2022, 2022, 2022, 2023],
        }
    )

    coaches_max_path = tmp_path / "coaches_max.csv"
    pd.DataFrame(
        {
            "Tm": ["BRK", "BRK", "PHO", "PHO"],
            "Season": ["20_21", "21_22", "21_22", "22_23"],
            "Coach": [
                "Coach BRK 2020",
                "Coach BRK 2021",
                "Coach PHO 2021",
                "Coach PHO 2022",
            ],
        }
    ).to_csv(coaches_max_path, index=False)

    coaches_first_path = tmp_path / "coaches_first.csv"
    pd.DataFrame(
        {
            "Tm": ["BRK", "BRK", "PHO", "PHO"],
            "Season": ["21_22", "22_23", "22_23", "23_24"],
            "Coach": [
                "Coach BRK 2021",
                "Coach BRK 2022",
                "Coach PHO 2022",
                "Coach PHO 2022",
            ],
        }
    ).to_csv(coaches_first_path, index=False)

    result = add_new_coach_flag(
        df,
        team_col="Team",
        season_col="season_start",
        player_col="Player-additional",
        coaches_max_path=str(coaches_max_path),
        coaches_first_path=str(coaches_first_path),
    )

    multi_team_flag = result.loc[
        (result["Team"] == "2TM") & (result["season_start"] == 2022),
        "new_coach",
    ].iloc[0]
    first_team_flag = result.loc[
        (result["Team"] == "BRK") & (result["season_start"] == 2022),
        "new_coach",
    ].iloc[0]
    later_team_flag = result.loc[
        (result["Team"] == "PHO") & (result["season_start"] == 2023),
        "new_coach",
    ].iloc[0]

    assert bool(multi_team_flag)
    assert bool(first_team_flag)
    assert bool(multi_team_flag) == bool(first_team_flag)
    assert not bool(later_team_flag)
    assert is_bool_dtype(result["new_coach"])


# def test_add_new_coach_flag_with_multiteam_player(tmp_path):
#     df = pd.DataFrame({
#         'Team': ['LAL', '2TM', 'LAL', 'GSW', 'GSW'],
#         'season_start': [2019, 2020, 2020, 2020, 2021],
#     })

#     coaches_max_path = tmp_path / 'coaches_max.csv'
#     pd.DataFrame({
#         'Tm': ['LAL', 'LAL', 'LAL', 'GSW', 'GSW', 'GSW'],
#         'Season': ['19_20', '20_21', '21_22', '19_20', '20_21', '21_22'],
#         'Coach': ['Coach A', 'Coach A', 'Coach A', 'Coach B', 'Coach B', 'Coach B'],
#     }).to_csv(coaches_max_path, index=False)

#     coaches_first_path = tmp_path / 'coaches_first.csv'
#     pd.DataFrame({
#         'Tm': ['LAL', 'LAL', 'LAL', 'GSW', 'GSW', 'GSW'],
#         'Season': ['19_20', '20_21', '21_22', '19_20', '20_21', '21_22'],
#         'Coach': ['Coach A', 'Coach A', 'Coach A', 'Coach B', 'Coach B', 'Coach B'],
#     }).to_csv(coaches_first_path, index=False)

#     result = add_new_coach_flag(
#         df,
#         team_col='Team',
#         season_col='season_start',
#         coaches_max_path=str(coaches_max_path),
#         coaches_first_path=str(coaches_first_path),
#     )

#     flag_2020_2tm = result.loc[(result['Team'] == '2TM') & (result['season_start'] == 2020), 'new_coach'].iloc[0]
#     flag_2021_gsw = result.loc[(result['Team'] == 'GSW') & (result['season_start'] == 2021), 'new_coach'].iloc[0]

#     assert not bool(flag_2020_2tm)
#     assert bool(flag_2021_gsw)
#     assert is_bool_dtype(result['new_coach'])


def test_add_new_team_flag():
    df = pd.DataFrame(
        {
            "Player-additional": ["A", "A", "A", "B", "B"],
            "season_start": [2020, 2021, 2023, 2021, 2022],
            "Team": ["ATL", "BOS", "BOS", "DAL", "DAL"],
        }
    )

    result = add_new_team_flag(df)

    first_a = result.loc[
        (result["Player-additional"] == "A") & (result["season_start"] == 2020),
        "new_team",
    ].iloc[0]
    second_a = result.loc[
        (result["Player-additional"] == "A") & (result["season_start"] == 2021),
        "new_team",
    ].iloc[0]
    third_a = result.loc[
        (result["Player-additional"] == "A") & (result["season_start"] == 2023),
        "new_team",
    ].iloc[0]
    first_b = result.loc[
        (result["Player-additional"] == "B") & (result["season_start"] == 2021),
        "new_team",
    ].iloc[0]
    second_b = result.loc[
        (result["Player-additional"] == "B") & (result["season_start"] == 2022),
        "new_team",
    ].iloc[0]

    assert first_a is pd.NA
    assert bool(second_a)
    assert not bool(third_a)
    assert first_b is pd.NA
    assert not bool(second_b)
    assert is_bool_dtype(result["new_team"])


def test_add_new_team_flag_gap_with_team_change():
    df = pd.DataFrame(
        {
            "Player-additional": ["adamsst01", "adamsst01", "adamsst01"],
            "season_start": [2021, 2022, 2024],
            "Team": ["MEM", "MEM", "HOU"],
        }
    )

    result = add_new_team_flag(df)

    first_season = result.loc[
        (result["Player-additional"] == "adamsst01") & (result["season_start"] == 2021),
        "new_team",
    ].iloc[0]
    second_season = result.loc[
        (result["Player-additional"] == "adamsst01") & (result["season_start"] == 2022),
        "new_team",
    ].iloc[0]
    third_season = result.loc[
        (result["Player-additional"] == "adamsst01") & (result["season_start"] == 2024),
        "new_team",
    ].iloc[0]

    assert first_season is pd.NA
    assert not bool(second_season)
    assert bool(third_season)
    assert is_bool_dtype(result["new_team"])


def test_add_new_team_flag_gap_with_team_change_inj():
    df = pd.DataFrame(
        {
            "Player-additional": ["adamsst01", "adamsst01", "adamsst01", "adamsst01"],
            "season_start": [2021, 2022, 2023, 2024],
            "Team": ["MEM", "MEM", pd.NA, "HOU"],
        }
    )

    result = add_new_team_flag(df)

    first_season = result.loc[
        (result["Player-additional"] == "adamsst01") & (result["season_start"] == 2021),
        "new_team",
    ].iloc[0]
    second_season = result.loc[
        (result["Player-additional"] == "adamsst01") & (result["season_start"] == 2022),
        "new_team",
    ].iloc[0]
    third_season = result.loc[
        (result["Player-additional"] == "adamsst01") & (result["season_start"] == 2024),
        "new_team",
    ].iloc[0]

    assert first_season is pd.NA
    assert not bool(second_season)
    assert bool(third_season)
    assert is_bool_dtype(result["new_team"])


def test_add_new_team_flag_multiple_teams():
    """Test players switching teams within a season"""
    df = pd.DataFrame(
        {
            "Player-additional": ["KD", "KD", "KD", "KD", "KD"],
            "season_start": [2021, 2022, 2022, 2022, 2023],
            "Team": ["BRK", "2TM", "BRK", "PHO", "PHO"],
        }
    )

    result = add_new_team_flag(df)

    kd_2021 = result.loc[
        (result["Player-additional"] == "KD") & (result["season_start"] == 2021),
        "new_team",
    ].iloc[0]
    kd_2022_2tm = result.loc[
        (result["Player-additional"] == "KD")
        & (result["season_start"] == 2022)
        & (result["Team"] == "2TM"),
        "new_team",
    ].iloc[0]
    kd_2022_brk = result.loc[
        (result["Player-additional"] == "KD")
        & (result["season_start"] == 2022)
        & (result["Team"] == "BRK"),
        "new_team",
    ].iloc[0]
    kd_2022_pho = result.loc[
        (result["Player-additional"] == "KD")
        & (result["season_start"] == 2022)
        & (result["Team"] == "PHO"),
        "new_team",
    ].iloc[0]
    kd_2023_pho = result.loc[
        (result["Player-additional"] == "KD") & (result["season_start"] == 2023),
        "new_team",
    ].iloc[0]

    assert kd_2021 is pd.NA
    assert not bool(kd_2022_2tm)
    assert kd_2022_brk is pd.NA
    assert kd_2022_pho is pd.NA
    assert bool(kd_2023_pho)  # TODO: open question if this should be True or False.
