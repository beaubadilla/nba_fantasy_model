import pandas as pd

from src.filter import filter_multi_team_stints, filter_years_in_nba_nonzero


def test_filter_multi_team_stints_removes_per_team_rows():
    df = pd.DataFrame(
        [
            {"Player-additional": "p1", "season_start": 2020, "Team": "LAL"},
            {"Player-additional": "p1", "season_start": 2020, "Team": "2TM"},
            {"Player-additional": "p2", "season_start": 2020, "Team": "NYK"},
            {"Player-additional": "p3", "season_start": 2021, "Team": "TOT"},
            {"Player-additional": "p3", "season_start": 2021, "Team": "BOS"},
        ]
    )

    result = filter_multi_team_stints(df)

    expected_teams = ["2TM", "NYK", "TOT"]
    assert result["Team"].tolist() == expected_teams


def test_filter_years_in_nba_nonzero_excludes_rookies():
    df = pd.DataFrame(
        [
            {"Player-additional": "p1", "season_start": 2020, "years_in_nba": 0},
            {"Player-additional": "p1", "season_start": 2021, "years_in_nba": 1},
            {"Player-additional": "p2", "season_start": 2020, "years_in_nba": 2},
            {"Player-additional": "p3", "season_start": 2020, "years_in_nba": pd.NA},
        ]
    )

    result = filter_years_in_nba_nonzero(df)

    expected_years = [1, 2, pd.NA]
    assert result["years_in_nba"].tolist() == expected_years
