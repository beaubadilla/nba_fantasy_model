import os
import pandas as pd
from typing import List

def add_years_in_nba(df: pd.DataFrame, id_col: str = 'Player-additional') -> pd.DataFrame:
    df = df.sort_values(by=[id_col, 'season_start']).copy()
    df['years_in_nba'] = df.groupby(id_col).cumcount()
    return df

def add_career_per_game_features(
    df: pd.DataFrame,
    per_game_cols: List[str],
    gp_col: str = 'G',
    id_col: str = 'Player-additional'
) -> pd.DataFrame:
    df = df.sort_values(by=[id_col, 'season_start']).copy()

    for col in per_game_cols:
        total_col = f'total_{col}'
        df[total_col] = df[col] * df[gp_col]

    # Running totals and GP up to previous season
    for col in per_game_cols:
        total_col = f'total_{col}'
        running_total = df.groupby(id_col)[total_col].cumsum() - df[total_col]
        running_gp = df.groupby(id_col)[gp_col].cumsum() - df[gp_col]
        df[f'career_{col}_pg'] = running_total / running_gp

    return df


def add_fantasy_points_per_game(
    df: pd.DataFrame,
    *,
    points_col: str = 'PTS',
    assists_col: str = 'AST',
    rebounds_col: str = 'TRB',
    blocks_col: str = 'BLK',
    steals_col: str = 'STL',
    turnovers_col: str = 'TOV',
    output_col: str = 'fantasy_points_per_game'
) -> pd.DataFrame:
    """Compute fantasy points per game using standard scoring weights."""
    required = {
        points_col,
        assists_col,
        rebounds_col,
        blocks_col,
        steals_col,
        turnovers_col,
    }
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f'DataFrame missing required columns: {sorted(missing)}')

    result = df.copy()
    to_numeric = lambda series: pd.to_numeric(series, errors='coerce')

    result[output_col] = (
        to_numeric(result[points_col]) * 1.0
        + to_numeric(result[assists_col]) * 1.5
        + to_numeric(result[rebounds_col]) * 1.2
        + to_numeric(result[blocks_col]) * 2.0
        + to_numeric(result[steals_col]) * 2.0
        - to_numeric(result[turnovers_col]) * 1.0
    )

    return result


def add_previous_season_lag_features(
    df: pd.DataFrame,
    lag_cols: List[str],
    *,
    id_col: str = 'Player-additional',
    season_col: str = 'season_start',
    team_col: str = 'Team'
) -> pd.DataFrame:
    """Create lag (previous season) columns for the specified stats."""
    if not lag_cols:
        return df.copy()

    required = {id_col, season_col, team_col, *lag_cols}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f'DataFrame missing required columns: {sorted(missing)}')

    result = df.copy()
    result['_row_order'] = range(len(result))

    season_numeric = pd.to_numeric(result[season_col], errors='coerce')
    if season_numeric.isna().any():
        raise ValueError(f'Unable to convert some values in {season_col} to numeric seasons.')
    result['_season_numeric'] = season_numeric.astype(int)

    team_series = result[team_col].astype(str).str.upper()
    aggregate_mask = team_series.str.endswith('TM') | team_series.eq('TOT')
    aggregate_mask &= result[team_col].notna()
    result['_is_aggregate'] = aggregate_mask

    sort_columns = [id_col, '_season_numeric', '_is_aggregate', '_row_order']
    season_rep = (
        result.sort_values(sort_columns, ascending=[True, True, False, True], kind='mergesort')
        .groupby([id_col, '_season_numeric'], as_index=False)
        .first()
    )

    season_rep.sort_values(by=[id_col, '_season_numeric'], inplace=True)

    lag_column_names = []
    for col in lag_cols:
        lag_col = f'lag1_{col}'
        season_rep[lag_col] = season_rep.groupby(id_col)[col].shift(1)
        lag_column_names.append(lag_col)

    merge_columns = [id_col, '_season_numeric'] + lag_column_names
    result = result.merge(season_rep[merge_columns], on=[id_col, '_season_numeric'], how='left')

    result.sort_values(by='_row_order', inplace=True)
    result.drop(columns=['_row_order', '_season_numeric', '_is_aggregate'], inplace=True)

    return result


def add_new_team_flag(
    df: pd.DataFrame,
    id_col: str = 'Player-additional',
    team_col: str = 'Team',
    season_col: str = 'season_start'
) -> pd.DataFrame:
    """Flag whether a player is on a new team compared to their previous season."""
    required = [id_col, team_col, season_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f'DataFrame missing required columns: {missing}')

    result = df.copy()

    # Identify aggregate team codes that denote multi-team seasons (e.g., '2TM', 'TOT').
    aggregate_codes = {'TOT'}
    aggregate_mask = result[team_col].astype(str).str.upper().str.endswith('TM')
    aggregate_mask |= result[team_col].astype(str).str.upper().isin(aggregate_codes)

    # Representative team for each player-season (aggregate row if available, else first occurrence).
    temp = result[[id_col, season_col, team_col]].copy()
    temp['_is_aggregate'] = aggregate_mask

    temp_sorted = temp.sort_values(
        by=[id_col, season_col, '_is_aggregate'],
        ascending=[True, True, False],
        kind='mergesort'
    )

    season_team = (
        temp_sorted.groupby([id_col, season_col], sort=False)
        .first()
        .reset_index()[[id_col, season_col, team_col]]
        .rename(columns={team_col: '_season_team'})
    )

    season_team_sorted = season_team.sort_values(by=[id_col, season_col], kind='mergesort')
    season_team_sorted['_previous_team'] = season_team_sorted.groupby(id_col)['_season_team'].transform(
        lambda group: group.ffill().shift()
    )
    previous_team_lookup = season_team_sorted[[id_col, season_col, '_previous_team']]

    result = result.merge(
        previous_team_lookup,
        how='left',
        on=[id_col, season_col]
    )

    season_counts = result.groupby([id_col, season_col])[team_col].transform('count')
    new_team = pd.Series(pd.NA, index=result.index, dtype='boolean')

    has_prev = result['_previous_team'].notna()
    is_aggregate_row = aggregate_mask
    is_unique_season = season_counts == 1
    has_team_value = result[team_col].notna()

    # Aggregate (multi-team) rows compare to previous season but are never treated as a new team.
    aggregate_comparable = has_prev & is_aggregate_row
    new_team.loc[aggregate_comparable] = False

    # Single-team seasons compare to previous season normally.
    comparable_single = has_prev & is_unique_season & (~is_aggregate_row) & has_team_value
    new_team.loc[comparable_single] = (
        result.loc[comparable_single, team_col] != result.loc[comparable_single, '_previous_team']
    )

    result['new_team'] = new_team
    result.drop(columns=['_previous_team', '_is_aggregate'], inplace=True, errors='ignore')

    return result


def _season_label_to_year(season_label: str) -> int:
    """Convert a season label like '22_23' into its starting season year (e.g. 2022)."""
    if not season_label or not isinstance(season_label, str):
        raise ValueError('Season label must be a non-empty string')

    normalized = season_label.strip().replace('-', '_')
    parts = normalized.split('_')
    if not parts or not parts[0]:
        raise ValueError(f'Invalid season label: {season_label}')

    start_fragment = parts[0]
    if len(start_fragment) == 4:
        return int(start_fragment)

    start_val = int(start_fragment)
    century = 1900 if start_val >= 50 else 2000
    return century + start_val


def add_new_coach_flag(
    df: pd.DataFrame,
    team_col: str = 'Team',
    season_col: str = 'season_start',
    player_col: str | None = 'Player-additional',
    coaches_max_path: str = os.path.join('data', 'interim', 'coaches_by_team_max_games.csv'),
    coaches_first_path: str = os.path.join('data', 'interim', 'coaches_by_team_first_only.csv')
) -> pd.DataFrame:
    """Flag whether a team has a different head coach compared to the player's most recent previous season with that team."""
    if team_col not in df.columns or season_col not in df.columns:
        missing = [col for col in (team_col, season_col) if col not in df.columns]
        raise KeyError(f'DataFrame missing required columns: {missing}')

    coaches_max = pd.read_csv(coaches_max_path)
    coaches_first = pd.read_csv(coaches_first_path)

    previous_lookup = coaches_max[['Tm', 'Season', 'Coach']].copy()
    previous_lookup['season_year'] = previous_lookup['Season'].apply(_season_label_to_year)
    previous_map = {
        (row['Tm'], row['season_year']): row['Coach']
        for _, row in previous_lookup.iterrows()
    }

    current_lookup = coaches_first[['Tm', 'Season', 'Coach']].copy()
    current_lookup['season_year'] = current_lookup['Season'].apply(_season_label_to_year)
    current_map = {
        (row['Tm'], row['season_year']): row['Coach']
        for _, row in current_lookup.iterrows()
    }

    has_player_col = bool(player_col and player_col in df.columns)

    result = df.copy()
    result['_row_order'] = range(len(result))

    season_numeric = pd.to_numeric(result[season_col], errors='coerce')
    if season_numeric.isna().any():
        raise ValueError(f'Unable to convert some values in {season_col} to numeric seasons.')
    season_numeric = season_numeric.astype(int)
    result['_season_numeric'] = season_numeric

    team_values = result[team_col].astype(str).str.upper()
    aggregate_codes = {'TOT'}
    aggregate_mask = result[team_col].notna() & (
        team_values.str.endswith('TM') | team_values.isin(aggregate_codes)
    )
    result['_is_aggregate'] = aggregate_mask

    season_group_keys: List[str] = []

    if has_player_col:
        season_group_keys.append(player_col)

    season_group_keys.append('_season_numeric')

    # Capture the list of actual teams a player suited up for in each season.
    if has_player_col:
        actual_rows = result.loc[
            (~aggregate_mask) & result[team_col].notna(),
            season_group_keys + [team_col, '_row_order']
        ].copy()

        if not actual_rows.empty:
            actual_rows.sort_values(by='_row_order', inplace=True)
            season_team_info = (
                actual_rows.groupby(season_group_keys, as_index=False)
                .agg({team_col: lambda s: list(dict.fromkeys(s.tolist()))})
                .rename(columns={team_col: '_season_team_list'})
            )

            season_team_info['_season_first_team'] = season_team_info['_season_team_list'].apply(
                lambda teams: teams[0] if teams else None
            )

            season_team_info.sort_values(by=season_group_keys, inplace=True)
            season_team_info['_previous_team_list'] = (
                season_team_info.groupby(season_group_keys[0])['_season_team_list'].shift(1)
            )

            result = result.merge(
                season_team_info,
                on=season_group_keys,
                how='left'
            )
        else:
            result['_season_team_list'] = pd.NA
            result['_season_first_team'] = pd.NA
            result['_previous_team_list'] = pd.NA
    else:
        result['_season_team_list'] = pd.NA
        result['_season_first_team'] = pd.NA
        result['_previous_team_list'] = pd.NA
    # Determine the team to use for coach lookups (aggregate rows adopt the first real team).
    result['_comparison_team'] = result[team_col]
    if has_player_col:
        aggregate_with_actual = result['_is_aggregate'] & result['_season_first_team'].notna()
        result.loc[aggregate_with_actual, '_comparison_team'] = result.loc[
            aggregate_with_actual, '_season_first_team'
        ]

    comparison_keys: List[str] = []
    if has_player_col:
        comparison_keys.append(player_col)
    comparison_keys.append('_comparison_team')

    history_columns = comparison_keys + ['_season_numeric', '_row_order']
    history = result.loc[result['_comparison_team'].notna(), history_columns].copy()

    if history.empty:
        result['_previous_season'] = pd.NA
    else:
        history.sort_values(by=comparison_keys + ['_season_numeric', '_row_order'], inplace=True)
        dedup_subset = comparison_keys + ['_season_numeric']
        history_unique = history.drop_duplicates(subset=dedup_subset, keep='first').copy()
        history_unique['_previous_season'] = history_unique.groupby(comparison_keys)['_season_numeric'].shift(1)

        merge_columns = comparison_keys + ['_season_numeric']
        result = result.merge(
            history_unique[merge_columns + ['_previous_season']],
            on=merge_columns,
            how='left'
        )

    result.sort_values(by='_row_order', inplace=True)
    result['_fallback_season'] = result['_season_numeric'] - 1

    current_coach = [
        current_map.get((team, int(season)))
        if pd.notna(team) and pd.notna(season) else None
        for team, season in zip(result['_comparison_team'], result['_season_numeric'])
    ]
    previous_coach = [
        (
            previous_map.get((team, int(prev_season)))
            if pd.notna(prev_season) else None
        )
        if pd.notna(team) else None
        for team, prev_season in zip(result['_comparison_team'], result['_previous_season'])
    ]

    # Try fallback to the immediately preceding season where no player-specific previous season exists.
    previous_team_lists = result['_previous_team_list']
    for idx, (team, coach, prev_season, fallback, team_list) in enumerate(
        zip(
            result['_comparison_team'],
            previous_coach,
            result['_previous_season'],
            result['_fallback_season'],
            previous_team_lists
        )
    ):
        if coach is None and pd.notna(team):
            if has_player_col:
                valid_prev_team = isinstance(team_list, (list, tuple)) and team in team_list
            else:
                valid_prev_team = True

            if pd.notna(fallback) and valid_prev_team:
                candidate = previous_map.get((team, int(fallback)))
                if candidate is not None:
                    previous_coach[idx] = candidate

    new_coach = pd.Series(pd.NA, index=result.index, dtype='boolean')

    current_series = pd.Series(current_coach, index=result.index, dtype='object')
    previous_series = pd.Series(previous_coach, index=result.index, dtype='object')

    comparable_mask = current_series.notna() & previous_series.notna()
    new_coach.loc[comparable_mask] = current_series.loc[comparable_mask] != previous_series.loc[comparable_mask]

    result['new_coach'] = new_coach

    result.drop(
        columns=[
            '_season_numeric',
            '_row_order',
            '_previous_season',
            '_fallback_season',
            '_comparison_team',
            '_season_team_list',
            '_previous_team_list',
            '_season_first_team',
            '_is_aggregate'
        ],
        inplace=True,
        errors='ignore'
    )

    return result
