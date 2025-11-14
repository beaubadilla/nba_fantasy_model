import pandas as pd

from src.utils.dataframe_formatting import round_continuous_features


def test_round_continuous_features_rounds_float_columns_only():
    df = pd.DataFrame(
        {
            "float_col": [1.234, 2.345],
            "int_col": [1, 2],
            "str_col": ["a", "b"],
        }
    )

    result = round_continuous_features(df, decimals=1)

    assert result["float_col"].tolist() == [1.2, 2.3]
    assert result["int_col"].tolist() == [1, 2]
    assert result["str_col"].tolist() == ["a", "b"]


def test_round_continuous_features_preserves_input_dataframe():
    df = pd.DataFrame({"float_col": [1.234, 5.678]})

    round_continuous_features(df, decimals=1)

    assert df["float_col"].tolist() == [1.234, 5.678]
