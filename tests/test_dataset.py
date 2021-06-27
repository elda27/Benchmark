import pytest
from utils.dataset import fetch_adult_census


def test_fetch_adult_census():
    df_list = fetch_adult_census()
    assert len(df_list) == 2
    for df in df_list:
        for column in df.columns:
            count = df[column].isna().value_counts().get(False, 0)
            assert count > len(df) * 0.90, \
                f'Too many elements missing at Column:"{column}"'


if __name__ == '__main__':
    pytest.main(__file__)
