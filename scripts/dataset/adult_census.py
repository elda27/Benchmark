from io import StringIO
import numpy as np
from typing import OrderedDict
from typing import Dict, List
from multiprocessing.sharedctypes import Value
import pandas as pd

from dataset.dataset import fetch_data_via_http


def _parse_adult_census_columns(data: bytes) -> OrderedDict[str, type]:
    """get columns and dtypes

    Parameters
    ----------
    data : bytes
        byte data of "adult.names"

    Returns
    -------
     Dict[str, type]
        column and dtypes
    """
    target_lines = filter(
        lambda line: not line.startswith('|') and len(line) > 0,
        data.decode('utf-8').split('\n')
    )
    dtypes = OrderedDict()
    for line in target_lines:
        if line == '>50K, <=50K.':
            continue
        else:
            key, values = line.strip('\r\n.').split(':')
            if values.strip() == 'continuous':
                dtypes[key] = np.float64
            else:
                dtypes[key] = pd.CategoricalDtype([
                    token.strip()
                    for token in values.split(',')
                ])

    dtypes['Target'] = pd.CategoricalDtype(['>50K', '<=50K'])
    return dtypes


def _parse_adult_census_data(data: bytes, is_test_data: bool, dtypes: Dict[str, type]) -> pd.DataFrame:
    """Parse adult census data from byte objects

    Parameters
    ----------
    data : bytes
        binary data
    is_test_data : bool
        If True, the data will parse as test set.
    """
    if is_test_data:
        data = data[data.find(b'\n') + 1:]
        data = data.replace(b'.', b'')
    df = pd.read_csv(
        StringIO(data.decode('utf-8')),
        names=dtypes.keys(),
        skipinitialspace=True,
        dtype=dtypes,
        na_values='?'
    )
    for column, dtype in dtypes.items():
        if df.dtypes[column] != dtype:
            df[column] = df[column].astype(dtype)
    return df


def fetch_adult_census(use_cache=True) -> List[pd.DataFrame]:
    """Fetch adult census dataset.

    Parameters
    ----------
    use_cache : bool, optional
        If True, load cache from local storage. by default True

    Returns
    -------
    List[pd.DataFrame]
        list of dataframe object.
    """
    _prefix = 'adult_census'
    train_data = fetch_data_via_http(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
        prefix=_prefix, use_cache=use_cache
    )
    names_data = fetch_data_via_http(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names',
        prefix=_prefix, use_cache=use_cache
    )
    test_data_file = fetch_data_via_http(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
        prefix=_prefix, use_cache=use_cache
    )

    dtypes = _parse_adult_census_columns(names_data)
    return [
        _parse_adult_census_data(
            train_data, dtypes=dtypes, is_test_data=False),
        _parse_adult_census_data(
            test_data_file, dtypes=dtypes, is_test_data=True),
    ]
