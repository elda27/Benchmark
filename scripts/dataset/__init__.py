from dataset.adult_census import fetch_adult_census
import pandas as pd
from typing import List

_datasets = {
    'adult': fetch_adult_census
}


def get_dataset_names() -> List[str]:
    """Get list of dataset names

    Returns
    -------
    List[str]
        name of datasets
    """
    return list(_datasets.keys())


def get_dataset(name: str, use_cache: bool = True) -> List[pd.Dataset]:
    """Get dataset from name

    Parameters
    ----------
    name : str
        name of dataset
    use_cache : bool, optional
        If True, use local storage., by default True

    Returns
    -------
    List[pd.Dataset]
        Fetched dataset
    """
    return _datasets[name](use_cache)
