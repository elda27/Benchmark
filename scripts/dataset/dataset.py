import requests
from pathlib import Path
import gzip
from logging import getLogger

_cache_dir = Path() / '.cache'

_logger = getLogger('benchmarks.utils.dataset')


def make_path(prefix: str, name: str) -> Path:
    """Make path from prefix and name

    Parameters
    ----------
    prefix : str
        prefix of the cache file
    name : str
        string of identification
    Returns
    -------
    Path
        generated path object
    """
    output_dir = _cache_dir / prefix
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir / f'{name}.gz'


def save_cache(prefix: str, name: str, data: bytes):
    """Save cache from file

    Parameters
    ----------
    prefix : str
        prefix of the cache file
    name : str
        string of identification
    data : bytes
        binary data to write
    """
    with gzip.open(make_path(prefix, name), 'wb+') as fp:
        fp.write(data)


def load_cache(prefix: str, name: str) -> bytes:
    """Load cache from file

    Parameters
    ----------
    prefix : str
        prefix of the cache file
    name : str
        string of identification

    Returns
    -------
    bytes
        load binary data
    """
    with gzip.open(make_path(prefix, name), 'rb') as fp:
        return fp.read()


def fetch_data_via_http(url: str, prefix: str, name: str = None, use_cache=True) -> bytes:
    """Fetch data via http

    Parameters
    ----------
    url : str
        string of URL
    prefix : str
        prefix of the cache file
    name : str, optional
        string of identification, if None the value will parse from url on backward, by default None
    use_cache : bool, optional
        If True, load cache from local storage, by default True

    Returns
    -------
    bytes
        binary data

    Raises
    ------
    ConnectionError
        4XX error on HTTP get request.
    """
    name = name or url.split('/')[-1]

    if make_path(prefix, name).exists() and use_cache:
        _logger.info(f'Loading cache: {prefix}/{name}')
        return load_cache(prefix, name)
    else:
        _logger.info(f'Fetching: {url}...')
        data = requests.get(url)
        if data.status_code != 200:
            raise ConnectionError(f'Failed to fetch data: {data.status_code}')
        save_cache(prefix, name, data.content)
        return data.content
