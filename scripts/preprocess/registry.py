from typing import NamedTuple, Type, List
from pydantic import BaseModel


_preprocessor_registry = {}


class PreprocessRegistry(NamedTuple):
    name: str
    preprocess_type: type
    props_type: Type[BaseModel]


def get_preprocessor_list() -> List[str]:
    """Enumerate registered preprocessor names

    Returns
    -------
    List[str]
        preprocessor names
    """
    return list(_preprocessor_registry.keys())


def get_register_preprocessor(name: str) -> PreprocessRegistry:
    """Get registry from name

    Parameters
    ----------
    name : str
        Name of preprocessor

    Returns
    -------
    Registry
        registry object namedtuple of preprocessor and property types.
    """
    return _preprocessor_registry[name]


def register_preprocessor(name: str, props: Type[BaseModel]):
    """Register preprocessor class 

    Parameters
    ----------
    name : str
        name of preprocessor
    props : Type[BaseModel]
        property class
    """
    def _(klass):
        _preprocessor_registry[name] = PreprocessRegistry(name, klass, props)
        return klass
    return _
