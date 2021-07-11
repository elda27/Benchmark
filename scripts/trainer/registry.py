from typing import NamedTuple, Type, List
from pydantic import BaseModel
from enum import Enum

_trainer_registry = {}


class Registry(NamedTuple):
    name: str
    trainer_type: type
    props_type: Type[BaseModel]


class Task(Enum):
    Regression = 'regression'
    BinaryClassification = 'binary_classification'
    Classification = 'classification'


def get_trainer_list() -> List[str]:
    """Enumerate registered trainer names

    Returns
    -------
    List[str]
        Trainer names
    """
    return list(_trainer_registry.keys())


def get_register_tariner(name: str) -> Registry:
    """Get registry from name

    Parameters
    ----------
    name : str
        Name of trainer

    Returns
    -------
    Registry
        registry object namedtuple of trainer and property types.
    """
    return _trainer_registry[name]


def get_trainer_props_type(name: str) -> Type[BaseModel]:
    return get_register_tariner(name).props_type


def register_trainer(name: str, props: Type[BaseModel]):
    """Register trainer class 

    Parameters
    ----------
    name : str
        name of trainer
    props : Type[BaseModel]
        property class
    """
    def _(klass):
        _trainer_registry[name] = Registry(name, klass, props)
        return klass
    return _
