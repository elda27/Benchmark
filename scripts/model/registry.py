from typing import NamedTuple, Type
from pydantic import BaseModel

_model_registry = {}


class Registry(NamedTuple):
    name: str
    trainer_type: type
    props_type: Type[BaseModel]


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
        _model_registry[name] = Registry(name, klass, props)
        return klass
    return _
