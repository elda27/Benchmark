from pydantic import BaseModel
from typing import Dict, Type, NamedTuple


class ModelRegistry(NamedTuple):
    model_name: str
    trainer_name: str
    preprocessor_name: str
    model_props: Type[BaseModel]


_model_registry: Dict[str, ModelRegistry] = {}


def register_model(
    model_name: str,
    trainer_name: str,
    preprocessor_name: str,
    model_props: Type[BaseModel]
):
    def _(klass):
        _model_registry[model_name] = ModelRegistry(
            model_name, trainer_name, preprocessor_name, model_props
        )
        return klass
    return _


def get_model_registry(name: str) -> ModelRegistry:
    return _model_registry[name]


def get_model_props_type(name: str) -> Type[BaseModel]:
    return get_model_registry(name).model_props
