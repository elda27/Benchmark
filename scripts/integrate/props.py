from pydantic import BaseModel
from pydantic.generics import GenericModel
from typing import Generic, TypeVar, Type
from integrate.registry_model import get_model_registry
from trainer.registry import get_trainer_props_type
from preprocess.registry import get_preprocessor_props_type

TrainerPropsT = TypeVar('TrainerPropsT')
PreprocessPropsT = TypeVar('PreprocessPropsT')
ModelPropsT = TypeVar('ModelPropsT')


class Props(GenericModel, Generic[TrainerPropsT, PreprocessPropsT, ModelPropsT]):
    dataset_type: str
    model_type: str

    trainer_props: TrainerPropsT
    preprocess_props: PreprocessPropsT
    model_props: ModelPropsT


def build_props_type(model_name: str, **kwargs) -> Type[BaseModel]:
    registry = get_model_registry(model_name)

    return Props[
        get_trainer_props_type(registry.trainer_name),
        get_preprocessor_props_type(registry.preprocessor_name),
        get_model_props_type(registry.model_name),
    ]
