from turtle import register_shape
from pydantic import BaseModel
from pydantic.generics import GenericModel
from typing import Generic, Tuple, TypeVar, Type
from trainer.registry import get_trainer_props_type, get_register_tariner
from trainer.abstract_trainer import AbstractTrainer
from preprocess.preprocessor import Preprocessor
from preprocess.registry import get_preprocessor_props_type
from integrate.registry_model import get_model_registry, get_model_props_type

TrainerPropsT = TypeVar('TrainerPropsT')
PreprocessPropsT = TypeVar('PreprocessPropsT')
ModelPropsT = TypeVar('ModelPropsT')


class Props(GenericModel, Generic[TrainerPropsT, PreprocessPropsT, ModelPropsT]):
    model_name: str
    dataset_name: str

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


def build_model(props: Type[Props]) -> Tuple[AbstractTrainer, Preprocessor]:
    data = get_model_registry(props.model_name)
    preprocessor_type = get_preprocessor_props_type(data.preprocessor_name)
    preprocessor = preprocessor_type(props.preprocess_props)
    trainer_type = get_register_tariner(data.trainer_name)
    if data.model_props is not None:
        model = data.model_type(data.model_props)
        trainer = trainer_type(model, props.trainer_props)
    else:
        trainer = trainer_type(props.trainer_props)
    return trainer, preprocessor
