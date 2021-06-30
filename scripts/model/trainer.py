from model.abstract_trainer import AbstractTrainer
from typing import Dict, Any, List, Union
from pydantic import BaseModel


class Trainer(AbstractTrainer):
    def __init__(self, props: BaseModel) -> None:
        super().__init__()
        self.props = props

    def _get_configs_from_keys(self, keys: List[str]) -> Dict[str, Any]:
        return self.props.dict(include=keys)

    def get_config_options(self) -> Dict[str, Any]:
        return self._get_config_from_keys(self._config_keys)

    def get_training_options(self) -> Dict[str, Any]:
        return self._get_config_from_keys(self._training_keys)

    def get_predict_options(self) -> Dict[str, Any]:
        return self._get_config_from_keys(self._predict_keys)
