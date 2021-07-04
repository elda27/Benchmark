from model.abstract_trainer import AbstractTrainer
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel

from model.runtime_params import RuntimeParams


class Trainer(AbstractTrainer):
    _config_keys: List[str]
    _training_keys: List[str]
    _predict_keys: List[str]
    _runtime_keys: List[RuntimeParams]

    def __init__(self, props: Optional[BaseModel]) -> None:
        super().__init__()
        self.props = props
        self._runtime_options = {}

    def get_required_runtime_options(self) -> List[RuntimeParams]:
        """Get requiring list of runtime options

        Returns
        -------
        List[RuntimeParams]
            runtime options
        """
        return self._runtime_options

    def add_runtime_option(self, key: RuntimeParams, value: Any):
        """Add runtime option

        Parameters
        ----------
        key : RuntimeParams
            [description]
        value : Any
            [description]
        """
        self._runtime_options[key] = value

    def _get_configs_from_keys(self, keys: List[str]) -> Dict[str, Any]:
        if self.props is None:
            return {}
        else:
            return self.props.dict(include=keys)

    def get_runtime_options(self) -> Dict[RuntimeParams, Any]:
        return self._runtime_options

    def get_config_options(self) -> Dict[str, Any]:
        return self._get_config_from_keys(self._config_keys)

    def get_training_options(self) -> Dict[str, Any]:
        return self._get_config_from_keys(self._training_keys)

    def get_predict_options(self) -> Dict[str, Any]:
        return self._get_config_from_keys(self._predict_keys)
