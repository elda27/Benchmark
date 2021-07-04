from abc import ABCMeta
from abc import ABCMeta, abstractmethod
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
from model.runtime_params import RuntimeParams


class AbstractTrainer(metaclass=ABCMeta):
    @abstractmethod
    def get_runtime_options(self) -> Dict[RuntimeParams, Any]:
        """Get runtime parameters

        Returns
        -------
        Dict[RuntimeParams, Any]
            configuration parameters
        """
        raise NotImplementedError()

    @abstractmethod
    def get_config_options(self) -> Dict[str, Any]:
        """Get initialize parameters

        Returns
        -------
        Dict[str, Any]
            configuration parameters
        """
        raise NotImplementedError()

    @abstractmethod
    def get_training_options(self) -> Dict[str, Any]:
        """Get training parameters

        Returns
        -------
        Dict[str, Any]
            parameters
        """
        raise NotImplementedError()

    @abstractmethod
    def get_predict_options(self) -> Dict[str, Any]:
        """Get prediction parameters

        Returns
        -------
        Dict[str, Any]
            prediction parameters
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate_contribute(
        self, xs: Union[np.ndarray, pd.DataFrame]
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Calcualate contribution of prediction from input features.

        Parameters
        ----------
        xs : Union[np.ndarray, pd.DataFrame]
            Input features

        Returns
        -------
        Union[np.ndarray, pd.DataFrame, List[pd.DataFrame]]
            Calculated contributions of prediction.
            If the trained task is multiclass classification, the contribution 

        Raises
        ------
        NotImplementedError
            [description]
        """
        raise NotImplementedError()
