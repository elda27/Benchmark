from abc import ABCMeta, abstractmethod
from ctypes import Union
from typing import List
import pandas as pd


class AbstractTrainer:
    @abstractmethod
    def get_config(self) -> dict:
        """Get training configuration as a json convertible dictionary

        Returns
        -------
        dict
            training configuraiton
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        """Training prediction model
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, xs: pd.DataFrame) -> pd.DataFrame:
        """Predict from input feature using trained model

        Parameters
        ----------
        xs : pd.DataFrame
            input feature

        Returns
        -------
        pd.DataFrame
            prediction value
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate_contribution(self, xs: pd.DataFrame) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Calcualate predicting contribution from input features.

        Parameters
        ----------
        xs : Union[np.ndarray, pd.DataFrame]
            Input features

        Returns
        -------
        Union[pd.DataFrame, List[pd.DataFrame]]
            Calculated contributions of prediction.
            If the trained task is multiclass classification, the contribution 
        """
        raise NotImplementedError()
