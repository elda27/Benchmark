from abc import ABCMeta
from sklearn.base import ClassifierMixin, RegressorMixin
from abc import ABCMeta, abstractmethod
from typing import Dict, Any, Union
import pandas as pd
import numpy as np


class AbstractTrainer(metaclass=ABCMeta):
    @abstractmethod
    def get_fit_options(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def get_evaluate_options(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def evaluate_contribute(self, xs: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        raise NotImplementedError()
