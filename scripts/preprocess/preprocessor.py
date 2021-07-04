from abc import ABCMeta, abstractmethod
from typing import Type
import pandas as pd
from pydantic import BaseModel
from sklearn.base import TransformerMixin


class Preprocessor:
    def __init__(self, props: Type[BaseModel]) -> None:
        self.props = props

    def transform(self, xs: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def fit(self, xs: pd.DataFrame) -> "Preprocessor":
        raise NotImplementedError()

    def fit_transform(self, xs: pd.DataFrame):
        return self.fit(xs).transform(xs)
