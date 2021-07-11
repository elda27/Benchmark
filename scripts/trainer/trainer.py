from typing import NamedTuple, Optional, Tuple

from catboost import train
from trainer.abstract_trainer import AbstractTrainer
import pandas as pd

DataPair = Tuple[pd.DataFrame, pd.DataFrame]


class Index(NamedTuple):
    train: pd.Index
    valid: pd.Index
    test: Optional[pd.Index]


class Trainer(AbstractTrainer):
    def __init__(self, data: DataPair, index: Index) -> None:
        """Construct trainer and 

        Parameters
        ----------
        data : DataPair
            Dataset for training or evaluation
        index : Index
            index of the datasets
        """
        super().__init__()
        self.data = data
        self.index = index
        self.xs_columns = self.data[0].columns
        self.ys_columns = self.data[1].columns

    @property
    def train_data(self) -> DataPair:
        return tuple(d.loc[self.index.train] for d in self.data)

    @property
    def valid_data(self) -> DataPair:
        return tuple(d.loc[self.index.valid] for d in self.data)

    @property
    def test_data(self) -> DataPair:
        return tuple(d.loc[self.index.test] for d in self.data)

    @property
    def xs(self) -> pd.DataFrame:
        return self.data[0]

    @property
    def ys(self) -> pd.DataFrame:
        return self.data[1]

    @property
    def train_xs(self) -> pd.DataFrame:
        return self.xs.loc[self.index.train]

    @property
    def valid_xs(self) -> pd.DataFrame:
        return self.xs.loc[self.index.valid]

    @property
    def test_xs(self) -> pd.DataFrame:
        return self.xs.loc[self.index.test]

    @property
    def train_ys(self) -> pd.DataFrame:
        return self.ys.loc[self.index.train]

    @property
    def valid_ys(self) -> pd.DataFrame:
        return self.ys.loc[self.index.valid]

    @property
    def test_ys(self) -> pd.DataFrame:
        return self.ys.loc[self.index.test]
