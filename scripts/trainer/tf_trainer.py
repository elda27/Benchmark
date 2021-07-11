from gc import callbacks
from typing import List, Any, Dict, Literal, Optional, Union
from lightgbm import early_stopping

import pydantic
import numpy as np
import pandas as pd
import tensorflow as tf

from trainer.trainer import Trainer
from trainer.registry import register_trainer, Task


class CommonProps(pydantic.BaseModel):
    using_device: str = '/GPU:0'
    model_type: str = 'TabNet'
    learning_rate: float = 1e-4
    batch_size: int = 64
    n_epochs: int = 20000
    early_stopping_epoch: int = 4000
    n_eval_freq: int = 10


class TFTrainer(Trainer):
    def __init__(self, model: tf.keras.Model, props: Optional[CommonProps]) -> None:
        self.props = props
        super(Trainer, self).__init__(props)

    def get_config(self) -> dict:
        d = self.props.dict()
        d['task'] = self.task
        return d

    def train(self):
        for ys_column, model in zip(self.ys_columns, self.models):
            model.fit(
                self.train_xs, self.train_ys[ys_column],
                eval_set=(self.valid_xs, self.valid_ys[ys_column]),
                early_stopping=self.props.early_stopping_rounds,
                callbacks=[]
            )

    def evaluate(self, xs: pd.DataFrame) -> pd.DataFrame:
        result = {}
        for model, col in zip(self.models, self.ys_columns):
            result[col] = model.predict(xs)
        return pd.DataFrame(result, index=xs.index)

    def evaluate_contribution(
        self, xs: Union[np.ndarray, pd.DataFrame]
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        results = []
        for model in self.models:
            contribs = model.get_booster().predict(xs, pred_contribs=True)
            results.append(pd.DataFrame(
                contribs,
                columns=list(xs.columns) + ['$Baseline']
            ))
        if len(self.models) > 1:
            return results
        else:
            return results[0]
