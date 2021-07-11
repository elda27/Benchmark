from gc import callbacks
from typing import List, Any, Dict, Literal, Optional, Union
from lightgbm import early_stopping

import pydantic
import numpy as np
import pandas as pd
from xgboost import XGBModel

from trainer.trainer import Trainer
from trainer.registry import register_trainer, Task


class XGBoostModelProps(pydantic.BaseModel):
    booster: Literal['gbtree', 'gblinear', 'dart'] = 'gbtree'
    verbosity: Optional[int] = None
    max_depth: int = None
    learning_rate: float = None
    n_estimators: int = 100
    objective = None
    tree_method = None
    n_jobs = None
    gamma = None
    min_child_weight = None
    max_delta_step = None
    subsample = None
    colsample_bytree = None
    colsample_bylevel = None
    colsample_bynode = None
    reg_alpha: Optional[float] = None
    reg_lambda: Optional[float] = None
    scale_pos_weight: float = None
    num_parallel_tree: Optional[int] = None


class XGBTrainerProps:
    early_stopping_rounds: int


@register_trainer('xgboost', XGBTrainerProps)
class XGBTrainer(Trainer):
    def __init__(self, task: Task, props: Optional[XGBTrainerProps]) -> None:
        self.props = props
        super(Trainer, self).__init__(props)
        self.task = task
        if task == Task.Regression:
            self.objective = 'reg:squarederror'
            self.metric = 'mae'
        if task == Task.Classification:
            self.objective = 'binary:logistic'
            self.metric = 'error'
        if task == Task.BinaryClassification:
            self.objective = 'multi:softmax'
            self.metric = 'merror'
        self.models = [
            XGBModel(objective=self.objective, **self.props.dict())
            for _ in self.ys_columns
        ]

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
