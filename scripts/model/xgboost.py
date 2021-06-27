from typing import List, Literal, Optional

import pydantic
import pandas as pd
from xgboost import XGBModel


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

    metric: Literal['mean_absolute_error']


class XGBTrainer:
    def __init__(self, props: XGBoostModelProps) -> None:
        self.props = props
        self.model = XGBModel(
            **props.dict()
        )
