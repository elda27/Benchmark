from typing import List, Any, Dict, Literal, Optional, Union

import pydantic
import numpy as np
import pandas as pd
from xgboost import XGBModel

from model.trainer import Trainer
from model.registry import register_trainer
from model.sklearn_compatible_mixin import SklearnCompatibleMixin


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


@register_trainer('xgboost', XGBoostModelProps)
class XGBTrainer(XGBModel, Trainer, SklearnCompatibleMixin):
    _config_keys = [
        'booster',
        'verbosity',
        'max_depth',
        'learning_rate',
        'n_estimators',
        'objective',
        'tree_method',
        'n_jobs',
        'gamma',
        'min_child_weight',
        'max_delta_step',
        'subsample',
        'colsample_bytree',
        'colsample_bylevel',
        'colsample_bynode',
        'reg_alpha',
        'reg_lambda',
        'scale_pos_weight',
        'num_parallel_tree',
    ]
    _training_keys = [
        'metric'
    ]
    _predict_keys = []
    _runtime_keys = []

    def __init__(self, props: Optional[XGBoostModelProps]) -> None:
        self.props = props
        super(Trainer, self).__init__(props)
        super(XGBModel, self).__init__(**self.get_config_options())
        super(SklearnCompatibleMixin, self).__init__()

    def evaluate_contribute(
        self, xs: Union[np.ndarray, pd.DataFrame]
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        contribs = self.get_booster().predict(xs, pred_contribs=True)
        if isinstance(xs, pd.DataFrame):
            def wrap_df(contrib):
                return pd.DataFrame(
                    contrib,
                    columns=list(xs.columns) + ['$Baseline']
                )
            if contribs.ndim == 3:
                return [
                    wrap_df(contrib)
                    for contrib in contribs
                ]
            else:
                return wrap_df(contribs)
        else:
            return contribs
