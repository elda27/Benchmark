from model.abstract_trainer import AbstractTrainer
from functools import partial


class SklearnCompatibleMixin:
    def __init__(self, use_predict_proba=True) -> None:
        """Mixin class for scikit learn compatible class such as XGBoost or LightGBM.
        Every sub class should derrive `AbstractTrainer`

        Parameters
        ----------
        use_predict_proba : bool, optional
            If True, the prediction use `predict_proba` instead of `predict`, by default False
        """
        super().__init__()
        self.fit = partial(self.fit, **self.get_training_options())
        if use_predict_proba:
            self.predict = partial(
                self.predict_proba, **self.get_predict_options()
            )
        else:
            self.predict = partial(
                self.predict, **self.get_predict_options()
            )
