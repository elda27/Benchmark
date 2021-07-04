# Summary

This script consists of 4 "core" modules in the following.

- dataset
- model
- preprocess
- evaluation

These modules have a registry of processing objects called with "registry" for command-line processing.
The "registry" will register by the decorator such as the following.

```python
from model.registry import register_trainer
@register_trainer('xgboost', XGBoostModelProps)
class XGBTrainer(XGBModel, Trainer, SklearnCompatibleMixin):
    ...
```

A `register_trainer` will register `XGBTrainer` and it will be constructed by command line args.
If you want to a new Trainer, you may define the class and `register_trainer`.

Arguments of `register_*` is different with each "core" module, so please refer to the document of source code.

# model

## Trainer and Props

This is the main feature of the "model" module.
It consists of Trainer and Props.
A "Trainer" class has three features.

- Training
- Prediction
- Predicting contribution of input features

The "Props" is a parameter of the training model.
It can be easy to convert `dict` object or JSON file so that it is derived `pydantic.BaseModel`.

### Detail of the Trainer

`Trainer` is a wrapper for scikit-learn, xgboost or TensorFlow models derived `AbstractTrainer`.
Almost all the implementation can be found a mixin class such as `SklearnCompatibleMixin`.
That is applied monkey patch to the `predict` method for obtaining predicting probability instead of predicting class index.

## Registry

`Registry` object contains a pair of Trainer and Props registered by `register_trainer`.
The `Registry` is obtained by `get_register_tariner`.

# preprocess

Almost architecture is similar to the "model" module.
This section is described only the difference.

## Preprocessor

The preprocessor has a process method to make a training dataset.
Every preprocessor has a `transform` method and it accepts `pandas.DataFrame` and return preprocessed `pandas.DataFrame`.

# evaluation

The evaluation module has metric of prediction results.
