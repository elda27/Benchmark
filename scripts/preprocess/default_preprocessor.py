import numpy as np
from preprocess.preprocessor import Preprocessor
from enum import Enum
from pydantic import BaseModel
import pandas as pd


class CategoricalEncodingType(str, Enum):
    Categorical = 'categorical'
    Onehot = 'onehot'
    Ordinal = 'ordinal'


class DefaultPreprocessorProps(BaseModel):
    categorical_encoding: CategoricalEncodingType


class DefaultPreprocessor(Preprocessor):
    def __init__(self, props: DefaultPreprocessorProps) -> None:
        super().__init__(props)
        self.props: DefaultPreprocessorProps

    def fit(self, xs: pd.DataFrame) -> "DefaultPreprocessor":
        category_columns = [
            col for col, dtype in xs.dtypes.items() if dtype.kind == 'O'
        ]
        cat_df = self.convert_categorical(
            xs[category_columns],
            self.props.categorical_encoding
        )

        value_columns = [
            col for col, dtype in xs.dtypes.items()
            if dtype.kind == 'O'
        ]
        value_df = self.convert_value(value_columns)
        return pd.concat([value_df, cat_df], axis=1)

    def convert_categorical(
        self, df: pd.DataFrame, encoding_type: CategoricalEncodingType
    ) -> pd.DataFrame:
        if encoding_type == CategoricalEncodingType.Categorical:
            return self._convert_categorical_dtype(df)
        elif encoding_type == CategoricalEncodingType.Onehot:
            return self._convert_onehot(df)
        elif encoding_type == CategoricalEncodingType.Ordinal:
            return self._convert_ordinal(df)

    def _convert_categorical_dtype(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            df[col] = df[col].astype('category')
        return df

    def _convert_onehot(self, df: pd.DataFrame) -> pd.DataFrame:
        data = []
        for col in df.columns:
            data.append(pd.get_dummies(df[col]))
        return pd.concat(data, axis=1)

    def _convert_ordinal(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([
            pd.factorize(df[col])[0]
            for col in df.columns
        ], axis=1)
