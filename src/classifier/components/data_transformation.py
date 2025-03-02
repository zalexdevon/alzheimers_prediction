import os
from classifier import logger
import pandas as pd
from classifier.entity.config_entity import DataTransformationConfig
from classifier.Mylib import myfuncs
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    LabelEncoder,
    PolynomialFeatures,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsTransformer, NearestNeighbors
from imblearn.over_sampling import SMOTE


# FEATURE
### PRE
class PreFeatureColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        X = X.drop(columns = ['do_get_Diabetes_bin', 'do_get_Hypertension_bin', 'do_Family_History_bin', 'Education_Level_num'])
        

        self.cols = X.columns.tolist()
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols


### AFTER
class AfterFeatureColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None) -> pd.DataFrame:

        self.cols = X.columns.tolist()
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols


# TARGET
### PRE
class PreTargetColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None) -> pd.DataFrame:

        self.cols = X.columns.tolist()
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols


class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):
        for col in X.columns:
            X[col] = X[col].cat.codes

        self.cols = X.columns.tolist()
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols


class DuringFeatureColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        cols = pd.Series(X.columns)
        numeric_cols = cols[cols.str.endswith("num")].tolist()
        nominal_cols = cols[cols.str.endswith("nom")].tolist()
        ordinal_cols = cols[
            cols.str.endswith("ord") | cols.str.endswith("bin")
        ].tolist()

        nominal_cols_pipeline = Pipeline(
            steps=[
                ("1", OneHotEncoder(sparse_output=False, drop="first")),
                # ("2", StandardScaler()),
            ]
        )

        ordinal_pipeline = Pipeline(
            steps=[
                ("1", CustomOrdinalEncoder()),
                # ("2", StandardScaler()),
            ]
        )

        self.column_transformer = ColumnTransformer(
            transformers=[
                ("1", StandardScaler(), numeric_cols),
                ("2", nominal_cols_pipeline, nominal_cols),
                ("3", ordinal_pipeline, ordinal_cols),
            ],
        )

        self.column_transformer.fit(X)

    def transform(self, X, y=None):
        X = self.column_transformer.transform(X)
        self.cols = myfuncs.get_real_column_name_from_get_feature_names_out(
            self.column_transformer.get_feature_names_out()
        )

        return pd.DataFrame(X, columns=self.cols)  # lỗi chỗ này

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols


class NamedColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_cols, target_col) -> None:
        super().__init__()
        self.feature_cols = feature_cols
        self.target_col = target_col

    def fit(self, X, y=None):

        feature_transformer = Pipeline(
            steps=[
                ("pre", PreFeatureColumnTransformer()),
                ("durig", DuringFeatureColumnTransformer()),
                ("after", AfterFeatureColumnTransformer()),
            ]
        )

        target_transformer = Pipeline(
            steps=[
                ("pre", PreTargetColumnTransformer()),
                ("during", CustomOrdinalEncoder()),
            ]
        )

        self.column_transformer = ColumnTransformer(
            transformers=[
                ("feature", feature_transformer, self.feature_cols),
                ("target", target_transformer, [self.target_col]),
            ]
        )

        self.column_transformer.fit(X)

    def transform(self, X, y=None):
        X = self.column_transformer.transform(X)

        return pd.DataFrame(
            X,
            columns=myfuncs.get_real_column_name_from_get_feature_names_out(
                self.column_transformer.get_feature_names_out()
            ),
        )

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def load_data(self):
        self.df_train = myfuncs.load_python_object(self.config.train_data_path)
        self.df_val = myfuncs.load_python_object(self.config.val_data_path)

        self.num_train_sample = len(self.df_train)

        self.feature_cols = list(
            set(self.df_train.columns) - set([self.config.target_col])
        )

        self.classes = self.df_train[self.config.target_col].cat.categories

    def create_preprocessor_for_train_data(self):
        self.preprocessor = NamedColumnTransformer(
            self.feature_cols, self.config.target_col
        )

    def transform_data(self):
        df = pd.concat([self.df_train, self.df_val], axis=0)

        df_transformed = self.preprocessor.fit_transform(df)

        df_train_transformed = df_transformed.iloc[: self.num_train_sample, :]
        df_val_transformed = df_transformed.iloc[self.num_train_sample :, :]

        df_train_feature = df_train_transformed.drop(columns=[self.config.target_col])
        df_train_target = df_train_transformed[self.config.target_col]

        df_val_feature = df_val_transformed.drop(columns=[self.config.target_col])
        df_val_target = df_val_transformed[self.config.target_col]

        myfuncs.save_python_object(self.config.preprocessor_path, self.preprocessor)
        myfuncs.save_python_object(self.config.classes_path, self.classes)
        myfuncs.save_python_object(self.config.train_features_path, df_train_feature)
        myfuncs.save_python_object(self.config.train_target_path, df_train_target)
        myfuncs.save_python_object(self.config.val_features_path, df_val_feature)
        myfuncs.save_python_object(self.config.val_target_path, df_val_target)
