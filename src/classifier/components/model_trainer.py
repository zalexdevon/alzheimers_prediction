import pandas as pd
import os
from classifier import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit, GridSearchCV
from classifier.entity.config_entity import ModelTrainerConfig
from classifier.Mylib import myfuncs
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from xgboost import XGBClassifier
from scipy.stats import randint
import random
from lightgbm import LGBMClassifier
from sklearn.model_selection import ParameterSampler
from sklearn import metrics
from sklearn.base import clone
import time


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train_randomised_train_val(self):
        random_search = RandomizedSearchCV(
            self.base_model,
            param_distributions=self.config.param_grid_model,
            n_iter=self.config.N_ITER,
            cv=self.trainval_splitter,
            random_state=42,
            scoring=self.config.metric,
            return_train_score=True,  # Lưu lại điểm train trong kết quả
            verbose=2,
        )

        self.find_best_model_and_save_result(random_search)

    def train_grid_train_val(self):
        grid_search = GridSearchCV(
            self.base_model,
            param_grid=self.config.param_grid_model,
            cv=self.trainval_splitter,
            scoring=self.config.metric,
            return_train_score=True,  # Lưu lại điểm train trong kết quả
            verbose=2,
        )

        self.find_best_model_and_save_result(grid_search)

    def train_randomisedcv(self):
        random_search = RandomizedSearchCV(
            self.base_model,
            param_distributions=self.config.param_grid_model,
            n_iter=self.config.N_ITER,
            cv=5,
            random_state=42,
            scoring=self.config.metric,
            return_train_score=True,  # Lưu lại điểm train trong kết quả
            verbose=2,
        )

        self.find_best_model_and_save_result(random_search)

    def train_gridcv(self):
        grid_search = GridSearchCV(
            self.base_model,
            param_grid=self.config.param_grid_model,
            cv=5,
            scoring=self.config.metric,
            return_train_score=True,  # Lưu lại điểm train trong kết quả
            verbose=2,
        )

        self.find_best_model_and_save_result(grid_search)

    def find_best_model_and_save_result(self, searcher):
        searcher.fit(self.features, self.target)
        best_model = searcher.best_estimator_
        cv_results = searcher.cv_results_

        scores = list(
            zip(cv_results["mean_train_score"], cv_results["mean_test_score"])
        )
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        self.update_best_val_score(scores)

        myfuncs.save_python_object(self.config.best_model_path, best_model)

    def update_best_val_score(self, scores):
        while True:
            if self.config.metric == "accuracy":
                self.best_val_accuracy = scores[0][1] * 100
                self.train_accuracy_follow_best_val = scores[0][0] * 100
                return

            if self.config.metric == "neg_log_loss":
                self.best_val_accuracy = -scores[0][1]
                self.train_accuracy_follow_best_val = -scores[0][0]
                return

            return

    def load_data_to_train(self):
        self.train_feature_data = myfuncs.load_python_object(
            self.config.train_feature_path
        )
        self.train_target_data = myfuncs.load_python_object(
            self.config.train_target_path
        )
        self.val_feature_data = myfuncs.load_python_object(self.config.val_feature_path)
        self.val_target_data = myfuncs.load_python_object(self.config.val_target_path)

        self.features = pd.concat(
            [self.train_feature_data, self.val_feature_data], axis=0
        )
        self.target = pd.concat([self.train_target_data, self.val_target_data], axis=0)

        self.base_model = myfuncs.get_base_model(self.config.model_name)

        self.trainval_splitter = PredefinedSplit(
            test_fold=[-1] * len(self.train_feature_data)
            + [0] * len(self.val_feature_data)
        )

        result = f"P: {self.config.data_transformation}<br>{self.config.model_name}<br>"
        result += myfuncs.get_monitor_desc(self.config.param_grid_model_desc)
        self.monitor_desc = result

    def save_list_monitor_components(self):
        if os.path.exists(self.config.list_monitor_components_path):

            self.list_monitor_components = myfuncs.load_python_object(
                self.config.list_monitor_components_path
            )
        else:
            self.list_monitor_components = []

        self.list_monitor_components += [
            (
                self.monitor_desc,
                self.train_accuracy_follow_best_val,
                self.best_val_accuracy,
            )
        ]
        myfuncs.save_python_object(
            self.config.list_monitor_components_path, self.list_monitor_components
        )

    def plot_monitor(self):
        myfuncs.plot_monitor(self.list_monitor_components)

    def get_detail_of_training_and_save_data(self, param_list):
        best_model = None
        self.best_val_accuracy = -np.inf
        self.train_accuracy_follow_best_val = 0

        for i, params in enumerate(param_list, 1):
            model = clone(self.base_model)
            model.set_params(**params)

            start_time = time.time()
            model.fit(self.train_feature_data, self.train_target_data)
            end_time = time.time()

            val_predictions = model.predict(self.val_feature_data)
            val_accuracy = metrics.accuracy_score(self.val_target_data, val_predictions)

            train_predictions = model.predict(self.train_feature_data)
            train_accuracy = metrics.accuracy_score(
                self.train_target_data, train_predictions
            )

            elapsed_time = round(end_time - start_time, 1)  # ⏱ Tính thời gian chạy

            print(
                f"{i} -> Train: {round(train_accuracy*100, 2)} Val: {round(val_accuracy*100, 2)} Time: {elapsed_time} (s)\n\n"
            )

            if self.best_val_accuracy < val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.train_accuracy_follow_best_val = train_accuracy
                best_model = model

        self.best_val_accuracy = round(self.best_val_accuracy * 100, 2)
        self.train_accuracy_follow_best_val = round(
            self.train_accuracy_follow_best_val * 100, 2
        )

        myfuncs.save_python_object(self.config.best_model_path, best_model)
