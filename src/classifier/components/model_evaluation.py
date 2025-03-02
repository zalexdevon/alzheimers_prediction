import pandas as pd
import os
from classifier import logger
from classifier.entity.config_entity import ModelEvaluationConfig
from classifier.Mylib import myfuncs
from sklearn import metrics


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate_model(self):
        df = myfuncs.load_python_object(self.config.test_data_path)
        preprocessor = myfuncs.load_python_object(self.config.preprocessor_path)
        df_transformed = preprocessor.transform(df)
        df_feature = df_transformed.drop(columns=[self.config.target_col])
        df_target = df_transformed[self.config.target_col]

        model = myfuncs.load_python_object(self.config.model_path)

        score = self.evaluate(df_target, model, df_feature)

        print(f"{self.config.evaluated_model_name}: {score}\n\n")

        with open(self.config.result, "a") as file:
            file.write(f"{self.config.evaluated_model_name}: {score}\n\n")

    def evaluate(self, df_target, model, df_feature):
        while True:
            if self.config.metric == "accuracy":
                df_feature_prediction = model.predict(df_feature)
                return metrics.accuracy_score(df_target, df_feature_prediction)

            if self.config.metric == "neg_log_loss":
                df_feature_prediction = model.predict_proba(df_feature)

                return metrics.log_loss(df_target, df_feature_prediction)

            return
