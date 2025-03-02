from classifier.constants import *
from classifier.Mylib.myfuncs import read_yaml, create_directories
from classifier.entity.config_entity import (
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    MonitorPlotterConfig,
)
from pathlib import Path
from classifier.Mylib import myfuncs


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):

        self.config = read_yaml(Path(config_filepath))
        self.params = read_yaml(Path(params_filepath))

        create_directories([self.config.artifacts_root])

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            train_data_path=config.train_data_path,
            val_data_path=config.val_data_path,
            root_dir=config.root_dir,
            preprocessor_path=config.preprocessor_path,
            classes_path=config.classes_path,
            train_features_path=config.train_features_path,
            train_target_path=config.train_target_path,
            val_features_path=config.val_features_path,
            val_target_path=config.val_target_path,
            target_col=self.params.target_col,
        )

        return data_transformation_config

    def get_model_trainer_config(
        self,
    ) -> ModelTrainerConfig:
        config = self.config.model_trainer

        create_directories([config.root_dir, config.monitor_desc_folder_path])

        param_grid_model = myfuncs.get_param_grid_model(
            self.params.param_grid_model_desc
        )

        model_trainer_config = ModelTrainerConfig(
            train_feature_path=config.train_feature_path,
            train_target_path=config.train_target_path,
            val_feature_path=config.val_feature_path,
            val_target_path=config.val_target_path,
            root_dir=config.root_dir,
            best_model_path=config.best_model_path,
            monitor_desc_folder_path=config.monitor_desc_folder_path,
            list_monitor_components_path=config.list_monitor_components_path,
            model_name=self.params.model_name,
            param_grid_model_desc=self.params.param_grid_model_desc,
            param_grid_model=param_grid_model,
            data_transformation=str(self.params.data_transformation),
            N_ITER=self.params.N_ITER,
            model_trainer_type=self.params.model_trainer_type,
            metric=self.params.metric,
        )

        return model_trainer_config

    # MODEL_EVALUATION
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        obj = ModelEvaluationConfig(
            test_data_path=config.test_data_path,
            preprocessor_path=config.preprocessor_path,
            model_path=config.model_path,
            root_dir=config.root_dir,
            result=config.result,
            target_col=config.target_col,
            metric=config.metric,
            evaluated_model_name=config.evaluated_model_name,
        )

        return obj

    def get_monitor_plot_config(self) -> MonitorPlotterConfig:
        config = self.params.monitor_plotter

        obj = MonitorPlotterConfig(
            target_val_value=config.target_val_value,
            max_val_value=config.max_val_value,
            height_for_annot=config.height_for_annot,
            dtick_y_value=config.dtick_y_value,
        )

        return obj
