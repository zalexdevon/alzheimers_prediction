from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataTransformationConfig:
    train_data_path: Path
    val_data_path: Path
    root_dir: Path
    preprocessor_path: Path
    classes_path: Path
    train_features_path: Path
    train_target_path: Path
    val_features_path: Path
    val_target_path: Path
    target_col: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    train_feature_path: Path
    train_target_path: Path
    val_feature_path: Path
    val_target_path: Path
    root_dir: Path
    best_model_path: Path
    monitor_desc_folder_path: Path
    list_monitor_components_path: Path

    N_ITER: int
    data_transformation: int
    model_name: str
    param_grid_model_desc: dict
    param_grid_model: dict
    model_trainer_type: str
    metric: str


# MODEL_EVALUATION
@dataclass(frozen=True)
class ModelEvaluationConfig:
    test_data_path: Path
    preprocessor_path: Path
    model_path: Path
    root_dir: Path
    result: Path
    target_col: str
    metric: str
    evaluated_model_name: str


@dataclass(frozen=True)
class MonitorPlotterConfig:
    target_val_value: float
    max_val_value: float
    height_for_annot: float
    dtick_y_value: float
