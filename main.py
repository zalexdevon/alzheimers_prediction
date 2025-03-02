from classifier import logger
from classifier.pipeline.stage_03_data_transformation import DataTransformationPipeline
from classifier.pipeline.stage_04_model_training_randomised import (
    ModelTrainerRandomisedTrainingPipeline,
)
from classifier.pipeline.stage_04_model_training_grid import (
    ModelTrainerGridTrainingPipeline,
)
from classifier.pipeline.stage_04_model_training_randomised_train_val import (
    ModelTrainerRandomisedTrainvalTrainingPipeline,
)
from classifier.pipeline.stage_04_model_training_grid_train_val import (
    ModelTrainerGridTrainvalTrainingPipeline,
)

from classifier.pipeline.stage_05_model_evaluation import (
    ModelEvaluationPipeline,
)


import warnings

warnings.filterwarnings("ignore")

STAGE_NAME = "Data Transformation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_transform = DataTransformationPipeline()
    data_transform.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Trainer Randomised stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_training_randomised = ModelTrainerRandomisedTrainingPipeline()
    model_training_randomised.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Trainer Grid stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainerGridTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Trainer Randomised train-val stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainerRandomisedTrainvalTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Trainer Grid train-val stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainerGridTrainvalTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Evaluation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelEvaluationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
