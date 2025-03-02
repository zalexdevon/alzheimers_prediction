from classifier.config.configuration import ConfigurationManager
from classifier.components.model_evaluation import ModelEvaluation
from classifier import logger


STAGE_NAME = "Model Evaluation stage"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        eval = ModelEvaluation(config=model_evaluation_config)
        eval.evaluate_model()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
