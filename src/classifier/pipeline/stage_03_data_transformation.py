from classifier.config.configuration import ConfigurationManager
from classifier.components.data_transformation import DataTransformation
from classifier import logger
from pathlib import Path


STAGE_NAME = "Data Transformation stage"


class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.load_data()
            data_transformation.create_preprocessor_for_train_data()
            data_transformation.transform_data()
        except Exception as e:
            print(e)


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_transform = DataTransformationPipeline()
        data_transform.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
