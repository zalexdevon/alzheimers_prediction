from classifier.config.configuration import ConfigurationManager
from classifier.components.model_trainer import (
    ModelTrainer,
)
from classifier import logger
from classifier.components.monitor_plotter import (
    MonitorPlotter,
)

STAGE_NAME = "Model Trainer stage"


class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.load_data_to_train()

        while True:
            if model_trainer.config.model_trainer_type == "rcv":
                model_trainer.train_randomisedcv()
                break

            if model_trainer.config.model_trainer_type == "gcv":
                model_trainer.train_gridcv()
                break

            if model_trainer.config.model_trainer_type == "r":
                model_trainer.train_randomised_train_val()
                break

            if model_trainer.config.model_trainer_type == "g":
                model_trainer.train_grid_train_val()
                break

        model_trainer.save_list_monitor_components()

        monitor_plot_config = config.get_monitor_plot_config()
        monitor_plot = MonitorPlotter(config=monitor_plot_config)
        monitor_plot.plot(model_trainer.list_monitor_components)


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
