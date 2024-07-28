from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import get_logger
from src.utils import ObjectControl
import os
import sys

logger = get_logger('Training_Pipeline')

if __name__ == '__main__':

    logger.info('------Training Pipeline Initialized------')

    data_ing_obj = DataIngestion()
    train_path, test_path = data_ing_obj.initiate_data_ingestion()

    data_transform_obj = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transform_obj.initialize_data_transformation(
        train_path, test_path)

    model_trainer_obj = ModelTrainer()
    model_trainer_obj.initiate_model_training(train_arr, test_arr)
    logger.info('------Model Training Successful------')
