import pandas as pd
import numpy as np
import os
import sys

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from src.logger import get_logger
from src.exception import CustomException
from src.utils import ObjectControl, ModelEvaluation
from dataclasses import dataclass

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

logger = get_logger('Model Trainer')


@dataclass
class ModelConfig:
    model_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_config = ModelConfig()

    def initiate_model_training(self, train_arr, test_arr):

        try:

            logger.info('Model training initialized')

            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            logger.info(f'''Train input: \n{
                        pd.DataFrame(X_train).head().to_string()}''')
            logger.info(f'''Train target: \n{
                        pd.DataFrame(y_train).head().to_string()}''')

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]
            logger.info(f'''Test input: \n{
                        pd.DataFrame(X_test).head().to_string()}''')
            logger.info(f'''Test target: \n{
                        pd.DataFrame(y_test).head().to_string}''')

            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'Knn Regressor': KNeighborsRegressor(),
                'Random Forest Regressor': RandomForestRegressor(),
                'Adaboost': AdaBoostRegressor(),
                'Gradient Boost': GradientBoostingRegressor(),
                'XGBoost': XGBRegressor(),
            }

            logger.info('Model Trainer initialized')
            model_trainer = ModelEvaluation()

            model_report = model_trainer.evaluate_model(
                models, X_train, y_train, X_test, y_test)

            print('='*60, '\n')
            model_trainer.print_report(model_report)
            print('='*60, '\n')
            logger.info(f'Model Training Report: \n{model_report}')

            best_model_name, best_score = model_trainer.best_model(
                model_report)
            print(f'''------Best model found------ \nModel Name: {
                  best_model_name} \nR2 Score: {best_score}\n''')
            print('='*60, '\n')
            logger.info(f'''Best Model: \n{
                        best_model_name} \nR2 Score: {best_score}''')

            best_model = models[best_model_name]

            ObjectControl.save_object(
                best_model,
                self.model_config.model_path
            )
            logger.info(f'''Model saved successfully at: {
                        self.model_config.model_path}''')

        except Exception as e:
            logger.error('Error in Model Training Stage')
            raise CustomException(e, sys)
