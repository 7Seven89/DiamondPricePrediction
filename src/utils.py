from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.exception import CustomException
import sys
import os
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = get_logger('Utils')


class VolumeCalculator:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    logger.info('Volume Calculator Initiated')

    def transform(slef, X, y=None):
        try:
            X = pd.DataFrame(
                X, columns=['carat', 'depth', 'table', 'x', 'y', 'z'])
            X['volume'] = X['x'] * X['y'] * X['z']
            X.drop(columns=['x', 'y', 'z'], inplace=True)
            return X

        except Exception as e:
            logger.info('Error in Volume Calculator Transformer')
            raise CustomException(e, sys)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            input_features = list(input_features)
            if 'x' in input_features and 'y' in input_features and 'z' in input_features:
                input_features.remove('x')
                input_features.remove('y')
                input_features.remove('z')
            return input_features + ['volume']
        return ['volume']


class ModelEvaluation:
    def evaluate_model(self, models, X_train, y_train, X_test, y_test):

        model_report = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2_scr = r2_score(y_test, y_pred)

            model_report[name] = {'MSE': float(mse), 'MAE': float(
                mae), 'RMSE': float(rmse), 'R2 Score': float(r2_scr)}

        return model_report

    def best_model(self, model_report):

        max_r2_scr = -float('inf')

        for name, score in model_report.items():
            r2_scr = score['R2 Score']
            if r2_scr > max_r2_scr:
                max_r2_scr = r2_scr
                best_model_name = name

        return best_model_name, max_r2_scr

    def print_report(self, model_report):
        num_models = len(model_report)-1
        for idx, (name, score) in enumerate(model_report.items()):
            print(name)
            print('Model Report:')
            for metric, value in score.items():
                print(f'{metric}: {value}')
            if idx < num_models:
                print('='*35)


class ObjectControl:

    @staticmethod
    def save_object(object, file_path):

        try:
            dir_path = os.path.dirname(file_path)

            os.makedirs(dir_path, exist_ok=True)

            with open(file_path, 'wb') as f:
                pickle.dump(object, f)
            logger.info(f'Object saved successfully at: {file_path}')

        except Exception as e:
            logger.error('Save object error')
            raise CustomException(e, sys)

    @staticmethod
    def load_object(file_path):

        try:
            with open(file_path, 'rb') as f:
                logger.info(f'Object loaded succefully from: {file_path}')
                return pickle.load(f)

        except Exception as e:
            logger.error('Load object error')
            raise CustomException(e, sys)
