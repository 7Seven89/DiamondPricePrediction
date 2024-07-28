from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import pandas as pd
import numpy as np
from dataclasses import dataclass
import os
import sys
from src.logger import get_logger
from src.exception import CustomException
from src.utils import VolumeCalculator, ObjectControl


logger = get_logger('Data Transformation')


@dataclass
class DataTransformationConfig:
    processor_object_path = os.path.join(
        'artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):

        try:
            logger.info('Preprocessor Initiated')

            num_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
            cat_cols = ['cut', 'color', 'clarity']

            cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1',
                                  'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            logger.info('Pipeline Initiated')

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('vol_calc', VolumeCalculator()),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OrdinalEncoder(categories=[
                     cut_categories, color_categories, clarity_categories])),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_cols),
                    ('cat_pipeline', cat_pipeline, cat_cols)
                ]
            )

            logger.info('Pipeline Completed')

            return preprocessor

        except Exception as e:
            logger.error('Error in preprocessor initialization')
            raise CustomException(e, sys)

    def initialize_data_transformation(self, train_path, test_path):

        try:
            logger.info('Data transformation initialized')

            train_data = pd.read_csv(train_path)
            logger.info(f'Train data read: \n{train_data.head().to_string()}')

            test_data = pd.read_csv(test_path)
            logger.info(f'Test data read: \n{test_data.head().to_string()}')

            logger.info('Input and Target split')
            target_col = 'price'
            drop_cols = ['id', target_col]

            input_train = train_data.drop(columns=drop_cols)
            target_train = train_data[target_col]
            logger.info(f'Train input: \n{input_train.head().to_string()}')
            logger.info(f'Train target: \n{target_train.head().to_string()}')

            input_test = test_data.drop(columns=drop_cols)
            target_test = test_data[target_col]
            logger.info(f'Test input: \n{input_test.head().to_string()}')
            logger.info(f'Test target: \n{target_test.head().to_string()}')

            logger.info('Preprocessor object initialize')
            preprocessor_obj = self.get_data_transformation_object()

            scaled_input_train = preprocessor_obj.fit_transform(input_train)
            scaled_input_test = preprocessor_obj.transform(input_test)
            logger.info('Data Transformed using preprocessor')

            train_arr = np.c_[scaled_input_train, np.array(target_train)]
            test_arr = np.c_[scaled_input_test, np.array(target_test)]
            logger.info(f'''Train array: \n{
                        pd.DataFrame(train_arr).head().to_string()}''')
            logger.info(f'''Test array: \n{
                        pd.DataFrame(test_arr).head().to_string()}''')

            ObjectControl.save_object(
                preprocessor_obj,
                self.data_transformation_config.processor_object_path
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.processor_object_path
            )

        except Exception as e:
            logger.error('Error in data transformation stage')
            raise CustomException(e, sys)
