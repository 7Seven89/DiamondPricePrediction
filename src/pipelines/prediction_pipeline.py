import pandas as pd
from src.exception import CustomException
from src.logger import get_logger
from src.utils import ObjectControl
import os
import sys


logger = get_logger('Prediction_Pipeline')


class GetDataframe:

    def __init__(self, carat: float, depth: float, table: float, x: float, y: float, z: float, cut: str, color: str, clarity: str):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def make_dataframe(self):

        try:
            custom_data_dict = {
                'carat': [self.carat],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity]
            }

            df = pd.DataFrame(custom_data_dict)
            logger.info('Input Data dataframe created successfully')

            return df

        except Exception as e:
            logger.error('Erro in dataframe transformation stage')
            raise CustomException(e, sys)


class GetPrediction:

    @staticmethod
    def make_prediction(custom_data):

        try:
            logger.info('Model prediction initialized')

            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor_obj = ObjectControl.load_object(preprocessor_path)
            model_obj = ObjectControl.load_object(model_path)

            sclaed_custom_data = preprocessor_obj.transform(custom_data)
            pred_val = model_obj.predict(sclaed_custom_data)

            logger.info('Model prediction successful')
            return pred_val

        except Exception as e:
            logger.error('Error in Prediction Pipeline')
            raise CustomException(e, sys)
