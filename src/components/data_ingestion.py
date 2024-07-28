import os
import sys
from src.logger import get_logger
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

logger = get_logger('Data Ingestion')


@dataclass
class DataIngestionPath:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingest_path = DataIngestionPath()

    def initiate_data_ingestion(self):
        logger.info('Data Ingestion initiated')

        try:
            df = pd.read_csv('notebooks/data/gemstone.csv')
            df.to_csv(self.ingest_path.raw_data_path, index=False)

            logger.info('Train Test Split')

            train_set, test_set = train_test_split(
                df, test_size=0.30, random_state=42)

            train_set.to_csv(self.ingest_path.train_data_path,
                             index=False, header=True)
            test_set.to_csv(self.ingest_path.test_data_path,
                            index=False, header=True)

            logger.info('Data ingestion completed')

            return (
                self.ingest_path.train_data_path,
                self.ingest_path.test_data_path
            )

        except Exception as e:
            logger.error('Error occured in Data Ingestion Path config')
