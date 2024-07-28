import logging
import os
from datetime import datetime


def get_logger(name):

    log_file_name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

    logs_path_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(logs_path_dir, exist_ok=True)

    logs_file_path = os.path.join(logs_path_dir, log_file_name)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

    file_handler = logging.FileHandler(logs_file_path)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
