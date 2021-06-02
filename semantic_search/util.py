from sentence_transformers         import LoggingHandler
from sentence_transformers         import util
from sentence_transformers.readers import InputExample
from torch.utils.data              import DataLoader
from zipfile                       import ZipFile 

import csv
import json
import logging
import os


def load_config(config_path=None):
    config_path = config_path if config_path else 'config.json'
    with open(config_path, 'r') as file:
      config = json.load(file)
    return config


def initialize_logger(config):
    logging.basicConfig(format=config['TRAIN']['LOGGER_FORMAT'],
                        datefmt=config['TRAIN']['LOGGER_DATE_FORMAT'],
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    return logging.getLogger(__name__)


def download_dataset(config, logger):
    if not os.path.exists(config['TRAIN']['DATASET_PATH']):
        logger.info("Dataset not found. Download")
        util.http_get(url=config['TRAIN']['DATASET_DOWNLOAD_URL'], path=config['TRAIN']['ZIP_SAVE_PATH'])
        with ZipFile(config['TRAIN']['ZIP_SAVE_PATH'], 'r') as zip:
            zip.extractall(config['TRAIN']['DATASET_PATH'])


def create_data_loader(config, data_type):
    train_data_path = os.path.join(config['TRAIN']['DATASET_PATH'], config['TRAIN']['DATASET_CLASSIFICATION_PATH'])
    
    train_samples = []
    with open(train_data_path, encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if data_type == 'MultipleNegativesRankingLoss':
                if row['is_duplicate'] == '1':
                    train_samples.append(InputExample(texts=[row['question1'], row['question2']], label=1))
                    train_samples.append(InputExample(texts=[row['question2'], row['question1']], label=1))  # if A is a duplicate of B, then B is a duplicate of A
            elif data_type == 'ConstrativeLoss':    
                train_samples.append(InputExample(texts=[row['question1'], row['question2']], label=int(row['is_duplicate'])))
            
    return DataLoader(train_samples, shuffle=True, batch_size=config['TRAIN']['TRAIN_BATCH_SIZE'])

