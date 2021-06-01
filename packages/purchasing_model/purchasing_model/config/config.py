from pathlib import Path

import purchasing_model


PACKAGE_ROOT = Path(purchasing_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT/'trained_models'
DATASET_DIR = PACKAGE_ROOT/'data'

# data
TRAINING_DATA_FILE = "bank-additional-full.csv"
TARGET = 'y'

FEATURES = [
    'age',
    'job',
    'marital',
    'education',
    'contact',
    'month',
    'day_of_week',
    'campaign',
    'pdays',
    'previous',
    'poutcome',
    'emp.var.rate',
    'cons.price.idx',
    'cons.conf.idx',
    'euribor3m',
    'nr.employed'
]

FEATURES_TO_CREATE_MISSING_LABELS = ['default', 'education']

FEATURES_WITH_NA = ['job', 'marital']

PIPELINE_NAME = 'xgb_purchasing'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output_v'
