import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from purchasing_model.config import config


def load_dataset(file_name):
    """Load in a csv file, separator is ':' """
    _data = pd.read_csv(f'{config.DATASET_DIR}/{file_name}')
    return _data


def replace_unknown_labels_to_nan(data):
    """Convert 'unknown' labels in dataframe to np.nan"""
    data.replace('unknown', np.nan, inplace=True)
    return data


def convert_target_to_binary(data):
    """Convert target variable to 0's and 1's"""
    data[config.TARGET] = data[config.TARGET].replace({'no': 0, 'yes': 1})
    return data


def create_missing_labels(data):
    """Create missing labels"""
    data[config.FEATURES_TO_CREATE_MISSING_LABELS] = data[
        config.FEATURES_TO_CREATE_MISSING_LABELS].fillna('Missing')
    return data


def drop_instances_missing(data):
    """Drop very little data points that are null"""
    data[config.FEATURES_WITH_NA] = data.dropna(subset=[config.FEATURES_WITH_NA])
    return data


def engineer_pdays(data):
    """Create new feature from 'pdays', remove original pdays feature"""
    data['prev_contact'] = data['pdays'].apply(
        lambda row: 'no' if row == 999 else 'yes'
    )
    data = data.drop('pdays', axis=1)


def remove_rare_instances(data):
    """Remove rare instances that account for less than 0.005% of all data points"""
    data = data[~data['education'].str.contains('illiterate')]
    return data


def save_pipeline(pipeline_to_persist):
    """Persist the pipeline."""
    save_file_name = f'{config.PIPELINE_SAVE_FILE}.pkl'
    save_path = config.TRAINED_MODEL_DIR/save_file_name
    joblib.dump(pipeline_to_persist, save_path)
