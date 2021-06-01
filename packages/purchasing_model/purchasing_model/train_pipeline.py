from purchasing_model import pipeline
from config import config
from processing.data_management import (
    load_dataset, replace_unknown_labels_to_nan, convert_target_to_binary,
    create_missing_labels, drop_instances_missing, engineer_pdays, remove_rare_instances, save_pipeline
)


def train_model():
    """Train the model"""
    data = load_dataset(config.TRAINING_DATA_FILE)
    data = replace_unknown_labels_to_nan(data)
    data = convert_target_to_binary(data)
    data = create_missing_labels(data)
    data = drop_instances_missing(data)
    data = engineer_pdays(data)
    data = remove_rare_instances(data)
    print('...fitting model')
    pipeline.purchasing_pipe.fit(
        data.drop(data[config.TARGET], axis=1), data[config.TARGET]
    )
    save_pipeline(pipeline.purchasing_pipe)


if __name__ == "__main__":
    train_model()
