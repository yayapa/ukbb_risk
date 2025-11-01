import os
from datetime import datetime

import pandas as pd
import numpy as np
import yaml
import sys
import os
sys.path.append("put_yours")
from PrepareDataset.DataEncoder.PreprocessLogger import PreprocessLogger
from ModelTrainer import ModelTrainer
from ModelTrainerNested import ModelTrainerNested
from ModelTrainerVal import ModelTrainerVal
from ModelTrainerCVFull import ModelTrainerCVFull
from pathlib import Path



import os

# Initialize logger

# create standard logger
logger = PreprocessLogger(PreprocessLogger.__name__, jupyter=False, file_name="experiment.log").logger


def prepare_data(data):
    """Prepare train and test sets with necessary transformations."""
    train_data = data[(data['split'] == 'train') | (data['split'] == 'val')].drop(
        columns=['split', 'eid', 'event', 'time_to_event']).copy()
    train_data['label'] = data[(data['split'] == 'train') | (data['split'] == 'val')]['event'].copy()

    test_data = data[data['split'] == 'test'].drop(
        columns=['split', 'eid', 'event', 'time_to_event']).copy()
    test_data['label'] = data[data['split'] == 'test']['event'].copy()

    X_train = train_data.drop(columns=['label'])
    y_train = train_data['label']
    X_test = test_data.drop(columns=['label'])
    y_test = test_data['label']

    return X_train, y_train, X_test, y_test

def prepare_data_val(data):
    col_to_drop = ['eid', 'event', 'time_to_event', 'split']
    X_train = data[data['split'] == 'train'].drop(columns=col_to_drop)
    y_train = data[data['split'] == 'train']['event']
    X_val = data[data['split'] == 'val'].drop(columns=col_to_drop)
    y_val = data[data['split'] == 'val']['event']
    X_test = data[data['split'] == 'test'].drop(columns=col_to_drop)
    y_test = data[data['split'] == 'test']['event']

    return X_train, y_train, X_val, y_val, X_test, y_test

def prepare_data_all(data):
    col_to_drop = ['eid', 'event', 'time_to_event', 'split']
    X = data.drop(columns=col_to_drop)
    y = data['event']
    return X, y

def remove_categories(data, categories):
    """Remove specified categories from features."""
    features = data.columns
    for category in categories:
        features = [feature for feature in features if category not in feature]
    return data[features]


def setup_directories(dataset_name, feature_set, model_name):
    """Create and return paths for saving results."""
    base_path = os.path.join(SAVE_PATH, dataset_name, feature_set)
    paths = {
        "predictions": os.path.join(base_path, "predictions", model_name),
        "feature_importances": os.path.join(base_path, "feature_importances", model_name),
        #"cv": os.path.join(base_path, "cv", model_name),
        "summary": base_path
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def save_summary(metrics_list, dataset_name, feature_set):
    """Calculate and save summary of results."""
    summary_path = os.path.join(SAVE_PATH, dataset_name, feature_set, "summary.csv")
    metrics_df = pd.DataFrame(metrics_list)
    summary = metrics_df.groupby('model').agg(['mean', 'std']).reset_index()
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary.to_csv(summary_path, index=False)
    logger.info(f"Saved summary for feature set {feature_set} in dataset {dataset_name} at {summary_path}")

def preprocess_feature_names(data):
    data.columns = data.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
    return data

import re

# Regular expression to match scientific notation or decimal numbers
numeric_pattern = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')

def convert_scientific_notation(config):
    for key, value in config.items():
        if isinstance(value, dict):
            # Recursive call for nested dictionaries
            convert_scientific_notation(value)
        elif isinstance(value, list):
            # Check each element in the list
            config[key] = [
                float(v) if isinstance(v, str) and numeric_pattern.match(v) else
                int(v) if isinstance(v, str) and v.isdigit() else
                v
                for v in value
            ]
        elif isinstance(value, str):
            # Convert scientific notation strings to floats, and integer-like strings to int
            if numeric_pattern.match(value):
                config[key] = float(value) if '.' in value or 'e' in value.lower() else int(value)

# switch off warnings
import warnings
warnings.filterwarnings('ignore')

with open("put_yours/config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

convert_scientific_notation(config)

current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
SAVE_PATH = os.path.join(config["paths"]["results"], config["experiment"]["name"], current_date)

#model_trainer = ModelTrainerNested(config, SAVE_PATH)
if config["mode"] == "val":
    model_trainer = ModelTrainerVal(config, SAVE_PATH)
elif config["mode"] == "cvfull":
    model_trainer = ModelTrainerCVFull(config, SAVE_PATH)

Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
# save config
with open(os.path.join(SAVE_PATH, "config.yaml"), "w") as config_file:
    yaml.dump(config, config_file)

# Iterate over datasets and feature pools
for dataset in config["datasets"]:
    dataset_path = os.path.join(config["paths"]["datasets"], dataset)
    eids_path = os.path.join(dataset_path, 'labels_with_val.csv')

    for feature_set, features in config["feature_pools"].items():
        data = pd.read_csv(eids_path)
        for feature in features:
            feature_path = os.path.join(dataset_path, 'tabular_final_preprocessed', feature)
            feature_data = pd.read_csv(feature_path)
            data = data.merge(feature_data, on='eid', how='inner')

        # Remove unwanted categories
        data = remove_categories(data, config["remove_categories"])
        data = preprocess_feature_names(data)

        #X_train, y_train, X_test, y_test = prepare_data(data)
        if config["mode"] == "val":
            X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_val(data)
        elif config["mode"] == "cvfull":
            X, y = prepare_data_all(data)
        metrics_summary = []
        for model_name in config["models"].keys():
            for seed in config["random_seed"]:
                np.random.seed(seed)
                paths = setup_directories(dataset, feature_set, model_name)
                #metrics = model_trainer.train_and_evaluate(X_train, y_train, X_test, y_test, model_name, dataset, feature_set, seed)
                if config["mode"] == "val":
                    metrics = model_trainer.train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, model_name, dataset, feature_set, seed)
                elif config["mode"] == "cvfull":
                    metrics = model_trainer.train_and_evaluate(X, y, model_name, dataset, feature_set, seed)
                metrics["model"] = model_name
                metrics["seed"] = seed
                metrics_summary.append(metrics)
                logger.info(f"Completed seed {seed} for model {model_name}")

        save_summary(metrics_summary, dataset, feature_set)
        logger.info(f"Completed feature set: {feature_set} in dataset: {dataset}")

logger.info("Experiment pipeline completed.")
