import sys
import os
import pandas as pd
import numpy as np
import pymrmr
import warnings

# Append project path if needed
sys.path.append("../")  # Adjust as necessary
from PrepareDataset.DataEncoder.PreprocessLogger import PreprocessLogger

# Initialize logger
RESULTS_PATH = PROJECT_DIR + "results_mrmr"
PROJECT_DIR = "/home/dmitrii/GitHub/ukbb_risk_assessment/"
logger = PreprocessLogger(PreprocessLogger.__name__, jupyter=False,
                          file_name=RESULTS_PATH + "/mrmr_extraction.log").logger
logger.info("Starting mRMR feature extraction...")

categories = pd.read_csv(PROJECT_DIR + 'results_pca/Total_Segmentator_Categories.csv')


# Set datasets and feature pools
DATASETS = ["pancreas_3m_3y", "liver_3m_3y", "cancer_3m_3y", "copd_3m_3y", "ckd_3m_3y"]
FEATURE_POOLS = {"total_segmentator": ["total_radiomics_tabular.csv"]}


# Helper function for mRMR feature selection
def select_features_with_mrmr(data, label_column, num_features=10):
    features_data = data.drop(columns=[label_column])
    features_data[label_column] = data[label_column]  # Append label column for pymrmr compatibility
    selected_features = pymrmr.mRMR(features_data, 'MIQ', num_features)
    return selected_features


# Main loop for each dataset
warnings.filterwarnings("ignore")
for dataset in DATASETS:
    cohort_path = f'{PROJECT_DIR}PrepareDataset/resources/3m_3y/{dataset}/'
    eids_path = cohort_path + 'labels_with_val.csv'
    data = pd.read_csv(eids_path)

    # Process each feature pool for total_segmentator only
    for feature_set, pool_features in FEATURE_POOLS.items():
        for pool_feature in pool_features:
            df_pool_feature = pd.read_csv(cohort_path + 'tabular_final_preprocessed/' + pool_feature)
            data = data.merge(df_pool_feature, on='eid', how='inner')

        # Perform mRMR feature selection
        logger.info(f"Selecting mRMR features for dataset: {dataset}, feature set: {feature_set}")
        selected_features = select_features_with_mrmr(data.drop(columns=['eid', 'time_to_event', 'split']),
                                                      label_column='event', num_features=100)
        logger.info(f"Selected features: {selected_features}")

        # Save selected features
        summary_path = os.path.join(RESULTS_PATH, dataset, feature_set)
        os.makedirs(summary_path, exist_ok=True)
        selected_features_path = os.path.join(summary_path, "selected_features.csv")
        pd.DataFrame(selected_features, columns=["selected_feature"]).to_csv(selected_features_path, index=False)
        logger.info(f"Saved selected features at {selected_features_path}")

logger.info("mRMR feature extraction completed.")
