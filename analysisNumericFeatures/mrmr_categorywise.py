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
PROJECT_DIR = "put yours"
RESULTS_PATH = "results_mrmr_categorywise"
logger = PreprocessLogger(PreprocessLogger.__name__, jupyter=False,
                          file_name=RESULTS_PATH + "/mrmr_extraction.log").logger
logger.info("Starting mRMR feature extraction...")

# Set datasets and feature pools
#DATASETS = ["cvd2_3m_3y", "pancreas_3m_3y", "liver_3m_3y", "cancer_3m_3y", "copd_3m_3y", "ckd_3m_3y", "osteoarthritis_3m_3y"]
DATASETS = ["cancer_3m_3y"]
FEATURE_POOLS = {
    #"nonimage": ["nonimage_tabular.csv"],
    "total_segmentator": ["total_radiomics_prostate_tabular.csv"]
}

categories = pd.read_csv(PROJECT_DIR + 'Total_Segmentator_Categories.csv')

# Helper function for mRMR feature selection
def select_features_with_mrmr(data, label_column, num_features=50):
    # Ensure the label column is the first column in the DataFrame for pymrmr compatibility
    features_data = data.copy()
    label_data = features_data.pop(label_column)
    features_data.insert(0, label_column, label_data)  # Insert label column at the beginning

    # Perform mRMR feature selection
    selected_features = pymrmr.mRMR(features_data, 'MIQ', num_features)
    return selected_features


def get_category_dict(dataset):
    CATEGORY_DICT = {}
    for i, row in categories.iterrows():
        category = row['CATEGORY']
        feature = row['NAME']
        if category in CATEGORY_DICT:
            CATEGORY_DICT[category].append(feature)
        else:
            CATEGORY_DICT[category] = [feature]


    features = pd.read_csv(f'{PROJECT_DIR}/PrepareDataset/resources/3m_3y/{dataset}/tabular_final_preprocessed/total_radiomics_prostate_tabular.csv')
    features_nonimage = pd.read_csv(f'{PROJECT_DIR}/PrepareDataset/resources/3m_3y/{dataset}/tabular_final_preprocessed/nonimage_tabular.csv')
    features = features.drop(columns=['eid'])
    features = features.columns
    category_dict = {}
    for category in CATEGORY_DICT:
        #for contrast in {'fat', 'wat'}:
            category_name = category #+ "_" + contrast
            features_category = [feature for feature in features if any(label in feature for label in CATEGORY_DICT[category])] #and (feature.endswith(contrast) or 'shape' in feature)]
            if len(features_category) > 0:
                category_dict[category_name] = features_category


    category_dict['nonimage'] = list(set(features_nonimage.columns) - {'eid'})
    return category_dict

#category_dict = get_category_dict()
#print(category_dict['nonimage'])

for dataset in DATASETS:
    cohort_path = f'{PROJECT_DIR}/PrepareDataset/resources/3m_3y/{dataset}/'
    eids_path = cohort_path + 'labels_with_val.csv'
    data = pd.read_csv(eids_path)
    category_dict = get_category_dict(dataset)

    # Process each feature pool for total_segmentator only
    for feature_set, pool_features in FEATURE_POOLS.items():
        for pool_feature in pool_features:
            df_pool_feature = pd.read_csv(cohort_path + 'tabular_final_preprocessed/' + pool_feature)
            data = data.merge(df_pool_feature, on='eid', how='inner')

        data = data[data['split'] == 'train']
        # Loop through each category in category_dict and apply mRMR
        for category, features in category_dict.items():
            if category != 'muscle':
                continue
            logger.info(f"Selecting mRMR features for dataset: {dataset}, feature set: {feature_set}, category: {category}")
            # remove nans rows
            print("Before dropna", data.shape)
            data = data.dropna(subset=features)
            print("After dropna", data.shape)

            category_data = data[features + ['event']]  # Select features specific to the category plus label column
            selected_features = select_features_with_mrmr(category_data, label_column='event', num_features=50)
            logger.info(f"Selected features for {category}: {selected_features}")

            # Save selected features
            summary_path = os.path.join(RESULTS_PATH, dataset, feature_set, category)
            os.makedirs(summary_path, exist_ok=True)
            selected_features_path = os.path.join(summary_path, "selected_features.csv")
            pd.DataFrame(selected_features, columns=["selected_feature"]).to_csv(selected_features_path, index=False)
            logger.info(f"Saved selected features for {category} at {selected_features_path}")

logger.info("mRMR feature extraction completed.")
