import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
import json

RESULTS_PATH = "/home/dmitrii/GitHub/ukbb_risk_assessment/analysisNumericFeatures/tabular_experiment/results/results_cv_fi_best_3"
#DATASET = "total_segmentator"
DATASET = "non_image+total_segmentator"
MODELS = {
    #"cvd2_3m_3y": {
    #    #"RF": "/home/dmitrii/GitHub/ukbb_risk_assessment/analysisNumericFeatures/tabular_experiment/results/results_val/2024-12-02 21:06:10/cvd2_3m_3y/total_segmentator/validation/RF",
    #    "XGB": f"/home/dmitrii/GitHub/ukbb_risk_assessment/analysisNumericFeatures/tabular_experiment/results/results_val/2024-12-02 21:06:10/cvd2_3m_3y/{DATASET}/validation/XGB",
    #},
    "pancreas_3m_3y": {
        #"RF": "/home/dmitrii/GitHub/ukbb_risk_assessment/analysisNumericFeatures/tabular_experiment/results/results_val/2024-11-18 15:14:25/pancreas_3m_3y/total_segmentator/validation/RF",
        "XGB": f"/home/dmitrii/GitHub/ukbb_risk_assessment/analysisNumericFeatures/tabular_experiment/results/results_val/2024-11-18 15:14:25/pancreas_3m_3y/{DATASET}/validation/XGB",
    },
    "liver_3m_3y": {
        "RF": f"/home/dmitrii/GitHub/ukbb_risk_assessment/analysisNumericFeatures/tabular_experiment/results/results_val/2024-11-18 15:14:25/liver_3m_3y/{DATASET}/validation/RF",
        #"XGB": "/home/dmitrii/GitHub/ukbb_risk_assessment/analysisNumericFeatures/tabular_experiment/results/results_val/2024-11-18 15:14:25/liver_3m_3y/total_segmentator/validation/XGB",
    },
    #"cancer_3m_3y": {
    #    #"RF": "/home/dmitrii/GitHub/ukbb_risk_assessment/analysisNumericFeatures/tabular_experiment/results/results_val/2024-11-18 15:14:25/cancer_3m_3y/total_segmentator/validation/RF",
    #    #"XGB": "/home/dmitrii/GitHub/ukbb_risk_assessment/analysisNumericFeatures/tabular_experiment/results/results_val/2024-11-18 15:14:25/cancer_3m_3y/total_segmentator/validation/XGB",
    #    'XGB': f"/home/dmitrii/GitHub/ukbb_risk_assessment/analysisNumericFeatures/tabular_experiment/results/results_val/2024-12-08 09:47:17/cancer_3m_3y/{DATASET}/validation/XGB"
    #},
    #"copd_3m_3y":  {
    #    #"RF": "/home/dmitrii/GitHub/ukbb_risk_assessment/analysisNumericFeatures/tabular_experiment/results/results_val/2024-12-02 21:06:10/copd_3m_3y/total_segmentator/validation/RF",
    #    #"XGB": "/home/dmitrii/GitHub/ukbb_risk_assessment/analysisNumericFeatures/tabular_experiment/results/results_val/2024-12-02 21:06:10/copd_3m_3y/total_segmentator/validation/XGB",
    #    "RF": "/home/dmitrii/GitHub/ukbb_risk_assessment/analysisNumericFeatures/tabular_experiment/results/results_val/2024-12-08 23:10:50/copd_3m_3y/non_image+total_segmentator/validation/RF",
    #},
    #"ckd_3m_3y": {
        #"RF": "/home/dmitrii/GitHub/ukbb_risk_assessment/analysisNumericFeatures/tabular_experiment/results/results_val/2024-11-18 15:14:25/ckd_3m_3y/total_segmentator/validation/RF",
    #    "XGB": f"/home/dmitrii/GitHub/ukbb_risk_assessment/analysisNumericFeatures/tabular_experiment/results/results_val/2024-11-18 15:14:25/ckd_3m_3y/{DATASET}/validation/XGB"
    #},
    #"osteoarthritis_3m_3y": {
    #    #"RF": "/home/dmitrii/GitHub/ukbb_risk_assessment/analysisNumericFeatures/tabular_experiment/results/results_val/2024-11-19 10:15:01/osteoarthritis_3m_3y/total_segmentator/validation/RF",
    #    #"XGB": "/home/dmitrii/GitHub/ukbb_risk_assessment/analysisNumericFeatures/tabular_experiment/results/results_val/2024-11-19 10:15:01/osteoarthritis_3m_3y/total_segmentator/validation/XGB"
    #    "XGB": "/home/dmitrii/GitHub/ukbb_risk_assessment/analysisNumericFeatures/tabular_experiment/results/results_val/2024-12-08 23:10:50/copd_3m_3y/non_image+total_segmentator/validation/XGB",
    #},
}

CATEGORIES = pd.read_csv('/home/dmitrii/GitHub/ukbb_risk_assessment/results_pca/Total_Segmentator_Categories.csv')

MODEL_DICT = {
    "RF": RandomForestClassifier,
    "XGB": XGBClassifier,
    "MLP": MLPClassifier,
}

BEST_PARAMS_DEFAULT = {
    "RF": {
        "bootstrap": True,
        "oob_score": True,
    },
    "XGB": {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "exact"
    },
    "MLP": {
        "activation": "relu",
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 10,
        "solver": "adam",
        "max_iter": 10000
    }
}

def get_category_dict(dataset):
    CATEGORY_DICT = {}
    for i, row in CATEGORIES.iterrows():
        category = row['CATEGORY']
        feature = row['NAME']
        if category in CATEGORY_DICT:
            CATEGORY_DICT[category].append(feature)
        else:
            CATEGORY_DICT[category] = [feature]


    features = pd.read_csv(f'/home/dmitrii/GitHub/ukbb_risk_assessment/PrepareDataset/resources/3m_3y/{dataset}/tabular_final_preprocessed/total_radiomics_tabular.csv', nrows=1)
    features_nonimage = pd.read_csv(f'/home/dmitrii/GitHub/ukbb_risk_assessment/PrepareDataset/resources/3m_3y/{dataset}/tabular_final_preprocessed/nonimage_tabular.csv')
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


def calculate_category_importance(predictor, X_test, y_test, metric_fn, category_dict, num_shuffles=100):
    if metric_fn == roc_auc_score:
        baseline_score = metric_fn(y_test, predictor.predict_proba(X_test)[:, 1])
    else:
        baseline_score = metric_fn(y_test, predictor.predict(X_test))  # Baseline performance
    category_importance = {}
    p_values = {}


    for category, features in category_dict.items():
        features_selected = [feature for feature in features if feature in X_test.columns]
        print("FI for category", category)
        shuffle_scores = []

        for _ in range(num_shuffles):
            # Shuffle all features in this category
            test_data_shuffled = X_test.copy()
            test_data_shuffled[features_selected] = test_data_shuffled[features_selected].apply(np.random.permutation)

            # Calculate performance with shuffled features
            if metric_fn == roc_auc_score:
                shuffled_score = metric_fn(y_test, predictor.predict_proba(test_data_shuffled)[:, 1])
            else:
                shuffled_score = metric_fn(y_test, predictor.predict(test_data_shuffled))
            shuffle_scores.append(baseline_score - shuffled_score)

        # Calculate average importance for the category
        avg_importance = np.mean(shuffle_scores)
        category_importance[category] = avg_importance
        # Conduct a one-sided t-test for the null hypothesis: importance = 0
        t_stat, p_value = ttest_1samp(shuffle_scores, 0, alternative='greater')
        p_values[category] = p_value

        # Calculate p-value as the proportion of shuffled scores greater than or equal to observed importance
        #p_values[category] = (np.sum(np.array(shuffle_scores) >= avg_importance) + 1) / (num_shuffles + 1)
        print("avg_importance", category_importance[category])
        print("p_values", p_values[category])
        print("FI for category", category, "done")
        # Save category importance and p-values
    category_importance_df = pd.DataFrame({
        'category': category_importance.keys(),
        'importance': category_importance.values(),
        'p_value': [p_values[cat] for cat in category_importance.keys()]
    })

    return category_importance_df


#category_dict = get_category_dict()


def calculate_feature_importances_val():
    for mode, mode_dict in MODELS.items():
        labels = f"/home/dmitrii/GitHub/ukbb_risk_assessment/PrepareDataset/resources/3m_3y/{mode}/labels_with_val.csv"
        data = f"/home/dmitrii/GitHub/ukbb_risk_assessment/PrepareDataset/resources/3m_3y/{mode}/tabular_final_preprocessed/total_radiomics_mrmr_20.csv"
        labels = pd.read_csv(labels)
        data = pd.read_csv(data)
        data = data.merge(labels, how="inner", on="eid")
        # based on "split in labels create train and val sets
        train_data = data[data["split"] == "train"].copy()
        train_data.drop(columns=["eid", "split", "time_to_event"], inplace=True)
        test_data = data[data["split"] == "test"].copy()
        test_data.drop(columns=["eid", "split", "time_to_event"], inplace=True)
        X_train = train_data.drop(columns=["event"])
        y_train = train_data["event"]
        X_test = test_data.drop(columns=["event"])
        y_test = test_data["event"]

        for model_name, exp_dir in mode_dict.items():

            results_all = {}

            model = MODEL_DICT[model_name]
            print("Model: ", model_name)
            results = {
                "balanced_accuracy": [],
                "f1": [],
                "roc_auc": [],
                "y_proba": [],
                "y_pred": []
            }
            for seed in [1514, 42, 0, 29847, 867228]:
                # read the best param json file
                best_param_path = os.path.join(exp_dir, f"best_params_seed_{str(seed)}.json")
                with open(best_param_path, "r") as f:
                    best_params = json.load(f)

                for k, v in BEST_PARAMS_DEFAULT[model_name].items():
                    best_params[k] = v

                model_trained = model(random_state=seed, **best_params)
                model_trained.fit(X_train, y_train)
                preds = model_trained.predict(X_test)
                preds_proba = model_trained.predict_proba(X_test)[:, 1]
                balanced_accuracy = balanced_accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds)
                roc_auc = roc_auc_score(y_test, preds_proba)

                results["balanced_accuracy"].append(balanced_accuracy)
                results["f1"].append(f1)
                results["roc_auc"].append(roc_auc)
                results["y_pred"].append(preds)
                results["y_proba"].append(preds_proba)
                feature_importances = calculate_category_importance(model_trained, X_test, y_test, balanced_accuracy_score, category_dict, num_shuffles=10)
                os.makedirs(f"{RESULTS_PATH}/{mode}/{DATASET}/feature_importances/{model_name}", exist_ok=True)
                feature_importances.to_csv(f"{RESULTS_PATH}/{mode}/{DATASET}/feature_importances/{model_name}/feature_importances_seed_{seed}.csv")
                os.makedirs(f"{RESULTS_PATH}/{mode}/{DATASET}/metrics/{model_name}", exist_ok=True)
                pd.DataFrame(results).to_csv(f"{RESULTS_PATH}/{mode}/{DATASET}/metrics/{model_name}/metrics_seed_{seed}.csv")
            results_all[model_name] = results

        for model_name, results in results_all.items():
            for metric, values in results.items():
                if metric not in ["y_proba", "y_pred"]:
                    results_all[model_name][metric] = f"{np.mean(values):.3f} ± {np.std(values):.3f}"
        pd.DataFrame(results_all).to_csv(f"{RESULTS_PATH}/{mode}/{DATASET}/metrics/all_metrics.csv")


def calculate_feature_importances_cv():
    for mode, mode_dict in MODELS.items():
        category_dict = get_category_dict(mode)
        #if mode != "cancer_3m_3y":
        #    continue
        labels = f"/home/dmitrii/GitHub/ukbb_risk_assessment/PrepareDataset/resources/3m_3y/{mode}/labels_with_val.csv"
        data = f"/home/dmitrii/GitHub/ukbb_risk_assessment/PrepareDataset/resources/3m_3y/{mode}/tabular_final_preprocessed/total_radiomics_mrmr_20.csv"
        data_nonimage = f"/home/dmitrii/GitHub/ukbb_risk_assessment/PrepareDataset/resources/3m_3y/{mode}/tabular_final_preprocessed/nonimage_tabular.csv"
        labels = pd.read_csv(labels)
        data = pd.read_csv(data)
        data_nonimage = pd.read_csv(data_nonimage)
        data = data.merge(data_nonimage, on="eid", how="inner")
        data = data.merge(labels, how="inner", on="eid")
        X = data.drop(columns=["eid", "time_to_event", "split", "event"])
        y = data["event"]
        # based on "split in labels create train and val sets

        for model_name, exp_dir in mode_dict.items():

            results_all = {}

            model = MODEL_DICT[model_name]
            print("Model: ", model_name)
            results = {
                "balanced_accuracy": [],
                "f1": [],
                "roc_auc": [],
                "y_proba": [],
                "y_pred": []
            }
            for seed in [1514, 42, 0, 29847, 867228]:
                outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                outer_metrics = []


                for fold_idx, (train_val_idx, test_idx) in enumerate(outer_cv.split(X, y)):
                    X_train_val, X_test = X.iloc[train_val_idx], X.iloc[test_idx]
                    y_train_val, y_test = y.iloc[train_val_idx], y.iloc[test_idx]

                    # read the best param json file
                    best_param_path = os.path.join(exp_dir, f"best_params_seed_{str(seed)}.json")
                    with open(best_param_path, "r") as f:
                        best_params = json.load(f)

                    for k, v in BEST_PARAMS_DEFAULT[model_name].items():
                        best_params[k] = v

                    model_trained = model(random_state=seed, **best_params)
                    model_trained.fit(X_train_val, y_train_val)
                    preds = model_trained.predict(X_test)
                    preds_proba = model_trained.predict_proba(X_test)[:, 1]
                    balanced_accuracy = balanced_accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds)
                    roc_auc = roc_auc_score(y_test, preds_proba)

                    results["balanced_accuracy"].append(balanced_accuracy)
                    results["f1"].append(f1)
                    results["roc_auc"].append(roc_auc)
                    results["y_pred"].append(preds)
                    results["y_proba"].append(preds_proba)
                    feature_importances = calculate_category_importance(model_trained, X_test, y_test, balanced_accuracy_score,
                                                                        category_dict, num_shuffles=50)
                    os.makedirs(f"{RESULTS_PATH}/{mode}/{DATASET}/feature_importances/{model_name}/{fold_idx}", exist_ok=True)
                    feature_importances.to_csv(
                        f"{RESULTS_PATH}/{mode}/{DATASET}/feature_importances/{model_name}/{fold_idx}/feature_importances_seed_{seed}.csv")
                    os.makedirs(f"{RESULTS_PATH}/{mode}/{DATASET}/metrics/{model_name}/{fold_idx}", exist_ok=True)
                    pd.DataFrame(results).to_csv(
                        f"{RESULTS_PATH}/{mode}/{DATASET}/metrics/{model_name}/{fold_idx}/metrics_seed_{seed}.csv")
                results_all[model_name] = results

            for model_name, results in results_all.items():
                for metric, values in results.items():
                    if metric not in ["y_proba", "y_pred"]:
                        results_all[model_name][metric] = f"{np.mean(values):.3f} ± {np.std(values):.3f}"
            pd.DataFrame(results_all).to_csv(f"{RESULTS_PATH}/{mode}/{DATASET}/metrics/all_metrics.csv")


if __name__ == "__main__":
    #calculate_feature_importances_val()
    calculate_feature_importances_cv()

