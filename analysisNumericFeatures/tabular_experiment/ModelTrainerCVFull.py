import optuna
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import importlib
import json
import os

class ModelTrainerCVFull:
    def __init__(self, config, base_path):
        self.config = config
        self.base_path = base_path

    def get_model(self, model_name, trial, seed):
        model_config = self.config["models"][model_name]
        model_class_path = model_config["type"]

        module_name, class_name = model_class_path.rsplit(".", 1)
        model_module = importlib.import_module(module_name)
        model_class = getattr(model_module, class_name)

        params = self._suggest_parameters(trial, model_config["parameters"])
        if "hidden_layer_sizes" in params and isinstance(params["hidden_layer_sizes"], list):
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])

        return model_class(**params, random_state=seed)

    def _suggest_parameters(self, trial, param_config):
        params = {}
        for param_name, param_values in param_config.items():
            if isinstance(param_values, list):
                if isinstance(param_values[0], list):
                    param_values = [tuple(val) for val in param_values]
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                elif isinstance(param_values[0], str):
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                elif isinstance(param_values[0], float):
                    params[param_name] = trial.suggest_float(param_name, *param_values, log=True)
                elif isinstance(param_values[0], int):
                    params[param_name] = trial.suggest_int(param_name, *param_values)
            else:
                params[param_name] = param_values
        return params

    def train_and_evaluate(self, X, y, model_name, dataset_name, feature_set, seed):
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        outer_results = []

        for fold_idx, (train_val_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            X_train_val, X_test = X.iloc[train_val_idx], X.iloc[test_idx]
            y_train_val, y_test = y.iloc[train_val_idx], y.iloc[test_idx]

            # Inner CV for hyperparameter optimization
            inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
            study = optuna.create_study(direction="maximize")
            model_func = lambda trial: self._objective_cv(trial, X_train_val, y_train_val, model_name, inner_cv, seed)
            study.optimize(model_func, n_trials=self.config["optimization"]["n_trials"],
                           timeout=self.config["optimization"]["timeout"], n_jobs=-1)

            # Train the best model on the entire training-validation set
            best_trial = study.best_trial
            best_model = self.get_model(model_name, best_trial, seed)
            best_model.fit(X_train_val, y_train_val)

            # Test set evaluation
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

            metrics_extended = metrics.copy()
            metrics_extended['fold'] = fold_idx
            outer_results.append(metrics_extended)

        # Aggregate outer fold results
        outer_results_df = pd.DataFrame(outer_results)
        summary_metrics = outer_results_df.mean().to_dict()  # Calculate mean metrics
        summary_metrics_std = outer_results_df.std().to_dict()  # Calculate std metrics

        # Combine mean and std metrics for easy logging and saving
        summary_metrics_combined = {f"{k}_mean": v for k, v in summary_metrics.items()}
        summary_metrics_combined.update({f"{k}_std": v for k, v in summary_metrics_std.items()})

        return summary_metrics_combined

    def _objective_cv(self, trial, X_train, y_train, model_name, cv, seed):
        """Objective function for hyperparameter optimization with cross-validation in the inner loop."""
        model = self.get_model(model_name, trial, seed)
        # Cross-validation in the inner loop to evaluate hyperparameters
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
        return np.mean(scores)

    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
        }

    def save_fold_results(self, fold_results, model_name, dataset_name, feature_set, seed, fold_idx):
        results_path = os.path.join(self.base_path, dataset_name, feature_set, "fold_results", model_name)
        os.makedirs(results_path, exist_ok=True)
        fold_results_df = pd.DataFrame([fold_results])
        fold_results_file = os.path.join(results_path, f"fold_{fold_idx}_results_seed_{seed}.csv")
        fold_results_df.to_csv(fold_results_file, index=False)

    def save_best_model_and_params(self, best_model, best_params, model_name, dataset_name, feature_set, seed, fold_idx):
        results_path = os.path.join(self.base_path, dataset_name, feature_set, "fold_results", model_name)
        os.makedirs(results_path, exist_ok=True)

        best_params_file = os.path.join(results_path, f"best_params_seed_{seed}_fold_{fold_idx}.json")
        with open(best_params_file, "w") as f:
            json.dump(best_params, f)
