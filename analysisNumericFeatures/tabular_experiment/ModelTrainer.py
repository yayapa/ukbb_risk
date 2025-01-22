import optuna
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import importlib
import json
import os


class ModelTrainer:
    def __init__(self, config, base_path):
        self.config = config
        self.base_path = base_path

    def get_model(self, model_name, trial, seed):
        model_config = self.config["models"][model_name]
        model_class_path = model_config["type"]

        # Dynamically import the model class
        module_name, class_name = model_class_path.rsplit(".", 1)
        model_module = importlib.import_module(module_name)
        model_class = getattr(model_module, class_name)

        # Suggest parameters for hyperparameter optimization
        params = self._suggest_parameters(trial, model_config["parameters"])
        return model_class(**params, random_state=seed)

    def _suggest_parameters(self, trial, param_config):
        """Suggest parameters based on configuration for Optuna trials."""
        params = {}
        for param_name, param_values in param_config.items():
            if isinstance(param_values, list):
                if isinstance(param_values[0], tuple):
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                elif isinstance(param_values[0], float):
                    params[param_name] = trial.suggest_float(param_name, *param_values, log=True)
                elif isinstance(param_values[0], int):
                    params[param_name] = trial.suggest_int(param_name, *param_values)
            else:
                params[param_name] = param_values
        return params

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, model_name, dataset_name, feature_set, seed):
        """Main training and evaluation function with Optuna optimization"""
        cv_config = self.config["cross_validation"]
        study = optuna.create_study(direction="maximize")
        model_func = lambda trial: self._objective(trial, X_train, y_train, model_name, seed)
        study.optimize(model_func, n_trials=cv_config["n_trials"], timeout=cv_config["timeout"], n_jobs=-1)

        best_trial = study.best_trial
        best_model = self.get_model(model_name, best_trial, seed)
        best_model.fit(X_train, y_train)

        self.save_cv_results(best_trial.user_attrs["cv_results"], model_name, dataset_name, feature_set, seed)
        self.save_best_model_and_params(best_model, best_trial.params, model_name, dataset_name, feature_set, seed)

        # Predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

        metrics_extended = metrics.copy()
        metrics_extended['y_pred'] = y_pred.tolist()
        metrics_extended['y_proba'] = y_pred_proba.tolist()
        metrics_extended['label'] = y_test.tolist()

        # save test predictions results
        predictions_path = os.path.join(self.base_path, dataset_name, feature_set, "predictions", model_name, f"predictions_seed_{seed}")
        pd.DataFrame([metrics_extended]).to_csv(predictions_path, index=False)

        return metrics

    def _objective(self, trial, X_train, y_train, model_name, seed):
        model = self.get_model(model_name, trial, seed)
        cv = StratifiedKFold(n_splits=self.config["cross_validation"]["n_folds"])
        cv_scores = []
        cv_results = []

        for fold_idx, (train_idx, val_idx ) in enumerate(cv.split(X_train, y_train)):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            y_pred_proba = model.predict_proba(X_val_fold)[:, 1]

            metrics = self._calculate_metrics(y_val_fold, y_pred, y_pred_proba)
            metrics["fold"] = fold_idx
            cv_results.append(metrics)
            cv_scores.append(metrics["balanced_accuracy"])  # Use a metric for optimization

        # Save CV results in trial attributes for later use
        trial.set_user_attr("cv_results", cv_results)

        return np.mean(cv_scores)

    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
        }


    def save_cv_results(self, cv_results, model_name, dataset_name, feature_set, seed):
        """Save cross-validation results as CSV with seed-specific directory."""
        results_path = os.path.join(self.base_path, dataset_name, feature_set, "cv", model_name)
        os.makedirs(results_path, exist_ok=True)
        cv_results_df = pd.DataFrame(cv_results)
        cv_results_file = os.path.join(results_path, f"cv_results_seed_{seed}.csv")
        cv_results_df.to_csv(cv_results_file, index=False)

    def save_best_model_and_params(self, best_model, best_params, model_name, dataset_name, feature_set, seed):
        """Save the best model and parameters with seed-specific directory."""
        results_path = os.path.join(self.base_path, dataset_name, feature_set, "cv", model_name)
        os.makedirs(results_path, exist_ok=True)

        # Save best hyperparameters as JSON
        best_params_file = os.path.join(results_path, f"best_params_seed_{seed}.json")
        with open(best_params_file, "w") as f:
            json.dump(best_params, f)

        # Save the best model using joblib
        #best_model_file = os.path.join(results_path, "best_model.pkl")
        #joblib.dump(best_model, best_model_file)
