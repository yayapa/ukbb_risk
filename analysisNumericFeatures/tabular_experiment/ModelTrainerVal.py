import optuna
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import importlib
import json
import os
import pickle


class ModelTrainerVal:
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
        # Convert hidden_layer_sizes to tuple if it's specified in params and is a list of lists
        if "hidden_layer_sizes" in params and isinstance(params["hidden_layer_sizes"], list):
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])

        return model_class(**params, random_state=seed)

    def _suggest_parameters(self, trial, param_config):
        params = {}
        for param_name, param_values in param_config.items():
            # Print param name and value to confirm structure before suggesting
            #print(f"Processing parameter {param_name} with values: {param_values}")
            if isinstance(param_values, list):
                if isinstance(param_values[0], list):  # For tuple-based categorical values
                    # Convert lists to tuples for Optuna compatibility
                    param_values = [tuple(val) for val in param_values]
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                elif isinstance(param_values[0], str):  # For string-based categorical values
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                elif isinstance(param_values[0], float):
                    params[param_name] = trial.suggest_float(param_name, *param_values, log=True)
                elif isinstance(param_values[0], int):
                    params[param_name] = trial.suggest_int(param_name, *param_values)
            else:
                params[param_name] = param_values
            #print(f"Suggested parameters for {param_name}: {params[param_name]}")
        return params

    def train_and_evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test, model_name, dataset_name, feature_set,
                           seed):
        """Main training and evaluation function with validation set optimization"""
        study = optuna.create_study(direction="maximize")
        model_func = lambda trial: self._objective(trial, X_train, y_train, X_val, y_val, model_name, seed)
        if model_name == "CatBoost":
            study.optimize(model_func, n_trials=round(self.config["optimization"]["n_trials"]/3),
                           timeout=self.config["optimization"]["timeout"], n_jobs=-1)
        else:
            study.optimize(model_func, n_trials=self.config["optimization"]["n_trials"],
                       timeout=self.config["optimization"]["timeout"], n_jobs=1)

        # Get the best model and train it
        best_trial = study.best_trial
        best_model = self.get_model(model_name, best_trial, seed)
        best_model.fit(X_train, y_train)

        # Save validation results
        val_metrics = best_trial.user_attrs["val_results"]
        self.save_val_results(val_metrics, model_name, dataset_name, feature_set, seed)
        self.save_best_model_and_params(best_model, best_trial.params, model_name, dataset_name, feature_set, seed)

        # Test set evaluation
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

        metrics_extended = metrics.copy()
        metrics_extended['y_pred'] = y_pred.tolist()
        metrics_extended['y_proba'] = y_pred_proba.tolist()
        metrics_extended['label'] = y_test.tolist()

        # Save test predictions
        predictions_path = os.path.join(self.base_path, dataset_name, feature_set, "predictions", model_name)
        os.makedirs(predictions_path, exist_ok=True)
        pd.DataFrame([metrics_extended]).to_csv(
            os.path.join(predictions_path, f"predictions_seed_{seed}.csv"),
            index=False
        )

        return metrics

    def _objective(self, trial, X_train, y_train, X_val, y_val, model_name, seed):
        """Objective function for hyperparameter optimization using validation set"""
        try:
            model = self.get_model(model_name, trial, seed)
            if model_name == "LightGBM":

                model.fit(X_train, y_train, feature_name="auto")
            else:
                model.fit(X_train, y_train)
        except RuntimeError as e:
            print(f"Error during model training: {e}")
            raise optuna.exceptions.TrialPruned()

        # Validate on validation set
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
        trial.set_user_attr("val_results", metrics)

        return metrics["balanced_accuracy"]  # Use balanced accuracy for optimization

    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
        }

    def save_val_results(self, val_results, model_name, dataset_name, feature_set, seed):
        """Save validation results"""
        results_path = os.path.join(self.base_path, dataset_name, feature_set, "validation", model_name)
        os.makedirs(results_path, exist_ok=True)
        val_results_df = pd.DataFrame([val_results])
        val_results_file = os.path.join(results_path, f"val_results_seed_{seed}.csv")
        val_results_df.to_csv(val_results_file, index=False)

    def save_best_model_and_params(self, best_model, best_params, model_name, dataset_name, feature_set, seed):
        """Save the best model parameters"""
        results_path = os.path.join(self.base_path, dataset_name, feature_set, "validation", model_name)
        os.makedirs(results_path, exist_ok=True)

        best_params_file = os.path.join(results_path, f"best_params_seed_{seed}.json")
        with open(best_params_file, "w") as f:
            json.dump(best_params, f)

        # Save the model using pickle
        model_file = os.path.join(results_path, f"best_model_seed_{seed}.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(best_model, f)

        print(f"Model and parameters saved to {results_path}")