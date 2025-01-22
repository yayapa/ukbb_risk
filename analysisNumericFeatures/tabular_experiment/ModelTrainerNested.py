import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score
import importlib
import os
import json


class ModelTrainerNested:
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

    # Keeping get_model and _suggest_parameters methods same as your original code

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, model_name, dataset_name, feature_set, seed):
        """Nested CV training and evaluation"""
        outer_cv = StratifiedKFold(n_splits=self.config["cross_validation"]["n_folds"])
        outer_scores = []
        outer_best_params = []

        # Outer CV loop
        for outer_fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train)):
            X_train_outer, X_val_outer = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_outer, y_val_outer = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Inner optimization with Optuna
            study = optuna.create_study(direction="maximize")
            model_func = lambda trial: self._inner_objective(trial, X_train_outer, y_train_outer, model_name, seed)
            study.optimize(model_func, n_trials=self.config["cross_validation"]["n_trials"], n_jobs=-1, timeout=self.config["cross_validation"]["timeout"])

            # Get best model from inner CV
            best_trial = study.best_trial
            best_model = self.get_model(model_name, best_trial, seed)
            best_model.fit(X_train_outer, y_train_outer)

            # Evaluate on outer fold
            y_val_pred = best_model.predict(X_val_outer)
            y_val_proba = best_model.predict_proba(X_val_outer)[:, 1]
            metrics = self._calculate_metrics(y_val_outer, y_val_pred, y_val_proba)
            metrics["outer_fold"] = outer_fold

            outer_scores.append(metrics)
            outer_best_params.append(best_trial.params)

        # Save outer CV results
        self.save_cv_results(outer_scores, model_name, dataset_name, feature_set, seed)

        # Train final model using most frequent best parameters
        final_params = self._get_most_frequent_params(outer_best_params)

        # Create final model directly with parameters
        model_config = self.config["models"][model_name]
        module_name, class_name = model_config["type"].rsplit(".", 1)
        model_module = importlib.import_module(module_name)
        model_class = getattr(model_module, class_name)
        final_model = model_class(**final_params, random_state=seed)

        final_model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = final_model.predict(X_test)
        y_pred_proba = final_model.predict_proba(X_test)[:, 1]
        test_metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

        # Save test predictions
        self._save_predictions(test_metrics, y_pred, y_pred_proba, y_test,
                               model_name, dataset_name, feature_set, seed)

        return test_metrics

    def _inner_objective(self, trial, X_train, y_train, model_name, seed):
        """Inner cross-validation for hyperparameter optimization"""
        inner_cv = StratifiedKFold(n_splits=self.config["cross_validation"]["n_folds"])
        model = self.get_model(model_name, trial, seed)
        scores = []

        for train_idx, val_idx in inner_cv.split(X_train, y_train):
            X_train_inner, X_val_inner = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_inner, y_val_inner = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_train_inner, y_train_inner)
            y_val_pred = model.predict(X_val_inner)
            scores.append(balanced_accuracy_score(y_val_inner, y_val_pred))

        return np.mean(scores)

    def _get_most_frequent_params(self, params_list):
        """Get most frequently selected parameters across outer folds"""
        param_counts = {}
        for params in params_list:
            param_str = json.dumps(params, sort_keys=True)
            param_counts[param_str] = param_counts.get(param_str, 0) + 1

        most_frequent = max(param_counts.items(), key=lambda x: x[1])[0]
        return json.loads(most_frequent)

    def _save_predictions(self, metrics, y_pred, y_pred_proba, y_test,
                          model_name, dataset_name, feature_set, seed):
        metrics_extended = metrics.copy()
        metrics_extended['y_pred'] = y_pred.tolist()
        metrics_extended['y_proba'] = y_pred_proba.tolist()
        metrics_extended['label'] = y_test.tolist()

        predictions_path = os.path.join(self.base_path, dataset_name, feature_set,
                                        "predictions", model_name, f"predictions_seed_{seed}")
        pd.DataFrame([metrics_extended]).to_csv(predictions_path, index=False)

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
