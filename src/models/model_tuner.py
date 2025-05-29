import yaml
import optuna
import optuna.visualization as vis
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from loggers import logger_main as logger
import joblib
import os


class ModelTunerEvaluator:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, n_splits=5, n_trials=50, random_state=0):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.studies = {}
        self.random_state = random_state
        self.cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.models_dir = 'models'
        os.makedirs(self.models_dir, exist_ok=True)

    def _objective_svm(self, trial):
        C = trial.suggest_float('C', 1e-3, 1e3, log=True)
        kernel = trial.suggest_categorical(
            'kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
        gamma = 'scale'
        if kernel in ['rbf', 'poly', 'sigmoid']:
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

        degree = 3
        if kernel == 'poly':
            degree = trial.suggest_int('degree', 2, 5)

        model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            class_weight='balanced',
            probability=True,
            random_state=self.random_state
        )

        scores = []
        for train_idx, val_idx in self.cv.split(self.X_train, self.y_train):
            X_train_fold, X_val_fold = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train_fold, y_val_fold = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            model.fit(X_train_fold, y_train_fold)
            y_pred_fold = model.predict(X_val_fold)
            scores.append(f1_score(y_val_fold, y_pred_fold,
                          average='weighted', zero_division=0))

        return np.mean(scores)

    def _objective_random_forest(self, trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 3, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        max_features = trial.suggest_categorical(
            'max_features', ['sqrt', 'log2', None])

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )

        scores = []
        for train_idx, val_idx in self.cv.split(self.X_train, self.y_train):
            X_train_fold, X_val_fold = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train_fold, y_val_fold = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            model.fit(X_train_fold, y_train_fold)

            y_pred_fold = model.predict(X_val_fold)
            scores.append(f1_score(y_val_fold, y_pred_fold,
                          average='weighted', zero_division=0))

        return np.mean(scores)

    def _objective_lightgbm(self, trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        learning_rate = trial.suggest_float(
            'learning_rate', 1e-3, 0.3, log=True)
        num_leaves = trial.suggest_int('num_leaves', 20, 150)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        reg_alpha = trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True)
        reg_lambda = trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        min_child_samples = trial.suggest_int('min_child_samples', 5, 50)

        model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            min_child_samples=min_child_samples,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )

        scores = []
        for train_idx, val_idx in self.cv.split(self.X_train, self.y_train):
            X_train_fold, X_val_fold = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train_fold, y_val_fold = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            callbacks = [early_stopping(stopping_rounds=10, verbose=False)]
            model.fit(X_train_fold, y_train_fold, eval_set=[
                      (X_val_fold, y_val_fold)], callbacks=callbacks)

            y_pred_fold = model.predict(X_val_fold)
            scores.append(f1_score(y_val_fold, y_pred_fold,
                          average='weighted', zero_division=0))

        return np.mean(scores)

    def _evaluate_model(self, model, X, y, prefix):
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix
        )

        y_pred = model.predict(X)

        try:
            y_prob = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_prob)
        except (AttributeError, IndexError, ValueError):
            auc = None

        try:
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        except ValueError:
            tn = fp = fn = tp = None

        metrics = {
            f'{prefix}_accuracy': accuracy_score(y, y_pred),
            f'{prefix}_precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            f'{prefix}_recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            f'{prefix}_f1_score': f1_score(y, y_pred, average='weighted', zero_division=0),
            f'{prefix}_auc_roc': auc,
            f'{prefix}_tp': int(tp) if tp is not None else None,
            f'{prefix}_tn': int(tn) if tn is not None else None,
            f'{prefix}_fp': int(fp) if fp is not None else None,
            f'{prefix}_fn': int(fn) if fn is not None else None
        }

        return metrics


    def tune_and_evaluate(self, model_name):
        logger.info(f"Starting hyperparameter tuning for {model_name}...")

        supported_types = ['svm', 'random_forest', 'lightgbm']
        model_type = next((m for m in supported_types if model_name.startswith(m)), None)

        objective_map = {
            'svm': self._objective_svm,
            'random_forest': self._objective_random_forest,
            'lightgbm': self._objective_lightgbm
        }

        if model_type not in objective_map:
            logger.error(f"Unsupported model type: {model_type}")
            return None, None, None

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(objective_map[model_type], n_trials=self.n_trials, n_jobs=-1)

        self.studies[model_name] = study

        best_params = study.best_params
        logger.info(f"Best hyperparameters for {model_name}: {best_params}")

        logger.info(f"Training final {model_name} model with best hyperparameters...")
        if model_type == 'svm':
            final_model = SVC(**best_params, class_weight='balanced', probability=True, random_state=self.random_state)
        elif model_type == 'random_forest':
            final_model = RandomForestClassifier(**best_params, class_weight='balanced', random_state=self.random_state, n_jobs=-1)
        elif model_type == 'lightgbm':
            final_model = LGBMClassifier(**best_params, class_weight='balanced', random_state=self.random_state, n_jobs=-1, verbose=-1)

        final_model.fit(self.X_train, self.y_train)

        model_filename = os.path.join(self.models_dir, f'{model_name}_tuned_model.joblib')
        joblib.dump(final_model, model_filename)
        logger.info(f"Saved tuned {model_name} model to {model_filename}")

        logger.info(f"Evaluating final {model_name} model...")
        train_metrics = self._evaluate_model(final_model, self.X_train, self.y_train, 'train')
        val_metrics = self._evaluate_model(final_model, self.X_val, self.y_val, 'val')
        test_metrics = self._evaluate_model(final_model, self.X_test, self.y_test, 'test')

        all_metrics = {**train_metrics, **val_metrics, **test_metrics}
        logger.info(f"{model_name} evaluation complete. Metrics: {all_metrics}")

        return final_model, best_params, all_metrics


    
    def run_from_config(self, config_path='config/training_config.yaml', plots_path = 'results/optuna'):
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        models_cfg = config['models']
        results = {}

        for model_name, model_cfg in models_cfg.items():
            tune = model_cfg.get('tune', False)
            static_params = model_cfg.get('params', {})

            if not tune:
                logger.info(f"Skipping tuning for {model_name} (tune: false)")
                continue

            logger.info(f"Tuning model: {model_name}")
            model, best_params, metrics = self.tune_and_evaluate(model_name)

            if model is None:
                logger.error(f"Tuning failed for {model_name}")
                continue

            self.save_tuning_plots(self.studies[model_name], model_name, output_dir=plots_path)

            results[model_name] = {
                'model': model,
                'static_params': static_params,
                'tuned_params': best_params,
                'final_params': best_params,
                'metrics': metrics
            }

        return results
    
    def save_tuning_plots(self, study, model_name, output_dir='results/optuna'):

        os.makedirs(output_dir, exist_ok=True)

        vis.plot_optimization_history(study).write_html(f"{output_dir}/{model_name}_opt_history.html")
        vis.plot_param_importances(study).write_html(f"{output_dir}/{model_name}_param_importance.html")
        vis.plot_contour(study).write_html(f"{output_dir}/{model_name}_contour.html")
        vis.plot_parallel_coordinate(study).write_html(f"{output_dir}/{model_name}_parallel.html")
        vis.plot_slice(study).write_html(f"{output_dir}/{model_name}_slice.html")

