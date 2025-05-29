import yaml
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score, roc_auc_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

from loggers import logger_trainer as logger



class Trainer:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, training_config_path):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.config = self._load_config(training_config_path)
        self.learning_curve = {
            'accuracy': [],
            'recall': [],
            'f1_score': []
        }

    def _load_config(self, path):
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    def train(self, model_name: str):
        if self.X_train is None or self.y_train is None or self.X_val is None or self.y_val is None:
            logger.error("Training data is not available.")
            return None, None

        if model_name not in self.config.get('models', {}):
            logger.warning(f"No training config for model '{model_name}'")
            return None, None

        available_model_types = ['svm', 'random_forest', 'lightgbm']
        matched_type = next((m for m in available_model_types if model_name.startswith(m)), None)

        if matched_type is None:
            logger.warning(f"No training method for model '{model_name}' (unrecognized prefix)")
            return None, None

        train_fn = getattr(self, f'_train_{matched_type}', None)
        if train_fn is None:
            logger.warning(f"No training function implemented for '{matched_type}'")
            return None, None

        model_config = self.config['models'][model_name]
        return train_fn(model_config)


    def _train_svm(self, config):
        params = config.get('params', {})

        logger.info(f"Initializing SVM model with config: {config}")
        model = SVC(**params)

        logger.info("Training SVM model...")
        model.fit(self.X_train, self.y_train)

        # --- TRAIN SET ---
        y_train_pred = model.predict(self.X_train)
        y_train_prob = model.predict_proba(self.X_train)[:, 1] if hasattr(model, 'predict_proba') else None
        train_acc = accuracy_score(self.y_train, y_train_pred)
        train_rec = recall_score(self.y_train, y_train_pred, average='weighted', zero_division=0)
        train_prec = precision_score(self.y_train, y_train_pred, average='weighted', zero_division=0)
        train_f1 = f1_score(self.y_train, y_train_pred, average='weighted', zero_division=0)
        train_auc = roc_auc_score(self.y_train, y_train_prob) if y_train_prob is not None else None
        try:
            tn_train, fp_train, fn_train, tp_train = confusion_matrix(self.y_train, y_train_pred).ravel()
        except ValueError:
            tn_train = fp_train = fn_train = tp_train = None

        logger.info(f"Training Accuracy: {train_acc:.4f}")

        # --- VALIDATION SET ---
        logger.info("SVM model training completed. Evaluating on validation set...")
        y_val_pred = model.predict(self.X_val)
        y_val_prob = model.predict_proba(self.X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        val_acc = accuracy_score(self.y_val, y_val_pred)
        val_rec = recall_score(self.y_val, y_val_pred, average='weighted', zero_division=0)
        val_prec = precision_score(self.y_val, y_val_pred, average='weighted', zero_division=0)
        val_f1 = f1_score(self.y_val, y_val_pred, average='weighted', zero_division=0)
        val_auc = roc_auc_score(self.y_val, y_val_prob) if y_val_prob is not None else None
        try:
            tn_val, fp_val, fn_val, tp_val = confusion_matrix(self.y_val, y_val_pred).ravel()
        except ValueError:
            tn_val = fp_val = fn_val = tp_val = None

        logger.info(f"Validation Accuracy: {val_acc:.4f}")

        # --- TEST SET ---
        logger.info("Evaluating on test set...")
        y_test_pred = model.predict(self.X_test)
        y_test_prob = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        test_acc = accuracy_score(self.y_test, y_test_pred)
        test_rec = recall_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        test_prec = precision_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        test_auc = roc_auc_score(self.y_test, y_test_prob) if y_test_prob is not None else None
        try:
            tn_test, fp_test, fn_test, tp_test = confusion_matrix(self.y_test, y_test_pred).ravel()
        except ValueError:
            tn_test = fp_test = fn_test = tp_test = None

        self.learning_curve['accuracy'].append(val_acc)
        self.learning_curve['recall'].append(val_rec)
        self.learning_curve['f1_score'].append(val_f1)

        return model, {
            # TRAIN
            'train_accuracy': train_acc,
            'train_precision': train_prec,
            'train_recall': train_rec,
            'train_f1_score': train_f1,
            'train_auc_roc': train_auc,
            'train_tp': int(tp_train) if tp_train is not None else None,
            'train_tn': int(tn_train) if tn_train is not None else None,
            'train_fp': int(fp_train) if fp_train is not None else None,
            'train_fn': int(fn_train) if fn_train is not None else None,
            # VAL
            'val_accuracy': val_acc,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_f1_score': val_f1,
            'val_auc_roc': val_auc,
            'val_tp': int(tp_val) if tp_val is not None else None,
            'val_tn': int(tn_val) if tn_val is not None else None,
            'val_fp': int(fp_val) if fp_val is not None else None,
            'val_fn': int(fn_val) if fn_val is not None else None,
            # TEST
            'test_accuracy': test_acc,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'test_f1_score': test_f1,
            'test_auc_roc': test_auc,
            'test_tp': int(tp_test) if tp_test is not None else None,
            'test_tn': int(tn_test) if tn_test is not None else None,
            'test_fp': int(fp_test) if fp_test is not None else None,
            'test_fn': int(fn_test) if fn_test is not None else None
        }



    
    def _train_random_forest(self, config):
        params = config.get('params', {})
        logger.info(f"Initializing Random Forest model with config: {params}")
        model = RandomForestClassifier(**params)
        model.fit(self.X_train, self.y_train)

        # --- TRAIN SET ---
        y_train_pred = model.predict(self.X_train)
        y_train_prob = model.predict_proba(self.X_train)[:, 1] if hasattr(model, 'predict_proba') else None

        train_acc = accuracy_score(self.y_train, y_train_pred)
        train_rec = recall_score(self.y_train, y_train_pred, average='weighted', zero_division=0)
        train_prec = precision_score(self.y_train, y_train_pred, average='weighted', zero_division=0)
        train_f1 = f1_score(self.y_train, y_train_pred, average='weighted', zero_division=0)
        train_auc = roc_auc_score(self.y_train, y_train_prob) if y_train_prob is not None else None

        try:
            tn_train, fp_train, fn_train, tp_train = confusion_matrix(self.y_train, y_train_pred).ravel()
        except ValueError:
            tn_train = fp_train = fn_train = tp_train = None

        logger.info(f"Training Accuracy: {train_acc:.4f}")

        # --- VALIDATION SET ---
        y_val_pred = model.predict(self.X_val)
        y_val_prob = model.predict_proba(self.X_val)[:, 1] if hasattr(model, 'predict_proba') else None

        val_acc = accuracy_score(self.y_val, y_val_pred)
        val_rec = recall_score(self.y_val, y_val_pred, average='weighted', zero_division=0)
        val_prec = precision_score(self.y_val, y_val_pred, average='weighted', zero_division=0)
        val_f1 = f1_score(self.y_val, y_val_pred, average='weighted', zero_division=0)
        val_auc = roc_auc_score(self.y_val, y_val_prob) if y_val_prob is not None else None

        try:
            tn_val, fp_val, fn_val, tp_val = confusion_matrix(self.y_val, y_val_pred).ravel()
        except ValueError:
            tn_val = fp_val = fn_val = tp_val = None

        logger.info(f"Validation Accuracy: {val_acc:.4f}")

        # --- TEST SET ---
        y_test_pred = model.predict(self.X_test)
        y_test_prob = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        test_acc = accuracy_score(self.y_test, y_test_pred)
        test_rec = recall_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        test_prec = precision_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        test_auc = roc_auc_score(self.y_test, y_test_prob) if y_test_prob is not None else None

        try:
            tn_test, fp_test, fn_test, tp_test = confusion_matrix(self.y_test, y_test_pred).ravel()
        except ValueError:
            tn_test = fp_test = fn_test = tp_test = None

        logger.info(f"Test Accuracy: {test_acc:.4f}")

        self.learning_curve['accuracy'].append(val_acc)
        self.learning_curve['recall'].append(val_rec)
        self.learning_curve['f1_score'].append(val_f1)

        return model, {
            # TRAIN
            'train_accuracy': train_acc,
            'train_precision': train_prec,
            'train_recall': train_rec,
            'train_f1_score': train_f1,
            'train_auc_roc': train_auc,
            'train_tp': int(tp_train) if tp_train is not None else None,
            'train_tn': int(tn_train) if tn_train is not None else None,
            'train_fp': int(fp_train) if fp_train is not None else None,
            'train_fn': int(fn_train) if fn_train is not None else None,
            # VAL
            'val_accuracy': val_acc,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_f1_score': val_f1,
            'val_auc_roc': val_auc,
            'val_tp': int(tp_val) if tp_val is not None else None,
            'val_tn': int(tn_val) if tn_val is not None else None,
            'val_fp': int(fp_val) if fp_val is not None else None,
            'val_fn': int(fn_val) if fn_val is not None else None,
            # TEST
            'test_accuracy': test_acc,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'test_f1_score': test_f1,
            'test_auc_roc': test_auc,
            'test_tp': int(tp_test) if tp_test is not None else None,
            'test_tn': int(tn_test) if tn_test is not None else None,
            'test_fp': int(fp_test) if fp_test is not None else None,
            'test_fn': int(fn_test) if fn_test is not None else None
        }




    
    def _train_lightgbm(self, config):
        from lightgbm import LGBMClassifier

        params = config.get('params', {})
        logger.info(f"Initializing LightGBM model with config: {params}")
        model = LGBMClassifier(**params)
        model.fit(self.X_train, self.y_train)

        # --- TRAIN SET ---
        y_train_pred = model.predict(self.X_train)
        y_train_prob = model.predict_proba(self.X_train)[:, 1] if hasattr(model, 'predict_proba') else None

        train_acc = accuracy_score(self.y_train, y_train_pred)
        train_rec = recall_score(self.y_train, y_train_pred, average='weighted', zero_division=0)
        train_prec = precision_score(self.y_train, y_train_pred, average='weighted', zero_division=0)
        train_f1 = f1_score(self.y_train, y_train_pred, average='weighted', zero_division=0)
        train_auc = roc_auc_score(self.y_train, y_train_prob) if y_train_prob is not None else None

        try:
            tn_train, fp_train, fn_train, tp_train = confusion_matrix(self.y_train, y_train_pred).ravel()
        except ValueError:
            tn_train = fp_train = fn_train = tp_train = None

        logger.info(f"Training Accuracy: {train_acc:.4f}")

        # --- VALIDATION SET ---
        y_val_pred = model.predict(self.X_val)
        y_val_prob = model.predict_proba(self.X_val)[:, 1] if hasattr(model, 'predict_proba') else None

        val_acc = accuracy_score(self.y_val, y_val_pred)
        val_rec = recall_score(self.y_val, y_val_pred, average='weighted', zero_division=0)
        val_prec = precision_score(self.y_val, y_val_pred, average='weighted', zero_division=0)
        val_f1 = f1_score(self.y_val, y_val_pred, average='weighted', zero_division=0)
        val_auc = roc_auc_score(self.y_val, y_val_prob) if y_val_prob is not None else None

        try:
            tn_val, fp_val, fn_val, tp_val = confusion_matrix(self.y_val, y_val_pred).ravel()
        except ValueError:
            tn_val = fp_val = fn_val = tp_val = None

        logger.info(f"Validation Accuracy: {val_acc:.4f}")

        # --- TEST SET ---
        y_test_pred = model.predict(self.X_test)
        y_test_prob = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        test_acc = accuracy_score(self.y_test, y_test_pred)
        test_rec = recall_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        test_prec = precision_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        test_auc = roc_auc_score(self.y_test, y_test_prob) if y_test_prob is not None else None

        try:
            tn_test, fp_test, fn_test, tp_test = confusion_matrix(self.y_test, y_test_pred).ravel()
        except ValueError:
            tn_test = fp_test = fn_test = tp_test = None

        logger.info(f"Test Accuracy: {test_acc:.4f}")

        self.learning_curve['accuracy'].append(val_acc)
        self.learning_curve['recall'].append(val_rec)
        self.learning_curve['f1_score'].append(val_f1)

        return model, {
            # --- TRAIN ---
            'train_accuracy': train_acc,
            'train_precision': train_prec,
            'train_recall': train_rec,
            'train_f1_score': train_f1,
            'train_auc_roc': train_auc,
            'train_tp': int(tp_train) if tp_train is not None else None,
            'train_tn': int(tn_train) if tn_train is not None else None,
            'train_fp': int(fp_train) if fp_train is not None else None,
            'train_fn': int(fn_train) if fn_train is not None else None,
            # --- VAL ---
            'val_accuracy': val_acc,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_f1_score': val_f1,
            'val_auc_roc': val_auc,
            'val_tp': int(tp_val) if tp_val is not None else None,
            'val_tn': int(tn_val) if tn_val is not None else None,
            'val_fp': int(fp_val) if fp_val is not None else None,
            'val_fn': int(fn_val) if fn_val is not None else None,
            # --- TEST ---
            'test_accuracy': test_acc,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'test_f1_score': test_f1,
            'test_auc_roc': test_auc,
            'test_tp': int(tp_test) if tp_test is not None else None,
            'test_tn': int(tn_test) if tn_test is not None else None,
            'test_fp': int(fp_test) if fp_test is not None else None,
            'test_fn': int(fn_test) if fn_test is not None else None
        }
    
    def get_default_params(self, model_name):
        if model_name.startswith("svm"):
            return SVC().get_params()
        elif model_name.startswith("random_forest"):
            return RandomForestClassifier().get_params()
        elif model_name.startswith("lightgbm"):
            return LGBMClassifier().get_params()
        return {}






