import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
from loggers import logger_transformer as logger

class DataTransformer:
    def __init__(self, config_path='config/processing_config.yaml'):
        self.config = self._load_config(config_path)
        self.scalers = {}
        self.power_transformers = {}
        self.fitted = False

    def _load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def _get_feature_names(self, transform_type):
        return self.config.get(transform_type, {}).get('names', [])

    def apply_power_transform(self, df, fit=True):
        features = self._get_feature_names('power_transform')
        df_transformed = df.copy()

        for feature in features:
            if feature in df_transformed.columns:
                transformer = PowerTransformer(
                    method='yeo-johnson', standardize=False)
                data_reshaped = df_transformed[feature].values.reshape(-1, 1)

                if fit:
                    transformer.fit(data_reshaped)
                    self.power_transformers[feature] = transformer

                if feature in self.power_transformers:
                    df_transformed[feature] = self.power_transformers[feature].transform(
                        data_reshaped)
                    logger.info(
                        f"Applied Yeo-Johnson transformation to {feature}")
                else:
                    logger.warning(
                        f"Transformer not fitted for {feature}.")
            else:
                logger.warning(
                    f"Feature {feature} not found in power transformations.")

        return df_transformed

    def apply_standard_scaling(self, df, fit=True):
        features = self._get_feature_names('standard_scale')
        df_scaled = df.copy()

        for feature in features:
            if feature in df_scaled.columns:
                scaler = StandardScaler()
                data_reshaped = df_scaled[feature].values.reshape(-1, 1)

                if fit:
                    scaler.fit(data_reshaped)
                    self.scalers[feature] = scaler

                if feature in self.scalers:
                    df_scaled[feature] = self.scalers[feature].transform(
                        data_reshaped)
                    logger.info(f"Applied Standard Scaler to {feature}")
                else:
                    logger.warning(
                        f"Scaler not fitted for {feature}.")
            else:
                logger.warning(
                    f"Feature {feature} not found in data for standard scaling.")

        return df_scaled

    def apply_one_hot_encoding(self, df):
        features = self._get_feature_names('one_hot_encoding')
        df_encoded = df.copy()
        features_to_encode = [f for f in features if f in df_encoded.columns]

        if features_to_encode:
            logger.info(f"Applying One-Hot Encoding to: {features_to_encode}")
            for col in features_to_encode:
                df_encoded[col] = pd.Categorical(df_encoded[col])

            df_encoded = pd.get_dummies(
                df_encoded, columns=features_to_encode, prefix=features_to_encode, dummy_na=False)
            logger.info(
                f"Columns after One-Hot Encoding: {df_encoded.columns.tolist()}")
        else:
            logger.warning(
                "No features to apply One-Hot Encoding.")

        return df_encoded

    def apply_min_max_scaling(self, df, fit=True):
        features = self._get_feature_names('min_max_scale')
        if not features:
            logger.info(
                "No features specified for MinMax scaling.")
            return df

        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(
            df[features]) if fit else scaler.transform(df[features])
        return df

    def apply_log_transform(self, df):
        features = self._get_feature_names('log_transform')
        if not features:
            logger.info(
                "No features specified for log transformation (placeholder).")
            return df

        df[features] = df[features].apply(lambda x: np.log1p(x))
        return df

    def transform(self, df, fit=True):
        df_temp = df.copy()
        df_temp = self.apply_power_transform(df_temp, fit=fit)
        df_temp = self.apply_standard_scaling(df_temp, fit=fit)
        # df_temp = self.apply_min_max_scaling(df_temp, fit=fit)
        # df_temp = self.apply_log_transform(df_temp)
        df_temp = self.apply_one_hot_encoding(df_temp)
        return df_temp

    def fit_transform(self, df):
        return self.transform(df, fit=True)
