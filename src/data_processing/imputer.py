import pandas as pd
from sklearn.impute import KNNImputer
from loggers import logger_imputer as logger
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class Imputer:
    def __init__(self, config):
        self.config = config
        self.fitted_params = {}
        self.fitted_imputers = {}
        self.numeric_cols_knn = None
        self.numeric_cols_mice = None
        self.fitted = False

    def fit(self, df):
        logger.info("Fitting imputer...")
        df_copy = df.copy()
        self.fitted_params = {}
        self.fitted_imputers = {}
        self.numeric_cols_knn = None
        self.numeric_cols_mice = None

        cols_to_impute = [
            col for col in df_copy.columns if df_copy[col].isna().sum() > 0
        ]

        mice_needed = any(self.config['columns'].get(col, {}).get(
            'imputation') == 'mice' for col in cols_to_impute)

        if mice_needed:
            logger.info("Fitting global MICE (Iterative Imputer)...")
            self.numeric_cols_mice = df_copy.select_dtypes(
                include=['float64', 'int64']).columns.tolist()
            if not self.numeric_cols_mice:
                logger.warning(
                    "No numeric columns found for MICE imputation. Skipping..")
            else:
                mice_imputer = IterativeImputer(max_iter=10, random_state=0)
                mice_imputer.fit(df_copy[self.numeric_cols_mice])
                self.fitted_imputers['mice'] = mice_imputer
                logger.info("Fitted global MICE imputer.")

        for col in df_copy.columns:
            strategy = self.config['columns'].get(
                col, {}).get('imputation', 'mean')

            if strategy == 'mean':
                self.fitted_params[col] = df_copy[col].mean()
                logger.info(
                    f"Calculated mean for '{col}': {self.fitted_params[col]}")

            elif strategy == 'median':
                self.fitted_params[col] = df_copy[col].median()
                logger.info(
                    f"Calculated median for '{col}': {self.fitted_params[col]}")

            elif strategy == 'mode':
                mode_val = df_copy[col].mode()
                self.fitted_params[col] = mode_val[0]
                logger.info(
                    f"Calculated mode for '{col}': {self.fitted_params[col]}")
                
            elif strategy == 'mice':
                pass

            elif strategy == 'knn':
                if self.numeric_cols_knn is None:
                    self.numeric_cols_knn = df_copy.select_dtypes(
                        include=['float64', 'int64']).columns.tolist()

                if 'knn' not in self.fitted_imputers:
                    logger.info("Fitting KNN imputer...")
                    knn_imputer = KNNImputer(n_neighbors=5)
                    knn_imputer.fit(df_copy[self.numeric_cols_knn])
                    self.fitted_imputers['knn'] = knn_imputer
                    logger.info("Fitted KNN imputer.")
            else:
                logger.warning(
                    f"Unknown imputation method '{strategy}' for '{col}' feature. Calculating mean.")
                self.fitted_params[col] = df_copy[col].mean()

        self.fitted = True
        logger.info("Imputer fitting complete.")
        return self

    def transform(self, df):
        logger.info("Transforming data using imputer that was fit...")
        df_copy = df.copy()

        mice_applied = False
        if 'mice' in self.fitted_imputers:
            logger.info("Global MICE transformation...")
            cols_present = [
                col for col in self.numeric_cols_mice if col in df_copy.columns]

            if cols_present:
                imputed_data = self.fitted_imputers['mice'].transform(
                    df_copy[cols_present])
                df_imputed_mice = pd.DataFrame(
                    imputed_data, columns=cols_present, index=df_copy.index)

                for col in cols_present:
                    df_copy[col] = df_imputed_mice[col]

                logger.info(
                    "Global MICE transformation applied.")
                df_copy[cols_present] = df_copy[cols_present].round()
                mice_applied = True

        knn_applied = False
        if 'knn' in self.fitted_imputers:
            logger.info("KNN transformation...")
            cols_present = [
                col for col in self.numeric_cols_knn if col in df_copy.columns]

            if cols_present:
                if df_copy[cols_present].isna().sum().sum() > 0:
                    imputed_data = self.fitted_imputers['knn'].transform(
                        df_copy[cols_present])
                    df_imputed_knn = pd.DataFrame(
                        imputed_data, columns=cols_present, index=df_copy.index)
                    for col in cols_present:
                        df_copy[col] = df_imputed_knn[col]
                    logger.info("KNN transformation applied.")
                    knn_applied = True
                else:
                    logger.info(
                        "No missing values found in KNN imputation method columns. Skipping.")

        for col in df_copy.columns:
            if df_copy[col].isna().sum() > 0:
                strategy = self.config['columns'].get(
                    col, {}).get('imputation', 'mean')

                if strategy == 'mice' and mice_applied and col in self.numeric_cols_mice and df_copy[col].isna().sum() == 0:
                    continue
                if strategy == 'knn' and knn_applied and col in self.numeric_cols_knn and df_copy[col].isna().sum() == 0:
                    continue

                if col in self.fitted_params:
                    fill_value = self.fitted_params[col]
                    df_copy[col] = df_copy[col].fillna(fill_value)
                    logger.info(
                        f"Imputed remaining nan values in '{col}' feature with calculated value: {fill_value}")
                elif f"{col}_knn_fallback" in self.fitted_params:
                    fill_value = self.fitted_params[f"{col}_knn_fallback"]
                    df_copy[col] = df_copy[col].fillna(fill_value)
                    logger.info(
                        f"Imputed remaining nan values in '{col}'  feature with KNN fallback value: {fill_value}")

        remaining_nans = df_copy.isna().sum().sum()
        if remaining_nans > 0:
            logger.warning(
                f"Remained nan values: {remaining_nans} after incomplete imputation.")
        else:
            logger.info("Every feature was imputed succesfully.")

        for col in df_copy.columns:
            col_config = self.config['columns'].get(col, {})
            col_type = col_config.get('type')

            if col_type == 'categorical':
                mapping = col_config.get('mapping', {})
                if mapping:
                    allowed_vals = list(mapping.keys())
                    if allowed_vals:
                        min_val, max_val = min(allowed_vals), max(allowed_vals)
                        if pd.api.types.is_numeric_dtype(df_copy[col]):
                            df_copy[col] = df_copy[col].clip(
                                lower=min_val, upper=max_val).round()
                            logger.info(
                                f"Clipped values in categorical column '{col}' to range [{min_val}, {max_val}] and rounded.")
                        else:
                            logger.warning(
                                f"Cannot clip categorical feature '{col}'.")

        return df_copy
