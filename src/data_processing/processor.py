import os
import pandas as pd
import numpy as np
import yaml
from src.data_processing.imputer import Imputer
from src.data_processing import feature_engineering as fe
from loggers import logger_processor as logger


class Processor:
    def __init__(self, config_path, processed_data_path):
        self.config = self._load_config(config_path)
        self.processed_data_path = processed_data_path
        self.column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        self.imputer = Imputer(self.config)
        self._initialize_paths(processed_data_path)

    def _load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def _initialize_paths(self, processed_data_path):
        os.makedirs(processed_data_path, exist_ok=True)

    def load_data(self, source_key=None):
        """Loading data from specified source/location (e.g. 'cleveland', 'switzerland' ...) or all sources if None is specified."""

        data_sources = self.config['data_sources']
        dataframes = []

        if source_key is not None:
            if source_key not in data_sources:
                logger.error(
                    f"Source key '{source_key}' not found in the configuration file"
                )
                return None

            sources_to_process = {source_key: data_sources[source_key]}
        else:
            # Load data from all the locations/sources
            sources_to_process = data_sources

        for key, source_config in sources_to_process.items():
            logger.info(f"Loading data from {key} source...")
            file_path = source_config['file_path']
            dataset_value = source_config['dataset_value']

            try:
                df = pd.read_csv(
                    file_path,
                    header=None,
                    names=self.column_names
                )
                df['dataset'] = dataset_value
                logger.info(f"Loaded {len(df)} records from {key}")
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Error loading {key} data: {str(e)}")

        if not dataframes:
            logger.error("No data was loaded")
            return None

        df_all = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Total dataset size: {len(df_all)} records")

        return df_all

    def clean_data(self, df):
        """Handling missing values, outliers and type conversions."""

        if df is None or df.empty:
            logger.error("No data to clean")
            return None

        logger.info("Cleaning data...")
        df_clean = df.copy()

        # '?' marker from raw data to NaN value
        missing_markers = self.config['missing_values']['markers']
        for marker in missing_markers:
            df_clean.replace(marker, np.nan, inplace=True)

        # Converting columns to appropriate types
        # Categorical data as encoded numeric values
        for col in df_clean.columns:
            if col in self.config['columns']:
                col_type = self.config['columns'][col].get('type')

                if col_type == 'numerical':
                    df_clean[col] = pd.to_numeric(
                        df_clean[col], errors='coerce')
                elif col_type == 'categorical':
                    df_clean[col] = pd.to_numeric(
                        df_clean[col], errors='coerce')

        # Mapping thal feature - ordinal type
        # from: 3 = normal; 6 = fixed defect; 7 = reversable defect
        # to: 0 = normal; 1 = reversable defect; 2 = fixed defect
        if 'thal' in df_clean.columns:
            original_to_new = self.config['columns']['thal']['original_to_new']
            df_clean['thal'] = df_clean['thal'].map(
                original_to_new).fillna(df_clean['thal'])

        missing_values = df_clean.isna().sum()
        features_with_missing = missing_values[missing_values > 0]
        if not features_with_missing.empty:
            logger.info("Features with missing values:")
            for feature, count in features_with_missing.items():
                logger.info(
                    f"  {feature}: {count} missing values ({count/len(df_clean):.2%})")
        else:
            logger.info("No missing values found in the dataset")

        # Changing outliers with non medical sence to nan values
        cols_with_zero_invalid = ['chol', 'trestbps']
        for col in cols_with_zero_invalid:
            num_zeros = (df_clean[col] == 0).sum()
            if num_zeros > 0:
                logger.info(f"{col}: Changing {num_zeros} zeros to nan value")
                df_clean.loc[df_clean[col] == 0, col] = np.nan

        return df_clean
    
    def convert_target_to_binary(self, df):
        if 'target' not in df.columns:
            logger.warning("No 'target' column found to binarize.")
            return df
        logger.info("Converting target to binary classification (0 vs 1+).")
        df = df.copy()
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        return df


    def impute_missing_values(self, df):
        logger.info("Imputing missing values...")
        return self.imputer.impute_all(df)

    def process_data(self, source_key=None, impute=True, save_csv=True, save_csv_source_key=''):
        df = self.load_data(source_key)
        if df is None:
            return None

        df_clean = self.clean_data(df)
        if df_clean is None:
            return None

        df_processed = df_clean

        if save_csv:
            self.save_to_csv(df_processed, save_csv_source_key)

        return df_processed

    def save_to_csv(self, df, source_key=None):
        """Saving processed dataframe to csv file/files. Source key is used for filename in this case."""
        if df is None or df.empty:
            logger.error("No data to save")
            return

        output_path = os.path.join(
            self.processed_data_path, 'data_original.csv'
        )
        if source_key:
            output_path = os.path.join(
                self.processed_data_path, f'data_original_{source_key}.csv'
            )

        if os.path.exists(output_path):
            logger.info(f"File {output_path} already exists.")
            return

        logger.info(f"Saving data to {output_path}")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} records to {output_path}")

        if source_key is None:
            for key in self.config['data_sources'].keys():
                source_df = df[df['dataset'] == key]
                if not source_df.empty:
                    source_output_path = os.path.join(
                        self.processed_data_path, f'data_original_{key}.csv'
                    )
                    if not os.path.exists(source_output_path):
                        logger.info(
                            f"Saving {key} data to {source_output_path}"
                        )
                        source_df.to_csv(source_output_path, index=False)
                        logger.info(
                            f"Saved {len(source_df)} records to {source_output_path}"
                        )
                    else:
                        logger.info(
                            f"File {source_output_path} already exists."
                        )

    def add_feature_engineering(self, df_cleaned, save_csv=True, source_str=None):
        if df_cleaned is None:
            return None

        df_processed = df_cleaned.copy()
        logger.info("Applying feature engineering transformations...")

        df_processed["framingham_score"] = fe.calculate_total_frs(
            df_processed["age"], df_processed["sex"],
            df_processed["chol"], df_processed["trestbps"], df_processed["fbs"]
        )
        df_processed["age_squared"] = fe.calculate_age_squared(
            df_processed["age"])
        df_processed["pain_related"] = fe.calculate_pain_related(
            df_processed["cp"], df_processed["exang"])
        df_processed["food_related"] = fe.calculate_food_related(
            df_processed["chol"], df_processed["fbs"])
        df_processed["exercise_related"] = fe.calculate_exercise_related(
            df_processed["thalach"], df_processed["trestbps"], df_processed["thal"]
        )
        df_processed["ecg_related"] = fe.calculate_ecg_related(
            df_processed["restecg"], df_processed["oldpeak"], df_processed["slope"]
        )
        df_processed["dead_cells_related"] = fe.calculate_dead_cells(
            df_processed["ca"], df_processed["thal"])

        if save_csv:
            output_path = os.path.join(
                self.processed_data_path, f'data_feature_engineered_{source_str}.csv')
            df_processed.to_csv(output_path, index=False)
            logger.info(f"Saved feature engineered data to {output_path}")

        return df_processed
