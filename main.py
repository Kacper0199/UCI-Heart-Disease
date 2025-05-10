import matplotlib
matplotlib.use('Agg')
import pandas as pd
from src.data_processing.processor import Processor
from src.data_processing.data_splitting import split_data
from src.data_processing.imputer import Imputer
from src.data_processing.data_transformation import DataTransformer
from src.data_processing.pca_transformer import PCATransformer
from src.models.model_tuner import ModelTunerEvaluator
from loggers import logger_main as logger
import os
from src.visualization.missing_values import plot_missing_values_bar_plot, plot_missing_values_matrix_plot, plot_missing_values_heatmap
from src.visualization.distributions import plot_feature_distribution, plot_feature_distribution_by_target
from src.visualization.correlation import plot_correlation_matrix, plot_correlation_coefficients
from src.visualization.evaluation import plot_metric_comparison, plot_confusion_matrix, plot_feature_importance


# Config paths
DATA_CONFIG_PATH = 'config/data_config.yaml'
PROCESSING_CONFIG_PATH = 'config/processing_config.yaml'
TRAINING_CONFIG_PATH = 'config/training_config.yaml'

# Data paths
PROCESSED_DATA_PATH = 'data/processed'

# Visualization paths
EDA_PATH = 'plots/01_basic_eda'
DIM_REDUCTION_PATH = 'plots/02_dimensionality_reduction'
EVALUATION_PLOT_PATH = 'plots/03_evaluation'


def main(source_key=None, feature_engineering=True, run_visualizations=False):
    logger.info("Starting analysis with refactored pipeline...")
    source_str = 'uci' if source_key is None else source_key

    # (I) Loading, cleaning data and convert target to binary
    logger.info("(I) Loading and cleaning data...")
    processor = Processor(DATA_CONFIG_PATH, PROCESSED_DATA_PATH)

    # Without imputation - imputation in (IV) step
    df_cleaned = processor.process_data(
        source_key=source_key,
        impute=False,
        save_csv=True,
        save_csv_source_key=f'{source_str}_cleaned'
    )
    if df_cleaned is None:
        logger.error("Data loading and cleaning failed.")
        return
    
    df_cleaned = processor.convert_target_to_binary(df_cleaned)

    # Basic EDA Visualizations on cleaned and whole dataset
    if run_visualizations:
        logger.info("Running initial visualizations on cleaned data...")
        os.makedirs(EDA_PATH, exist_ok=True)
        logger.info("Generating missing values bar plot...")
        plot_missing_values_bar_plot(
            df_cleaned,
            output_dir=EDA_PATH,
            filename=f'{source_str}_missing_values_bar_plot.png'
        )

    # (II) Data split
    logger.info("(II) Splitting dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        df_cleaned,
        target_col='target',
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=0
    )
    if X_train is None:
        logger.error("Dataset split failed.")
        return

    # (III) Feature engineering
    logger.info("(III) Applying feature engineering...")
    if feature_engineering:
        X_train = processor.add_feature_engineering(X_train, save_csv=False)
        X_val = processor.add_feature_engineering(X_val, save_csv=False)
        X_test = processor.add_feature_engineering(X_test, save_csv=False)
        if X_train is None or X_val is None or X_test is None:
            logger.error(
                "Failed feature engineering.")
            return
        logger.info(
            "Feature engineering applied to train, validation, and test datasets.")

    # (IV) Missing values imputation
    logger.info("(IV) Missing values imputation...")
    imputer = Imputer(processor.config)
    imputer.fit(X_train)
    X_train_imp = imputer.transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)
    if X_train_imp is None or X_val_imp is None or X_test_imp is None:
        logger.error("Imputation failed on one or more data splits.")
        return
    logger.info("Imputation applied to train, validation, and test sets.")

    # Next visualizations
    if run_visualizations:
        logger.info(
            "Visualizations on imputed and/or feature engineered datasets...")
        os.makedirs(EDA_PATH, exist_ok=True)

        df_train_imp = pd.concat([X_train_imp, y_train], axis=1)
        df_val_imp = pd.concat([X_val_imp, y_val], axis=1)
        df_test_imp = pd.concat([X_test_imp, y_test], axis=1)

        target_col_name = y_train.name if y_train.name else 'target'

        for split_name, df_split in [('train', df_train_imp), ('val', df_val_imp), ('test', df_test_imp)]:
            # Distribution plots
            logger.info(
                f"Generating distribution plots for {split_name} set...")
            for column in df_split.columns:
                plot_feature_distribution(
                    df_split,
                    column,
                    output_dir=EDA_PATH,
                    filename=f'{source_str}_{split_name}_distributions_{column}.png'
                )
            plot_feature_distribution_by_target(
                df_split,
                output_dir=EDA_PATH,
                filename=f'{source_str}_{split_name}_feature_distribution_by_target.png'
            )

            # Correlation plots
            logger.info(
                f"Generating correlation plots for {split_name} set...")
            plot_correlation_matrix(
                df_split,
                output_dir=EDA_PATH,
                filename=f'{source_str}_{split_name}_correlation_matrix.png'
            )

            if target_col_name in df_split.columns:
                plot_correlation_coefficients(
                    df_split,
                    output_dir=EDA_PATH,
                    filename=f'{source_str}_{split_name}_feature_correlation_coefficients.png'
                )

    # (V) Data transformations
    logger.info("(V) Applying data transformations...")
    transformer = DataTransformer(config_path=PROCESSING_CONFIG_PATH)
    transformer.transform(X_train_imp, fit=True)
    X_train_trans = transformer.transform(X_train_imp, fit=False)
    X_val_trans = transformer.transform(X_val_imp, fit=False)
    X_test_trans = transformer.transform(X_test_imp, fit=False)
    if X_train_trans is None or X_val_trans is None or X_test_trans is None:
        logger.error("Failed data transformation.")
        return
    logger.info(
        "Data transformations applied to train, validation, and test datasets.")

    # (VI) Dimensionality reduction with PCA
    logger.info("(VI) Dimensionality reduction with PCA...")
    pca_transformer = PCATransformer(variance_threshold=0.9, random_state=0)
    pca_transformer.fit(
        X_train_trans, plot_results=run_visualizations, source_str=f'{source_str}_train')
    df_train_pca = pca_transformer.transform(
        X_train_trans, y_train, plot_scatter=run_visualizations, source_str=f'{source_str}_train')
    df_val_pca = pca_transformer.transform(
        X_val_trans, y_val, plot_scatter=run_visualizations, source_str=f'{source_str}_val')
    df_test_pca = pca_transformer.transform(
        X_test_trans, y_test, plot_scatter=False)

    if df_train_pca is None or df_val_pca is None or df_test_pca is None:
        logger.error("Failed PCA transformation.")
        return

    target_col = y_train.name if y_train.name else 'target'
    X_train_pca = df_train_pca.drop(columns=[target_col])
    y_train_pca = df_train_pca[target_col]
    X_val_pca = df_val_pca.drop(columns=[target_col])
    y_val_pca = df_val_pca[target_col]
    X_test_pca = df_test_pca.drop(columns=[target_col])
    y_test_pca = df_test_pca[target_col]

    logger.info("PCA applied to train, validation, and test datasets.")

    # (VII) Model tuning and evaluation
    logger.info("(VII) Model tuning and evaluation...")
    tuner_evaluator = ModelTunerEvaluator(
        X_train=X_train_pca, y_train=y_train_pca,
        X_val=X_val_pca, y_val=y_val_pca,
        X_test=X_test_pca, y_test=y_test_pca,
        n_trials=5
    )

    models_to_run = ['svm', 'random_forest', 'lightgbm']
    results = {}

    for model_name in models_to_run:
        model, best_params, metrics = tuner_evaluator.tune_and_evaluate(
            model_name)
        if model is None:
            logger.error(f"Tuning and evaluation failed for {model_name}.")
            continue
        results[model_name] = {
            'model': model,
            'best_params': best_params,
            'metrics': metrics
        }

    logger.info("Completed tuning and evaluation for all specified models.")

    # (VIII) Tuned model visualization
    if run_visualizations and results:
        logger.info("(VIII) Generating tuned model metrics visualizations...")
        os.makedirs(EVALUATION_PLOT_PATH, exist_ok=True)

        # Plot metric comparisons
        plot_metric_comparison(
            results, EVALUATION_PLOT_PATH, source_str=f'{source_str}_')

        class_names = ['No Disease', 'Disease']
        feature_names = X_test_pca.columns.tolist()

        for model_name, data in results.items():
            plot_confusion_matrix(
                model=data['model'],
                X_test=X_test_pca,
                y_test=y_test_pca,
                model_name=model_name,
                class_names=class_names,
                output_dir=EVALUATION_PLOT_PATH,
                source_str=f'{source_str}_'
            )

            # Feature importances for ensemble models
            if model_name in ['random_forest', 'lightgbm']:
                plot_feature_importance(
                    model=data['model'],
                    feature_names=feature_names,
                    model_name=model_name,
                    output_dir=EVALUATION_PLOT_PATH,
                    top_n=5,
                    source_str=f'{source_str}_'
                )
        logger.info("Completed generating evaluation visualizations.")

    logger.info("Analysis finished successfully.")


if __name__ == "__main__":
    main(run_visualizations=True, feature_engineering=True)
