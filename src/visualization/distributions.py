import matplotlib.pyplot as plt
import seaborn as sns
import os
from loggers import logger_visualizations as logger


def plot_feature_distribution(df, feature_name, output_dir='plots/01_basic_eda', filename=None):
    if df is None or df.empty:
        logger.error("No data to plot feature distribution.")
        return
    if feature_name not in df.columns:
        logger.error(f"Feature '{feature_name}' not found in DataFrame.")
        return

    logger.info(f"Plotting distribution for feature: {feature_name}")
    plt.figure(figsize=(8, 6))
    sns.histplot(df[feature_name], kde=True)
    plt.title(f'Distribution of {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        filename = f'uci_distributions_{feature_name}.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    logger.info(f"Feature distribution plot saved to {output_path}")
    plt.close()


def plot_feature_distribution_by_target(df, output_dir='plots/01_basic_eda', filename=None):
    if df is None or df.empty:
        logger.error("No data to plot feature distribution.")
        return

    logger.info(f"Plotting feature distribution by target")
    plt.figure(figsize=(17, 14))
    palette = sns.color_palette("coolwarm", n_colors=df['target'].nunique())
    for i, feature in enumerate(df.columns.drop(['dataset', 'target']), 1):
        plt.subplot(5, 4, i)
        sns.kdeplot(data=df, x=feature, hue='target', common_norm=False,
                    fill=True, alpha=0.4, palette=palette, warn_singular=False)
        plt.title(feature)

    plt.suptitle('Feature Distributions by target')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        filename = f'uci_feature_distribution_by_target.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    logger.info(f"Feature distribution plot saved to {output_path}")
    plt.close()


def plot_all_distributions(df, output_dir='plots/01_basic_eda'):
    if df is None or df.empty:
        logger.error("No data to plot feature distributions.")
        return

    logger.info("Plotting distributions for all features...")
    for column in df.columns:
        plot_feature_distribution(df, column, output_dir)
    plot_feature_distribution_by_target(df, output_dir)
