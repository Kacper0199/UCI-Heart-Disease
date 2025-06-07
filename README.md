# UCI Heart Disease Classification

This repository contains a comprehensive machine learning project focused on classifying heart disease using the well-known UCI Heart Disease dataset. The project implements an end-to-end pipeline, from data loading and extensive preprocessing to model training, hyperparameter tuning with Optuna, and results evaluation. It emphasizes configurability through YAML files and modular Python code.

**Read the full article detailing this project on Medium:** [ARTICLE](https://medium.com/@kacperkozik999/uci-heart-disease-classification-a-practical-guide-to-building-a-machine-learning-pipeline-dd6c60f9ad58)

## Project Overview

The primary objective is the binary classification of heart disease (presence vs. absence). The project explores various data processing techniques and their impact on three different classification models: Support Vector Machines (SVM), Random Forest, and LightGBM.

Key features of this project include:
*   **Configurable Data Processing:** Control data sources, imputation strategies, and transformation steps via YAML configuration files.
*   **Feature Engineering:** Implementation of several domain-inspired and statistical features to enhance predictive power.
*   **Multiple Imputation Strategies:** Support for MICE, KNN, median, and mode imputation.
*   **Advanced Data Transformation:** Includes One-Hot Encoding, Yeo-Johnson Power Transformation, and Standard Scaling.
*   **Dimensionality Reduction:** Optional PCA to reduce feature space while retaining variance.
*   **Baseline and Tuned Models:** Training of baseline models with static parameters and subsequent hyperparameter optimization using Optuna.
*   **Comprehensive Evaluation:** Detailed analysis of model performance using metrics like accuracy, precision, recall, F1-score, and AUC-ROC.
*   **Visualizations:** Generation of various plots for EDA, PCA analysis, and model evaluation.
*   **Structured Codebase:** Modular Python scripts for clarity, maintainability, and reusability.

## Directory Structure

```
UCI-Heart-Disease/
├── config/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   ├── processing_config.yaml
│   └── training_config.yaml
├── data/
│   ├── processed/
│   └── raw/
├── documentation/
│   └── features_description.md
├── loggers.py
├── main.py
├── models/
├── notebooks/
│   ├── report_data_processing.ipynb
│   ├── report_parameters.ipynb
│   └── report_tuned.ipynb
├── requirements.txt
├── results/
│   ├── different_processing_results/
│   └── optuna/
└── src/
    ├── data_processing/
    │   ├── data_splitting.py
    │   ├── data_transformation.py
    │   ├── feature_engineering.py
    │   ├── imputer.py
    │   ├── pca_transformer.py
    │   └── processor.py
    ├── models/
    │   ├── model_trainer.py
    │   └── model_tuner.py
    ├── pipelines/
    │   └── pipelines.py
    ├── train/
    │   └── trainer.py
    └── visualization/
        ├── correlation.py
        ├── dimensionality_reduction.py
        ├── distributions.py
        ├── evaluation.py
        └── missing_values.py
```


## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. It's recommended to use a virtual environment.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Kacper0199/UCI-Heart-Disease.git
    cd UCI-Heart-Disease
    ```
2.  Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Pipeline

The main pipeline is executed via `main.py`.

```bash
python main.py
```

The pipeline execution involves the following key steps, largely driven by the main.py script and configurations:

1.  **Data Loading and Cleaning (I):**
    *   Loads data from raw files specified in `config/data_config.yaml`.
    *   Handles initial missing value markers.
    *   Converts the target variable to binary.
2.  **Data Splitting (II):**
    *   Splits data into training, validation, and test sets using stratified splitting.
3.  **Feature Engineering (III):**
    *   Creates new features based on logic in `src/data_processing/feature_engineering.py`.
4.  **Missing Value Imputation (IV):**
    *   Imputes missing values based on strategies in `config/data_config.yaml`.
    *   The `Imputer` fits on the training set and transforms all sets.
5.  **Data Transformation (V):**
    *   Applies One-Hot Encoding, Power Transformation, and Standard Scaling based on `config/processing_config.yaml`.
    *   The `DataTransformer` fits on the training set and transforms all sets.
6.  **Dimensionality Reduction with PCA (VI):**
    *   Reduces dimensionality, aiming for a variance threshold (e.g. 90%).
7.  **Baseline Model Training (VII):**
    *   Trains SVM, Random Forest, LightGBM with static hyperparameters from `config/training_config.yaml`.
8.  **Model Tuning with Optuna (VIII):**
    *   For models flagged with `tune: true`, Optuna optimizes hyperparameters.
    *   Search spaces are in `src/models/model_tuner.py`.
9.  **Saving Results (IX):**
    *   Metrics and parameters are saved to CSVs in `results/`.
    *   Tuned models are saved as `.joblib` files in `models/`.
  
## Configuration Files

The program's behavior is mainly controlled by YAML configuration files in `config/` directory:

*   **`data_config.yaml`**:
    *   Defines `missing_values` markers, `data_sources`, and per-column `imputation` strategies.
    *   Example:
        ```yaml
        # data_config.yaml
        columns:
          age: {type: "numerical", imputation: "mice"}
          sex: {type: "categorical", imputation: "mode"}
        ```

*   **`processing_config.yaml`**:
    *   Lists features for `one_hot_encoding`, `power_transform`, and `standard_scale`.
    *   Example:
        ```yaml
        # processing_config.yaml
        one_hot_encoding:
          names: [cp, thal, sex]
        power_transform:
          names: [oldpeak, trestbps]
        ```

*   **`training_config.yaml`**:
    *   Defines `models` with a `tune` flag (true/false) and `params` for baseline training.
    *   Example:
        ```yaml
        # training_config.yaml
        models:
          svm_1:
            tune: false
            params: {C: 1.0, kernel: rbf}
          svm_11:
            tune: true
        ```

## Key Learnings & Observations

This project provided several valuable insights into the process of building a heart disease classification model with the UCI dataset:

*   **Data Understanding is Crucial:** Initial Exploratory Data Analysis (EDA) was essential. Visualizing feature distributions and correlations helped in understanding feature characteristics, their relationships with the target variable, and identifying initial challenges like missing data. For instance, features like `cp`, `oldpeak`, and `exang` showed notable linear correlations with the presence of heart disease.
*   **Feature Engineering Impact:** Creating new, domain-inspired features (such as a `framingham_score`, `pain_related`, and `ecg_related` combinations) generally proved beneficial. Our analysis showed that feature engineering often led to slightly higher or comparable performance for baseline models, particularly improving results for the SVM model on test data. This highlights the value of incorporating domain knowledge to enhance model inputs.
*   **Imputation Strategy Matters (Subtly):** While all tested imputation methods (MICE, KNN, median, mean) provided reasonable starting points, the more sophisticated techniques like MICE and KNN seemed to offer a slight edge in overall performance or generalization for this dataset. The differences were not dramatic, suggesting that for datasets of this nature, simpler imputation might suffice if computational cost is a major concern. MICE imputation, for example, often showed good consistency, though a slight increase in the train-test score gap was observed for MICE with Random Forest and LightGBM; however, these differences were generally small.
*   **Dimensionality Reduction (PCA) - Context is Key:** Applying PCA (aiming to retain 90% variance, as visualized with explained variance plots) did not consistently improve model performance. In some instances, it led to a slightly larger gap between training and test scores, suggesting a tendency to overfit a bit more. Given the moderate number of features in this dataset after initial preprocessing, PCA appeared to be an optional step, and using the full feature set was often preferable. This underscores that dimensionality reduction is not universally beneficial and its utility depends heavily on the dataset's characteristics. Scatter plots of the first two principal components showed some general tendencies but also considerable class overlap.
*   **Model Performance & Tuning:** All three chosen classifiers (SVM, Random Forest, LightGBM) demonstrated strong performance after hyperparameter optimization with Optuna. There was no single "best" model that dramatically outperformed the others. Each, when properly tuned with relevant parameters (like `C` for SVM; `n_estimators` and `max_depth` for Random Forest; or `learning_rate`, `num_leaves`, and `max_depth` for LightGBM), proved capable of effectively learning patterns. Confusion matrices for these tuned models showed reliable predictions with low false positive and false negative rates, indicating the robustness of these standard algorithms for this classification task.
*   **Importance of Proper Workflow:** The project reinforced the critical importance of a structured machine learning pipeline. This includes strict separation of training, validation, and test data, and fitting all preprocessing steps (imputation, scaling, PCA) *only* on the training data to prevent data leakage. An initial oversight in this regard led to artificially inflated metrics, serving as a practical reminder of this common pitfall.
*   **Configurability:** Designing the pipeline with YAML configuration files for data sources, imputation, transformations, and model parameters proved highly effective for organization, reproducibility, and facilitating systematic experimentation with different processing choices.

## Authors

*   Kacper Kozik
*   Mikołaj Pniak
