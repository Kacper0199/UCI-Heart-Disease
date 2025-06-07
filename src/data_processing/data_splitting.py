from sklearn.model_selection import train_test_split
from loggers import logger_main as logger


def split_data(df, target_col='target', train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=0):
    logger.info(
        f"Dataset split into train/validation/test sets with ratio: {train_ratio}/{val_ratio}/{test_ratio}")

    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and 0 < test_ratio < 1):
        logger.error("Ratios must be between 0 and 1.")
        return None, None, None, None, None, None

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if y.isna().any():
        logger.warning(
            "Deleting target rows with nan values.")
        not_nan_indices = y.dropna().index
        X = X.loc[not_nan_indices]
        y = y.loc[not_nan_indices]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        stratify=y,
        random_state=random_state
    )

    val_size_adjusted = val_ratio / (train_ratio + val_ratio)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=random_state
    )

    logger.info(f"Data split done:")
    logger.info(
        f"  Train set size: {len(X_train)} ({len(X_train)/len(df):.1%})")
    logger.info(
        f"  Validation set size: {len(X_val)} ({len(X_val)/len(df):.1%})")
    logger.info(f"  Test set size: {len(X_test)} ({len(X_test)/len(df):.1%})")

    return X_train, y_train, X_val, y_val, X_test, y_test
