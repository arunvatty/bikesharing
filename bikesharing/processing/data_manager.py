"""
Data management functionality.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any

from config.core import load_config

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(file_path)


def split_data(
    data: pd.DataFrame, target_col: str, unused_cols: list, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    
    Args:
        data: DataFrame to split
        target_col: Target column name
        unused_cols: Columns to drop
        test_size: Proportion of data to use for testing
        random_state: Random state for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    # Remove any unused columns
    drop_cols = unused_cols + [target_col]
    X = data.drop(columns=drop_cols, errors='ignore')
    y = data[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for model training.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    config = load_config()
    
    # Load dataset
    data = load_dataset(config["data"]["dataset_path"])
    
    # Apply initial preprocessing (to be refactored)
    from processing.features import DateFeatureExtractor, WeekdayImputer, WeathersitImputer, Mapper
    
    # Initial preprocessing before splitting data
    data = WeekdayImputer().fit_transform(data)
    data = DateFeatureExtractor().fit_transform(data)
    data = WeathersitImputer().fit_transform(data)
    data = Mapper().fit_transform(data)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(
        data=data,
        target_col=config["features"]["target_col"],
        unused_cols=config["features"]["unused_cols"],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )
    
    return X_train, X_test, y_train, y_test