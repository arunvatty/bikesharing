"""
Conftest.py for pytest fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config.core import load_config


@pytest.fixture(scope="session")
def sample_input_data():
    """Sample input data for testing."""
    
    # Create a small sample dataset
    return pd.DataFrame({
        "dteday": ["2011-01-01", "2011-01-02", "2011-01-03", "2011-01-04", "2011-01-05"],
        "season": ["winter", "winter", "winter", "winter", "winter"],
        "yr": [2011, 2011, 2011, 2011, 2011],
        "mnth": ["january", "january", "january", "january", "january"],
        "holiday": ["no", "no", "no", "no", "yes"],
        "weekday": ["Sat", "Sun", "Mon", "Tue", np.nan],  # One missing for testing imputation
        "workingday": ["no", "no", "yes", "yes", "no"],
        "weathersit": ["clear", "clear", "mist", np.nan, "light rain"],  # One missing for testing imputation
        "temp": [0.22, 0.2, 0.24, 0.24, 0.24],
        "atemp": [0.27, 0.24, 0.29, 0.29, 0.29],
        "hum": [0.8, 0.9, 0.75, 0.75, 0.8],
        "windspeed": [0.0, 0.0, 0.1, 0.1, 0.0],
        "casual": [331, 131, 120, 108, 88],
        "registered": [654, 276, 1229, 1454, 1518],
        "cnt": [985, 407, 1349, 1562, 1606]
    })


@pytest.fixture(scope="session")
def sample_train_test_split(sample_input_data):
    """Split the sample data into train and test sets."""
    
    config = load_config()
    
    # Initial preprocessing before splitting
    from processing.features import (
        DateFeatureExtractor, 
        WeekdayImputer, 
        WeathersitImputer, 
        Mapper
    )
    
    data = WeekdayImputer().fit_transform(sample_input_data)
    data = DateFeatureExtractor().fit_transform(data)
    data = WeathersitImputer().fit_transform(data)
    data = Mapper().fit_transform(data)
    
    # Split data - Use errors='ignore' to skip columns that don't exist
    target_col = config["features"]["target_col"]
    unused_cols = config["features"]["unused_cols"].copy()  # Make a copy to avoid modifying the original
    
    # Only include columns that actually exist in the dataframe
    columns_to_drop = [col for col in unused_cols + [target_col] if col in data.columns]
    
    X = data.drop(columns=columns_to_drop)
    y = data[target_col]
    
    # IMPORTANT: Force the test set to have at least 2 samples for valid R² calculation
    # For a small sample dataset, we use a fixed split rather than a random one
    X_train = X.iloc[:3]  # Use first 3 samples for training
    X_test = X.iloc[3:]   # Use remaining samples for testing (should be at least 2)
    y_train = y.iloc[:3]
    y_test = y.iloc[3:]
    
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def prediction_input():
    """Sample input data for prediction testing."""
    
    return pd.DataFrame([{
        "season": 1,
        "yr": 1,
        "mnth": 1,
        "holiday": 0,
        "weekday": "Mon",
        "workingday": 1,
        "weathersit": 1,
        "temp": 0.35,
        "atemp": 0.38,
        "hum": 0.45,
        "windspeed": 0.2,
        "hr": 12
    }])