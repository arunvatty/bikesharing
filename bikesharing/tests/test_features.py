"""
Test features processing.
"""

import numpy as np
import pandas as pd
import pytest

from processing.features import (
    DateFeatureExtractor,
    WeekdayImputer,
    WeathersitImputer,
    Mapper,
    OutlierHandler,
    WeekdayOneHotEncoder
)


def test_date_feature_extractor(sample_input_data):
    """Test the DateFeatureExtractor."""
    
    # Create transformer
    transformer = DateFeatureExtractor()
    
    # Transform data
    result = transformer.transform(sample_input_data.copy())
    
    # Check that new columns are created
    assert "yr" in result.columns
    assert "mnth" in result.columns
    
    # Check that dteday is dropped
    assert "dteday" not in result.columns
    
    # Check that the transformed values are correct
    assert result["yr"].iloc[0] == 2011
    assert result["mnth"].iloc[0] == "January"


def test_weekday_imputer(sample_input_data):
    """Test the WeekdayImputer."""
    
    # Check that we have at least one missing value to test imputation
    assert sample_input_data["weekday"].isna().any()
    
    # Create transformer
    transformer = WeekdayImputer()
    
    # Transform data
    result = transformer.transform(sample_input_data.copy())
    
    # Check that all missing values are imputed
    assert not result["weekday"].isna().any()
    
    # Check that the imputed value is correct
    # Last row had NaN for weekday, and dteday of "2011-01-05" which is a Wednesday
    assert result["weekday"].iloc[-1] == "Wed"


def test_weathersit_imputer(sample_input_data):
    """Test the WeathersitImputer."""
    
    # Check that we have at least one missing value to test imputation
    assert sample_input_data["weathersit"].isna().any()
    
    # Create transformer
    transformer = WeathersitImputer()
    
    # Fit transformer
    transformer.fit(sample_input_data)
    
    # Check that the most frequent value is set
    assert transformer.most_frequent_ is not None
    
    # Transform data
    result = transformer.transform(sample_input_data.copy())
    
    # Check that all missing values are imputed
    assert not result["weathersit"].isna().any()


def test_mapper(sample_input_data):
    """Test the Mapper."""
    
    # Create transformer
    transformer = Mapper()
    
    # Transform data
    result = transformer.transform(sample_input_data.copy())
    
    # Check that categorical columns are converted to numbers
    assert pd.api.types.is_numeric_dtype(result["season"])
    assert pd.api.types.is_numeric_dtype(result["holiday"])
    assert pd.api.types.is_numeric_dtype(result["workingday"])
    assert pd.api.types.is_numeric_dtype(result["weathersit"])
    
    # Check specific mappings
    assert result["season"].iloc[0] == 4  # winter -> 4
    assert result["holiday"].iloc[0] == 0  # no -> 0
    assert result["holiday"].iloc[-1] == 1  # yes -> 1


def test_outlier_handler():
    """Test the OutlierHandler."""
    
    # Create test data with outliers
    test_data = pd.DataFrame({
        "col1": [1, 2, 3, 4, 5, 100],  # 100 is an outlier
        "col2": [10, 20, 30, 40, 50, -100]  # -100 is an outlier
    })
    
    # Create transformer
    transformer = OutlierHandler(factor=1.5)
    
    # Fit transformer
    transformer.fit(test_data)
    
    # Transform data
    result = transformer.transform(test_data.copy())
    
    # Check that outliers are clipped
    assert result["col1"].max() < 100
    assert result["col2"].min() > -100


def test_weekday_one_hot_encoder():
    """Test the WeekdayOneHotEncoder."""
    
    # Create test data
    test_data = pd.DataFrame({
        "weekday": ["Mon", "Tue", "Wed", "Thu", "Fri"]
    })
    
    # Create transformer
    transformer = WeekdayOneHotEncoder()
    
    # Fit transformer
    transformer.fit(test_data)
    
    # Transform data
    result = transformer.transform(test_data.copy())
    
    # Check that one-hot encoded columns are created
    assert "weekday_mon" in result.columns
    assert "weekday_tue" in result.columns
    
    # Check that the original column is dropped
    assert "weekday" not in result.columns
    
    # Check that the one-hot encoding is correct
    assert result["weekday_mon"].iloc[0] == 1
    assert result["weekday_mon"].iloc[1] == 0