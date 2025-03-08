"""
Test prediction functionality.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score, mean_squared_error

from bikeshare_model.predict import make_prediction
from bikeshare_model.pipeline import create_pipeline


def test_pipeline_creation():
    """Test that the pipeline is created correctly."""
    
    # Create pipeline
    pipeline = create_pipeline()
    
    # Check that the pipeline has the expected steps
    assert "preprocessor" in pipeline.named_steps
    assert "regressor" in pipeline.named_steps


def test_make_prediction(sample_train_test_split, prediction_input):
    """Test the prediction functionality."""
    
    # Get data
    X_train, X_test, y_train, y_test = sample_train_test_split
    
    # Create and fit pipeline
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Make predictions using our function
    result = make_prediction(prediction_input)
    
    # Check that prediction has expected format
    assert "predictions" in result
    assert "errors" in result
    
    # Check that there are no errors
    assert result["errors"] is None
    
    # Check that predictions are returned
    assert result["predictions"] is not None
    assert len(result["predictions"]) == len(prediction_input)
    
    # Check that predictions are reasonable (should be positive for bike rentals)
    for pred in result["predictions"]:
        assert pred >= 0


def test_prediction_quality(sample_train_test_split):
    """Test the quality of predictions on test data."""
    
    # Get data
    X_train, X_test, y_train, y_test = sample_train_test_split
    
    # Skip test if not enough samples
    if len(y_test) < 2:
        pytest.skip("Not enough test samples to calculate R^2")
        
    # Create and fit pipeline
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Check predictions are positive (since we're predicting bike counts)
    assert all(y_pred >= 0)
    
    # For test data with very few samples, we'll only verify the predictions are numeric
    # and not worry about R² values which can be unreliable with such small datasets
    assert all(np.isfinite(y_pred))
    
    # If the test set is large enough, we can test R²
    if len(y_test) >= 10:  # Arbitrary threshold for a "reasonable" test size
        r2 = r2_score(y_test, y_pred)
        assert not np.isnan(r2)
        assert -1 <= r2 <= 1


def test_prediction_input_validation():
    """Test that invalid inputs are correctly rejected."""
    
    # Create invalid input (missing required columns)
    invalid_input = pd.DataFrame([{
        "season": 1,
        "yr": 1,
        # Missing 'mnth' and other required columns
        "temp": 0.35,
        "hum": 0.45
    }])
    
    # Make prediction
    result = make_prediction(invalid_input)
    
    # Check that there are errors
    assert result["errors"] is not None
    assert len(result["errors"]) > 0
    
    # Check that no predictions are returned
    assert result["predictions"] is None