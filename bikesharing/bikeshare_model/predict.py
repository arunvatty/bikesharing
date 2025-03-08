"""
Make predictions with the trained bikeshare model.
"""

import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Union, List

from processing.validation import validate_inputs


def load_model(model_path: str = "trained_models/bikeshare_model.pkl"):
    """
    Load the trained pipeline.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    model = joblib.load(model_path)
    return model


def make_prediction(input_data: Union[Dict, List, pd.DataFrame]) -> Dict:
    """
    Make a prediction using the saved model.
    
    Args:
        input_data: Input data for prediction
        
    Returns:
        Dict with predictions and metadata
    """
    # Convert input to DataFrame if it's a dict or list
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        input_data = pd.DataFrame(input_data)
        
    # Validate inputs
    validation_result = validate_inputs(input_data)
    
    if not validation_result["success"]:
        return {
            "predictions": None,
            "errors": validation_result["errors"]
        }
    
    # Load model
    model = load_model()
    
    # Make predictions
    predictions = model.predict(input_data)
    
    # Return predictions
    return {
        "predictions": predictions.tolist(),
        "errors": None
    }


if __name__ == "__main__":
    # Example of usage
    import sys
    
    # Check if a file path is provided as argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Loading data from {file_path}")
        
        # Load data from CSV
        input_data = pd.read_csv(file_path)
    else:
        # Use sample data
        input_data = pd.DataFrame([{
            'season': 1,  # Spring
            'yr': 1,      # 2012
            'mnth': 5,    # May
            'holiday': 0, # No
            'weekday': 'Mon',
            'workingday': 1, # Yes
            'weathersit': 1, # Clear
            'temp': 0.35,   # Normalized temperature
            'atemp': 0.38,  # Normalized feeling temperature
            'hum': 0.45,    # Normalized humidity
            'windspeed': 0.2, # Normalized wind speed
            'hr': 12 # Hour
        }])
        print("Using sample data:")
        print(input_data)
    
    # Make prediction
    result = make_prediction(input_data)
    
    # Display result
    if result["errors"]:
        print("Errors:", result["errors"])
    else:
        # Create a DataFrame to display predictions
        if isinstance(input_data, pd.DataFrame) and len(input_data) == len(result["predictions"]):
            results_df = input_data.copy()
            results_df["predicted_cnt"] = result["predictions"]
            print("\nPrediction results:")
            print(results_df)
        else:
            print("\nPredictions:", result["predictions"])