"""
Data validation functionality.
"""

import pandas as pd
from typing import Dict, List, Union


def validate_inputs(data: pd.DataFrame) -> Dict[str, Union[bool, List[str]]]:
    """
    Validate input data for model prediction.
    
    Args:
        data: Input data to validate
        
    Returns:
        Dict containing validation results
    """
    # Initialize validation dict
    validated_data = {"success": True, "errors": []}
    
    # Check for required columns
    required_columns = [
        'yr', 'mnth', 'season', 'weathersit', 'holiday', 
        'workingday', 'hr', 'temp', 'atemp', 'hum', 'windspeed'
    ]
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        validated_data["success"] = False
        validated_data["errors"].append(
            f"Missing required columns: {', '.join(missing_columns)}"
        )
    
    # Check for data types (simplified)
    if validated_data["success"]:
        # Check numerical columns
        numerical_cols = ['temp', 'atemp', 'hum', 'windspeed']
        for col in numerical_cols:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                validated_data["success"] = False
                validated_data["errors"].append(
                    f"Column {col} must be numeric but found {data[col].dtype}"
                )
    
    return validated_data