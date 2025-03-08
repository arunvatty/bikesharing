"""
Pipeline creation functionality.
"""

from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.core import load_config
from processing.features import OutlierHandler, WeekdayOneHotEncoder


def create_pipeline() -> Pipeline:
    """
    Create a pipeline for the bikeshare model.
    
    Returns:
        sklearn Pipeline
    """
    config = load_config()
    
    # Get configuration
    num_features = config["features"]["numerical_features"]
    weekday_col = config["features"]["weekday_col"]
    model_params = config["model"]["params"]
    outlier_factor = config["pipeline"]["outlier_factor"]
    
    # Define the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('outlier_handler', OutlierHandler(factor=outlier_factor), num_features),
            ('weekday_encoder', WeekdayOneHotEncoder(), weekday_col),
            ('scaler', StandardScaler(), num_features)
        ],
        remainder='passthrough'  # Include other columns as-is
    )
    
    # Create the full pipeline
    full_pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(**model_params))
        ]
    )
    
    return full_pipeline