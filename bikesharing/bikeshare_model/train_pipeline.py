"""
Train the bikeshare prediction pipeline.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

from config.core import load_config
from processing.data_manager import prepare_data
from .pipeline import create_pipeline


def run_training() -> None:
    """
    Train the model and save it.
    """
    # Load config
    config = load_config()
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Create pipeline
    pipeline = create_pipeline()
    
    # Train model
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model on test data...")
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Sample comparison of actual vs predicted
    comparison_df = pd.DataFrame({
        'Actual': y_test, 
        'Predicted': y_pred, 
        'Difference': y_test - y_pred,
        'Percent Error': abs((y_test - y_pred) / y_test) * 100
    })
    
    print("\nSample of predictions (10 random samples):")
    print(comparison_df.sample(10))
    
    # Save the model
    save_path = Path("trained_models/bikeshare_model.pkl")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    run_training()