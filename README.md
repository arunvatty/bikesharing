# Bikeshare Model

A machine learning model for predicting bike rental demand.

## Project Structure

'''
Application/
├── bikeshare_model/
│   ├── config.yml
│   ├── pipeline.py
│   ├── predict.py
│   ├── train_pipeline.py
│   ├── VERSION
│   ├── __init__.py
│   ├── config/
│   │   ├── core.py
│   │   ├── __init__.py
│   ├── datasets/
│   │   ├── bike-rental-dataset.csv
│   │   ├── __init__.py
│   ├── processing/
│   │   ├── data_manager.py
│   │   ├── features.py
│   │   ├── validation.py
│   │   ├── __init__.py
│   ├── trained_models/
│   │   ├── __init__.py
├── tests/
│   ├── conftest.py
│   ├── test_features.py
│   ├── test_predictions.py
├── MANIFEST.in
├── mypy.ini
├── pyproject.toml
├── README.md
├── requirements.txt
├── setup.py
├── test_requirements.txt
'''

## Installation

To install the package:

'''bash
pip install -e .
'''

## Running Tests

To run tests:

'''bash
pytest
'''

To run tests with coverage:

'''bash
pytest --cov=bikeshare_model
'''

## Building the Package

To build the package:

'''bash
pip install build
python -m build
'''

## Usage

After installing the package, you can use it to train a model:

'''python
from bikeshare_model.train_pipeline import run_training

run_training()
'''

And make predictions:

'''python
import pandas as pd
from bikeshare_model.predict import make_prediction

# Prepare input data
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
    'hr': 12       # Hour of the day
}])

# Make prediction
result = make_prediction(input_data)
print(result)
'''

## License

MIT

## Author

#Arun Vasudeva Rao <arunv.inc@gmail.com>
