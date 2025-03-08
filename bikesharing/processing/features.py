"""
Feature transformation and preprocessing classes.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts 'yr' and 'mnth' from 'dteday' and ensures output includes all columns.
    """

    def __init__(self, drop_original=True):
        self.drop_original = drop_original  # Option to drop 'dteday'

    def fit(self, X, y=None):
        return self  # No fitting necessary

    def transform(self, X):
        X = X.copy()
        if 'dteday' in X.columns:
            X['dteday'] = pd.to_datetime(X['dteday'], format='%Y-%m-%d')
            X['yr'] = X['dteday'].dt.year
            X['mnth'] = X['dteday'].dt.month_name()

            if self.drop_original:
                X = X.drop(columns=['dteday'], errors='ignore')
                
        return X


class WeekdayImputer(BaseEstimator, TransformerMixin):
    """Impute missing values in 'weekday' column by extracting dayname from 'dteday' column."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Impute missing 'weekday' values only if 'dteday' is available."""
        X_transformed = X.copy()

        if 'dteday' in X_transformed.columns and 'weekday' in X_transformed.columns:
            missing_mask = X_transformed['weekday'].isna()
            missing_indices = X_transformed[missing_mask].index

            # Ensure 'dteday' is datetime
            if not pd.api.types.is_datetime64_any_dtype(X_transformed['dteday']):
                X_transformed['dteday'] = pd.to_datetime(X_transformed['dteday'])

            # Fill missing weekday values
            for idx in missing_indices:
                X_transformed.loc[idx, 'weekday'] = X_transformed.loc[idx, 'dteday'].day_name()[:3]

        return X_transformed


class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self):
        """Initialize the imputer"""
        self.most_frequent_ = None

    def fit(self, X, y=None):
        """
        Fit the imputer by computing the most frequent value.
        """
        if 'weathersit' in X.columns:
            # Find the most frequent category in the weathersit column
            self.most_frequent_ = X['weathersit'].mode()[0]
        return self

    def transform(self, X, y=None):
        """
        Transform X by imputing missing weathersit values.
        """
        # Create a copy of the dataframe to avoid modifying the original
        X_transformed = X.copy()

        # Fill missing values with the most frequent category if column exists
        if 'weathersit' in X_transformed.columns and self.most_frequent_ is not None:
            X_transformed['weathersit'].fillna(self.most_frequent_, inplace=True)

        return X_transformed

    def impute(self, dataframe):
        """
        Directly impute missing weathersit values in the dataframe.
        """
        return self.fit(dataframe).transform(dataframe)


class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as an ordinal categorical variable and assign values accordingly.
    """

    def __init__(self):
        self.mappings = {
            'yr': {2011: 0, 2012: 1},  # Direct mapping for integers
            'mnth': {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
                     'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12},
            'season': {'spring': 1, 'summer': 2, 'fall': 3, 'winter': 4},
            'weathersit': {'clear': 1, 'mist': 2, 'light rain': 3, 'heavy rain': 4},
            'holiday': {'no': 0, 'yes': 1},
            'workingday': {'no': 0, 'yes': 1}
        }

    def fit(self, X, y=None):
        return self  # No fitting needed, mappings are predefined

    def transform(self, X):
        X_transformed = X.copy()

        # Ensure categorical columns are treated as strings before mapping
        for col, mapping in self.mappings.items():
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].astype(str).str.lower().str.strip()
                X_transformed[col] = X_transformed[col].replace(mapping)
                X_transformed[col] = X_transformed[col].astype(float)

        # Convert hr column separately (ensure it's always numeric)
        if 'hr' in X_transformed.columns:
            X_transformed['hr'] = X_transformed['hr'].apply(self._convert_hour)
            X_transformed['hr'] = pd.to_numeric(X_transformed['hr'], errors='coerce')  # Ensure hr is numeric

        return X_transformed

    def _convert_hour(self, hr):
        """Convert '6am' -> 6, '4pm' -> 16, '12am' -> 0, '12pm' -> 12"""
        try:
            hr = str(hr).strip().lower()
            if hr.endswith('am'):
                return 0 if hr == '12am' else int(hr[:-2])  # '12am' -> 0, '6am' -> 6
            elif hr.endswith('pm'):
                return 12 if hr == '12pm' else int(hr[:-2]) + 12  # '12pm' -> 12, '4pm' -> 16
            return int(hr)  # If it's already a number, just convert to int
        except (ValueError, AttributeError):
            return None  # Handle unexpected cases gracefully


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change outlier values:
        - to upper-bound, if the value is higher than the upper-bound
        - to lower-bound, if the value is lower than the lower-bound
    Uses the IQR method to determine outlier bounds.
    """

    def __init__(self, factor=1.5):
        self.factor = factor  # Factor for IQR (default = 1.5)
        self.bounds = {}  # Store column-wise bounds

    def fit(self, X, y=None):
        """Compute IQR-based outlier bounds for each numerical column."""
        X_num = X.select_dtypes(include=[np.number])  # Only consider numerical columns

        for col in X_num.columns:
            Q1 = X_num[col].quantile(0.25)
            Q3 = X_num[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - self.factor * IQR
            upper_bound = Q3 + self.factor * IQR
            self.bounds[col] = (lower_bound, upper_bound)

        return self

    def transform(self, X):
        """Clip values to computed bounds."""
        X_transformed = X.copy()

        for col, (lower, upper) in self.bounds.items():
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].clip(lower=lower, upper=upper)

        return X_transformed


class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode the 'weekday' column """

    def __init__(self):
        self.categories_ = None  # Store unique categories for one-hot encoding

    def fit(self, X, y=None):
        """Learn unique values of 'weekday' for encoding"""
        if 'weekday' in X.columns:
            # Ensure all values are treated as strings before processing
            self.categories_ = sorted(X['weekday'].astype(str).str.lower().str.strip().unique())
        return self

    def transform(self, X):
        """One-hot encode 'weekday' column"""
        X_transformed = X.copy()

        if 'weekday' in X_transformed.columns:
            # Convert to string before applying string operations
            X_transformed['weekday'] = X_transformed['weekday'].astype(str).str.lower().str.strip()

            # Perform one-hot encoding
            weekday_dummies = pd.get_dummies(X_transformed['weekday'], prefix='weekday')

            # Ensure column order consistency
            if self.categories_:
                weekday_dummies = weekday_dummies.reindex(columns=[f'weekday_{day}' for day in self.categories_], fill_value=0)

            # Drop original 'weekday' column and concatenate one-hot encoded columns
            X_transformed = X_transformed.drop(columns=['weekday']).join(weekday_dummies)

        return X_transformed