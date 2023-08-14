
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierFixTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to handle outliers in input data.

    This transformer identifies outliers in the input data and clips them to a specified range.
    It calculates lower and upper percentiles for each feature and then replaces values
    outside these percentiles with the corresponding percentiles.

    Parameters:
    -----------
    None

    Attributes:
    -----------
    perlow : pandas.Series
        Series containing the lower percentile values for each feature.
    perupp : pandas.Series
        Series containing the upper percentile values for each feature.

    Methods:
    --------
    fit(X, y=None)
        Calculate the lower and upper percentiles for each feature based on the input data.

        Parameters:
        X : pandas.DataFrame or numpy.ndarray
            Input data with features.
        y : None
            Not used in this transformer.

        Returns:
        self : OutlierFixTransformer
            The fitted transformer instance.

    transform(X)
        Clip the input data to the calculated lower and upper percentiles for each feature.

        Parameters:
        X : pandas.DataFrame or numpy.ndarray
            Input data with features.

        Returns:
        X_copy : pandas.DataFrame
            Transformed data with outliers clipped.

    fit_transform(X, y=None)
        Fit the transformer and apply the transformation to the input data.

        Parameters:
        X : pandas.DataFrame or numpy.ndarray
            Input data with features.
        y : None
            Not used in this transformer.

        Returns:
        X_copy : pandas.DataFrame
            Transformed data with outliers clipped.

    get_feature_names_out(input_features=None)
        Get the names of the output features.

        Parameters:
        input_features : array-like or None
            Names of input features. Not used in this method.

        Returns:
        input_features : array-like
            Names of the output features (same as input features).
    """

    def __init__(self):
        """
        Initializes an instance of the OutlierFixTransformer.
        """
        return None

    def fit(self, X, y=None):
        """
        Fit the transformer by calculating the lower and upper percentiles for each feature.

        Parameters:
        X : pandas.DataFrame or numpy.ndarray
            Input data with features.
        y : None
            Not used in this transformer.

        Returns:
        self : OutlierFixTransformer
            The fitted transformer instance.
        """
        if isinstance(X, pd.DataFrame):
            self.perlow = X.quantile(0.03)
            self.perupp = X.quantile(0.97)
        elif isinstance(X, np.ndarray):
            self.perlow = pd.DataFrame(X).quantile(0.03)
            self.perupp = pd.DataFrame(X).quantile(0.97)
        else:
            raise ValueError(
                "Input X must be a pandas DataFrame or a numpy array.")
        return self

    def transform(self, X):
        """
        Clip the input data to the calculated lower and upper percentiles for each feature.

        Parameters:
        X : pandas.DataFrame or numpy.ndarray
            Input data with features.

        Returns:
        X_copy : pandas.DataFrame
            Transformed data with outliers clipped.
        """
        X_copy = pd.DataFrame(X).copy()
        for col in X_copy.columns:
            X_copy[col] = X_copy[col].clip(
                lower=self.perlow[col], upper=self.perupp[col])
        return X_copy

    def fit_transform(self, X, y=None):
        """
        Fit the transformer and apply the transformation to the input data.

        Parameters:
        X : pandas.DataFrame or numpy.ndarray
            Input data with features.
        y : None
            Not used in this transformer.

        Returns:
        X_copy : pandas.DataFrame
            Transformed data with outliers clipped.
        """
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        """
        Get the names of the output features.

        Parameters:
        input_features : array-like or None
            Names of input features. Not used in this method.

        Returns:
        input_features : array-like
            Names of the output features (same as input features).
        """
        return input_features


class CFixedValues(BaseEstimator, TransformerMixin):
    """
    Custom transformer to enforce fixed values for specific columns.

    This transformer applies fixed value thresholds to certain columns in the input data.
    If a value in a specified column falls below a threshold, it is replaced by the threshold.

    Parameters:
    -----------
    None

    Methods:
    --------
    fit(X, y=None)
        Placeholder method. No fitting is performed.

        Parameters:
        X : array-like
            Input data with features.
        y : None
            Not used in this transformer.

        Returns:
        self : CFixedValues
            The transformer instance.

    transform(X)
        Apply fixed value thresholds to specified columns.

        Parameters:
        X : array-like
            Input data with features.

        Returns:
        X_copy : pandas.DataFrame
            Transformed data with fixed value thresholds applied.

    fit_transform(X, y=None)
        Fit the transformer (placeholder) and apply the transformation to the input data.

        Parameters:
        X : array-like
            Input data with features.
        y : None
            Not used in this transformer.

        Returns:
        X_copy : pandas.DataFrame
            Transformed data with fixed value thresholds applied.

    get_feature_names_out(input_features=None)
        Get the names of the output features.

        Parameters:
        input_features : array-like or None
            Names of input features. Not used in this method.

        Returns:
        input_features : array-like
            Names of the output features (same as input features).
    """

    def __init__(self):
        """
        Initializes an instance of the CFixedValues transformer.
        """
        return None

    def fit(self, X, y=None):
        """
        Placeholder method. No fitting is performed.

        Parameters:
        X : array-like
            Input data with features.
        y : None
            Not used in this transformer.

        Returns:
        self : CFixedValues
            The transformer instance.
        """
        return self

    def transform(self, X):
        """
        Apply fixed value thresholds to specified columns.

        Parameters:
        X : array-like
            Input data with features.

        Returns:
        X_copy : pandas.DataFrame
            Transformed data with fixed value thresholds applied.
        """
        X_copy = pd.DataFrame(X).copy()
        for col in X_copy.columns:
            if col == 'px_height':
                X_copy[col] = np.where(X_copy[col] < 217, 217, X_copy[col])
            elif col == 'sc_w':
                X_copy[col] = np.where(X_copy[col] < 2.5, 2.5, X_copy[col])
        return X_copy

    def fit_transform(self, X, y=None):
        """
        Fit the transformer (placeholder) and apply the transformation to the input data.

        Parameters:
        X : array-like
            Input data with features.
        y : None
            Not used in this transformer.

        Returns:
        X_copy : pandas.DataFrame
            Transformed data with fixed value thresholds applied.
        """
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        """
        Get the names of the output features.

        Parameters:
        input_features : array-like or None
            Names of input features. Not used in this method.

        Returns:
        input_features : array-like
            Names of the output features (same as input features).
        """
        return input_features
