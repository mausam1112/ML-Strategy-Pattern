import logging
import pandas as pd

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig


def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame|pd.Series,
    y_test: pd.DataFrame|pd.Series,
    config: ModelNameConfig
) -> RegressorMixin:
    """
    Trains the model.

    Args: 
        X_train: pd.DataFrame
        X_test: pd.DataFrame
        y_train: pd.DataFrame
        y_test: pd.DataFrame
        config: ModelNameConfig
    """
    try:
        model = None
        if config.model_name == 'LinearRegression':
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise NotImplementedError(f"Model {config.model_name} is not implemented.")
    except Exception as e:
        logging.error(f"Error in training model. {e}")
        raise e
    