import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    @abstractmethod
    def calculate_scores(self, y_true: pd.Series|np.ndarray, y_pred: np.ndarray):
        """
        Calculate the scores for the model
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """
    Evaluation Strategyg using Mean Squared Error.
    """
    def calculate_scores(self, y_true: pd.Series|np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating mean squared error.")
            mse = mean_squared_error(y_true, y_pred)
            return mse
        except Exception as e:
            logging.error(f"Error calculating mse. {e}")
            raise e

class RMSE(Evaluation):
    """
    Evaluation Strategyg using Root Mean Squared Error.
    """
    def calculate_scores(self, y_true: pd.Series|np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating mean squared error.")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            return rmse
        except Exception as e:
            logging.error(f"Error calculating rmse. {e}")
            raise e

class R2(Evaluation):
    """
    Evaluation Strategyg using R2 Score.
    """
    def calculate_scores(self, y_true: pd.Series|np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 score.")
            r2 = r2_score(y_true, y_pred)
            return r2
        except Exception as e:
            logging.error(f"Error calculating R2 score. {e}")
            raise e
