import logging
import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin
from src.model_eval import MSE, RMSE, R2
from typing import Tuple

def eval_model(
        model: RegressorMixin, 
        X_test: pd.DataFrame|np.ndarray,
        y_test: pd.Series|np.ndarray
) -> Tuple:
    """
    Evaluates model on test data.
    Args:
        model: RegressorMixin 
        X_test: pd.DataFrame
        y_test: pd.Series|np.ndarray
    Returns:
        float: rmse
        float: mse
        float: r2
    """

    try:
        prediction = model.predict(X_test)
        mse = MSE().calculate_scores(y_test, prediction)
        rmse = RMSE().calculate_scores(y_test, prediction)
        r2 = R2().calculate_scores(y_test, prediction)

        return rmse, mse, r2
    except Exception as e:
        logging.error(f"Error calculating R2 score. {e}")
        raise e
