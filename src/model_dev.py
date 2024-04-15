import logging
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression

class Model(ABC):
    """Abstract class for all models."""
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model.
        Args: 
            X_train: Training data features.
            y_train: Training data labels.
        Returns: 
            None
        """
        pass

class LinearRegressionModel(Model):
    
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model.
        Args: 
            X_train: Training data features.
            y_train: Training data labels.
        Returns: 
            None
        """
        try:
            model_reg = LinearRegression(**kwargs)
            model_reg.fit(X_train, y_train)
            logging.info("Model training completed.")
            return model_reg
        except Exception as e:
            logging.error(f"Error in training model with error {e}")
            raise e
        

         