import logging
from abc import ABC, abstractmethod
from typing import Tuple
from typing_extensions import Annotated

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame | pd.Series:
        pass


class DataPreProcessStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame | pd.Series:
        """
        Preprocesses the data. (Here Label encoding.)
        Args:
            df: pandas.Dataframe
        """
        try:
            label_encoder = LabelEncoder()
            data['variety'] = label_encoder.fit_transform(data['variety'])
            return data
        except Exception as e:
            logging.error(f"Error in preprocessing the data: {e}")
            raise e

class DataSplitStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame, test_size=0.2) -> Tuple[
        Annotated[pd.DataFrame, "X_train"],
        Annotated[pd.DataFrame, "X_test"],
        Annotated[pd.Series, "y_train"],
        Annotated[pd.Series, "y_test"],
    ]:
        """
        Splits the data in train and test sets.

        Args:
            df: pandas.Dataframe
        """
        try:
            X = data.drop(['variety'], axis=1)
            y = data['variety']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in preprocessing the data: {e}")
            raise e
        
class DataCleaning:
    """
    Prepocessing the data and then splits the data
    """
    def __init__(self, data: pd.DataFrame | pd.Series, strategy: DataStrategy) -> None:
        self.data = data
        self.strategy = strategy

    def handle_data(self):
        """
        Handling the data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in preprocessing the data: {e}")
            raise e
    
if __name__ == "__main__":
    data = pd.read_csv('./data/iris.csv')
    data_cleaning = DataCleaning(data, DataPreProcessStrategy())
    data = data_cleaning.handle_data()
    print(data.tail())

        
        
                