import logging
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
from src.data_cleaning import DataCleaning, DataPreProcessStrategy, DataSplitStrategy

def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Component for cleaing data.
    Args:
        df: pandas.Dataframe
    """
    try:
        dc = DataCleaning(df, DataPreProcessStrategy())
        processed_df = dc.handle_data()
        dc = DataCleaning(processed_df, DataSplitStrategy())
        X_train, X_test, y_train, y_test = dc.handle_data()
        logging.info("Data cleaning completed.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in preprocessing the data: {e}")
        raise e