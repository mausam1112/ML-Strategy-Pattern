from components.ingest_data import ingest_data
from components.clean_data import clean_data
from components.train_model import train_model
from components.eval_model import eval_model
from components.config import ModelNameConfig

def train_pipeline(data_filepath: str):
    """
    Pipeline for training data.
    Args:
        data_filepath: str, path to data file 
    """
    df = ingest_data(data_filepath)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test, ModelNameConfig())
    rmse, mse, r2 = eval_model(model, X_test, y_test)