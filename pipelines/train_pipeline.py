from components.ingest_data import ingest_data
from components.clean_data import clean_data
from components.train_model import train_model
from components.eval_model import eval_model
from components.config import ModelNameConfig

def train_pipeline(data_filepath: str):
    df = ingest_data(data_filepath)
    X_train, X_test, y_train, y_test = clean_data(df)
    train_model(X_train, X_test, y_train, y_test, ModelNameConfig())
    eval_model(df)