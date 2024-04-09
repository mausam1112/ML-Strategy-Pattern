from components.ingest_data import ingest_data
from components.clean_data import clean_data
from components.train_model import train_model
from components.eval_model import eval_model

def train_pipeline(data_filepath: str):
    df = ingest_data(data_filepath)
    clean_data(df)
    train_model(df)
    eval_model(df)