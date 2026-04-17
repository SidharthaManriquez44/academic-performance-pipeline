import pandas as pd
from src.ingestion.load_data import load_data
from pathlib import Path

from src.preprocessing.data_cleaning import clean_data
from src.validation.data_validation import validate_data
from src.features.select_features import select_features
from src.features.feature_engineering import transform
from src.utils.logger_config import get_logger
from src.models.train import train
from src.evaluation.metrics import evaluate


BASE_DIR = Path(__file__).resolve().parents[2]
logger = get_logger(__name__)

if __name__ == "__main__":
    file_path = BASE_DIR / "data/raw/students.csv"
    df = load_data(file_path)


    df = clean_data(df)
    validate_data(df)

    df = select_features(df)
    df = transform(df)

    model, X_test, y_test = train(df)

    evaluate(model, X_test, y_test)
