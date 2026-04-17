import pandas as pd
from src.utils.logger_config import get_logger

logger = get_logger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data")

    # 1. Delete completely empty rows
    df = df.dropna(how="all")

    # 2. Fill numeric values with mean
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        df[col] = df[col].fillna(df[col].mean())

    # 3. Fill categorical with mode
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    logger.info("Cleaning data")
    return df
