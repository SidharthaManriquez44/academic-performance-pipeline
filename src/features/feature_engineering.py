import pandas as pd
from src.utils.logger_config import get_logger

logger = get_logger(__name__)


def transform(df):
    logger.info("Transforming data")

    # feature util
    df["study_efficiency"] = df["Study_Hours_per_Week"] / (df["Attendance (%)"] + 1)

    # target
    df["target"] = df["Grade"].isin(["D", "F"]).astype(int)

    df = df.drop(columns=["Grade"])

    # one-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    logger.info("Transformation complete")
    return df
