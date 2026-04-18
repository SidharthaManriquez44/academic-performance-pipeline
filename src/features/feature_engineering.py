import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.utils.logger_config import get_logger

logger = get_logger(__name__)


def transform(df: pd.DataFrame, kind: str = "multiclass"):
    logger.info(f"Transforming data | mode={kind}")

    # Avoid modify teh original
    df = df.copy()

    # Util feature
    df["study_efficiency"] = df["Study_Hours_per_Week"] / (df["Attendance (%)"] + 1)

    # Target
    if kind == "multiclass":
        le = LabelEncoder()
        df["target"] = le.fit_transform(df["Grade"])

        logger.info(f"Classes: {le.classes_}")

    elif kind == "binary":
        df["target"] = df["Grade"].isin(["D", "F"]).astype(int)
        le = None  # consistency

    else:
        raise ValueError(f"Invalid kind: {kind}. Use 'multiclass' or 'binary'")

    # Drop no used columns
    df = df.drop(columns=["Grade"])

    # one-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    logger.info("Transformation complete")

    return df, le
