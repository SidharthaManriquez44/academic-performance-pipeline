from src.utils.logger_config import get_logger

logger = get_logger(__name__)

def select_features(df):
    logger.info("Selecting features")

    drop_cols = [
        "Student_ID",
        "First_Name",
        "Last_Name",
        "Email",
        "Final_Score",
        "Total_Score",
    ]

    df = df.drop(columns=drop_cols)

    logger.info("Feature selection complete")
    return df
