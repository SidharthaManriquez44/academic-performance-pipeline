from models.xgboost.pipeline import build_pipeline
from sklearn.model_selection import train_test_split
from src.utils.logger_config import get_logger
import joblib

logger = get_logger(__name__)


def train(df):
    print("Training pipeline...")

    X = df.drop("target", axis=1)
    y = df["target"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    pipeline = build_pipeline(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, "model.pkl") 

    logger.info("Pipeline trained successfully!")

    return pipeline, X_test, y_test