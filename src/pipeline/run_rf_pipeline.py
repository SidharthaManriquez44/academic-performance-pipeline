from pathlib import Path
from sklearn.metrics import classification_report

from src.ingestion.load_data import load_data
from src.preprocessing.data_cleaning import clean_data
from src.validation.data_validation import validate_data
from src.features.select_features import select_features
from src.features.feature_engineering import transform
from src.features.confusion_matrix import plot_confusion_matrix

from src.utils.logger_config import get_logger
from models.random_forest.train_random import train
from src.evaluation.metrics import evaluate

logger = get_logger(__name__)
BASE_DIR = Path(__file__).resolve().parents[2]


def run_pipeline(kind: str = "binary"):
    logger.info(f"Running Random Forest pipeline | mode={kind}")

    # Load
    file_path = BASE_DIR / "data/raw/students.csv"
    df = load_data(file_path)

    # Clean & validate
    df = clean_data(df)
    validate_data(df)

    # Feature selection
    df = select_features(df)

    # Feature engineering
    df, le = transform(df, kind)

    # Train
    model, X_test, y_test = train(df)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    logger.info(classification_report(y_test, y_pred))
    evaluate(model, X_test, y_test)

    # Confusion Matrix
    if kind == "multiclass":
        y_pred_labels = le.inverse_transform(y_pred)
        y_test_labels = le.inverse_transform(y_test)

        plot_confusion_matrix(
            y_test_labels,
            y_pred_labels,
            "Confusion Matrix - Random Forest (Multiclass)"
        )
    else:
        plot_confusion_matrix(
            y_test,
            y_pred,
            "Confusion Matrix - Random Forest (Binary)"
        )

    return model


if __name__ == "__main__":
    # run_pipeline(kind="multiclass")
    run_pipeline(kind="binary")