from pathlib import Path
from sklearn.metrics import classification_report

from src.utils.logger_config import get_logger
from src.ingestion.load_data import load_data
from src.preprocessing.data_cleaning import clean_data
from src.validation.data_validation import validate_data
from src.features.select_features import select_features
from src.features.prepare_data import prepare_data
from src.features.confusion_matrix import plot_confusion_matrix

from src.models.logistic_regression.train_logistic import train_logistic_model


logger = get_logger(__name__)
BASE_DIR = Path(__file__).resolve().parents[2]


def run_logistic_pipeline(kind="binary"):

    logger.info(f"Running Logistic | mode={kind}")

    # Load
    file_path = BASE_DIR / "data/raw/students.csv"
    df = load_data(file_path)

    # Clean
    df = clean_data(df)
    validate_data(df)

    # Features
    df = select_features(df)

    # Prepare
    X, y, num_features, cat_features, le = prepare_data(df, kind)

    # Train
    model, X_test, y_test, y_pred = train_logistic_model(
        X, y, num_features, cat_features
    )

    # Metrics
    logger.info(classification_report(y_test, y_pred))

    # Multiclass labels
    if kind == "multiclass":
        y_pred_labels = le.inverse_transform(y_pred)
        y_test_labels = le.inverse_transform(y_test)

        plot_confusion_matrix(
            y_test_labels,
            y_pred_labels,
            "Confusion Matrix - Logistic (Multiclass)"
        )
    else:
        plot_confusion_matrix(
            y_test,
            y_pred,
            "Confusion Matrix - Logistic (Binary)"
        )

    return model


if __name__ == "__main__":
    run_logistic_pipeline("binary")
    # run_logistic_pipeline("multiclass")
