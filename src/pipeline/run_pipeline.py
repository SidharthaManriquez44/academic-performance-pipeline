from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from src.ingestion.load_data import load_data
from pathlib import Path

from src.preprocessing.data_cleaning import clean_data
from src.validation.data_validation import validate_data
from src.features.select_features import select_features
from src.features.feature_engineering import transform

from src.utils.logger_config import get_logger
from models.xgboost.train import train
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


    model, X_test, y_test = train(df)

    # Generated predictions
    y_pred = model.predict(X_test)

    # Get metrics
    print(classification_report(y_test, y_pred))


    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()
