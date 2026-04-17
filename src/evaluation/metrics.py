from sklearn.metrics import classification_report
from src.utils.logger_config import get_logger

logger = get_logger(__name__)

def evaluate(model, X_test, y_test):
    pred_data = model.predict(X_test)
    logger.info(classification_report(y_test, pred_data))
