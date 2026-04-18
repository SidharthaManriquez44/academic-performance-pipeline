from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.utils.logger_config import get_logger

logger = get_logger(__name__)

def train(df):
    logger.info("Training Random Forest model")

    # Split features & target
    X = df.drop(columns=["target"])
    y = df["target"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=200,          # more trees = best generalization
        max_depth=10,              # avoid overfitting
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1                 # use all cores
    )

    model.fit(X_train, y_train)

    logger.info("Random Forest training completed")

    return model, X_test, y_test
