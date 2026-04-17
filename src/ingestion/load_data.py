from pathlib import Path
import pandas as pd
from src.utils.logger_config import get_logger

logger = get_logger(__name__)
BASE_DIR = Path(__file__).resolve().parents[2]

def load_data(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    logger.info(f"Data loaded successfully: {path}")
    logger.info(data.head())
    return data


if __name__ == "__main__":
    file_path = BASE_DIR / "data/raw/students.csv"
    df = load_data(file_path)
