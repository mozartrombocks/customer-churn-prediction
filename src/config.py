from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "customer_churn.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

TARGET_COLUMN  = "Churn Label"

RANDOM_STATE = 42 
TEST_SIZE = 0.20 
VALID_SIZE = 0.25 # 25% of remaining train split 

