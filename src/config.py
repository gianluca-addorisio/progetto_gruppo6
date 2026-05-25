from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"

MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_VALUES_FILE = RAW_DATA_DIR / "train_values.csv"
TRAIN_LABELS_FILE = RAW_DATA_DIR / "train_labels.csv"
TEST_VALUES_FILE = RAW_DATA_DIR / "test_values.csv"
SUBMISSION_FORMAT_FILE = RAW_DATA_DIR / "submission_format.csv"

ID_COL = "building_id"
TARGET_COL = "damage_grade"
RANDOM_STATE = 42
