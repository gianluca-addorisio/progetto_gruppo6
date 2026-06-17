from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"


TRAIN_VALUES_FILE = RAW_DATA_DIR / "train_values.csv"
TRAIN_LABELS_FILE = RAW_DATA_DIR / "train_labels.csv"
TEST_VALUES_FILE = RAW_DATA_DIR / "test_values.csv"
SUBMISSION_FORMAT_FILE = RAW_DATA_DIR / "submission_format.csv"

RESULTS_COMPARISON_FILE = METRICS_DIR / "results_comparison.csv"
FINAL_SUBMISSION_FILE = SUBMISSIONS_DIR / "final_submission.csv"

ID_COL = "building_id"
TARGET_COL = "damage_grade"
RANDOM_STATE = 42

FINAL_MODEL_NAME = "XGBoost"
FINAL_SPLIT_STRATEGY = 2
FINAL_FEATURE_SELECTION = False
FINAL_USE_SAMPLE_WEIGHT = False
FINAL_USE_PCA = False
FINAL_DO_TUNING = False

VALID_MODEL_NAMES = (
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "VotingEnsemble",
    "StackingEnsemble",
)
