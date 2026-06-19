"""
Configurazione centralizzata del progetto.

Questo modulo definisce percorsi dei dati, directory di output, nomi delle
colonne principali, seed di riproducibilità e parametri della configurazione
finale del modello. Centralizzare questi valori evita duplicazioni e mantiene
allineati pipeline sperimentale, training finale e generazione della submission.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
MODELS_DIR = PROJECT_ROOT / "models"


TRAIN_VALUES_FILE = RAW_DATA_DIR / "train_values.csv"
TRAIN_LABELS_FILE = RAW_DATA_DIR / "train_labels.csv"
TEST_VALUES_FILE = RAW_DATA_DIR / "test_values.csv"
SUBMISSION_FORMAT_FILE = RAW_DATA_DIR / "submission_format.csv"

RESULTS_COMPARISON_FILE = METRICS_DIR / "results_comparison.csv"
FINAL_SUBMISSION_FILE = SUBMISSIONS_DIR / "final_submission.csv"
FINAL_PIPELINE_FILE = MODELS_DIR / "final_pipeline.joblib"
FINAL_MODEL_CONFIG_FILE = METRICS_DIR / "final_model_config.json"

ID_COL = "building_id"
TARGET_COL = "damage_grade"
RANDOM_STATE = 42

FINAL_MODEL_NAME = "StackingEnsemble"
FINAL_SPLIT_STRATEGY = 4
FINAL_FEATURE_SELECTION = True
FINAL_FS_METHOD = "ctb"
FINAL_FS_THRESHOLD = 0.005
FINAL_MAX_FEATURES_TO_HOLD = 30
FINAL_USE_SAMPLE_WEIGHT = False
FINAL_USE_PCA = False
FINAL_DO_TUNING = True
FINAL_TUNING_ITER = 15
FINAL_TUNING_SAMPLE_SIZE = 50000

VALID_MODEL_NAMES = (
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "VotingEnsemble",
    "StackingEnsemble",
)
