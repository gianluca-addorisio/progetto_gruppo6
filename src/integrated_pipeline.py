import pandas as pd
import logging
import sys
from pathlib import Path
from src.data_loader import DataLoader
from src.preprocessing import prepare_features
from src.feature_selection import FeatureSelection
from src.pipeline_training_model import run_training_pipeline
from src.utils import ensure_dir
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay

# Configurazione Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("--- Avvio Pipeline Integrata ---")
    
    # 1. CARICAMENTO DATI
    dl = DataLoader()
    X, y = dl.load_train_test()
    X_train, X_val, y_train, y_val = dl.split_dataset_by_strategy(2, X, y)
    
    # 2. Target Encoding (Identico a notebook 5)
    geo_cols = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
    for col in geo_cols:
        risk_map = y_train.groupby(X_train[col]).mean()
        std_map = y_train.groupby(X_train[col]).std()
        X_train[f'{col}_risk_mean'] = X_train[col].map(risk_map).fillna(0)
        X_val[f'{col}_risk_mean'] = X_val[col].map(risk_map).fillna(0)
        X_train[f'{col}_risk_std'] = X_train[col].map(std_map).fillna(0)
        X_val[f'{col}_risk_std'] = X_val[col].map(std_map).fillna(0)
        X_train = X_train.drop(columns=[col])
        X_val = X_val.drop(columns=[col])
    
    X_processed_train = prepare_features(X_train)
    X_processed_val = prepare_features(X_val)
    
    # 3. SETUP SELEZIONE
    fs = FeatureSelection()
    target_col = 'damage_grade'
    
    # Definiamo le strategie
    strategies = {
        "RandomForest": lambda: fs.random_forest_selection(X_processed_train.drop(columns=[target_col], errors='ignore'), y_train),
        "XGBoost": lambda: fs.xgboost_selection(X_processed_train.drop(columns=[target_col], errors='ignore'), y_train),
        "CatBoost": lambda: fs.catboost_selection(X_processed_train.drop(columns=[target_col], errors='ignore'), y_train)
    }

    final_results = []
    
    # CICLO SULLE STRATEGIE
    for strategy_name, selection_func in strategies.items():
        logger.info(f"--- Selezione con {strategy_name} ---")
        
        # 1. Esegui la selezione
        X_selected_train = selection_func()
        
        # 2. Pulizia: rimuovi il target se presente
        cols_to_keep = [c for c in X_selected_train.columns if c != target_col]
        X_sel_train = X_selected_train[cols_to_keep]
        
        # 3. TRASFORMAZIONE CRITICA: Converti tutto in numeri
        # get_dummies gestisce le stringhe ('r', 'v', ecc.) trasformandole in colonne 0/1
        X_sel_train = pd.get_dummies(X_sel_train)
        
        # 4. Allinea il validation set alle nuove colonne dummy create
        X_sel_val = X_processed_val.reindex(columns=X_sel_train.columns, fill_value=0)
        
        # 4. TRAINING E VALUTAZIONE (Loop sui tuoi modelli)
        y_train_idx = y_train - 1
        y_val_idx = y_val - 1
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=25, min_samples_leaf=5, random_state=42, n_jobs=-1),
            "XGBoost": XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=10, eval_metric='mlogloss', n_jobs=-1, random_state=42),
            "LightGBM": LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, n_jobs=-1, random_state=42, verbose=-1)
        }

        for model_name, model in models.items():
            target_train = y_train_idx if model_name in ["XGBoost", "LightGBM"] else y_train
            target_val = y_val_idx if model_name in ["XGBoost", "LightGBM"] else y_val

            model.fit(X_sel_train, target_train)
            y_pred = model.predict(X_sel_val)

            score = f1_score(target_val, y_pred, average='micro')
            final_results.append({"Strategy": strategy_name, "Model": model_name, "Micro-F1": score})
            print(f"{strategy_name} + {model_name} | F1: {score:.4f}")
    
    # 5. SALVATAGGIO
    pd.DataFrame(final_results).to_csv("outputs/metrics/final_experiment_results.csv", index=False)
    
if __name__ == "__main__":
    main()
