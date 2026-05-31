from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from .data_loader import DataLoader
from .preprocessing import make_complete_pipeline # Usiamo la funzione helper per pipeline piatte
from .evaluation import evaluate_predictions
from .models import get_random_forest_model, get_xgboost_model, get_lightgbm_model
from .config import RANDOM_STATE

def run_training_pipeline(feature_selection: bool = False, split_strategy: int = 4):
    """
    Main pipeline for loading data, training models and comparing results.
    """
    print("--- 1. Loading Data ---")
    data_loader = DataLoader()
    X, y = data_loader.load_train_test()
    
    # Map labels to [0, 1, 2] for XGBoost/LightGBM compatibility
    y = y - 1
    
    if feature_selection:
        print("--- 2. Feature Selection (Placeholder) ---")
        pass

    print(f"--- 3. Splitting and Training (Strategy {split_strategy}) ---")
    
    models_to_compare = {
        "RandomForest": get_random_forest_model(),
        "XGBoost": get_xgboost_model(),
        "LightGBM": get_lightgbm_model()
    }

    results = []

    if split_strategy in [1, 2]:
        # --- HOLD-OUT STRATEGY ---
        X_train, X_val, y_train, y_val = data_loader.split_dataset_by_strategy(split_strategy, X, y)
        print(f"Train size: {X_train.shape}, Validation size: {X_val.shape}")
        
        for name, model in models_to_compare.items():
            print(f"Training {name} con Nuova Pipeline Modulare...")
            
            # Creiamo una pipeline "piatta" (Preprocessing + Modello)
            full_pipeline = make_complete_pipeline(model)
            
            full_pipeline.fit(X_train, y_train)
            y_pred = full_pipeline.predict(X_val)
            
            metrics = evaluate_predictions(y_val, y_pred, name)
            results.append(metrics)
            print(f"Done. Micro-F1: {metrics['micro_f1']:.4f}")

    else:
        # --- CROSS-VALIDATION STRATEGY ---
        splits = data_loader.split_dataset_by_strategy(split_strategy, X, y)

        for name, model in models_to_compare.items():
            print(f"Evaluating {name} via CV con Nuova Pipeline Modulare...")
            fold_results = []
            
            for fold, (train_idx, val_idx) in enumerate(splits):
                X_train_f, X_val_f = X.iloc[train_idx], X.iloc[val_idx]
                y_train_f, y_val_f = y.iloc[train_idx], y.iloc[val_idx]
                
                # Creiamo una pipeline fresca e piatta per ogni fold
                model_fold = clone(model)
                full_pipeline = make_complete_pipeline(model_fold)
                
                full_pipeline.fit(X_train_f, y_train_f)
                y_pred = full_pipeline.predict(X_val_f)
                
                fold_metrics = evaluate_predictions(y_val_f, y_pred, name)
                fold_results.append(fold_metrics)
                print(f"  Fold {fold+1}: Micro-F1 = {fold_metrics['micro_f1']:.4f}")

            avg_metrics = {
                "model": name,
                "micro_f1": np.mean([r["micro_f1"] for r in fold_results]),
                "macro_f1": np.mean([r["macro_f1"] for r in fold_results]),
                "weighted_f1": np.mean([r["weighted_f1"] for r in fold_results]),
            }
            results.append(avg_metrics)
            print(f"Average Micro-F1 for {name}: {avg_metrics['micro_f1']:.4f}\n")

    comparison_df = pd.DataFrame(results)
    print("\n--- 4. Results Comparison ---")
    print(comparison_df.to_string(index=False))
    
    return comparison_df

if __name__ == "__main__":
    run_training_pipeline()
