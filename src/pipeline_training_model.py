from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from .data_loader import DataLoader
from .preprocessing import prepare_features, make_model_pipeline
from .evaluation import evaluate_predictions
from .models import get_random_forest_model, get_xgboost_model, get_lightgbm_model
from .config import RANDOM_STATE

def run_training_pipeline(feature_selection: bool = False, split_strategy: int = 4):
    """
    Main pipeline for loading data, training models and comparing results.
    
    Args:
        feature_selection: Whether to apply feature selection (placeholder).
        split_strategy: Choice from DataLoader.split_dataset_by_strategy:
            1: Holdout, 2: Stratified Holdout, 3: K-Fold, 4: Stratified K-Fold...
    """
    print("--- 1. Loading Data ---")
    data_loader = DataLoader()
    X, y = data_loader.load_train_test()
    
    print("--- 2. Preparing Features ---")
    X_prepared = prepare_features(X)
    
    # Map labels to [0, 1, 2] for XGBoost/LightGBM compatibility (they expect 0-indexed classes)
    y = y - 1
    
    if feature_selection:
        print("--- 2b. Feature Selection (Placeholder) ---")
        # Qui potrai applicare la tua logica di feature selection
        # Es: fs = FeatureSelection()
        # X_prepared = fs.select(X_prepared, y)
        pass

    print(f"--- 3. Splitting Data (Strategy {split_strategy}) ---")
    
    # Define models to compare using factory functions from models.py
    models_to_compare = {
        "RandomForest": get_random_forest_model(),
        "XGBoost": get_xgboost_model(),
        "LightGBM": get_lightgbm_model()
    }

    results = []

    if split_strategy in [1, 2]:
        # --- HOLD-OUT STRATEGY ---
        X_train, X_val, y_train, y_val = data_loader.split_dataset_by_strategy(split_strategy, X_prepared, y)
        print(f"Train size: {X_train.shape}, Validation size: {X_val.shape}")
        
        print("\n--- 4. Training and Evaluation (Holdout) ---")
        for name, model in models_to_compare.items():
            print(f"Training {name}...")
            pipeline = make_model_pipeline(model, X_train)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            
            metrics = evaluate_predictions(y_val, y_pred, name)
            results.append(metrics)
            print(f"Done. Micro-F1: {metrics['micro_f1']:.4f}")

    else:
        # --- CROSS-VALIDATION STRATEGY ---
        print(f"\n--- 4. Training and Evaluation (CV Strategy {split_strategy}) ---")
        # DataLoader returns list of (train_idx, val_idx)
        splits = data_loader.split_dataset_by_strategy(split_strategy, X_prepared, y)

        for name, model in models_to_compare.items():
            print(f"Evaluating {name} via CV...")
            fold_results = []
            
            for fold, (train_idx, val_idx) in enumerate(splits):
                X_train_f, X_val_f = X_prepared.iloc[train_idx], X_prepared.iloc[val_idx]
                y_train_f, y_val_f = y.iloc[train_idx], y.iloc[val_idx]
                
                # Clone model to ensure a fresh start for each fold
                model_fold = clone(model)
                pipeline = make_model_pipeline(model_fold, X_train_f)
                
                pipeline.fit(X_train_f, y_train_f)
                y_pred = pipeline.predict(X_val_f)
                
                fold_metrics = evaluate_predictions(y_val_f, y_pred, name)
                fold_results.append(fold_metrics)
                print(f"  Fold {fold+1}: Micro-F1 = {fold_metrics['micro_f1']:.4f}")

            # Aggregate fold results
            avg_metrics = {
                "model": name,
                "micro_f1": np.mean([r["micro_f1"] for r in fold_results]),
                "macro_f1": np.mean([r["macro_f1"] for r in fold_results]),
                "weighted_f1": np.mean([r["weighted_f1"] for r in fold_results]),
            }
            results.append(avg_metrics)
            print(f"Average Micro-F1 for {name}: {avg_metrics['micro_f1']:.4f}\n")

    # Create comparison table
    comparison_df = pd.DataFrame(results)
    
    print("\n--- 5. Results Comparison ---")
    print(comparison_df.to_string(index=False))
    
    return comparison_df

if __name__ == "__main__":
    run_training_pipeline()
