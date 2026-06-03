from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight

from .data_loader import DataLoader
from .preprocessing.pipeline import make_complete_pipeline, make_complete_pipeline_with_selection
from .evaluation import evaluate_predictions
from .models import get_random_forest_model, get_xgboost_model, get_lightgbm_model, get_dummy_classifier, get_logistic_regression, get_decision_tree
from .config import RANDOM_STATE

def run_training_pipeline(feature_selection: bool = True, fs_method: str = 'rf', split_strategy: int = 4):
    """
    Main pipeline for loading data, training models and comparing results.
    """
    print(f"--- 1. Loading Data (FS Method: {fs_method if feature_selection else 'None'}) ---")
    data_loader = DataLoader()
    X, y = data_loader.load_train_test()
    
    # Map labels to [0, 1, 2] for XGBoost/LightGBM compatibility
    y = y - 1

    print(f"--- 2. Splitting and Training (Strategy {split_strategy}) ---")
    
    models_to_compare = {
        'DummyClassifier': get_dummy_classifier(),
        'LogisticRegression': get_logistic_regression(),
        'DescisionTree': get_decision_tree(),
        "RandomForest": get_random_forest_model(),
        "XGBoost": get_xgboost_model(),
        "LightGBM": get_lightgbm_model()
    }

    results = []

    if split_strategy in [1, 2]:
        # --- HOLD-OUT STRATEGY ---
        X_train, X_val, y_train, y_val = data_loader.split_dataset_by_strategy(split_strategy, X, y)
        print(f"Train size: {X_train.shape}, Validation size: {X_val.shape}")
        
        weights_train = compute_sample_weight(class_weight='balanced', y=y_train)
        
        for name, model in models_to_compare.items():
            print(f"Training {name}...")
            
            # Creiamo la pipeline completa (Preprocessing [+ Selezione] + Modello)
            if feature_selection and name not in ['DummyClassifier', 'LogisticRegression']:
                full_pipeline = make_complete_pipeline_with_selection(model, fs_method=fs_method, max_features=30)
            else:
                full_pipeline = make_complete_pipeline(model)
            
            # Fit della pipeline (Preprocessing e Selezione imparano solo dal Train)
            full_pipeline.fit(X_train, y_train, model__sample_weight=weights_train)
            y_pred = full_pipeline.predict(X_val)
            
            metrics = evaluate_predictions(y_val, y_pred, name)
            results.append(metrics)
            print(f"Done. Micro-F1: {metrics['micro_f1']:.4f} | Macro-F1: {metrics['macro_f1']:.4f}")

    else:
        # --- CROSS-VALIDATION STRATEGY ---
        splits = data_loader.split_dataset_by_strategy(split_strategy, X, y)

        for name, model in models_to_compare.items():
            print(f"Evaluating {name} via CV...")
            fold_results = []
            
            for fold, (train_idx, val_idx) in enumerate(splits):
                X_train_f, X_val_f = X.iloc[train_idx], X.iloc[val_idx]
                y_train_f, y_val_f = y.iloc[train_idx], y.iloc[val_idx]
                
                weights_fold = compute_sample_weight(class_weight='balanced', y=y_train_f)
                
                model_fold = clone(model)
                if feature_selection and name not in ['DummyClassifier', 'LogisticRegression']:
                    full_pipeline = make_complete_pipeline_with_selection(model_fold, fs_method=fs_method, max_features=30)
                else:
                    full_pipeline = make_complete_pipeline(model_fold)
                
                full_pipeline.fit(X_train_f, y_train_f, model__sample_weight=weights_fold)
                y_pred = full_pipeline.predict(X_val_f)
                
                fold_metrics = evaluate_predictions(y_val_f, y_pred, name)
                fold_results.append(fold_metrics)
                print(f"  Fold {fold+1}: Micro-F1 = {fold_metrics['micro_f1']:.4f} | Macro-F1 = {fold_metrics['macro_f1']:.4f}")

            avg_metrics = {
                "model": name,
                "micro_f1": np.mean([r["micro_f1"] for r in fold_results]),
                "macro_f1": np.mean([r["macro_f1"] for r in fold_results]),
                "weighted_f1": np.mean([r["weighted_f1"] for r in fold_results]),
            }
            results.append(avg_metrics)
            print(f"Average Micro-F1: {avg_metrics['micro_f1']:.4f} | Average Macro-F1: {avg_metrics['macro_f1']:.4f}\n")

    comparison_df = pd.DataFrame(results)
    print("\n--- 3. Results Comparison ---")
    print(comparison_df.to_string(index=False))
    
    return comparison_df

if __name__ == "__main__":
    # Esempio: puoi cambiare fs_method qui per testare diversi selettori
    run_training_pipeline(feature_selection=True, fs_method='rf', split_strategy=4)
