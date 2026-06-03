from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight # Per gestire lo sbilanciamento delle classi

from .data_loader import DataLoader
from .preprocessing.pipeline import make_complete_pipeline_from_features
from .evaluation import evaluate_predictions
from .models import get_random_forest_model, get_xgboost_model, get_lightgbm_model
from .config import RANDOM_STATE
from .featureselector import FeatureSelector
from .preprocessing import pipeline

def run_training_pipeline(feature_selection: bool = True, split_strategy: int = 2):
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
        preprocessor = pipeline.get_preprocessing_pipeline(scale_numeric=True)
        X = preprocessor.fit_transform(X)
        f_selector_rf = FeatureSelector('rf', 0.005, 20)
        f_selector_rf.fit(X, y)
        print("---. Feature rf: \n", f_selector_rf.get_feature_names_out())
        f_selector_xgb = FeatureSelector('xgb', 0.005, 20)
        f_selector_xgb.fit(X, y)
        print("---. Feature xgb: \n", f_selector_xgb.get_feature_names_out())
        f_selector_ctb = FeatureSelector('ctb', 0.005, 20)
        f_selector_ctb.fit(X, y)
        print("--- 3. Feature ctb: \n", f_selector_ctb.get_feature_names_out())

        f_selector_corr = FeatureSelector('corr_matrix', 0.005, 20)
        f_selector_corr.fit(X, y)
        print("--- 3. Feature Correlation: \n", f_selector_corr.get_feature_names_out())

        f_selector_chi2 = FeatureSelector('chi2', 0.005, 20)
        f_selector_chi2.fit(X, y)
        print("--- 3. Feature Chi-Square: \n", f_selector_chi2.get_feature_names_out())

        f_selector_mu = FeatureSelector('mu', 0.005, 20)
        f_selector_mu.fit(X, y)
        print("--- 3. Feature Mutual Info: \n", f_selector_mu.get_feature_names_out())

        # ReliefF può essere molto lento su dataset grandi, lo mettiamo per ultimo
        #f_selector_rlf = FeatureSelector('rlf', 0.005, 20)
        #f_selector_rlf.fit(X, y)
        #sprint("--- 3. Feature ReliefF: \n", f_selector_rlf.get_feature_names_out())
        pass


    print(f"--- 3. Splitting and Training (Strategy {split_strategy}) ---")
    
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
        
        # BILANCIAMENTO: Calcoliamo i pesi per ogni riga del training set.
        # Le classi meno frequenti (come la Classe 1) riceveranno un peso maggiore.
        # Questo costringe il modello a dare più importanza agli errori sulle classi rare.
        weights_train = compute_sample_weight(class_weight='balanced', y=y_train)
        
        for name, model in models_to_compare.items():
            print(f"Training {name} con Pesi Bilanciati (per Macro-F1)...")
            
            # Creiamo una pipeline "piatta" (Preprocessing + Modello)
            full_pipeline = make_complete_pipeline_from_features(model, X_train)
            
            # Passiamo i pesi calcolati allo step 'model' della pipeline
            full_pipeline.fit(X_train, y_train, model__sample_weight=weights_train)
            y_pred = full_pipeline.predict(X_val)
            
            metrics = evaluate_predictions(y_val, y_pred, name)
            results.append(metrics)
            print(f"Done. Micro-F1: {metrics['micro_f1']:.4f} | Macro-F1: {metrics['macro_f1']:.4f}")

    else:
        # --- CROSS-VALIDATION STRATEGY ---
        splits = data_loader.split_dataset_by_strategy(split_strategy, X, y)

        for name, model in models_to_compare.items():
            print(f"Evaluating {name} via CV con Pesi Bilanciati...")
            fold_results = []
            
            for fold, (train_idx, val_idx) in enumerate(splits):
                X_train_f, X_val_f = X.iloc[train_idx], X.iloc[val_idx]
                y_train_f, y_val_f = y.iloc[train_idx], y.iloc[val_idx]
                
                # Calcoliamo i pesi bilanciati specifici per questo fold
                weights_fold = compute_sample_weight(class_weight='balanced', y=y_train_f)
                
                # Creiamo una pipeline fresca e piatta per ogni fold
                model_fold = clone(model)
                full_pipeline = make_complete_pipeline_from_features(model_fold, X_train_f)
                
                # Applichiamo i pesi nel fit
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
    print("\n--- 4. Results Comparison ---")
    print(comparison_df.to_string(index=False))
    
    return comparison_df

if __name__ == "__main__":
    run_training_pipeline()
