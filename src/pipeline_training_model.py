from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight # Per gestire lo sbilanciamento delle classi

from src.hyperparameter_tuning import tune_random_forest
from .data_loader import DataLoader
from .preprocessing.pipeline import make_complete_pipeline_from_features
from .evaluation import evaluate_predictions
from .models import get_random_forest_model, get_xgboost_model, get_lightgbm_model
from .config import RANDOM_STATE
from .featureselector import FeatureSelector
from .preprocessing import pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from .hyperparameter_tuning_feature_selection import hyperparameter_tune_fs

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
        print("--- 2. Feature Selection & Tuning ---")
        preprocessor = pipeline.get_preprocessing_pipeline(scale_numeric=True)
        X_encoded = preprocessor.fit_transform(X)
        
        print("Esecuzione tuning automatico veloce (sampling 20k righe, 5 iterazioni)...")
        tune_pipe = Pipeline([
            ('selector', FeatureSelector(fs_method='rf')),
            ('model', RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1))
        ])
        
        # Sampling per velocizzare il tuning
        sample_size = min(20000, len(X_encoded))
        X_tune = X_encoded.sample(n=sample_size, random_state=RANDOM_STATE)
        y_tune = y.iloc[X_tune.index]
        
        best_fs_params = hyperparameter_tune_fs(tune_pipe, X_tune, y_tune, num_iter=5)
        print(f"Parametri ottimali trovati: {best_fs_params}")
        
        # Rimuoviamo il prefisso 'selector__' dai parametri per poterli usare nel singolo oggetto
        clean_fs_params = {k.replace('selector__', ''): v for k, v in best_fs_params.items() if k.startswith('selector__')}
        
        f_selector_rf = FeatureSelector('rf')
        f_selector_rf.set_params(**clean_fs_params)
        f_selector_rf.fit(X_encoded, y)
        print("---. Feature rf (TUNED): \n", f_selector_rf.get_feature_names_out())

        f_selector_xgb = FeatureSelector('xgb', 0.005, 20)
        f_selector_xgb.fit(X_encoded, y)
        print("---. Feature xgb: \n", f_selector_xgb.get_feature_names_out())
        
        f_selector_ctb = FeatureSelector('ctb', 0.005, 20)
        f_selector_ctb.fit(X_encoded, y)
        print("---. Feature ctb: \n", f_selector_ctb.get_feature_names_out())

        f_selector_corr = FeatureSelector('corr_matrix', 0.005, 20)
        f_selector_corr.fit(X_encoded, y)
        print("---. Feature Correlation: \n", f_selector_corr.get_feature_names_out())

        f_selector_chi2 = FeatureSelector('chi2', 0.005, 20)
        f_selector_chi2.fit(X_encoded, y)
        print("---. Feature Chi-Square: \n", f_selector_chi2.get_feature_names_out())

        #f_selector_mu = FeatureSelector('mu', 0.005, 20)
        #f_selector_mu.fit(X_encoded, y)
        #print("---. Feature Mutual Info: \n", f_selector_mu.get_feature_names_out())

        # Aggiorniamo X con le feature selezionate dal selettore tuned per il training successivo
        X = f_selector_rf.transform(X_encoded)
        print(f"Dataset finale per il training: {X.shape[1]} feature.")



    print(f"--- 3. Splitting and Training (Strategy {split_strategy}) ---")
    
    # --- MODEL TUNING (Esempio per RandomForest) ---
    print("Esecuzione tuning iperparametri per RandomForest...")
    # Prepariamo i dati per il tuning (già filtrati dal selettore)
    # Usiamo un sampling anche qui per velocità se necessario, o procediamo sul dataset filtrato
    tuned_rf_model, best_rf_score, _ = tune_random_forest(X, y, n_iter=10)
    print(f"Miglior Macro-F1 trovato per RF: {best_rf_score:.4f}")

    models_to_compare = {
        "RandomForest_Tuned": tuned_rf_model,
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
