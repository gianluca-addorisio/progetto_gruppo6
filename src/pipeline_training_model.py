from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight

from .data_loader import DataLoader
from .preprocessing.pipeline import make_complete_pipeline
from .evaluation import evaluate_predictions
from .models import (
    get_random_forest_model, get_xgboost_model, get_lightgbm_model, 
    get_dummy_classifier, get_decision_tree, get_logistic_regression,
    get_stacking_ensemble, get_voting_ensemble
)
from .featureselector import FeatureSelector
from .config import RANDOM_STATE
from .hyperparameter_tuning import ModelTuner
from .hyperparameter_tuning_feature_selection import FeatureSelectionTuner

from sklearn.model_selection import train_test_split

def run_training_pipeline(
    feature_selection: bool = True,
    split_strategy: int = 4,
    use_sample_weight: bool = False,
    fs_method: str = "ctb",
    fs_threshold: float = 0.005,
    max_features_to_hold: int = 30,
    use_pca: bool = False,
    pca_n_components: int = 40,
    do_tuning: bool = False,
    tuning_iter: int = 50,
    tuning_sample_size: int = 30000
):
    """
    Pipeline principale ottimizzata:
    1. Tuning globale (opzionale) su subset per trovare i parametri migliori (Modello + FS).
    2. Valutazione robusta usando i parametri SPECIFICI per ogni modello.
    """
    print("--- 1. Loading Data ---")
    data_loader = DataLoader()
    X, y = data_loader.load_train_test()
    y = y - 1

    # Inizializziamo i Tuner
    tuner = ModelTuner(random_state=RANDOM_STATE)
    fs_tuner = FeatureSelectionTuner(random_state=RANDOM_STATE)
    
    # Dizionario per memorizzare la configurazione ottimizzata di ogni modello
    # Ogni entry sarà: { 'model': estimator, 'fs_params': { 'threshold': ..., 'max_features_to_hold': ... } }
    model_configs = {
        "RandomForest": {
            "model": get_random_forest_model(),
            "fs_params": {"threshold": fs_threshold, "max_features_to_hold": max_features_to_hold}
        },
        "XGBoost": {
            "model": get_xgboost_model(),
            "fs_params": {"threshold": fs_threshold, "max_features_to_hold": max_features_to_hold}
        },
        "LightGBM": {
            "model": get_lightgbm_model(),
            "fs_params": {"threshold": fs_threshold, "max_features_to_hold": max_features_to_hold}
        }
    }

    # --- 2. GLOBAL TUNING (se richiesto) ---
    if do_tuning:
        print(f"\n--- 2. Global Tuning attivo ({tuning_iter} iterazioni su {tuning_sample_size} campioni) ---")
        X_tune, _, y_tune, _ = train_test_split(
            X, y, train_size=min(tuning_sample_size, len(X)), 
            stratify=y, random_state=RANDOM_STATE
        )

        for name in ["RandomForest", "XGBoost", "LightGBM"]:
            print(f"  > Tuning {name} (Modello + Feature Selection)...")
            
            # Creiamo una pipeline temporanea con i parametri di default per il tuning
            temp_fs = FeatureSelector(fs_method=fs_method) if feature_selection else None
            temp_pipeline = make_complete_pipeline(
                model_configs[name]["model"], 
                feature_selector=temp_fs,
                use_pca=use_pca, 
                pca_n_components=pca_n_components
            )
            
            param_grid = tuner.get_param_grid(name)
            best_pipeline, best_params, _ = fs_tuner.tune_pipeline(
                temp_pipeline, name, param_grid, X_tune, y_tune, n_iter=tuning_iter
            )
            
            # SALVATAGGIO CONFIGURAZIONE SPECIFICA
            model_configs[name]["model"] = best_pipeline.named_steps['model']
            if feature_selection:
                model_configs[name]["fs_params"] = {
                    "threshold": best_params.get('feature_selector__threshold', fs_threshold),
                    "max_features_to_hold": best_params.get('feature_selector__max_features_to_hold', max_features_to_hold)
                }
            print(f"    Tuning completato per {name}.")

    # Preparazione Ensembles
    # Per semplicità, gli ensemble useranno i parametri di FS del modello con performance tipicamente migliore (LightGBM)
    ensemble_fs_params = model_configs["LightGBM"]["fs_params"] if do_tuning else {"threshold": fs_threshold, "max_features_to_hold": max_features_to_hold}
    
    model_configs["VotingEnsemble"] = {
        "model": get_voting_ensemble(
            model_configs["RandomForest"]["model"],
            model_configs["XGBoost"]["model"],
            model_configs["LightGBM"]["model"]
        ),
        "fs_params": ensemble_fs_params
    }
    model_configs["StackingEnsemble"] = {
        "model": get_stacking_ensemble(
            model_configs["RandomForest"]["model"],
            model_configs["XGBoost"]["model"],
            model_configs["LightGBM"]["model"]
        ),
        "fs_params": ensemble_fs_params
    }

    print(f"\n--- 3. Evaluation (Strategy {split_strategy}) ---")
    results = []

    if split_strategy in [1, 2]:
        # --- HOLD-OUT FLOW ---
        X_train, X_val, y_train, y_val = data_loader.split_dataset_by_strategy(split_strategy, X, y)
        print(f"Train size: {X_train.shape}, Validation size: {X_val.shape}")
        
        for name, config in model_configs.items():
            print(f"  [Model: {name}] Training con FS dedicata...")
            
            # Applichiamo la FS specifica del modello
            fs_p = config["fs_params"]
            fs = FeatureSelector(
                fs_method=fs_method, 
                threshold=fs_p["threshold"], 
                max_features_to_hold=fs_p["max_features_to_hold"]
            ) if feature_selection else None
            
            pipeline = make_complete_pipeline(
                config["model"], 
                feature_selector=fs, 
                use_pca=use_pca, 
                pca_n_components=pca_n_components
            )
            
            if use_sample_weight and name not in ["VotingEnsemble", "StackingEnsemble"]:
                weights = compute_sample_weight(class_weight='balanced', y=y_train)
                pipeline.fit(X_train, y_train, model__sample_weight=weights)
            else:
                pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_val)
            metrics = evaluate_predictions(y_val, y_pred, name)
            results.append(metrics)
            print(f"    Micro-F1: {metrics['micro_f1']:.4f} | FS: {fs_p['max_features_to_hold']} feat")

    else:
        # --- CROSS-VALIDATION FLOW ---
        splits = data_loader.split_dataset_by_strategy(split_strategy, X, y)
        
        for name, config in model_configs.items():
            print(f"  [Model: {name}] CV Evaluation con FS dedicata...")
            fold_results = []
            fs_p = config["fs_params"]
            
            for fold, (train_idx, val_idx) in enumerate(splits):
                X_train_f, X_val_f = X.iloc[train_idx], X.iloc[val_idx]
                y_train_f, y_val_f = y.iloc[train_idx], y.iloc[val_idx]
                
                # Clone del modello ottimizzato
                model_fold = clone(config["model"])
                
                # Ricreiamo la FS specifica per il fold
                fs = FeatureSelector(
                    fs_method=fs_method, 
                    threshold=fs_p["threshold"], 
                    max_features_to_hold=fs_p["max_features_to_hold"]
                ) if feature_selection else None
                
                pipeline = make_complete_pipeline(
                    model_fold, 
                    feature_selector=fs, 
                    use_pca=use_pca, 
                    pca_n_components=pca_n_components
                )

                if use_sample_weight and name not in ["VotingEnsemble", "StackingEnsemble"]:
                    weights = compute_sample_weight(class_weight="balanced", y=y_train_f)
                    pipeline.fit(X_train_f, y_train_f, model__sample_weight=weights)
                else:
                    pipeline.fit(X_train_f, y_train_f)

                y_pred = pipeline.predict(X_val_f)
                fold_metrics = evaluate_predictions(y_val_f, y_pred, name)
                fold_results.append(fold_metrics)

            avg_metrics = {
                "model": name,
                "micro_f1": np.mean([r["micro_f1"] for r in fold_results]),
                "macro_f1": np.mean([r["macro_f1"] for r in fold_results]),
                "weighted_f1": np.mean([r["weighted_f1"] for r in fold_results]),
            }
            results.append(avg_metrics)
            print(f"    Avg Micro-F1: {avg_metrics['micro_f1']:.4f} | FS: {fs_p['max_features_to_hold']} feat")

    comparison_df = pd.DataFrame(results)
    print("\n--- 4. Final Comparison ---")
    print(comparison_df.to_string(index=False))
    
    return comparison_df

if __name__ == "__main__":
    run_training_pipeline(
        do_tuning=True, 
        tuning_iter=15, 
        tuning_sample_size=50000,
        split_strategy=4
    )
