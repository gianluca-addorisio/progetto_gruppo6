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
    tuning_iter: int = 50
):
    """
    Pipeline principale per caricamento dati, addestramento modelli e confronto risultati.
    Include l'integrazione con le classi di Hyperparameter Tuning e Ensemble.
    """
    print("--- 1. Loading Data ---")
    data_loader = DataLoader()
    X, y = data_loader.load_train_test()
    
    # Mappa i label a [0, 1, 2] per compatibilità
    y = y - 1

    if do_tuning:
        print(f"--- Tuning attivo ({tuning_iter} iterazioni) ---")

    if feature_selection:
        print(
            f"--- 2. Feature Selection attiva "
            f"({fs_method}, threshold={fs_threshold}, max_features={max_features_to_hold}) ---"
        )
    else:
        print("--- 2. Feature Selection non attiva ---")

    if use_pca:
        print(f"--- PCA attiva ({pca_n_components} componenti) ---")

    print(f"--- 3. Splitting and Training (Strategy {split_strategy}) ---")

    model_factories = {
        "RandomForest": get_random_forest_model,
        "XGBoost": get_xgboost_model,
        "LightGBM": get_lightgbm_model,
        "VotingEnsemble": get_voting_ensemble(),
        "StackingEnsemble": get_stacking_ensemble,
        #"DummyClassifier": get_dummy_classifier,
        #"DecisionTree": get_decision_tree,
        #"LogisticRegression": get_logistic_regression
    }

    models_to_compare = {}
    for name, factory in model_factories.items():
        try:
            models_to_compare[name] = factory()
        except Exception as exc:
            print(f"Skipping {name}: {exc}")

    results = []
    # Inizializziamo i Tuner (Classi Apposite)
    tuner = ModelTuner(random_state=RANDOM_STATE)
    fs_tuner = FeatureSelectionTuner(random_state=RANDOM_STATE)

    if split_strategy in [1, 2]:
        # --- HOLD-OUT STRATEGY ---
        X_train, X_val, y_train, y_val = data_loader.split_dataset_by_strategy(split_strategy, X, y)
        print(f"Train size: {X_train.shape}, Validation size: {X_val.shape}")
        
        for name, model in models_to_compare.items():
            print(f"\n[Model: {name}]")

            feature_selector = None
            if feature_selection:
                feature_selector = FeatureSelector(
                    fs_method=fs_method,
                    threshold=fs_threshold,
                    max_features_to_hold=max_features_to_hold,
                )

            full_pipeline = make_complete_pipeline(
                model,
                feature_selector=feature_selector,
                use_pca=use_pca,
                pca_n_components=pca_n_components
            )

            # Se il tuning è attivo, usiamo il FeatureSelectionTuner per ottimizzare congiuntamente
            if do_tuning and name in ["RandomForest", "XGBoost", "LightGBM", "StackingEnsemble"]:
                print(f"  > Tuning hyperparameters for {name} and Feature Selection...")
                param_grid_model = tuner.get_param_grid(name)
                best_pipeline, best_params, _ = fs_tuner.tune_pipeline(
                    full_pipeline, name, param_grid_model, X_train, y_train, n_iter=tuning_iter
                )
                full_pipeline = best_pipeline
                print(f"  > Best params: {best_params}")

            # Fit finale del modello (ottimizzato o meno)
            if use_sample_weight:
                weights_train = compute_sample_weight(class_weight='balanced', y=y_train)
                full_pipeline.fit(X_train, y_train, model__sample_weight=weights_train)
            else:
                full_pipeline.fit(X_train, y_train)

            y_pred = full_pipeline.predict(X_val)
            metrics = evaluate_predictions(y_val, y_pred, name)
            results.append(metrics)
            print(f"  > Micro-F1: {metrics['micro_f1']:.4f} | Macro-F1: {metrics['macro_f1']:.4f}")

    else:
        # --- CROSS-VALIDATION STRATEGY ---
        # Nota: Il tuning congiunto in ogni fold della CV è computazionalmente costoso.
        # In questa implementazione, la CV viene usata con i parametri di default o passati.
        splits = data_loader.split_dataset_by_strategy(split_strategy, X, y)

        for name, model in models_to_compare.items():
            print(f"\n[CV Model: {name}]")
            fold_results = []
            
            for fold, (train_idx, val_idx) in enumerate(splits):
                X_train_f, X_val_f = X.iloc[train_idx], X.iloc[val_idx]
                y_train_f, y_val_f = y.iloc[train_idx], y.iloc[val_idx]
                
                model_fold = clone(model)
                feature_selector = None
                if feature_selection:
                    feature_selector = FeatureSelector(
                        fs_method=fs_method,
                        threshold=fs_threshold,
                        max_features_to_hold=max_features_to_hold,
                    )

                full_pipeline = make_complete_pipeline(
                    model_fold,
                    feature_selector=feature_selector,
                    use_pca=use_pca,
                    pca_n_components=pca_n_components,
                )

                if use_sample_weight:
                    weights_fold = compute_sample_weight(class_weight="balanced", y=y_train_f)
                    full_pipeline.fit(X_train_f, y_train_f, model__sample_weight=weights_fold)
                else:
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
            print(f"Average Micro-F1: {avg_metrics['micro_f1']:.4f}")

    comparison_df = pd.DataFrame(results)
    print("\n--- 4. Results Comparison ---")
    print(comparison_df.to_string(index=False))
    
    return comparison_df

if __name__ == "__main__":
    # Esempio di esecuzione con tuning attivo e più iterazioni per qualità maggiore
    run_training_pipeline(do_tuning=False, tuning_iter=20)
