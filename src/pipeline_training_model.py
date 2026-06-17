from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight

from .data_loader import DataLoader
from .preprocessing.pipeline import make_complete_pipeline
from .evaluation import evaluate_predictions
from .models import (
    get_random_forest_model,
    get_xgboost_model,
    get_lightgbm_model,
    get_stacking_ensemble,
    get_voting_ensemble,
)
from .featureselector import FeatureSelector
from .config import RANDOM_STATE, TARGET_COL
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
    tuning_sample_size: int = 30000,
    models_to_run=None,
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

    valid_models = ["RandomForest", "XGBoost", "LightGBM", "VotingEnsemble", "StackingEnsemble"]

    if models_to_run is None:
        models_to_run = valid_models.copy()
    elif isinstance(models_to_run, str):
        models_to_run = [models_to_run]
    else:
        models_to_run = list(models_to_run)

    invalid_models = [model for model in models_to_run if model not in valid_models]
    if invalid_models:
        raise ValueError(
            f"Modelli non riconosciuti: {invalid_models}. "
            f"Valori ammessi: {valid_models}"
        )

    # --- 2. GLOBAL TUNING (se richiesto) ---
    if do_tuning:
        print(f"\n--- 2. Global Tuning attivo ({tuning_iter} iterazioni su {tuning_sample_size} campioni) ---")
        X_tune, _, y_tune, _ = train_test_split(
            X, y, train_size=min(tuning_sample_size, len(X)), 
            stratify=y, random_state=RANDOM_STATE
        )

        for name in [m for m in ["RandomForest", "XGBoost", "LightGBM"] if m in models_to_run]:
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

            if feature_selection:
                best_pipeline, best_params, best_score = fs_tuner.tune_pipeline(
                    temp_pipeline,
                    name,
                    param_grid,
                    X_tune,
                    y_tune,
                    n_iter=tuning_iter,
                )
            else:
                best_pipeline, best_params, best_score = tuner.tune_pipeline(
                    temp_pipeline,
                    name,
                    X_tune,
                    y_tune,
                    n_iter=tuning_iter,
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
    
    if "VotingEnsemble" in models_to_run:
        model_configs["VotingEnsemble"] = {
            "model": get_voting_ensemble(
                model_configs["RandomForest"]["model"],
                model_configs["XGBoost"]["model"],
                model_configs["LightGBM"]["model"]
            ),
            "fs_params": ensemble_fs_params
        }

    if "StackingEnsemble" in models_to_run:
        model_configs["StackingEnsemble"] = {
            "model": get_stacking_ensemble(
                model_configs["RandomForest"]["model"],
                model_configs["XGBoost"]["model"],
                model_configs["LightGBM"]["model"]
            ),
            "fs_params": ensemble_fs_params
        }

    # Tiene solo i modelli richiesti
    model_configs = {
        name: config
        for name, config in model_configs.items()
        if name in models_to_run
    }

    if not model_configs:
        raise ValueError("Nessun modello valido selezionato.")

    print(f"\n--- 3. Evaluation (Strategy {split_strategy}) ---")
    results = []

    if split_strategy in [1, 2]:
        # --- HOLD-OUT FLOW ---
        X_train, X_val, y_train, y_val = data_loader.split_dataset_by_strategy(split_strategy, X, y)
        print(f"Train size: {X_train.shape}, Validation size: {X_val.shape}")
        
        for name, config in model_configs.items():
            fs_label = (
                f"con FS dedicata ({fs_method}, max {config['fs_params']['max_features_to_hold']} feat)"
                if feature_selection
                else "senza feature selection"
            )
            pca_label = f" + PCA({pca_n_components})" if use_pca else ""
            print(f"  [Model: {name}] Training {fs_label}{pca_label}...")
            
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
            fs_result = (
                f" | FS: {fs_p['max_features_to_hold']} feat"
                if feature_selection
                else " | FS: non usata"
            )
            pca_result = f" | PCA: {pca_n_components} comp." if use_pca else " | PCA: non usata"
            print(f"    Micro-F1: {metrics['micro_f1']:.4f}{fs_result}{pca_result}")

    else:
        # --- CROSS-VALIDATION FLOW ---
        splits = data_loader.split_dataset_by_strategy(split_strategy, X, y)
        
        for name, config in model_configs.items():
            fs_label = (
                f"con FS dedicata ({fs_method}, max {config['fs_params']['max_features_to_hold']} feat)"
                if feature_selection
                else "senza feature selection"
            )
            pca_label = f" + PCA({pca_n_components})" if use_pca else ""
            print(f"  [Model: {name}] CV Evaluation {fs_label}{pca_label}...")
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
            fs_result = (
                f" | FS: {fs_p['max_features_to_hold']} feat"
                if feature_selection
                else " | FS: non usata"
            )
            pca_result = f" | PCA: {pca_n_components} comp." if use_pca else " | PCA: non usata"
            print(f"    Avg Micro-F1: {avg_metrics['micro_f1']:.4f}{fs_result}{pca_result}")

    comparison_df = pd.DataFrame(results)
    print("\n--- 4. Final Comparison ---")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def _get_model_by_name(model_name: str):
    """Restituisce il modello richiesto per la pipeline finale."""
    models = {
        "RandomForest": get_random_forest_model,
        "XGBoost": get_xgboost_model,
        "LightGBM": get_lightgbm_model,
        "VotingEnsemble": get_voting_ensemble,
        "StackingEnsemble": get_stacking_ensemble,
    }

    if model_name not in models:
        raise ValueError(
            f"Modello non riconosciuto: {model_name}. "
            f"Valori ammessi: {list(models)}"
        )

    return models[model_name]()


def generate_final_submission(
    model_name: str = "XGBoost",
    output_path: str | Path = "outputs/submissions/final_submission.csv",
    feature_selection: bool = False,
    fs_method: str = "rf",
    fs_threshold: float = 0.005,
    max_features_to_hold: int = 30,
    use_pca: bool = False,
    pca_n_components: int = 40,
):
    """
    Addestra il modello finale su tutto il training set e genera il file di submission.

    Configurazione finale consigliata:
    XGBoost senza feature selection, senza PCA e senza tuning.
    """
    print("--- Generazione submission finale ---")

    data_loader = DataLoader()
    X, y = data_loader.load_train_test()

    # Le classi originali sono 1, 2, 3. I modelli vengono addestrati su 0, 1, 2.
    y = y - 1

    model = _get_model_by_name(model_name)

    fs = (
        FeatureSelector(
            fs_method=fs_method,
            threshold=fs_threshold,
            max_features_to_hold=max_features_to_hold,
        )
        if feature_selection
        else None
    )

    pipeline = make_complete_pipeline(
        model,
        feature_selector=fs,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
    )

    print(f"Training finale modello: {model_name}")
    print(f"Feature selection: {'attiva' if feature_selection else 'non usata'}")
    print(f"PCA: {'attiva' if use_pca else 'non usata'}")

    pipeline.fit(X, y)

    test_values = data_loader.test_values_df.copy()
    submission = data_loader.submission_format_df.copy()

    predictions = pipeline.predict(test_values)

    # Rimappatura da 0, 1, 2 alle classi richieste dalla competizione: 1, 2, 3.
    submission[TARGET_COL] = predictions + 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"Submission salvata in: {output_path}")
    print(submission[TARGET_COL].value_counts().sort_index())

    return submission

if __name__ == "__main__":
    results = run_training_pipeline(
        feature_selection=False,
        split_strategy=2,
        use_sample_weight=False,
        use_pca=False,
        do_tuning=False,
        models_to_run=["XGBoost"],
    )

    print(results)
