"""
Pipeline sperimentale per validazione interna e confronto modelli.

Questo modulo gestisce esperimenti, ablation study, confronto tra modelli,
feature selection opzionale, PCA opzionale e tuning opzionale. Non genera la
submission finale e non salva il modello conclusivo: queste responsabilità sono
separate in final_model.py, così gli esperimenti restano isolati dal workflow
finale di training e inferenza.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
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
from .config import (
    FINAL_DO_TUNING,
    FINAL_FEATURE_SELECTION,
    FINAL_FS_METHOD,
    FINAL_FS_THRESHOLD,
    FINAL_MAX_FEATURES_TO_HOLD,
    FINAL_SPLIT_STRATEGY,
    FINAL_TUNING_ITER,
    FINAL_TUNING_SAMPLE_SIZE,
    FINAL_USE_PCA,
    FINAL_USE_SAMPLE_WEIGHT,
    RANDOM_STATE,
    VALID_MODEL_NAMES,
)
from .hyperparameter_tuning import ModelTuner
from .hyperparameter_tuning_feature_selection import FeatureSelectionTuner


def run_training_pipeline(
    feature_selection: bool = FINAL_FEATURE_SELECTION,
    split_strategy: int = FINAL_SPLIT_STRATEGY,
    use_sample_weight: bool = FINAL_USE_SAMPLE_WEIGHT,
    fs_method: str = FINAL_FS_METHOD,
    fs_threshold: float = FINAL_FS_THRESHOLD,
    max_features_to_hold: int = FINAL_MAX_FEATURES_TO_HOLD,
    use_pca: bool = FINAL_USE_PCA,
    pca_n_components: int = 40,
    do_tuning: bool = FINAL_DO_TUNING,
    tuning_iter: int = FINAL_TUNING_ITER,
    tuning_sample_size: int = FINAL_TUNING_SAMPLE_SIZE,
    models_to_run=None,
):
    """
    Esegue un workflow configurabile di training e validazione.

    La funzione viene usata per confrontare modelli, valutare configurazioni
    alternative e svolgere ablation study. Può riprodurre sia configurazioni
    semplici senza tuning e feature selection, sia configurazioni più vicine
    a quella finale con tuning, feature selection e cross-validation.
    """
    print("--- 1. Loading Data ---")
    data_loader = DataLoader()
    X, y = data_loader.load_train_test()

    # Le label originali 1, 2 e 3 vengono ricodificate in 0, 1 e 2 per
    # mantenere compatibilità con modelli come XGBoost.
    y = y - 1

    # Tuner opzionali usati solo quando do_tuning=True.
    tuner = ModelTuner(random_state=RANDOM_STATE)
    fs_tuner = FeatureSelectionTuner(random_state=RANDOM_STATE)

    # Configurazioni dei modelli base usate sia singolarmente sia negli ensemble.
    model_configs = {
        "RandomForest": {
            "model": get_random_forest_model(),
            "fs_params": {
                "threshold": fs_threshold,
                "max_features_to_hold": max_features_to_hold,
            },
        },
        "XGBoost": {
            "model": get_xgboost_model(),
            "fs_params": {
                "threshold": fs_threshold,
                "max_features_to_hold": max_features_to_hold,
            },
        },
        "LightGBM": {
            "model": get_lightgbm_model(),
            "fs_params": {
                "threshold": fs_threshold,
                "max_features_to_hold": max_features_to_hold,
            },
        },
    }

    valid_models = list(VALID_MODEL_NAMES)

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

    # Tuning globale opzionale su un sottoinsieme stratificato.
    if do_tuning:
        print(
            f"\n--- 2. Global Tuning attivo "
            f"({tuning_iter} iterazioni su {tuning_sample_size} campioni) ---"
        )
        X_tune, _, y_tune, _ = train_test_split(
            X,
            y,
            train_size=min(tuning_sample_size, len(X)),
            stratify=y,
            random_state=RANDOM_STATE,
        )

        tunable_models = [
            model_name
            for model_name in ["RandomForest", "XGBoost", "LightGBM"]
            if model_name in models_to_run
        ]

        for name in tunable_models:
            print(f"  > Tuning {name} (Modello + Feature Selection)...")

            # Pipeline temporanea usata durante il tuning.
            temp_fs = FeatureSelector(fs_method=fs_method) if feature_selection else None
            temp_pipeline = make_complete_pipeline(
                model_configs[name]["model"],
                feature_selector=temp_fs,
                use_pca=use_pca,
                pca_n_components=pca_n_components,
            )

            param_grid = tuner.get_param_grid(name)

            if feature_selection:
                best_pipeline, best_params, _ = fs_tuner.tune_pipeline(
                    temp_pipeline,
                    name,
                    param_grid,
                    X_tune,
                    y_tune,
                    n_iter=tuning_iter,
                )
            else:
                best_pipeline, best_params, _ = tuner.tune_pipeline(
                    temp_pipeline,
                    name,
                    X_tune,
                    y_tune,
                    n_iter=tuning_iter,
                )

            # Salva lo stimatore tunato e, se attiva, i parametri di feature selection.            model_configs[name]["model"] = best_pipeline.named_steps["model"]
            if feature_selection:
                model_configs[name]["fs_params"] = {
                    "threshold": best_params.get(
                        "feature_selector__threshold",
                        fs_threshold,
                    ),
                    "max_features_to_hold": best_params.get(
                        "feature_selector__max_features_to_hold",
                        max_features_to_hold,
                    ),
                }
            print(f"    Tuning completato per {name}.")

    # Gli ensemble vengono costruiti solo se richiesti esplicitamente.
        ensemble_fs_params = (
        model_configs["LightGBM"]["fs_params"]
        if do_tuning
        else {
            "threshold": fs_threshold,
            "max_features_to_hold": max_features_to_hold,
        }
    )

    if "VotingEnsemble" in models_to_run:
        model_configs["VotingEnsemble"] = {
            "model": get_voting_ensemble(
                model_configs["RandomForest"]["model"],
                model_configs["XGBoost"]["model"],
                model_configs["LightGBM"]["model"],
            ),
            "fs_params": ensemble_fs_params,
        }

    if "StackingEnsemble" in models_to_run:
        model_configs["StackingEnsemble"] = {
            "model": get_stacking_ensemble(
                model_configs["RandomForest"]["model"],
                model_configs["XGBoost"]["model"],
                model_configs["LightGBM"]["model"],
            ),
            "fs_params": ensemble_fs_params,
        }

    # Mantiene solo i modelli richiesti dall'utente.
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
        # Validazione Hold-out.
        X_train, X_val, y_train, y_val = data_loader.split_dataset_by_strategy(
            split_strategy,
            X,
            y,
        )
        print(f"Train size: {X_train.shape}, Validation size: {X_val.shape}")

        for name, config in model_configs.items():
            fs_label = (
                (
                    f"con FS dedicata "
                    f"({fs_method}, max {config['fs_params']['max_features_to_hold']} feat)"
                )
                if feature_selection
                else "senza feature selection"
            )
            pca_label = f" + PCA({pca_n_components})" if use_pca else ""
            print(f"  [Model: {name}] Training {fs_label}{pca_label}...")

            # Applica parametri di feature selection specifici per il modello.
            fs_p = config["fs_params"]
            fs = (
                FeatureSelector(
                    fs_method=fs_method,
                    threshold=fs_p["threshold"],
                    max_features_to_hold=fs_p["max_features_to_hold"],
                )
                if feature_selection
                else None
            )

            pipeline = make_complete_pipeline(
                config["model"],
                feature_selector=fs,
                use_pca=use_pca,
                pca_n_components=pca_n_components,
            )

            if use_sample_weight and name not in ["VotingEnsemble", "StackingEnsemble"]:
                weights = compute_sample_weight(class_weight="balanced", y=y_train)
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
            pca_result = (
                f" | PCA: {pca_n_components} comp."
                if use_pca
                else " | PCA: non usata"
            )
            print(f"    Micro-F1: {metrics['micro_f1']:.4f}{fs_result}{pca_result}")

    else:
        # Cross-validation.
        splits = data_loader.split_dataset_by_strategy(split_strategy, X, y)

        for name, config in model_configs.items():
            fs_label = (
                (
                    f"con FS dedicata "
                    f"({fs_method}, max {config['fs_params']['max_features_to_hold']} feat)"
                )
                if feature_selection
                else "senza feature selection"
            )
            pca_label = f" + PCA({pca_n_components})" if use_pca else ""
            print(f"  [Model: {name}] CV Evaluation {fs_label}{pca_label}...")

            fold_results = []
            fs_p = config["fs_params"]

            for fold, (train_idx, val_idx) in enumerate(splits):
                X_train_f = X.iloc[train_idx]
                X_val_f = X.iloc[val_idx]
                y_train_f = y.iloc[train_idx]
                y_val_f = y.iloc[val_idx]

                # Clona lo stimatore per il fold corrente.
                model_fold = clone(config["model"])

                # Ricrea la feature selection dentro ogni fold per evitare leakage.
                fs = (
                    FeatureSelector(
                        fs_method=fs_method,
                        threshold=fs_p["threshold"],
                        max_features_to_hold=fs_p["max_features_to_hold"],
                    )
                    if feature_selection
                    else None
                )

                pipeline = make_complete_pipeline(
                    model_fold,
                    feature_selector=fs,
                    use_pca=use_pca,
                    pca_n_components=pca_n_components,
                )

                if use_sample_weight and name not in ["VotingEnsemble", "StackingEnsemble"]:
                    weights = compute_sample_weight(
                        class_weight="balanced",
                        y=y_train_f,
                    )
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
            pca_result = (
                f" | PCA: {pca_n_components} comp."
                if use_pca
                else " | PCA: non usata"
            )
            print(f"    Avg Micro-F1: {avg_metrics['micro_f1']:.4f}{fs_result}{pca_result}")

    comparison_df = pd.DataFrame(results)
    print("\n--- 4. Final Comparison ---")
    print(comparison_df.to_string(index=False))

    return comparison_df
