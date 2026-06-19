"""
Training finale e generazione della submission.

Questo modulo contiene il workflow conclusivo del progetto: tuning dei modelli
base, costruzione dello StackingEnsemble finale, applicazione della feature
selection, fit sul training set completo, salvataggio della pipeline finale e
generazione della submission. È separato dalla pipeline sperimentale per evitare
side effect durante confronti, ablation study e validazione interna.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from .config import (
    FINAL_FEATURE_SELECTION,
    FINAL_FS_METHOD,
    FINAL_FS_THRESHOLD,
    FINAL_MAX_FEATURES_TO_HOLD,
    FINAL_MODEL_CONFIG_FILE,
    FINAL_PIPELINE_FILE,
    FINAL_SUBMISSION_FILE,
    FINAL_TUNING_ITER,
    FINAL_TUNING_SAMPLE_SIZE,
    FINAL_USE_PCA,
    RANDOM_STATE,
    TARGET_COL,
)
from .data_loader import DataLoader
from .featureselector import FeatureSelector
from .hyperparameter_tuning import ModelTuner
from .hyperparameter_tuning_feature_selection import FeatureSelectionTuner
from .models import (
    get_lightgbm_model,
    get_random_forest_model,
    get_stacking_ensemble,
    get_xgboost_model,
)
from .preprocessing.pipeline import make_complete_pipeline


def _json_ready(value):
    """Converte valori numpy in oggetti Python serializzabili in JSON."""
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def train_final_model(
    model_output_path: str | Path = FINAL_PIPELINE_FILE,
    config_output_path: str | Path = FINAL_MODEL_CONFIG_FILE,
    feature_selection: bool = FINAL_FEATURE_SELECTION,
    fs_method: str = FINAL_FS_METHOD,
    fs_threshold: float = FINAL_FS_THRESHOLD,
    max_features_to_hold: int = FINAL_MAX_FEATURES_TO_HOLD,
    use_pca: bool = FINAL_USE_PCA,
    pca_n_components: int = 40,
    tuning_iter: int = FINAL_TUNING_ITER,
    tuning_sample_size: int = FINAL_TUNING_SAMPLE_SIZE,
):
    """
    Addestra e salva la pipeline finale basata su StackingEnsemble tunato.

    Il modello finale viene costruito ottimizzando prima RandomForest, XGBoost
    e LightGBM, per poi usare questi stimatori come base dello StackingEnsemble.
    Se la feature selection è attiva, vengono riutilizzati i parametri ottimizzati
    durante il tuning per mantenere coerenza con la configurazione sperimentale
    finale.
    """

    # Le label originali della competizione sono 1, 2 e 3. Alcuni modelli,
    # in particolare XGBoost, lavorano più stabilmente con classi indicizzate
    # da 0; per questo il target viene temporaneamente ricodificato.
    print("--- Training final tuned StackingEnsemble ---")

    data_loader = DataLoader()
    X, y = data_loader.load_train_test()
    y = y - 1

    tuner = ModelTuner(random_state=RANDOM_STATE)
    fs_tuner = FeatureSelectionTuner(random_state=RANDOM_STATE)

    # I tre modelli base vengono ottimizzati separatamente prima di essere
    # inseriti nello StackingEnsemble finale. In questo modo lo stacking non
    # combina stimatori default, ma versioni già adattate al problema.
    base_configs = {
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

    if tuning_sample_size >= len(X):
        X_tune = X
        y_tune = y
    else:
        X_tune, _, y_tune, _ = train_test_split(
            X,
            y,
            train_size=tuning_sample_size,
            stratify=y,
            random_state=RANDOM_STATE,
        )

    tuning_summary = {}

    for name, config in base_configs.items():
        print(f"\nTuning base model: {name}")

        feature_selector = (
            FeatureSelector(fs_method=fs_method)
            if feature_selection
            else None
        )

        pipeline = make_complete_pipeline(
            config["model"],
            feature_selector=feature_selector,
            use_pca=use_pca,
            pca_n_components=pca_n_components,
        )

        param_grid = tuner.get_param_grid(name)

        if feature_selection:
            best_pipeline, best_params, best_score = fs_tuner.tune_pipeline(
                pipeline,
                name,
                param_grid,
                X_tune,
                y_tune,
                n_iter=tuning_iter,
            )
        else:
            best_pipeline, best_params, best_score = tuner.tune_pipeline(
                pipeline,
                name,
                X_tune,
                y_tune,
                n_iter=tuning_iter,
            )

        base_configs[name]["model"] = best_pipeline.named_steps["model"]

        if feature_selection:
            base_configs[name]["fs_params"] = {
                "threshold": best_params.get(
                    "feature_selector__threshold",
                    fs_threshold,
                ),
                "max_features_to_hold": best_params.get(
                    "feature_selector__max_features_to_hold",
                    max_features_to_hold,
                ),
            }

        tuning_summary[name] = {
            "best_score": best_score,
            "best_params": best_params,
            "fs_params": base_configs[name]["fs_params"],
        }

        print(f"Best tuning micro-F1: {best_score:.6f}")
        if feature_selection:
            print(f"FS params: {base_configs[name]['fs_params']}")

    # I parametri di feature selection ricavati dal tuning vengono riutilizzati
    # per costruire una configurazione finale coerente e non scelta manualmente.
    ensemble_fs_params = (
        base_configs["LightGBM"]["fs_params"]
        if feature_selection
        else {
            "threshold": fs_threshold,
            "max_features_to_hold": max_features_to_hold,
        }
    )

    final_model = get_stacking_ensemble(
        base_configs["RandomForest"]["model"],
        base_configs["XGBoost"]["model"],
        base_configs["LightGBM"]["model"],
    )

    final_feature_selector = (
        FeatureSelector(
            fs_method=fs_method,
            threshold=ensemble_fs_params["threshold"],
            max_features_to_hold=ensemble_fs_params["max_features_to_hold"],
        )
        if feature_selection
        else None
    )

    # Dopo la selezione della configurazione finale, la pipeline viene fittata
    # su tutto il training set disponibile prima della submission.
    final_pipeline = make_complete_pipeline(
        final_model,
        feature_selector=final_feature_selector,
        use_pca=use_pca,
        pca_n_components=pca_n_components,
    )

    print("\nFitting final pipeline on full training data...")
    final_pipeline.fit(X, y)

    model_output_path = Path(model_output_path)
    config_output_path = Path(config_output_path)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    config_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Il salvataggio della pipeline completa permette di rigenerare la
    # submission senza ripetere tuning e training finale.
    joblib.dump(final_pipeline, model_output_path)

    metadata = {
        "final_model": "StackingEnsemble",
        "pipeline_file": str(model_output_path),
        "feature_selection": feature_selection,
        "fs_method": fs_method if feature_selection else None,
        "ensemble_fs_source": "LightGBM" if feature_selection else None,
        "ensemble_fs_params": ensemble_fs_params if feature_selection else None,
        "use_pca": use_pca,
        "tuning_iter": tuning_iter,
        "tuning_sample_size": tuning_sample_size,
        "random_state": RANDOM_STATE,
        "base_models": tuning_summary,
    }

    with config_output_path.open("w", encoding="utf-8") as file:
        json.dump(_json_ready(metadata), file, indent=4)

    print(f"Final pipeline saved to: {model_output_path}")
    print(f"Final model config saved to: {config_output_path}")

    return final_pipeline, metadata


def create_submission_from_pipeline(
    pipeline,
    output_path: str | Path = FINAL_SUBMISSION_FILE,
):
    """
    Genera il file di submission a partire da una pipeline già addestrata.

    La funzione applica al test set la stessa pipeline usata nel training finale
    e salva le predizioni nel formato richiesto dalla competizione.
    """

    # La submission usa la stessa pipeline fittata sul training set completo,
    # garantendo coerenza tra preprocessing, feature selection e modello.
    data_loader = DataLoader()
    data_loader.load_train_test()

    test_values = data_loader.test_values_df.copy()
    submission = data_loader.submission_format_df.copy()

    predictions = pipeline.predict(test_values)
    submission[TARGET_COL] = (predictions + 1).astype(int)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"Submission saved to: {output_path}")
    print(submission[TARGET_COL].value_counts().sort_index())

    return submission


def generate_submission_from_saved_model(
    model_path: str | Path = FINAL_PIPELINE_FILE,
    output_path: str | Path = FINAL_SUBMISSION_FILE,
):
    """
    Carica una pipeline finale salvata e rigenera la submission.

    Questa funzione permette di separare inferenza e training, evitando di
    rieseguire tuning e fit quando il modello finale è già disponibile su disco.
    """

    # Questa funzione separa inferenza e training: se il modello finale è già
    # salvato, la submission può essere rigenerata senza rifare il fit.
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved final pipeline not found: {model_path}. "
            "Run train_final_model() first."
        )

    print(f"Loading saved final pipeline from: {model_path}")
    pipeline = joblib.load(model_path)

    return create_submission_from_pipeline(
        pipeline=pipeline,
        output_path=output_path,
    )

if __name__ == "__main__":
    train_final_model()
