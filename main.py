from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import (
    FINAL_MODEL_CONFIG_FILE,
    FINAL_PIPELINE_FILE,
    FINAL_SPLIT_STRATEGY,
    FINAL_SUBMISSION_FILE,
    PROJECT_ROOT,
    RESULTS_COMPARISON_FILE,
    SUBMISSION_FORMAT_FILE,
    TEST_VALUES_FILE,
    TRAIN_LABELS_FILE,
    TRAIN_VALUES_FILE,
    VALID_MODEL_NAMES,
)
from src.final_model import (
    create_submission_from_pipeline,
    generate_submission_from_saved_model,
    train_final_model,
)
from src.pipeline_training_model import run_training_pipeline

def _check_required_data_files() -> None:
    """Verify that the raw competition files are available before running."""
    required_files = [
        TRAIN_VALUES_FILE,
        TRAIN_LABELS_FILE,
        TEST_VALUES_FILE,
        SUBMISSION_FORMAT_FILE,
    ]

    missing_files = [Path(path) for path in required_files if not Path(path).exists()]

    if missing_files:
        missing = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            "Missing required raw data files:\n"
            f"{missing}\n\n"
            "Place the competition CSV files inside data/raw/ before running."
        )


def _parse_models(value: str | None) -> list[str] | None:
    """Parse a comma-separated model list."""
    if value is None:
        return None

    models = [item.strip() for item in value.split(",") if item.strip()]
    invalid_models = [model for model in models if model not in VALID_MODEL_NAMES]

    if invalid_models:
        allowed = ", ".join(VALID_MODEL_NAMES)
        invalid = ", ".join(invalid_models)
        raise argparse.ArgumentTypeError(
            f"Invalid model(s): {invalid}. Allowed values: {allowed}."
        )

    return models


def _format_path_for_help(path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _save_metrics(results: pd.DataFrame, output_path: Path | None) -> None:
    """Save metrics to CSV when an output path is provided."""
    if output_path is None:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nMetrics saved to: {output_path}")


def evaluate_final(args: argparse.Namespace) -> None:
    """Run a quick validation of the selected model without tuning."""
    _check_required_data_files()

    results = run_training_pipeline(
        feature_selection=False,
        split_strategy=args.split_strategy,
        use_sample_weight=False,
        use_pca=False,
        do_tuning=False,
        models_to_run=[args.model],
    )

    print("\nQuick validation results:")
    print(results.to_string(index=False))

    _save_metrics(results, args.output)


def compare_models(args: argparse.Namespace) -> None:
    """Compare selected models using the configurable validation pipeline."""
    _check_required_data_files()

    models_to_run = _parse_models(args.models)

    results = run_training_pipeline(
        feature_selection=False,
        split_strategy=args.split_strategy,
        use_sample_weight=False,
        use_pca=False,
        do_tuning=False,
        models_to_run=models_to_run,
    )

    print("\nModel comparison results:")
    print(results.to_string(index=False))

    _save_metrics(results, args.output)


def train_final(args: argparse.Namespace) -> None:
    """Train and serialize the final tuned StackingEnsemble pipeline."""
    _check_required_data_files()

    pipeline, metadata = train_final_model(
        model_output_path=args.model_output,
        config_output_path=args.config_output,
    )

    print("\nFinal model training completed.")
    print(f"Final model: {metadata['final_model']}")
    print(f"Pipeline saved to: {args.model_output}")
    print(f"Metadata saved to: {args.config_output}")


def make_submission(args: argparse.Namespace) -> None:
    """Generate a competition submission from a saved or newly trained final model."""
    _check_required_data_files()

    output_path = Path(args.output)

    if args.from_saved_model:
        submission = generate_submission_from_saved_model(
            model_path=args.model_path,
            output_path=output_path,
        )
    else:
        final_pipeline, _ = train_final_model(
            model_output_path=args.model_path,
        )
        submission = create_submission_from_pipeline(
            pipeline=final_pipeline,
            output_path=output_path,
        )

    print("\nSubmission preview:")
    print(submission.head().to_string(index=False))
    print(f"\nSubmission shape: {submission.shape}")
    print(f"Submission saved to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "FIA Earthquake Damage Predictor. "
            "Final configuration: tuned StackingEnsemble with feature selection. "
            "Use train-final to fit and save the final model, then make-submission "
            "--from-saved-model to regenerate the submission without retraining."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate-final",
        help="Run a quick validation of the selected model without tuning.",
    )
    evaluate_parser.add_argument(
        "--model",
        choices=VALID_MODEL_NAMES,
        default="XGBoost",
        help="Model to evaluate without tuning. Default: XGBoost.",
    )
    evaluate_parser.add_argument(
        "--split-strategy",
        type=int,
        default=FINAL_SPLIT_STRATEGY,
        help=f"Data split strategy. Default: {FINAL_SPLIT_STRATEGY}.",
    )
    evaluate_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path where validation metrics will be saved.",
    )
    evaluate_parser.set_defaults(func=evaluate_final)
    train_parser = subparsers.add_parser(
        "train-final",
        help="Train and save the final tuned StackingEnsemble pipeline.",
    )
    train_parser.add_argument(
        "--model-output",
        type=Path,
        default=FINAL_PIPELINE_FILE,
        help=f"Path where the fitted pipeline will be saved. Default: {_format_path_for_help(FINAL_PIPELINE_FILE)}.",
    )
    train_parser.add_argument(
        "--config-output",
        type=Path,
        default=FINAL_MODEL_CONFIG_FILE,
        help=f"Path where final model metadata will be saved. Default: {_format_path_for_help(FINAL_MODEL_CONFIG_FILE)}.",
    )
    train_parser.set_defaults(func=train_final)

    compare_parser = subparsers.add_parser(
        "compare-models",
        help="Compare selected models using the configurable validation pipeline.",
    )
    compare_parser.add_argument(
        "--models",
        default=None,
        help=(
            "Optional comma-separated model list. "
            "Example: XGBoost,LightGBM,StackingEnsemble. "
            "Default: all advanced models."
        ),
    )
    compare_parser.add_argument(
        "--split-strategy",
        type=int,
        default=FINAL_SPLIT_STRATEGY,
        help=f"Data split strategy. Default: {FINAL_SPLIT_STRATEGY}.",
    )
    compare_parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_COMPARISON_FILE,
        help=f"CSV path where comparison metrics will be saved. Default: {_format_path_for_help(RESULTS_COMPARISON_FILE)}.",
    )
    compare_parser.set_defaults(func=compare_models)

    submission_parser = subparsers.add_parser(
        "make-submission",
        help="Generate a competition submission from the final model.",
    )
    submission_parser.add_argument(
        "--from-saved-model",
        action="store_true",
        help="Load the saved final pipeline instead of retraining it.",
    )
    submission_parser.add_argument(
        "--model-path",
        type=Path,
        default=FINAL_PIPELINE_FILE,
        help=f"Saved final pipeline path. Default: {_format_path_for_help(FINAL_PIPELINE_FILE)}.",
    )
    submission_parser.add_argument(
        "--output",
        type=Path,
        default=FINAL_SUBMISSION_FILE,
        help=f"Submission CSV path. Default: {_format_path_for_help(FINAL_SUBMISSION_FILE)}.",
    )
    submission_parser.set_defaults(func=make_submission)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
