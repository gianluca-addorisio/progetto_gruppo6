from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import (
    FINAL_MODEL_NAME,
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
from src.pipeline_training_model import (
    generate_final_submission,
    run_training_pipeline,
)




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
    """Evaluate the final candidate model on the validation split."""
    _check_required_data_files()

    results = run_training_pipeline(
        feature_selection=False,
        split_strategy=args.split_strategy,
        use_sample_weight=False,
        use_pca=False,
        do_tuning=False,
        models_to_run=[args.model],
    )

    print("\nFinal-model validation results:")
    print(results.to_string(index=False))

    _save_metrics(results, args.output)


def compare_models(args: argparse.Namespace) -> None:
    """Compare the selected advanced models with the final preprocessing setup."""
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


def make_submission(args: argparse.Namespace) -> None:
    """Train the selected model on the full training set and generate a submission."""
    _check_required_data_files()

    output_path = Path(args.output)

    submission = generate_final_submission(
        model_name=args.model,
        output_path=output_path,
        feature_selection=False,
        use_pca=False,
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
            "Default configuration: XGBoost, no feature selection, no PCA, no tuning."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate-final",
        help="Evaluate the final candidate model on the validation split.",
    )
    evaluate_parser.add_argument(
        "--model",
        choices=VALID_MODEL_NAMES,
        default=FINAL_MODEL_NAME,
        help=f"Model to evaluate. Default: {FINAL_MODEL_NAME}.",
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

    compare_parser = subparsers.add_parser(
        "compare-models",
        help="Compare advanced models using the final preprocessing setup.",
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
        help="Train on the full training set and generate a competition submission.",
    )
    submission_parser.add_argument(
        "--model",
        choices=VALID_MODEL_NAMES,
        default=FINAL_MODEL_NAME,
        help=f"Model used for final training. Default: {FINAL_MODEL_NAME}.",
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
