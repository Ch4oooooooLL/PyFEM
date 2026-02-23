from __future__ import annotations

import argparse
import json
import os

from Condition_prediction.pipelines.condition_pipeline import run_condition_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run coupled FEM + Deep Learning condition prediction pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="condition_case.yaml",
        help="Path to condition config YAML in project root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional override for output root directory.",
    )
    args = parser.parse_args()

    summary = run_condition_pipeline(config_path=args.config, output_dir_override=args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nCompleted. Artifacts: {os.path.abspath(summary['run_dir'])}")


if __name__ == "__main__":
    main()
