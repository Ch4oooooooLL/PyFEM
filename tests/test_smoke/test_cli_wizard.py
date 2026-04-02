from __future__ import annotations

from pathlib import Path

from tools.cli import build_command_wizard_namespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_train_wizard_accepts_numeric_choices_and_defaults() -> None:
    answers = iter(["2", "", "5", "", "", "3", "2"])
    prompts: list[str] = []

    namespace = build_command_wizard_namespace(
        "train",
        input_fn=lambda prompt: prompts.append(prompt) or next(answers),
        output_fn=lambda _: None,
    )

    assert namespace is not None
    assert namespace.command == "train"
    assert namespace.model == "pinn"
    assert namespace.config == PROJECT_ROOT / "dataset_config.yaml"
    assert namespace.epochs == 5
    assert namespace.batch_size is None
    assert namespace.lr is None
    assert namespace.device == "gpu"
    assert namespace.eval_only is False
    assert len(prompts) == 7


def test_dataset_wizard_supports_back_navigation() -> None:
    answers = iter(["custom.yaml", "back", "", "8", "2"])

    namespace = build_command_wizard_namespace(
        "dataset",
        input_fn=lambda _: next(answers),
        output_fn=lambda _: None,
    )

    assert namespace is not None
    assert namespace.command == "dataset"
    assert namespace.config == PROJECT_ROOT / "dataset_config.yaml"
    assert namespace.jobs == 8
    assert namespace.seq is False


def test_predict_wizard_can_be_cancelled() -> None:
    answers = iter(["cancel"])

    namespace = build_command_wizard_namespace(
        "predict",
        input_fn=lambda _: next(answers),
        output_fn=lambda _: None,
    )

    assert namespace is None
