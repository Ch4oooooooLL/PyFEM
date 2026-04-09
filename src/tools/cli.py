from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
except ImportError:
    Console = None
    Panel = None
    Table = None
    box = None

if sys.platform == "win32":
    Console = None
    Panel = None
    Table = None
    box = None


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
ROOT_DIR = SRC_DIR.parent
CONFIGS_DIR = ROOT_DIR / "configs"
OUTPUTS_DIR = ROOT_DIR / "outputs"
PYFEM_DIR = SRC_DIR / "PyFEM_Dynamics"
DEEP_LEARNING_DIR = SRC_DIR / "Deep_learning"
CONDITION_PREDICTION_DIR = SRC_DIR / "Condition_prediction"

DEFAULT_CONDA_ENV_NAME = os.environ.get("FEM_DEFAULT_CONDA_ENV", "fem")
DEFAULT_CONDA_PYTHON_VERSION = os.environ.get("FEM_DEFAULT_PYTHON", "3.11")
REQUIREMENT_FILES = (
    ROOT_DIR / "requirements.txt",
    DEEP_LEARNING_DIR / "requirements.txt",
)
BOOTSTRAP_IMPORTS = (
    "numpy",
    "scipy",
    "yaml",
    "torch",
    "tqdm",
    "rich",
    "matplotlib",
)

console = Console() if Console is not None else None


def _print(message: str) -> None:
    if console is not None:
        console.print(message)
        return
    print(message)


def _print_panel(title: str, message: str) -> None:
    if console is not None and Panel is not None:
        console.print(Panel(message, title=title, border_style="cyan"))
        return
    print(f"[{title}] {message}")


def _print_status_rows(rows: list[tuple[str, str, str]]) -> None:
    if console is not None and Table is not None and box is not None:
        table = Table(box=box.ROUNDED)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")
        for component, status, details in rows:
            table.add_row(component, status, details)
        console.print(table)
        return

    width = max(len(component) for component, _, _ in rows)
    for component, status, details in rows:
        print(f"{component.ljust(width)} | {status} | {details}")


def _build_project_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(SRC_DIR)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_streaming_command(
    cmd: Sequence[str],
    description: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    _print(f"{description}...")
    process = subprocess.Popen(
        list(cmd),
        cwd=str(cwd or ROOT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    output_lines: list[str] = []
    for line in process.stdout:
        clean_line = line.rstrip()
        output_lines.append(clean_line)
        print(clean_line)
    process.wait()
    if process.returncode == 0:
        _print(f"{description} completed.")
    else:
        joined_output = "\n".join(output_lines)
        if "CondaToSNonInteractiveError" in joined_output:
            _print("Conda channel terms must be accepted before bootstrap can create environments.")
            _print("Run:")
            _print("  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main")
            _print("  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r")
            _print("  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2")
        _print(f"{description} failed with code {process.returncode}.")
    return process.returncode


def run_capture_command(
    cmd: Sequence[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd or ROOT_DIR),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def get_conda_executable() -> str:
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        return conda_exe
    conda_path = shutil.which("conda")
    if conda_path:
        return conda_path
    raise FileNotFoundError("Conda executable not found. Install Conda and ensure `conda` is on PATH.")


def get_active_conda_env_name() -> str | None:
    active_env = os.environ.get("CONDA_DEFAULT_ENV")
    if active_env:
        return active_env.strip() or None
    return None


def resolve_conda_env_name(explicit_env_name: str | None, active_env_name: str | None) -> str:
    if explicit_env_name:
        return explicit_env_name
    if active_env_name and active_env_name.lower() != "base":
        return active_env_name
    return DEFAULT_CONDA_ENV_NAME


def conda_env_exists(env_name: str) -> bool:
    command = [get_conda_executable(), "env", "list", "--json"]
    completed = run_capture_command(command)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "Unable to inspect conda environments.")
    payload = json.loads(completed.stdout or "{}")
    for env_path in payload.get("envs", []):
        if Path(env_path).name.lower() == env_name.lower():
            return True
    return False


def build_conda_run_prefix(env_name: str, live_output: bool = False) -> list[str]:
    prefix = [get_conda_executable(), "run"]
    if live_output:
        prefix.append("--no-capture-output")
    prefix.extend(["-n", env_name])
    return prefix


def ensure_conda_env(env_name: str, python_version: str) -> int:
    if conda_env_exists(env_name):
        _print(f"Conda environment `{env_name}` already exists.")
        return 0
    return run_streaming_command(
        [
            get_conda_executable(),
            "create",
            "-y",
            "-n",
            env_name,
            f"python={python_version}",
            "pip",
        ],
        f"Creating conda environment `{env_name}`",
    )


def _imports_available_in_target_env(env_name: str) -> bool:
    import_checks = [
        f"importlib.util.find_spec('{module}') is not None"
        for module in BOOTSTRAP_IMPORTS
    ]
    snippet = (
        "import importlib.util, sys; "
        f"checks=[{', '.join(import_checks)}]; "
        "sys.exit(0 if all(checks) else 1)"
    )

    active_env = get_active_conda_env_name()
    if active_env == env_name:
        completed = run_capture_command([sys.executable, "-c", snippet], env=_build_project_env())
    else:
        completed = run_capture_command(build_conda_run_prefix(env_name) + ["python", "-c", snippet])
    return completed.returncode == 0


def install_dependencies(env_name: str, force_install: bool = False) -> int:
    if not force_install and _imports_available_in_target_env(env_name):
        _print(f"Dependencies already available in `{env_name}`.")
        return 0

    for requirement_file in REQUIREMENT_FILES:
        if not requirement_file.exists():
            raise FileNotFoundError(f"Requirements file not found: {requirement_file}")

    active_env = get_active_conda_env_name()
    for requirement_file in REQUIREMENT_FILES:
        relative_path = requirement_file.relative_to(ROOT_DIR)
        if active_env == env_name:
            command = [sys.executable, "-m", "pip", "install", "-r", str(requirement_file)]
            env = _build_project_env()
        else:
            command = build_conda_run_prefix(env_name, live_output=True) + [
                "python",
                "-m",
                "pip",
                "install",
                "-r",
                str(requirement_file),
            ]
            env = None
        exit_code = run_streaming_command(
            command,
            f"Installing dependencies from `{relative_path.as_posix()}`",
            env=env,
        )
        if exit_code != 0:
            return exit_code
    return 0


def bootstrap_conda_env(env_name: str, python_version: str, force_install: bool = False) -> int:
    exit_code = ensure_conda_env(env_name, python_version)
    if exit_code != 0:
        return exit_code
    return install_dependencies(env_name, force_install=force_install)


def print_banner() -> None:
    _print_panel(
        "FEM CLI",
        "FEM Solver + Deep Learning + Condition Prediction",
    )


def cmd_env(args: argparse.Namespace) -> int:
    target_env = resolve_conda_env_name(args.env_name, get_active_conda_env_name())
    rows = [
        ("Conda Executable", "OK", get_conda_executable()),
        ("Active Env", "INFO", get_active_conda_env_name() or "None"),
        ("Target Env", "INFO", target_env),
        ("Target Exists", "YES" if conda_env_exists(target_env) else "NO", target_env),
    ]
    _print_status_rows(rows)
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    target_env = resolve_conda_env_name(args.env_name, get_active_conda_env_name())
    return bootstrap_conda_env(
        env_name=target_env,
        python_version=args.python_version,
        force_install=args.force_install,
    )


def cmd_install(args: argparse.Namespace) -> int:
    target_env = resolve_conda_env_name(args.env_name, get_active_conda_env_name())
    exit_code = ensure_conda_env(target_env, args.python_version)
    if exit_code != 0:
        return exit_code
    return install_dependencies(target_env, force_install=args.force_install)


def cmd_static(args: argparse.Namespace) -> int:
    _print_panel("Static", "Running FEM static analysis")
    structure_file = args.structure or CONFIGS_DIR / "structure.yaml"

    from PyFEM_Dynamics.main import run_single_static_test

    try:
        run_single_static_test(structure_yaml=str(structure_file))
        _print("Static analysis completed.")
        return 0
    except Exception as exc:
        _print(f"Static analysis failed: {exc}")
        return 1


def cmd_dataset(args: argparse.Namespace) -> int:
    _print_panel("Dataset", "Generating training dataset")
    config_file = args.config or CONFIGS_DIR / "dataset_config.yaml"

    from PyFEM_Dynamics.pipeline.data_gen import generate_dataset

    try:
        generate_dataset(
            config_path=str(config_file),
            n_jobs=args.jobs,
            sequential=args.seq,
        )
        _print("Dataset generation completed.")
        return 0
    except Exception as exc:
        _print(f"Dataset generation failed: {exc}")
        return 1


def normalize_device_value(device: str | None) -> str:
    if device is None:
        return "auto"
    normalized = str(device).strip().lower()
    if normalized == "gpu":
        return "cuda"
    return normalized or "auto"


def build_device_validation_error(
    device: str,
    *,
    torch_available: bool,
    cuda_available: bool,
) -> str | None:
    if not str(device).startswith("cuda"):
        return None
    if not torch_available:
        return "PyTorch is not installed in the current environment. Run `bootstrap` or `install` first."
    if cuda_available:
        return None
    return (
        "CUDA/GPU training was requested, but CUDA is not available in the current environment. "
        "Use `--device cpu` or install a CUDA-enabled PyTorch build in this conda environment."
    )


def validate_requested_device(device: str) -> str | None:
    normalized = normalize_device_value(device)
    if not normalized.startswith("cuda"):
        return None
    try:
        import torch
    except ImportError:
        return build_device_validation_error(
            normalized,
            torch_available=False,
            cuda_available=False,
        )
    return build_device_validation_error(
        normalized,
        torch_available=True,
        cuda_available=bool(torch.cuda.is_available()),
    )


def cmd_train(args: argparse.Namespace) -> int:
    _print_panel("Train", "Training deep learning models")
    model = args.model or "gt"
    config = args.config or CONFIGS_DIR / "dataset_config.yaml"
    device = normalize_device_value(args.device)
    device_error = validate_requested_device(device)
    if device_error is not None:
        _print(device_error)
        return 1

    command = [
        sys.executable,
        "-u",
        "-m",
        "Deep_learning.train",
        "--config",
        str(config),
        "--model",
        model,
        "--device",
        device,
    ]

    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        command.extend(["--batch_size", str(args.batch_size)])
    if args.lr is not None:
        command.extend(["--lr", str(args.lr)])
    if args.eval_only:
        command.append("--eval_only")

    return run_streaming_command(
        command,
        f"Training `{model}`",
        env=_build_project_env(),
    )


def cmd_predict(args: argparse.Namespace) -> int:
    _print_panel("Predict", "Running condition prediction")
    config = args.config or CONFIGS_DIR / "condition_case.yaml"

    from Condition_prediction.pipelines.condition_pipeline import run_condition_pipeline

    try:
        summary = run_condition_pipeline(
            config_path=str(config),
            output_dir_override=args.output_dir,
        )
    except Exception as exc:
        _print(f"Condition prediction failed: {exc}")
        return 1

    rows: list[tuple[str, str, str]] = [("Artifacts", "OK", summary.get("run_dir", "N/A"))]
    for key, value in summary.get("metrics", {}).items():
        rows.append((key, "OK", f"{value:.4f}" if isinstance(value, float) else str(value)))
    _print_status_rows(rows)
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    _print_panel("Pipeline", "Running the full FEM to DL pipeline")
    steps: list[tuple[str, callable]] = []

    if not args.skip_static:
        steps.append(
            (
                "Static Analysis",
                lambda: cmd_static(SimpleNamespace(structure=args.structure)),
            )
        )
    if not args.skip_dataset:
        steps.append(
            (
                "Dataset Generation",
                lambda: cmd_dataset(
                    SimpleNamespace(
                        config=args.dataset_config,
                        jobs=args.jobs,
                        seq=args.seq,
                    )
                ),
            )
        )
    if not args.skip_train:
        steps.append(
            (
                "Model Training",
                lambda: cmd_train(
                    SimpleNamespace(
                        config=args.dataset_config,
                        model=args.model,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        device=args.device,
                        eval_only=args.eval_only,
                    )
                ),
            )
        )
    if not args.skip_predict:
        steps.append(
            (
                "Condition Prediction",
                lambda: cmd_predict(
                    SimpleNamespace(
                        config=args.condition_config,
                        output_dir=args.output_dir,
                    )
                ),
            )
        )

    for index, (name, step) in enumerate(steps, start=1):
        _print(f"[{index}/{len(steps)}] {name}")
        exit_code = step()
        if exit_code != 0:
            _print(f"Pipeline stopped at `{name}`.")
            return exit_code

    _print("Pipeline completed.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    target_env = resolve_conda_env_name(args.env_name, get_active_conda_env_name())
    rows = [
        ("Conda Env", "READY" if conda_env_exists(target_env) else "MISSING", target_env),
        ("Active Env", "INFO", get_active_conda_env_name() or "None"),
    ]

    structure_file = CONFIGS_DIR / "structure.yaml"
    rows.append(("Structure Config", "READY" if structure_file.exists() else "MISSING", str(structure_file)))

    dataset_config = CONFIGS_DIR / "dataset_config.yaml"
    rows.append(("Dataset Config", "READY" if dataset_config.exists() else "MISSING", str(dataset_config)))

    dataset_file = ROOT_DIR / "dataset" / "train.npz"
    dataset_details = str(dataset_file)
    if dataset_file.exists():
        try:
            import numpy as np

            with np.load(dataset_file) as dataset:
                dataset_details = f"{dataset['load'].shape[0]} samples"
        except Exception:
            dataset_details = str(dataset_file)
    rows.append(("Training Dataset", "READY" if dataset_file.exists() else "MISSING", dataset_details))

    gt_ckpt = OUTPUTS_DIR / "checkpoints" / "gt_best.pth"
    pinn_v2_ckpt = OUTPUTS_DIR / "checkpoints" / "pinn_v2_best.pth"
    legacy_pinn_ckpt = OUTPUTS_DIR / "checkpoints" / "pinn_best.pth"
    rows.append(("GT Checkpoint", "READY" if gt_ckpt.exists() else "MISSING", str(gt_ckpt)))
    rows.append(("PINN V2 Checkpoint", "READY" if pinn_v2_ckpt.exists() else "MISSING", str(pinn_v2_ckpt)))
    rows.append(("Legacy PINN Checkpoint", "READY" if legacy_pinn_ckpt.exists() else "OPTIONAL", str(legacy_pinn_ckpt)))

    condition_config = CONFIGS_DIR / "condition_case.yaml"
    rows.append(("Condition Config", "READY" if condition_config.exists() else "MISSING", str(condition_config)))

    _print_status_rows(rows)
    return 0


@dataclass
class WizardField:
    name: str
    label: str
    kind: str
    default: Any = None
    choices: list[tuple[str, Any]] | None = None
    optional: bool = False
    help_text: str | None = None


WIZARD_CANCEL = object()
WIZARD_EXIT = object()
WIZARD_BACK = object()


def detect_default_device_choice() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "gpu" if torch.cuda.is_available() else "cpu"


def resolve_wizard_default(default: Any) -> Any:
    return default() if callable(default) else default


def format_wizard_default(value: Any) -> str:
    if value is None:
        return "skip"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def format_wizard_value(field: WizardField, value: Any) -> str:
    if field.kind in {"choice", "bool"} and field.choices is not None:
        for label, choice_value in field.choices:
            if choice_value == value:
                return label
    return format_wizard_default(value)


def build_wizard_fields(command_name: str) -> list[WizardField]:
    choice_yes_no = [("yes", True), ("no", False)]
    command_fields: dict[str, list[WizardField]] = {
        "env": [
            WizardField(
                name="env_name",
                label="Target conda environment",
                kind="text",
                default=lambda: resolve_conda_env_name(None, get_active_conda_env_name()),
                help_text="Press Enter to inspect the default target environment.",
            ),
        ],
        "status": [
            WizardField(
                name="env_name",
                label="Target conda environment",
                kind="text",
                default=lambda: resolve_conda_env_name(None, get_active_conda_env_name()),
                help_text="Press Enter to inspect the default target environment.",
            ),
        ],
        "bootstrap": [
            WizardField(
                name="env_name",
                label="Target conda environment",
                kind="text",
                default=lambda: resolve_conda_env_name(None, get_active_conda_env_name()),
            ),
            WizardField(
                name="python_version",
                label="Python version",
                kind="text",
                default=DEFAULT_CONDA_PYTHON_VERSION,
            ),
            WizardField(
                name="force_install",
                label="Reinstall dependencies even if already present",
                kind="bool",
                default=False,
                choices=choice_yes_no,
            ),
        ],
        "install": [
            WizardField(
                name="env_name",
                label="Target conda environment",
                kind="text",
                default=lambda: resolve_conda_env_name(None, get_active_conda_env_name()),
            ),
            WizardField(
                name="python_version",
                label="Python version used if the environment must be created",
                kind="text",
                default=DEFAULT_CONDA_PYTHON_VERSION,
            ),
            WizardField(
                name="force_install",
                label="Reinstall dependencies even if already present",
                kind="bool",
                default=False,
                choices=choice_yes_no,
            ),
        ],
        "static": [
            WizardField(
                name="structure",
                label="Structure YAML path",
                kind="path",
                default=CONFIGS_DIR / "structure.yaml",
            ),
        ],
        "dataset": [
            WizardField(
                name="config",
                label="Dataset config path",
                kind="path",
                default=CONFIGS_DIR / "dataset_config.yaml",
            ),
            WizardField(
                name="jobs",
                label="Parallel worker count",
                kind="int",
                default=-1,
                help_text="-1 means auto.",
            ),
            WizardField(
                name="seq",
                label="Run sequentially",
                kind="bool",
                default=False,
                choices=choice_yes_no,
            ),
        ],
        "train": [
            WizardField(
                name="model",
                label="Model type",
                kind="choice",
                default="gt",
                choices=[
                    ("gt", "gt"),
                    ("pinn", "pinn"),
                    ("pinn_v2", "pinn_v2"),
                    ("legacy_pinn", "legacy_pinn"),
                    ("both", "both"),
                ],
            ),
            WizardField(
                name="config",
                label="Training config path",
                kind="path",
                default=CONFIGS_DIR / "dataset_config.yaml",
            ),
            WizardField(
                name="epochs",
                label="Epoch count",
                kind="int",
                default=100,
            ),
            WizardField(
                name="batch_size",
                label="Batch size",
                kind="int",
                default=None,
                optional=True,
                help_text="Press Enter to keep the config default.",
            ),
            WizardField(
                name="lr",
                label="Learning rate",
                kind="float",
                default=None,
                optional=True,
                help_text="Press Enter to keep the config default.",
            ),
            WizardField(
                name="device",
                label="Training device",
                kind="choice",
                default=detect_default_device_choice,
                choices=[
                    ("auto", "auto"),
                    ("cpu", "cpu"),
                    ("gpu", "gpu"),
                ],
            ),
            WizardField(
                name="eval_only",
                label="Evaluation only",
                kind="bool",
                default=False,
                choices=choice_yes_no,
            ),
        ],
        "predict": [
            WizardField(
                name="config",
                label="Condition config path",
                kind="path",
                default=CONFIGS_DIR / "condition_case.yaml",
            ),
            WizardField(
                name="output_dir",
                label="Output directory override",
                kind="text",
                default=None,
                optional=True,
                help_text="Press Enter to use the config default output directory.",
            ),
        ],
        "pipeline": [
            WizardField(
                name="structure",
                label="Structure YAML path",
                kind="path",
                default=CONFIGS_DIR / "structure.yaml",
            ),
            WizardField(
                name="dataset_config",
                label="Dataset config path",
                kind="path",
                default=CONFIGS_DIR / "dataset_config.yaml",
            ),
            WizardField(
                name="condition_config",
                label="Condition config path",
                kind="path",
                default=CONFIGS_DIR / "condition_case.yaml",
            ),
            WizardField(
                name="jobs",
                label="Parallel worker count",
                kind="int",
                default=-1,
                help_text="-1 means auto.",
            ),
            WizardField(
                name="seq",
                label="Generate dataset sequentially",
                kind="bool",
                default=False,
                choices=choice_yes_no,
            ),
            WizardField(
                name="model",
                label="Training model",
                kind="choice",
                default="gt",
                choices=[
                    ("gt", "gt"),
                    ("pinn", "pinn"),
                    ("pinn_v2", "pinn_v2"),
                    ("legacy_pinn", "legacy_pinn"),
                    ("both", "both"),
                ],
            ),
            WizardField(
                name="epochs",
                label="Epoch count",
                kind="int",
                default=None,
                optional=True,
                help_text="Press Enter to keep the command default.",
            ),
            WizardField(
                name="batch_size",
                label="Batch size",
                kind="int",
                default=None,
                optional=True,
                help_text="Press Enter to keep the config default.",
            ),
            WizardField(
                name="lr",
                label="Learning rate",
                kind="float",
                default=None,
                optional=True,
                help_text="Press Enter to keep the config default.",
            ),
            WizardField(
                name="device",
                label="Training device",
                kind="choice",
                default=detect_default_device_choice,
                choices=[
                    ("auto", "auto"),
                    ("cpu", "cpu"),
                    ("gpu", "gpu"),
                ],
            ),
            WizardField(
                name="eval_only",
                label="Evaluation only",
                kind="bool",
                default=False,
                choices=choice_yes_no,
            ),
            WizardField(
                name="output_dir",
                label="Prediction output directory override",
                kind="text",
                default=None,
                optional=True,
                help_text="Press Enter to use the config default output directory.",
            ),
            WizardField(
                name="skip_static",
                label="Skip static analysis",
                kind="bool",
                default=False,
                choices=choice_yes_no,
            ),
            WizardField(
                name="skip_dataset",
                label="Skip dataset generation",
                kind="bool",
                default=False,
                choices=choice_yes_no,
            ),
            WizardField(
                name="skip_train",
                label="Skip model training",
                kind="bool",
                default=False,
                choices=choice_yes_no,
            ),
            WizardField(
                name="skip_predict",
                label="Skip condition prediction",
                kind="bool",
                default=False,
                choices=choice_yes_no,
            ),
        ],
    }
    return command_fields.get(command_name, [])


def prompt_wizard_field(
    field: WizardField,
    *,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> Any:
    default_value = resolve_wizard_default(field.default)
    if field.help_text:
        output_fn(field.help_text)

    while True:
        if field.kind in {"choice", "bool"} and field.choices is not None:
            output_fn(field.label)
            default_index = 1
            for index, (label, value) in enumerate(field.choices, start=1):
                marker = " (default)" if value == default_value else ""
                output_fn(f"  {index}. {label}{marker}")
                if value == default_value:
                    default_index = index
            raw_value = input_fn(f"Select an option [{default_index}]: ").strip()
        else:
            default_label = format_wizard_default(default_value)
            raw_value = input_fn(f"{field.label} [{default_label}]: ").strip()

        lowered = raw_value.lower()
        if lowered in {"cancel"}:
            return WIZARD_CANCEL
        if lowered in {"exit", "quit"}:
            return WIZARD_EXIT
        if lowered == "back":
            return WIZARD_BACK

        if raw_value == "":
            return default_value

        try:
            if field.kind in {"choice", "bool"} and field.choices is not None:
                if raw_value.isdigit():
                    index = int(raw_value) - 1
                    if 0 <= index < len(field.choices):
                        return field.choices[index][1]
                for label, value in field.choices:
                    if raw_value.lower() == str(label).lower():
                        return value
                raise ValueError("Enter one of the listed option numbers.")
            if field.kind == "int":
                return int(raw_value)
            if field.kind == "float":
                return float(raw_value)
            if field.kind == "path":
                return Path(raw_value)
            return raw_value
        except ValueError as exc:
            output_fn(f"Invalid input: {exc}")


def print_wizard_summary(command_name: str, fields: list[WizardField], namespace: argparse.Namespace) -> None:
    _print(f"Command summary for `{command_name}`:")
    for field in fields:
        _print(f"  {field.label}: {format_wizard_value(field, getattr(namespace, field.name))}")


def build_command_wizard_namespace(
    command_name: str,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = _print,
) -> argparse.Namespace | None:
    parser = build_parser()
    namespace = parser.parse_args([command_name])
    fields = build_wizard_fields(command_name)
    if not fields:
        return namespace

    output_fn(f"Starting `{command_name}` wizard.")
    output_fn("Press Enter to accept the default value. Use `back`, `cancel`, or `exit` when needed.")

    index = 0
    while index < len(fields):
        field = fields[index]
        value = prompt_wizard_field(field, input_fn=input_fn, output_fn=output_fn)
        if value is WIZARD_BACK:
            if index == 0:
                output_fn("Already at the first question.")
                continue
            index -= 1
            continue
        if value is WIZARD_CANCEL:
            output_fn(f"`{command_name}` cancelled.")
            return None
        if value is WIZARD_EXIT:
            raise SystemExit(0)
        setattr(namespace, field.name, value)
        index += 1

    print_wizard_summary(command_name, fields, namespace)
    return namespace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.bat / ./run.sh",
        description="Unified FEM project CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    run.bat bootstrap
    run.bat pipeline --jobs 4
    run.bat train --model gt --epochs 100
    ./run.sh predict --config configs/condition_case.yaml
        """.strip(),
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    env_parser = subparsers.add_parser("env", help="Show conda environment information")
    env_parser.add_argument("--env-name", type=str, help="Override target conda environment name")

    bootstrap_parser = subparsers.add_parser("bootstrap", help="Create or reuse the conda environment and install dependencies")
    bootstrap_parser.add_argument("--env-name", type=str, help="Override target conda environment name")
    bootstrap_parser.add_argument(
        "--python-version",
        type=str,
        default=DEFAULT_CONDA_PYTHON_VERSION,
        help="Python version to use when creating the conda environment",
    )
    bootstrap_parser.add_argument("--force-install", action="store_true", help="Reinstall dependencies even if imports already succeed")

    install_parser = subparsers.add_parser("install", help="Install project dependencies into the target conda environment")
    install_parser.add_argument("--env-name", type=str, help="Override target conda environment name")
    install_parser.add_argument(
        "--python-version",
        type=str,
        default=DEFAULT_CONDA_PYTHON_VERSION,
        help="Python version to use if the conda environment must be created first",
    )
    install_parser.add_argument("--force-install", action="store_true", help="Reinstall dependencies even if imports already succeed")

    static_parser = subparsers.add_parser("static", help="Run FEM static analysis")
    static_parser.add_argument("--structure", type=Path, help="Path to structure YAML file")

    dataset_parser = subparsers.add_parser("dataset", help="Generate training dataset")
    dataset_parser.add_argument("--config", type=Path, help="Path to dataset config YAML")
    dataset_parser.add_argument("-j", "--jobs", type=int, default=-1, help="Number of parallel workers")
    dataset_parser.add_argument("--seq", action="store_true", help="Run sequentially")

    train_parser = subparsers.add_parser("train", help="Train deep learning models")
    train_parser.add_argument("--model", choices=["gt", "pinn", "legacy_pinn", "pinn_v2", "both"], help="Model type")
    train_parser.add_argument("--config", type=Path, help="Path to training config YAML")
    train_parser.add_argument("--epochs", type=int, help="Number of epochs")
    train_parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, help="Batch size")
    train_parser.add_argument("--lr", type=float, help="Learning rate")
    train_parser.add_argument("--device", type=str, help="Device: auto/cpu/gpu/cuda/cuda:0")
    train_parser.add_argument("--eval-only", "--eval_only", dest="eval_only", action="store_true", help="Skip training and evaluate checkpoints only")

    predict_parser = subparsers.add_parser("predict", help="Run condition prediction")
    predict_parser.add_argument("--config", type=Path, help="Path to condition case YAML")
    predict_parser.add_argument("--output-dir", type=str, help="Output directory override")

    pipeline_parser = subparsers.add_parser("pipeline", help="Run the full pipeline")
    pipeline_parser.add_argument("--structure", type=Path, help="Path to structure YAML file")
    pipeline_parser.add_argument("--dataset-config", type=Path, help="Path to dataset config YAML")
    pipeline_parser.add_argument("--condition-config", type=Path, help="Path to condition case YAML")
    pipeline_parser.add_argument("-j", "--jobs", type=int, default=-1, help="Number of parallel workers for dataset generation")
    pipeline_parser.add_argument("--seq", action="store_true", help="Generate the dataset sequentially")
    pipeline_parser.add_argument("--model", choices=["gt", "pinn", "legacy_pinn", "pinn_v2", "both"], default="gt", help="Model type for training")
    pipeline_parser.add_argument("--epochs", type=int, help="Number of epochs")
    pipeline_parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, help="Batch size")
    pipeline_parser.add_argument("--lr", type=float, help="Learning rate")
    pipeline_parser.add_argument("--device", type=str, help="Device: auto/cpu/gpu/cuda/cuda:0")
    pipeline_parser.add_argument("--eval-only", "--eval_only", dest="eval_only", action="store_true", help="Skip training and evaluate checkpoints only")
    pipeline_parser.add_argument("--output-dir", type=str, help="Output directory override for prediction")
    pipeline_parser.add_argument("--skip-static", action="store_true", help="Skip static analysis")
    pipeline_parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset generation")
    pipeline_parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    pipeline_parser.add_argument("--skip-predict", action="store_true", help="Skip condition prediction")

    status_parser = subparsers.add_parser("status", help="Check project status")
    status_parser.add_argument("--env-name", type=str, help="Override target conda environment name")

    return parser


def run_interactive_shell(parser: argparse.ArgumentParser) -> int:
    _print("Interactive FEM CLI")
    _print("Type `help` to show available commands.")
    _print("Type `exit` or `quit` to leave.")
    _print("Enter a bare command like `train`, `dataset`, `predict`, or `pipeline` to start the guided wizard.")
    _print("Inside the wizard, press Enter to accept defaults and use option numbers for menu choices.")
    _print("Examples:")
    _print("  status")
    _print("  train")
    _print("  dataset")
    _print("  pipeline")
    _print("  train --model gt --epochs 100 --device gpu")

    while True:
        try:
            raw_command = input("Enter a command> ").strip()
        except EOFError:
            _print("")
            return 0
        except KeyboardInterrupt:
            _print("")
            return 0

        if not raw_command:
            continue
        if raw_command.lower() in {"exit", "quit"}:
            return 0
        if raw_command.lower() == "help":
            parser.print_help()
            continue

        try:
            tokens = shlex.split(raw_command)
        except ValueError as exc:
            _print(f"Invalid command: {exc}")
            continue

        if len(tokens) == 1 and tokens[0] in COMMANDS:
            try:
                parsed_args = build_command_wizard_namespace(tokens[0])
            except SystemExit as exc:
                return int(exc.code or 0)
            if parsed_args is None:
                continue
            exit_code = COMMANDS[parsed_args.command](parsed_args)
            if exit_code != 0:
                _print(f"Command exited with code {exit_code}.")
            continue

        try:
            parsed_args = parser.parse_args(tokens)
        except SystemExit:
            continue

        if parsed_args.command is None:
            parser.print_help()
            continue

        exit_code = COMMANDS[parsed_args.command](parsed_args)
        if exit_code != 0:
            _print(f"Command exited with code {exit_code}.")


COMMANDS = {
    "env": cmd_env,
    "bootstrap": cmd_bootstrap,
    "install": cmd_install,
    "static": cmd_static,
    "dataset": cmd_dataset,
    "train": cmd_train,
    "predict": cmd_predict,
    "pipeline": cmd_pipeline,
    "status": cmd_status,
}


def main() -> int:
    argv = sys.argv[1:]
    parser = build_parser()

    print_banner()

    if not argv:
        return run_interactive_shell(parser)

    args = parser.parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
