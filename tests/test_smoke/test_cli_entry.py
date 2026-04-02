from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_root_wrappers_replace_root_cli() -> None:
    assert (PROJECT_ROOT / "run.bat").exists()
    assert (PROJECT_ROOT / "run.sh").exists()
    assert not (PROJECT_ROOT / "cli.py").exists()


def test_run_wrappers_include_interactive_help_entrypoint() -> None:
    run_bat = (PROJECT_ROOT / "run.bat").read_text(encoding="utf-8").lower()
    run_sh = (PROJECT_ROOT / "run.sh").read_text(encoding="utf-8").lower()

    assert "python -m tools.cli" in run_bat
    assert "python -m tools.cli" in run_sh
    assert " activate " in run_bat
    assert "conda activate" in run_sh


def test_tools_cli_defines_python_repl_prompt() -> None:
    cli_source = (PROJECT_ROOT / "tools" / "cli.py").read_text(encoding="utf-8").lower()

    assert "enter a command> " in cli_source
    assert "type `help`" in cli_source
    assert "type `exit` or `quit` to leave" in cli_source
    assert "guided wizard" in cli_source
    assert "train" in cli_source
    assert "use option numbers for menu choices" in cli_source


def test_tools_cli_importable() -> None:
    from tools import cli

    assert cli.main is not None


def test_resolve_conda_env_name_prefers_explicit_value() -> None:
    from tools.cli import resolve_conda_env_name

    assert resolve_conda_env_name("custom-env", "active-env") == "custom-env"


def test_resolve_conda_env_name_uses_active_env_when_available() -> None:
    from tools.cli import resolve_conda_env_name

    assert resolve_conda_env_name(None, "active-env") == "active-env"


def test_resolve_conda_env_name_falls_back_to_default_when_needed() -> None:
    from tools.cli import DEFAULT_CONDA_ENV_NAME, resolve_conda_env_name

    assert resolve_conda_env_name(None, None) == DEFAULT_CONDA_ENV_NAME
    assert resolve_conda_env_name(None, "base") == DEFAULT_CONDA_ENV_NAME


def test_normalize_device_aliases_gpu_to_cuda() -> None:
    from tools.cli import normalize_device_value

    assert normalize_device_value("gpu") == "cuda"
    assert normalize_device_value("GPU") == "cuda"
    assert normalize_device_value("cpu") == "cpu"
    assert normalize_device_value("cuda:1") == "cuda:1"


def test_build_device_validation_error_for_missing_cuda() -> None:
    from tools.cli import build_device_validation_error

    message = build_device_validation_error("cuda", torch_available=True, cuda_available=False)

    assert message is not None
    assert "cuda" in message.lower()
    assert "--device cpu" in message


def test_build_device_validation_error_for_cpu_request_is_none() -> None:
    from tools.cli import build_device_validation_error

    assert build_device_validation_error("cpu", torch_available=True, cuda_available=False) is None
