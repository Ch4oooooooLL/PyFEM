#!/usr/bin/env python3
"""
统一CLI入口 - FEM结构损伤识别全流程工具

Usage:
    python cli.py --help
    python cli.py static              # 运行FEM静态分析
    python cli.py dataset             # 生成训练数据集
    python cli.py train               # 训练深度学习模型
    python cli.py predict             # 运行条件预测
    python cli.py pipeline            # 运行完整流水线
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich import box
from collections import deque

console = Console()

SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT_DIR = SCRIPT_DIR
PYFEM_DIR = ROOT_DIR / "PyFEM_Dynamics"
DEEP_LEARNING_DIR = ROOT_DIR / "Deep_learning"
CONDITION_PREDICTION_DIR = ROOT_DIR / "Condition_prediction"


def print_banner():
    """打印项目横幅"""
    banner = """
    +-----------------------------------------------------------+
    |                                                           |
    |    FEM Structural Damage Identification System            |
    |    Structural Damage Identification System                |
    |                                                           |
    |    [FEM Solver] -> [Deep Learning] -> [Predict Model]     |
    |                                                           |
    +-----------------------------------------------------------+
    """
    console.print(Panel(banner, border_style="cyan", title="Welcome", title_align="center"))


def run_command(cmd: list[str], description: str, cwd: Optional[Path] = None, max_log_lines: int = 50) -> int:
    """运行命令并显示进度，实时显示子进程输出日志"""
    from rich.console import Group
    from rich.text import Text
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)
    log_lines: deque[str] = deque(maxlen=max_log_lines)
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )
    task = progress.add_task(f"[cyan]{description}...", total=None)
    
    def make_renderable():
        if log_lines:
            log_text = Text()
            for line in log_lines:
                log_text.append(line + "\n")
            log_panel = Panel(log_text, title="[dim]Log Output[/dim]", border_style="dim", padding=(0, 1))
            return Group(progress, log_panel)
        return Group(progress)
    
    with Live(make_renderable(), console=console, refresh_per_second=10) as live:
        process = subprocess.Popen(
            cmd,
            cwd=cwd or ROOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            env=env,
            bufsize=1,
        )
        
        while True:
            if process.stdout is None:
                break
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                stripped = line.rstrip()
                if stripped:
                    log_lines.append(stripped)
                    live.update(make_renderable())
        
        process.wait()
        progress.update(task, completed=True)
        live.update(make_renderable())
    
    if process.returncode != 0:
        console.print(f"[red][X] {description} failed with code {process.returncode}")
    else:
        console.print(f"[green][OK] {description} completed")
    
    return process.returncode


def cmd_static(args: argparse.Namespace) -> int:
    """运行FEM静态分析"""
    console.print(Panel("[bold blue]FEM Static Analysis[/bold blue]", border_style="blue"))
    
    structure_file = args.structure or ROOT_DIR / "structure.yaml"
    
    with console.status("[bold green]Running FEM static analysis...") as status:
        sys.path.insert(0, str(PYFEM_DIR))
        from PyFEM_Dynamics.main import run_single_static_test
        
        try:
            run_single_static_test(structure_yaml=str(structure_file))
            console.print("[green][OK] Static analysis completed successfully")
            return 0
        except Exception as e:
            console.print(f"[red][X] Error: {e}")
            return 1


def cmd_dataset(args: argparse.Namespace) -> int:
    """生成训练数据集"""
    console.print(Panel("[bold blue]Dataset Generation[/bold blue]", border_style="blue"))
    
    config_file = args.config or ROOT_DIR / "dataset_config.yaml"
    
    sys.path.insert(0, str(PYFEM_DIR))
    from PyFEM_Dynamics.pipeline.data_gen import generate_dataset
    
    try:
        with console.status("[bold green]Generating training dataset..."):
            generate_dataset(
                config_path=str(config_file),
                n_jobs=args.jobs,
                sequential=args.seq,
            )
        console.print("[green][OK] Dataset generation completed")
        return 0
    except Exception as e:
        console.print(f"[red][X] Error: {e}")
        return 1


def cmd_train(args: argparse.Namespace) -> int:
    """训练深度学习模型"""
    console.print(Panel("[bold blue]Model Training[/bold blue]", border_style="blue"))
    
    model = args.model or "gt"
    epochs = args.epochs or 100
    config = args.config or ROOT_DIR / "dataset_config.yaml"
    
    cmd_args = [
        sys.executable,
        str(DEEP_LEARNING_DIR / "train.py"),
        "--config", str(config),
        "--model", model,
        "--epochs", str(epochs),
    ]
    
    if args.batch_size:
        cmd_args.extend(["--batch_size", str(args.batch_size)])
    if args.lr:
        cmd_args.extend(["--lr", str(args.lr)])
    if args.device:
        cmd_args.extend(["--device", args.device])
    if args.eval_only:
        cmd_args.append("--eval_only")
    
    return run_command(cmd_args, f"Training {model.upper()} model")


def cmd_predict(args: argparse.Namespace) -> int:
    """运行条件预测"""
    console.print(Panel("[bold blue]Condition Prediction[/bold blue]", border_style="blue"))
    
    config = args.config or ROOT_DIR / "condition_case.yaml"
    
    sys.path.insert(0, str(ROOT_DIR))
    from Condition_prediction.pipelines.condition_pipeline import run_condition_pipeline
    
    try:
        with console.status("[bold green]Running condition prediction pipeline..."):
            summary = run_condition_pipeline(
                config_path=str(config),
                output_dir_override=args.output_dir,
            )
        
        console.print("[green][OK] Prediction completed")
        
        table = Table(title="Prediction Results", box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        if "metrics" in summary:
            for key, value in summary["metrics"].items():
                table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
        
        console.print(table)
        console.print(f"[blue]Results saved to: {summary.get('run_dir', 'N/A')}")
        
        return 0
    except Exception as e:
        console.print(f"[red][X] Error: {e}")
        return 1


def cmd_pipeline(args: argparse.Namespace) -> int:
    """运行完整流水线"""
    console.print(Panel("[bold magenta]Full Pipeline Execution[/bold magenta]", border_style="magenta"))
    
    steps = []
    if not args.skip_static:
        steps.append(("Static Analysis", lambda: cmd_static(args)))
    if not args.skip_dataset:
        steps.append(("Dataset Generation", lambda: cmd_dataset(args)))
    if not args.skip_train:
        steps.append(("Model Training", lambda: cmd_train(args)))
    if not args.skip_predict:
        steps.append(("Condition Prediction", lambda: cmd_predict(args)))
    
    total_steps = len(steps)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="green", finished_style="bright_green"),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task("[cyan]Pipeline Progress", total=total_steps)
        
        for i, (step_name, step_func) in enumerate(steps, 1):
            progress.update(overall_task, description=f"[cyan]Step {i}/{total_steps}: {step_name}")
            
            result = step_func()
            if result != 0:
                console.print(f"[red][X] Pipeline failed at step: {step_name}")
                return result
            
            progress.advance(overall_task)
    
    console.print(Panel("[bold green][OK] Full pipeline completed successfully![/bold green]", border_style="green"))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """检查项目状态"""
    console.print(Panel("[bold blue]Project Status[/bold blue]", border_style="blue"))
    
    table = Table(box=box.ROUNDED)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")
    
    # Check structure file
    structure_file = ROOT_DIR / "structure.yaml"
    if structure_file.exists():
        table.add_row("Structure Config", "[OK] Ready", str(structure_file))
    else:
        table.add_row("Structure Config", "[X] Missing", "Please create structure.yaml")
    
    # Check dataset config
    dataset_config = ROOT_DIR / "dataset_config.yaml"
    if dataset_config.exists():
        table.add_row("Dataset Config", "[OK] Ready", str(dataset_config))
    else:
        table.add_row("Dataset Config", "[X] Missing", "Please create dataset_config.yaml")
    
    # Check dataset
    dataset_file = ROOT_DIR / "dataset" / "train.npz"
    if dataset_file.exists():
        import numpy as np
        data = np.load(dataset_file)
        table.add_row("Training Dataset", "[OK] Ready", f"{len(data.files)} arrays, {data['load'].shape[0]} samples")
    else:
        table.add_row("Training Dataset", "[X] Missing", "Run 'cli.py dataset' to generate")
    
    # Check model checkpoints
    gt_ckpt = DEEP_LEARNING_DIR / "checkpoints" / "gt_best.pth"
    pinn_ckpt = DEEP_LEARNING_DIR / "checkpoints" / "pinn_best.pth"
    
    if gt_ckpt.exists():
        table.add_row("GT Model", "[OK] Ready", str(gt_ckpt))
    else:
        table.add_row("GT Model", "[X] Missing", "Run 'cli.py train --model gt'")
    
    if pinn_ckpt.exists():
        table.add_row("PINN Model", "[OK] Ready", str(pinn_ckpt))
    else:
        table.add_row("PINN Model", "[X] Missing", "Run 'cli.py train --model pinn'")
    
    # Check condition config
    condition_config = ROOT_DIR / "condition_case.yaml"
    if condition_config.exists():
        table.add_row("Condition Config", "[OK] Ready", str(condition_config))
    else:
        table.add_row("Condition Config", "[X] Missing", "Please create condition_case.yaml")
    
    console.print(table)
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="FEM Structural Damage Identification CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cli.py static                           # 运行静态分析
    python cli.py dataset -j 4                     # 并行生成数据集
    python cli.py train --model gt --epochs 100    # 训练GT模型
    python cli.py train --model both               # 训练两个模型
    python cli.py predict                          # 运行条件预测
    python cli.py pipeline                         # 运行完整流程
    python cli.py status                           # 检查项目状态
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # static command
    static_parser = subparsers.add_parser("static", help="Run FEM static analysis")
    static_parser.add_argument("--structure", type=Path, help="Path to structure YAML file")
    
    # dataset command
    dataset_parser = subparsers.add_parser("dataset", help="Generate training dataset")
    dataset_parser.add_argument("--config", type=Path, help="Path to dataset config YAML")
    dataset_parser.add_argument("-j", "--jobs", type=int, default=-1, help="Number of parallel workers")
    dataset_parser.add_argument("--seq", action="store_true", help="Run sequentially")
    
    # train command
    train_parser = subparsers.add_parser("train", help="Train deep learning models")
    train_parser.add_argument("--model", choices=["gt", "pinn", "both"], help="Model type")
    train_parser.add_argument("--config", type=Path, help="Path to config YAML")
    train_parser.add_argument("--epochs", type=int, help="Number of epochs")
    train_parser.add_argument("--batch_size", type=int, help="Batch size")
    train_parser.add_argument("--lr", type=float, help="Learning rate")
    train_parser.add_argument("--device", type=str, help="Device (auto/cpu/cuda)")
    train_parser.add_argument("--eval_only", action="store_true", help="Evaluation only")
    
    # predict command
    predict_parser = subparsers.add_parser("predict", help="Run condition prediction")
    predict_parser.add_argument("--config", type=Path, help="Path to condition case YAML")
    predict_parser.add_argument("--output-dir", type=str, help="Output directory override")
    
    # pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline")
    pipeline_parser.add_argument("--skip-static", action="store_true", help="Skip static analysis")
    pipeline_parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset generation")
    pipeline_parser.add_argument("--skip-train", action="store_true", help="Skip training")
    pipeline_parser.add_argument("--skip-predict", action="store_true", help="Skip prediction")
    
    # status command
    status_parser = subparsers.add_parser("status", help="Check project status")
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    commands = {
        "static": cmd_static,
        "dataset": cmd_dataset,
        "train": cmd_train,
        "predict": cmd_predict,
        "pipeline": cmd_pipeline,
        "status": cmd_status,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
