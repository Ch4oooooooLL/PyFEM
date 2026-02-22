"""
Publication-grade postprocessing utilities for training results.
"""

import argparse
import csv
import glob
import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


LOWER_BETTER_METRICS = {"mae", "rmse", "mse", "damage_mae", "loss"}
DEFAULT_PRIMARY_METRICS = ["mae", "rmse", "f1", "iou", "accuracy", "balanced_accuracy"]


def _apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.linewidth": 1.0,
            "grid.alpha": 0.30,
            "grid.linestyle": "--",
        }
    )


def _parse_results_payload(raw: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Support both schemas:
    1) legacy: {model_name: {...}}
    2) new:    {"metadata": {...}, "models": {model_name: {...}}}
    """
    if "models" in raw and isinstance(raw["models"], dict):
        return raw["models"], raw.get("metadata", {})

    # Legacy fallback
    model_like = {
        k: v
        for k, v in raw.items()
        if isinstance(v, dict) and "history" in v and "test_metrics" in v
    }
    if model_like:
        metadata = raw.get("metadata", {}) if isinstance(raw.get("metadata"), dict) else {}
        return model_like, metadata

    raise ValueError("Unrecognized results schema.")


def _load_results_file(results_file: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    with open(results_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return _parse_results_payload(raw)


def _safe_mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr)) if len(arr) else 0.0
    if len(arr) <= 1:
        return mean, 0.0
    return mean, float(np.std(arr, ddof=1))


def aggregate_runs(result_files: Iterable[str]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple result files.
    """
    metric_pool: Dict[str, Dict[str, List[float]]] = {}
    file_count = 0

    for file_path in result_files:
        models, _ = _load_results_file(file_path)
        if not models:
            continue
        file_count += 1
        for model_name, model_data in models.items():
            metric_pool.setdefault(model_name, {})
            metrics = model_data.get("test_metrics", {})
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_pool[model_name].setdefault(metric_name, []).append(float(value))

    aggregated: Dict[str, Any] = {"n_runs": file_count, "models": {}}
    for model_name, model_metrics in metric_pool.items():
        aggregated["models"][model_name] = {}
        for metric_name, values in model_metrics.items():
            mean, std = _safe_mean_std(values)
            ci95 = 0.0 if len(values) <= 1 else float(1.96 * std / np.sqrt(len(values)))
            aggregated["models"][model_name][metric_name] = {
                "mean": mean,
                "std": std,
                "ci95": ci95,
                "n": len(values),
            }
    return aggregated


def _resolve_prediction_file(
    results_file: str, metadata: Dict[str, Any], explicit_prediction_file: Optional[str]
) -> Optional[str]:
    if explicit_prediction_file:
        return explicit_prediction_file if os.path.exists(explicit_prediction_file) else None

    pred_from_meta = metadata.get("prediction_file")
    if isinstance(pred_from_meta, str) and pred_from_meta:
        if os.path.isabs(pred_from_meta):
            return pred_from_meta if os.path.exists(pred_from_meta) else None
        candidate = os.path.join(os.path.dirname(results_file), pred_from_meta)
        if os.path.exists(candidate):
            return candidate

    # Fallback: infer from filename timestamp
    base = os.path.basename(results_file)
    if base.startswith("results_") and base.endswith(".json"):
        stamp = base[len("results_") : -len(".json")]
        candidate = os.path.join(os.path.dirname(results_file), f"predictions_{stamp}.npz")
        if os.path.exists(candidate):
            return candidate
    return None


def _load_prediction_bundle(prediction_file: str) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    data = np.load(prediction_file)
    predictions: Dict[str, np.ndarray] = {}
    ground_truth: Optional[np.ndarray] = None

    for key in data.files:
        value = data[key]
        if key.endswith("_pred"):
            model_name = key[: -len("_pred")]
            predictions[model_name] = value
        elif key in {"true_damage", "ground_truth", "labels"}:
            ground_truth = value

    if ground_truth is None:
        # Fallback for possible model-specific label keys
        for key in data.files:
            if key.endswith("_true"):
                ground_truth = data[key]
                break

    return predictions, ground_truth


class TrainingVisualizer:
    """Generate publication-grade figures and summary artifacts."""

    def __init__(
        self,
        save_dir: Optional[str] = None,
        threshold: float = 0.95,
        export_formats: Tuple[str, ...] = ("png", "pdf"),
        random_seed: int = 42,
    ):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_dir = os.path.join(os.path.dirname(script_dir), "figures")
        self.save_dir = save_dir or default_dir
        self.threshold = threshold
        self.export_formats = export_formats
        self.rng = np.random.default_rng(random_seed)
        os.makedirs(self.save_dir, exist_ok=True)
        _apply_publication_style()

    def _save_figure(self, fig: plt.Figure, save_stem: str) -> List[str]:
        dir_path = os.path.dirname(save_stem)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        base, ext = os.path.splitext(save_stem)
        if ext:
            fig.savefig(save_stem)
            return [save_stem]

        saved_files: List[str] = []
        for fmt in self.export_formats:
            file_path = f"{base}.{fmt}"
            fig.savefig(file_path)
            saved_files.append(file_path)
        return saved_files

    def plot_training_history(self, history: Dict[str, List[float]], model_name: str, save_stem: str) -> List[str]:
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        train_loss = np.asarray(history["train_loss"], dtype=float)
        val_loss = np.asarray(history["val_loss"], dtype=float)
        val_mae = np.asarray(history["val_mae"], dtype=float)
        val_f1 = np.asarray(history["val_f1"], dtype=float)

        best_idx = int(np.argmin(val_loss))
        best_epoch = int(epochs[best_idx])

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

        ax = axes[0, 0]
        ax.plot(epochs, train_loss, label="Train Loss", linewidth=2)
        ax.plot(epochs, val_loss, label="Val Loss", linewidth=2)
        ax.scatter([best_epoch], [val_loss[best_idx]], color="black", s=25, zorder=5, label="Best Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{model_name.upper()} Loss Curves")
        ax.legend()
        ax.grid(True)

        ax = axes[0, 1]
        ax.plot(epochs, val_mae, color="#2a9d8f", linewidth=2, label="Val MAE")
        ax2 = ax.twinx()
        ax2.plot(epochs, val_f1, color="#e76f51", linewidth=1.8, label="Val F1")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE")
        ax2.set_ylabel("F1")
        ax.set_title(f"{model_name.upper()} Validation Metrics")
        lines = ax.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc="best")
        ax.grid(True)

        ax = axes[1, 0]
        gap = val_loss - train_loss
        ax.plot(epochs, gap, color="#6a4c93", linewidth=2)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Loss - Train Loss")
        ax.set_title(f"{model_name.upper()} Generalization Gap")
        ax.grid(True)

        ax = axes[1, 1]
        rel_improve = (val_loss[0] - val_loss) / (abs(val_loss[0]) + 1e-12)
        ax.plot(epochs, rel_improve, color="#264653", linewidth=2)
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Relative Improvement")
        ax.set_title(f"{model_name.upper()} Validation Improvement")
        ax.grid(True)

        fig.tight_layout()
        return self._save_figure(fig, save_stem)

    def plot_model_comparison(self, results: Dict[str, Dict[str, Any]], save_stem: str) -> List[str]:
        model_names = list(results.keys())
        metrics = [m for m in DEFAULT_PRIMARY_METRICS if any(m in results[k]["test_metrics"] for k in model_names)]
        if not metrics:
            raise ValueError("No comparable metrics found in results.")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        x = np.arange(len(metrics))
        width = 0.8 / max(len(model_names), 1)

        ax = axes[0]
        for i, model_name in enumerate(model_names):
            values = [results[model_name]["test_metrics"].get(m, np.nan) for m in metrics]
            offset = (i - (len(model_names) - 1) / 2.0) * width
            ax.bar(x + offset, values, width=width, label=model_name.upper(), alpha=0.90)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics], rotation=20)
        ax.set_title("Raw Test Metrics")
        ax.set_ylabel("Metric Value")
        ax.grid(True, axis="y")
        ax.legend()

        ax = axes[1]
        normalized = np.zeros((len(model_names), len(metrics)), dtype=float)
        for j, metric in enumerate(metrics):
            values = np.asarray([results[m]["test_metrics"].get(metric, np.nan) for m in model_names], dtype=float)
            if np.all(np.isnan(values)):
                continue
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            vmin, vmax = float(np.min(finite)), float(np.max(finite))
            if np.isclose(vmin, vmax):
                normalized[:, j] = 0.5
            else:
                if metric in LOWER_BETTER_METRICS:
                    normalized[:, j] = (vmax - values) / (vmax - vmin)
                else:
                    normalized[:, j] = (values - vmin) / (vmax - vmin)

        for i, model_name in enumerate(model_names):
            ax.plot(metrics, normalized[i], marker="o", linewidth=2, label=model_name.upper())
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Normalized Score (higher is better)")
        ax.set_title("Direction-Aware Normalized Comparison")
        ax.grid(True)
        ax.legend()

        fig.tight_layout()
        return self._save_figure(fig, save_stem)

    def plot_run_stability(self, aggregate_stats: Dict[str, Any], save_stem: str) -> Optional[List[str]]:
        n_runs = int(aggregate_stats.get("n_runs", 0))
        if n_runs < 2:
            return None

        models = aggregate_stats.get("models", {})
        if not models:
            return None

        metric_candidates = [("mae", "MAE"), ("f1", "F1")]
        available_metrics = [
            (metric_key, label)
            for metric_key, label in metric_candidates
            if all(metric_key in model_stats for model_stats in models.values())
        ]
        if not available_metrics:
            return None

        fig, axes = plt.subplots(1, len(available_metrics), figsize=(6 * len(available_metrics), 4))
        if len(available_metrics) == 1:
            axes = [axes]

        model_names = list(models.keys())
        x = np.arange(len(model_names))

        for ax, (metric, metric_label) in zip(axes, available_metrics):
            means = [models[m][metric]["mean"] for m in model_names]
            ci95 = [models[m][metric]["ci95"] for m in model_names]
            ax.bar(x, means, yerr=ci95, capsize=5, alpha=0.9, color="#457b9d")
            ax.set_xticks(x)
            ax.set_xticklabels([m.upper() for m in model_names])
            ax.set_title(f"{metric_label} Across Runs (95% CI)")
            ax.set_ylabel(metric_label)
            ax.grid(True, axis="y")

        fig.tight_layout()
        return self._save_figure(fig, save_stem)

    def plot_prediction_comparison(
        self,
        predictions: Dict[str, np.ndarray],
        ground_truth: np.ndarray,
        save_stem: str,
        sample_indices: Optional[List[int]] = None,
        max_samples: int = 4,
    ) -> Optional[List[str]]:
        if ground_truth is None or not predictions:
            return None

        n_total = int(ground_truth.shape[0])
        if n_total == 0:
            return None

        if sample_indices is None:
            n_samples = min(max_samples, n_total)
            sample_indices = self.rng.choice(n_total, n_samples, replace=False).tolist()
        else:
            sample_indices = [idx for idx in sample_indices if 0 <= idx < n_total]
            if not sample_indices:
                return None

        n_samples = len(sample_indices)
        model_names = list(predictions.keys())
        n_cols = 1 + len(model_names)
        fig, axes = plt.subplots(n_samples, n_cols, figsize=(3.5 * n_cols, 2.8 * n_samples), squeeze=False)

        threshold = self.threshold
        element_ids = np.arange(ground_truth.shape[1])

        for row, sample_id in enumerate(sample_indices):
            true_vec = ground_truth[sample_id]
            true_state = true_vec < threshold

            ax_true = axes[row, 0]
            ax_true.bar(
                element_ids,
                true_vec,
                color=np.where(true_state, "#d62828", "#2a9d8f"),
                edgecolor="black",
                linewidth=0.4,
            )
            ax_true.axhline(threshold, color="gray", linestyle="--", linewidth=1)
            ax_true.set_ylim(0.0, 1.05)
            ax_true.set_title("Ground Truth" if row == 0 else "")
            ax_true.set_ylabel(f"Sample {sample_id}\nDamage Ratio")
            if row == n_samples - 1:
                ax_true.set_xlabel("Element ID")

            for col, model_name in enumerate(model_names, start=1):
                pred_vec = predictions[model_name][sample_id]
                pred_state = pred_vec < threshold
                correct = pred_state == true_state
                ax = axes[row, col]
                ax.bar(
                    element_ids,
                    pred_vec,
                    color=np.where(correct, "#2a9d8f", "#d62828"),
                    edgecolor="black",
                    linewidth=0.4,
                )
                ax.axhline(threshold, color="gray", linestyle="--", linewidth=1)
                ax.set_ylim(0.0, 1.05)
                mae = float(np.mean(np.abs(pred_vec - true_vec)))
                ax.text(
                    0.98,
                    0.96,
                    f"MAE={mae:.3f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                )
                if row == 0:
                    ax.set_title(f"{model_name.upper()} Prediction")
                if row == n_samples - 1:
                    ax.set_xlabel("Element ID")

        fig.tight_layout()
        return self._save_figure(fig, save_stem)

    def plot_calibration(
        self, predictions: Dict[str, np.ndarray], ground_truth: np.ndarray, save_stem: str
    ) -> Optional[List[str]]:
        if ground_truth is None or not predictions:
            return None

        true_binary = (ground_truth < self.threshold).astype(float).reshape(-1)
        if true_binary.size == 0:
            return None

        fig, ax = plt.subplots(figsize=(6.5, 5))
        bins = np.linspace(0.0, 1.0, 11)

        for model_name, pred in predictions.items():
            # Predicted damage probability: higher => more likely damaged
            prob_damage = np.clip(1.0 - pred, 0.0, 1.0).reshape(-1)
            bin_ids = np.digitize(prob_damage, bins) - 1
            x_points = []
            y_points = []
            for bin_idx in range(len(bins) - 1):
                mask = bin_ids == bin_idx
                if np.sum(mask) == 0:
                    continue
                x_points.append(float(np.mean(prob_damage[mask])))
                y_points.append(float(np.mean(true_binary[mask])))

            if x_points:
                ax.plot(x_points, y_points, marker="o", linewidth=2, label=model_name.upper())

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted Damage Probability")
        ax.set_ylabel("Observed Damage Frequency")
        ax.set_title("Calibration Curve")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        return self._save_figure(fig, save_stem)

    def export_metrics_table(self, results: Dict[str, Dict[str, Any]], save_path: str) -> str:
        model_names = list(results.keys())
        metric_names: List[str] = []
        for model_name in model_names:
            for key in results[model_name].get("test_metrics", {}).keys():
                if key not in metric_names and key not in {"threshold", "support_damaged", "support_undamaged"}:
                    metric_names.append(key)

        rows = []
        for model_name in model_names:
            row = {"model": model_name}
            row.update(results[model_name].get("test_metrics", {}))
            rows.append(row)

        with open(save_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["model"] + metric_names)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
        return save_path

    def _generate_text_report(
        self,
        results: Dict[str, Dict[str, Any]],
        aggregate_stats: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]],
    ) -> str:
        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("Structural Damage Identification: Postprocessing Report")
        lines.append("=" * 70)
        lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Decision threshold: {self.threshold:.4f}")
        if metadata:
            lines.append(f"Run timestamp: {metadata.get('created_at', 'N/A')}")
            lines.append(f"Seed: {metadata.get('seed', 'N/A')}")
            lines.append(f"Device: {metadata.get('device', 'N/A')}")
        lines.append("")

        lines.append("-" * 70)
        lines.append("Single-Run Test Metrics")
        lines.append("-" * 70)
        header = f"{'Model':<12} {'MAE':<10} {'RMSE':<10} {'F1':<10} {'IoU':<10} {'Acc':<10}"
        lines.append(header)
        lines.append("-" * 70)
        for model_name, model_data in results.items():
            m = model_data.get("test_metrics", {})
            lines.append(
                f"{model_name.upper():<12} "
                f"{m.get('mae', float('nan')):<10.4f} "
                f"{m.get('rmse', float('nan')):<10.4f} "
                f"{m.get('f1', float('nan')):<10.4f} "
                f"{m.get('iou', float('nan')):<10.4f} "
                f"{m.get('accuracy', float('nan')):<10.4f}"
            )
        lines.append("")

        for model_name, model_data in results.items():
            history = model_data.get("history", {})
            val_loss = history.get("val_loss", [])
            if not val_loss:
                continue
            best_epoch = int(np.argmin(np.asarray(val_loss)) + 1)
            lines.append(
                f"{model_name.upper()} best epoch: {best_epoch} | "
                f"best val loss: {np.min(val_loss):.6f}"
            )

        if aggregate_stats and aggregate_stats.get("n_runs", 0) >= 2:
            lines.append("")
            lines.append("-" * 70)
            lines.append(f"Cross-Run Stability Summary (n={aggregate_stats['n_runs']})")
            lines.append("-" * 70)
            for model_name, model_stats in aggregate_stats.get("models", {}).items():
                mae_stats = model_stats.get("mae")
                f1_stats = model_stats.get("f1")
                if mae_stats and f1_stats:
                    lines.append(
                        f"{model_name.upper()}: "
                        f"MAE={mae_stats['mean']:.4f}±{mae_stats['ci95']:.4f} (95% CI), "
                        f"F1={f1_stats['mean']:.4f}±{f1_stats['ci95']:.4f} (95% CI)"
                    )

        if results:
            best_mae = min(results.items(), key=lambda item: item[1]["test_metrics"].get("mae", float("inf")))
            best_f1 = max(results.items(), key=lambda item: item[1]["test_metrics"].get("f1", float("-inf")))
            lines.append("")
            lines.append("-" * 70)
            lines.append("Selection Recommendation")
            lines.append("-" * 70)
            lines.append(
                f"Primary regression criterion (lowest MAE): {best_mae[0].upper()} "
                f"({best_mae[1]['test_metrics'].get('mae', float('nan')):.4f})"
            )
            lines.append(
                f"Primary classification criterion (highest F1): {best_f1[0].upper()} "
                f"({best_f1[1]['test_metrics'].get('f1', float('nan')):.4f})"
            )

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    def create_report(
        self,
        results: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None,
        aggregate_stats: Optional[Dict[str, Any]] = None,
        predictions: Optional[Dict[str, np.ndarray]] = None,
        ground_truth: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = save_path or os.path.join(self.save_dir, f"report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)

        # Per-model training curves
        for model_name, data in results.items():
            self.plot_training_history(
                data["history"],
                model_name=model_name,
                save_stem=os.path.join(report_dir, f"{model_name}_history"),
            )

        # Cross-model summary
        self.plot_model_comparison(results, save_stem=os.path.join(report_dir, "model_comparison"))

        if aggregate_stats:
            self.plot_run_stability(aggregate_stats, save_stem=os.path.join(report_dir, "run_stability"))

        if predictions is not None and ground_truth is not None:
            self.plot_prediction_comparison(
                predictions=predictions,
                ground_truth=ground_truth,
                save_stem=os.path.join(report_dir, "prediction_comparison"),
            )
            self.plot_calibration(
                predictions=predictions,
                ground_truth=ground_truth,
                save_stem=os.path.join(report_dir, "calibration_curve"),
            )

        metrics_csv = self.export_metrics_table(results, os.path.join(report_dir, "metrics_table.csv"))

        report_text = self._generate_text_report(results, aggregate_stats, metadata)
        report_txt_path = os.path.join(report_dir, "report.txt")
        with open(report_txt_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        summary_payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "threshold": self.threshold,
            "metrics_table": os.path.basename(metrics_csv),
            "n_models": len(results),
            "n_runs_aggregated": int(aggregate_stats.get("n_runs", 0)) if aggregate_stats else 0,
            "metadata": metadata or {},
        }
        with open(os.path.join(report_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)

        print(f"Report generated: {report_dir}")
        return report_dir


def visualize_results(
    results_file: str,
    prediction_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    aggregate_files: Optional[List[str]] = None,
    threshold: Optional[float] = None,
) -> str:
    results, metadata = _load_results_file(results_file)
    threshold_value = float(threshold) if threshold is not None else float(metadata.get("threshold", 0.95))

    pred_file = _resolve_prediction_file(results_file, metadata, prediction_file)
    predictions = None
    ground_truth = None
    if pred_file:
        predictions, ground_truth = _load_prediction_bundle(pred_file)

    if aggregate_files is None:
        aggregate_files = [results_file]
    aggregate_stats = aggregate_runs(aggregate_files)

    vis = TrainingVisualizer(save_dir=output_dir, threshold=threshold_value)
    return vis.create_report(
        results=results,
        save_path=None if output_dir is None else os.path.join(output_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        aggregate_stats=aggregate_stats,
        predictions=predictions,
        ground_truth=ground_truth,
        metadata=metadata,
    )


def _choose_result_files(checkpoints_dir: str, latest_n: int, aggregate_all: bool) -> Tuple[str, List[str]]:
    result_files = sorted(glob.glob(os.path.join(checkpoints_dir, "results_*.json")), key=os.path.getmtime)
    if not result_files:
        raise FileNotFoundError(f"No results_*.json found in {checkpoints_dir}")

    latest = result_files[-1]
    if aggregate_all:
        return latest, result_files

    n = max(int(latest_n), 1)
    return latest, result_files[-n:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-grade reports from training results.")
    parser.add_argument("--results_file", type=str, default=None, help="Path to one results_*.json")
    parser.add_argument("--checkpoints_dir", type=str, default=None, help="Directory containing results_*.json")
    parser.add_argument("--prediction_file", type=str, default=None, help="Optional predictions_*.npz override")
    parser.add_argument("--output_dir", type=str, default=None, help="Output figures directory")
    parser.add_argument("--threshold", type=float, default=None, help="Override damage threshold")
    parser.add_argument("--latest_n", type=int, default=5, help="Aggregate latest N runs")
    parser.add_argument("--aggregate_all", action="store_true", help="Aggregate all runs in checkpoints dir")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_ckpt_dir = os.path.join(os.path.dirname(script_dir), "checkpoints")
    checkpoints_dir = args.checkpoints_dir or default_ckpt_dir

    if args.results_file:
        results_file = args.results_file
        if args.aggregate_all:
            aggregate_files = sorted(
                glob.glob(os.path.join(checkpoints_dir, "results_*.json")), key=os.path.getmtime
            )
        else:
            aggregate_files = sorted(
                glob.glob(os.path.join(checkpoints_dir, "results_*.json")), key=os.path.getmtime
            )[-max(int(args.latest_n), 1) :]
            if not aggregate_files:
                aggregate_files = [results_file]
    else:
        results_file, aggregate_files = _choose_result_files(
            checkpoints_dir=checkpoints_dir,
            latest_n=args.latest_n,
            aggregate_all=args.aggregate_all,
        )

    print(f"Using results file: {results_file}")
    print(f"Aggregating {len(aggregate_files)} run(s)")

    report_dir = visualize_results(
        results_file=results_file,
        prediction_file=args.prediction_file,
        output_dir=args.output_dir,
        aggregate_files=aggregate_files,
        threshold=args.threshold,
    )
    print(f"Final report directory: {report_dir}")


if __name__ == "__main__":
    main()
