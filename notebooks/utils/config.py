import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_notebooks_dir = os.path.dirname(_current_dir)
ROOT_DIR = os.path.dirname(_notebooks_dir)
DATASET_PATH = os.path.join(ROOT_DIR, "dataset", "train.npz")
STRUCTURE_PATH = os.path.join(ROOT_DIR, "structure.yaml")
CONFIG_PATH = os.path.join(ROOT_DIR, "dataset_config.yaml")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "Deep_learning", "checkpoints")

DT = 0.01
TOTAL_TIME = 2.0

PLOT_STYLE = "paper"

RESULTS_DIR = CHECKPOINT_DIR

LATEST_RESULTS = os.path.join(CHECKPOINT_DIR, "results_combined_latest_split_20260223_232547.json")
LATEST_PREDICTIONS = os.path.join(CHECKPOINT_DIR, "predictions_combined_latest_split_20260223_232547.npz")
