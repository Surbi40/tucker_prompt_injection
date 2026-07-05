import json
import random
import numpy as np
from pathlib import Path


def get_project_root():
    """
    Returns the project root directory (tucker_prompt_injection/),
    assuming this file lives at src/utils/config.py.
    """
    return Path(__file__).resolve().parents[2]


def load_config():
    """
    Loads configs/experiment_config.json from the project root.
    """
    config_path = get_project_root() / "configs" / "experiment_config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def set_seed(seed):
    """
    Sets random seeds for reproducibility across random and numpy.
    """
    random.seed(seed)
    np.random.seed(seed)