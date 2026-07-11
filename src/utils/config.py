import json
import random
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")  # loads GROQ_API_KEY


def get_project_root():
    """
    Returns the project root directory (tucker_prompt_injection/),
    assuming this file lives at src/utils/config.py.
    """
    return _ROOT


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