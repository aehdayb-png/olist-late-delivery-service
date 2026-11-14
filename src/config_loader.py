import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path=None):
    """Load YAML config. APP_CONFIG env var overrides the default (useful in Docker)."""
    path = Path(path or os.getenv("APP_CONFIG", DEFAULT_CONFIG))
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path):
    """Turn a config-relative path into an absolute one from the project root."""
    return PROJECT_ROOT / relative_path