"""Loading and merging of the pipeline configuration.

``pipeline.default.json`` holds the defaults and ``pipeline.json`` the explicit
user configuration. The user file is merged over the defaults, so it only needs
to declare the values the user actually wants to change.
"""

import json
from copy import deepcopy
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PIPELINE_DIR / "pipeline.default.json"
USER_CONFIG_PATH = PIPELINE_DIR / "pipeline.json"


def _deep_merge(base, override):
    """Recursively merge *override* into *base*; override always wins."""
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path=None):
    """Load and merge the pipeline configuration.

    Args:
        path: optional path to a user config file. When omitted,
            ``pipeline/pipeline.json`` is used if it exists.
    """
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as handle:
        config = json.load(handle)

    user_path = Path(path) if path else USER_CONFIG_PATH
    if user_path.is_file():
        with open(user_path, encoding="utf-8") as handle:
            config = _deep_merge(config, json.load(handle))
    return config