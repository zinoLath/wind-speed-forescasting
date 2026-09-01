"""Loading and merging of the pipeline configuration.

``pipeline.default.json`` holds the defaults and ``pipeline.json`` the explicit
user configuration. The user file is merged over the defaults, so it only needs
to declare the values the user actually wants to change.

Values shared by several stages (seed, window sizes, target column, loss, ...)
live in the ``common`` section and are inherited by every stage section whose
value is not explicitly set. Stage-level values always win over ``common``.
"""

import json
from copy import deepcopy
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PIPELINE_DIR / "pipeline.default.json"
USER_CONFIG_PATH = PIPELINE_DIR / "pipeline.json"

# Top-level sections that are not stages and therefore do not inherit from
# ``common``.
NON_STAGE_SECTIONS = ("paths", "gpu", "common")


def _deep_merge(base, override):
    """Recursively merge *override* into *base*; override always wins."""
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _inherit_common(config):
    """Fill each stage section with values from the shared ``common`` section.

    Resolution happens once at load time, so stages simply read their keys
    (``cfg["seed"]``, ...) without knowing where the value came from.
    """
    common = config.pop("common", {})
    for name, section in config.items():
        if name not in NON_STAGE_SECTIONS and isinstance(section, dict):
            config[name] = {**common, **section}
    return config


def load_config(path=None):
    """Load, merge and resolve the pipeline configuration.

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
    return _inherit_common(config)