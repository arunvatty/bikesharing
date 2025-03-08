"""
Core configuration functionality.
"""

import os
from pathlib import Path
import yaml
from typing import Dict, Any

# Project directories
ROOT = Path(__file__).resolve().parent.parent


def get_config_path() -> Path:
    """Get the configuration file path."""
    return ROOT / "config.yml"


def load_config() -> Dict[str, Any]:
    """Load the configuration from config.yml."""
    with open(get_config_path(), "r") as file:
        config = yaml.safe_load(file)
    
    return config