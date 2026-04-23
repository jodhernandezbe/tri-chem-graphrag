# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Configuration management module.

This module provides the ConfigManager class for loading and managing
configuration data from YAML files.

Author: Jose D. Hernandez-Betancur
Date: 2026-04-23

"""

import pathlib
from typing import Any, Dict, Optional

import yaml

CURRENT_DIR = pathlib.Path(__file__).parent


class ConfigManager:
    """Singleton configuration manager for loading YAML configuration files."""

    def __new__(cls):
        """Create a new instance using singleton pattern.

        Returns:
            ConfigManager: The singleton instance.

        """
        if not hasattr(cls, "instance"):
            cls.instance = super(ConfigManager, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self._config: Optional[Dict[str, Any]] = None

    @property
    def config(self) -> Dict[str, Any]:
        """Load the configuration from the config.yaml file.

        Returns:
            Dict[str, Any]: The configuration.

        """
        if self._config is None:
            config_path = CURRENT_DIR / "config" / "config.yaml"
            try:
                with open(config_path, "r") as file:
                    loaded_config = yaml.safe_load(file)
                    if not isinstance(loaded_config, dict):
                        raise ValueError("Config file is not a dictionary")
                    self._config = loaded_config
            except FileNotFoundError:
                raise FileNotFoundError(f"Config file not found at {config_path}")
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error loading config file: {e}")
        return self._config
